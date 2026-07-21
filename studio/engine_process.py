"""Engine child process — owns the FoundationStereo model and the CUDA context.

Launched as:  python -m studio.engine_process <port> <authkey_hex>

Runs inference in its OWN main thread (like a normal script), which is why it
never hits the QThread + CUDA hang. Communicates with the GUI over a
multiprocessing.connection socket: it receives (cmd, *args) tuples and sends
(event, payload) tuples back.
"""
from __future__ import annotations

import sys
import traceback
from multiprocessing.connection import Client


def _explain(exc: Exception) -> str:
    """Turn a GPU-memory failure into something actionable.

    Matching on text (not the exception type) on purpose: the same condition
    surfaces as torch.cuda.OutOfMemoryError, as a cuBLAS alloc failure, or — per
    FoundationStereo's own README FAQ — as a cuDNN 'CUDNN_STATUS_NOT_SUPPORTED',
    which is what a full-resolution pair actually raises on a 12 GB card.
    """
    s = str(exc)
    low = s.lower()
    if ("out of memory" in low or "cudnn_status_not_supported" in low
            or "cublas_status_alloc_failed" in low):
        return (
            "The GPU ran out of memory for this run.\n\n"
            "Lower Scale (e.g. 0.5 → 0.35), or lower Max disparity if the model "
            "has one — GPU memory scales with pixels × max-disparity, so either "
            "knob cuts it roughly proportionally.\n\n"
            "(FoundationStereo's FAQ notes the cuDNN 'CUDNN_STATUS_NOT_SUPPORTED' "
            "error is usually an out-of-memory in disguise.)\n\n"
            f"Original error: {s}"
        )
    return s


def main() -> None:
    # pythonw.exe gives this child no console -> sys.stdout/stderr are None and
    # torch.hub's stderr.write() would crash. Fix before importing torch.
    from studio._streams import ensure_streams
    ensure_streams("fs_studio_engine")

    # Windows: libraries in this worker spawn console child processes — Triton's
    # ptxas.exe most visibly, once per kernel it compiles — and from a windowless
    # parent every one of them FLASHES a black console at the user mid-run. This
    # is a headless compute worker; no child of it ever needs a window.
    import os as _os
    if _os.name == "nt":
        import subprocess as _sp
        _init = _sp.Popen.__init__

        def _no_window_init(self, *a, **kw):
            kw["creationflags"] = kw.get("creationflags", 0) | _sp.CREATE_NO_WINDOW
            return _init(self, *a, **kw)

        _sp.Popen.__init__ = _no_window_init

    address = ("127.0.0.1", int(sys.argv[1]))
    authkey = bytes.fromhex(sys.argv[2])
    conn = Client(address, authkey=authkey)

    # imported after connecting so the parent's accept() returns quickly
    from studio import cloud
    from studio.backends import load_backend
    from studio.infer import run_inference

    backend = None
    last_result = None

    def progress(msg: str) -> None:
        try:
            conn.send(("progress", msg))
        except Exception:
            pass

    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            break

        try:
            cmd = msg[0]   # inside the try: a malformed frame must not kill the child
            if cmd == "load":
                _, backend_key, ckpt = msg
                backend = load_backend(backend_key)
                backend.load(ckpt, None, progress=progress)
                conn.send(("loaded", backend.device_name()))

            elif cmd == "infer":
                _, left, right, params = msg
                if backend is not None:
                    backend.reset_peak_vram()
                result = run_inference(backend, left, right, params, progress=progress)
                if backend is not None:   # what this run actually cost, for the UI
                    result.timing["peak_vram_gb"] = backend.peak_vram_gb()
                last_result = result
                conn.send(("inference", result))
                cloud_res = (cloud.build_cloud(result, params, progress=progress)
                             if params.has_calibration else None)
                # Hand the run's scratch VRAM back BEFORE announcing the cloud —
                # otherwise the allocator keeps the whole peak reserved and the card
                # reads as full until exit. Order matters: the GUI treats the cloud as
                # the end of the run and, mid-comparison, tears this child down the
                # instant it lands — putting the kill inside empty_cache(). Go idle,
                # then report.
                if backend is not None:
                    try:
                        backend.release_cache()
                    except Exception:   # noqa: BLE001
                        pass   # a cache hiccup must not report a good run as failed
                conn.send(("cloud", cloud_res))
                conn.send(("progress", "Done."))

            elif cmd == "rebuild":
                params = msg[1]
                supplied = msg[2] if len(msg) > 2 else None
                if supplied is not None:
                    # the GUI is showing a result this child didn't produce (a model
                    # comparison) — adopt it so this and later rebuilds match it
                    last_result = supplied
                cloud_res = (cloud.build_cloud(last_result, params, progress=progress)
                             if last_result is not None else None)
                conn.send(("cloud", cloud_res))
                # terminal message, like the infer path's "Done." — without it the
                # status bar stays stuck on build_cloud's last tick ("Projecting to
                # 3D…" / "Denoising cloud…") forever after every live slider tweak.
                conn.send(("progress", "Cloud updated."))

            elif cmd == "rescale":   # mm⇄m unit switch — keep the cached result in step
                factor = float(msg[1])
                if last_result is not None:
                    if last_result.depth is not None:
                        last_result.depth = last_result.depth * factor
                    last_result.baseline = float(last_result.baseline) * factor
                # no reply: the GUI rescales its own copy in parallel and doesn't wait

            elif cmd == "vram":
                conn.send(("vram", backend.vram_gb() if backend is not None else (0.0, 0.0)))

            elif cmd == "quit":
                break

            else:   # unknown command — reply so the GUI never waits forever
                conn.send(("error", f"unknown command {cmd!r}"))

        except Exception as exc:  # noqa: BLE001
            # a half-finished run can be holding gigabytes — let it go before the
            # user retries, or the retry starts already starved
            if backend is not None:
                try:
                    backend.release_cache()
                except Exception:
                    pass
            try:
                conn.send(("error", f"{_explain(exc)}\n\n{traceback.format_exc()}"))
            except Exception:
                break   # connection is gone — stop rather than spin

    try:
        conn.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
