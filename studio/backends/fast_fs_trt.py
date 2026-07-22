"""Fast-FoundationStereo · TensorRT backend (Jetson-first).

Wraps upstream's single-ONNX export (``scripts/make_single_onnx.py``) and a
minimal TensorRT-10 runner modeled on ``scripts/run_demo_single_trt.py``
(both Apache-2.0). Engines are size-specific, so they are built LAZILY per
(padded working size, valid_iters, max_disp) the first time a run needs one —
several minutes, once per config per device — and cached on disk next to the
checkpoint (``weights/<run>/trt/``). Unlike the torch backend's triton
warm-up, the cache survives reboots: after the one-time build this backend
starts cold in seconds.

The export strips ImageNet normalization out of the graph (see upstream's
docstring), so this runner owes ``(pixel - mean) / std`` to every input —
constants below are upstream's, in 0-255 scale.
"""
from __future__ import annotations

import os

# The ONNX export runs a real traced forward through the EAGER cost volume —
# dynamo/inductor must stay out of it (mirrors upstream make_single_onnx.py).
# Module level on purpose: it has to land before torch's first import in this
# engine child, and this backend never wants compilation anyway.
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import subprocess
import sys
import time
from shutil import which
from typing import Optional

import numpy as np

from ..dtypes import DisparityResult, Progress, StereoParams, tick
from .base import StereoBackend
from .fast_foundation_stereo import backfill_pickled_args

_HERE = os.path.dirname(os.path.abspath(__file__))
_FS_REPO = os.path.dirname(os.path.dirname(_HERE))
FAST_REPO = os.path.join(os.path.dirname(_FS_REPO), "Fast-FoundationStereo")

# upstream's ImageNet constants (0-255 scale) — the graph no longer applies
# them, the runner must (make_single_onnx.py strips normalize_image).
_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def _log_last_line(path: str, maxread: int = 8192) -> str:
    """Last non-empty line of a growing log, cheaply (tail of the file)."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - maxread))
            lines = [ln for ln in f.read().decode("utf-8", "replace").splitlines()
                     if ln.strip()]
        return lines[-1][-140:] if lines else "(no output yet)"
    except OSError:
        return "(log unreadable)"


def _log_tail(path: str, n: int = 15) -> str:
    try:
        with open(path, "r", errors="replace") as f:
            return "\n".join(f.read().splitlines()[-n:])
    except OSError:
        return "(log unreadable)"


def _proc_rss_mb(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return 0


def _find_trtexec() -> Optional[str]:
    # JetPack installs trtexec outside PATH; check its home first.
    fixed = "/usr/src/tensorrt/bin/trtexec"
    if os.access(fixed, os.X_OK):
        return fixed
    return which("trtexec")


def _trt_version() -> str:
    import tensorrt as trt
    return ".".join(trt.__version__.split(".")[:2])


class _EngineRunner:
    """One deserialized engine + execution context, torch tensors as buffers.

    Distilled from upstream scripts/run_demo_single_trt.py
    (SingleEngineTrtRunner, Apache-2.0): set_input_shape → allocate outputs →
    set_tensor_address → execute_async_v3 on torch's current stream.
    """

    def __init__(self, engine_path: str, torch_mod) -> None:
        import tensorrt as trt
        self._trt = trt
        self._torch = torch_mod
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(
                f"Could not deserialize {os.path.basename(engine_path)} — it was "
                f"likely built by a different TensorRT version than the installed "
                f"{trt.__version__}. Delete the file and rerun; it rebuilds.")
        self.context = self.engine.create_execution_context()

    def _dtype(self, dt):
        trt, torch = self._trt, self._torch
        table = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16,
                 trt.DataType.INT32: torch.int32, trt.DataType.INT8: torch.int8,
                 trt.DataType.BOOL: torch.bool}
        bf16 = getattr(trt.DataType, "BF16", None)
        if bf16 is not None:
            table[bf16] = torch.bfloat16
        if dt not in table:
            raise RuntimeError(f"unsupported TRT dtype {dt}")
        return table[dt]

    def infer(self, inputs: dict) -> dict:
        trt, torch = self._trt, self._torch
        for name, t in inputs.items():
            want = self._dtype(self.engine.get_tensor_dtype(name))
            if t.dtype != want:
                t = t.to(want)
            inputs[name] = t.contiguous()
            self.context.set_input_shape(name, tuple(inputs[name].shape))
        outputs = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT:
                continue
            shape = tuple(self.context.get_tensor_shape(name))
            outputs[name] = torch.empty(
                shape, device="cuda", dtype=self._dtype(self.engine.get_tensor_dtype(name)))
        for name, t in {**inputs, **outputs}.items():
            self.context.set_tensor_address(name, int(t.data_ptr()))
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 returned False")
        return outputs


class FastFsTrtBackend(StereoBackend):
    def __init__(self) -> None:
        self.ckpt_path: Optional[str] = None
        self._runner: Optional[_EngineRunner] = None
        self._runner_key = None
        self._padder_cls = None
        self._progress: Progress = None   # the child's socket-progress closure,
                                          # kept so lazy engine builds can narrate

    # ------------------------------------------------------------------ load
    def load(self, ckpt_path: str, params: Optional[StereoParams] = None,
             progress: Progress = None) -> None:
        if FAST_REPO not in sys.path:
            sys.path.insert(0, FAST_REPO)
        import torch
        try:
            import tensorrt  # noqa: F401 — fail HERE with a useful message
        except ImportError as e:
            raise RuntimeError(
                "TensorRT python bindings are not visible in this environment. "
                "On Jetson, re-run ./setup_jetson.sh (it links the system "
                "bindings into the venv); this backend is Jetson/Linux-only."
            ) from e
        if not torch.cuda.is_available():
            raise RuntimeError("The TensorRT backend needs the CUDA device.")
        from core.utils.utils import InputPadder
        self._torch = torch
        self._device = "cuda"
        self._padder_cls = InputPadder
        self._progress = progress
        torch.autograd.set_grad_enabled(False)
        self.ckpt_path = ckpt_path
        tick(progress, "TensorRT backend ready — engines build lazily per "
                       "input size (first run per size takes minutes, then "
                       "it is cached on disk for good).")

    @property
    def is_loaded(self) -> bool:
        return self.ckpt_path is not None

    def unload(self) -> None:
        self._runner = None
        self._runner_key = None
        super().unload()

    # --------------------------------------------------- engine provisioning
    def _engine_path(self, hp: int, wp: int, iters: int, max_disp: int) -> str:
        d = os.path.join(os.path.dirname(self.ckpt_path), "trt")
        return os.path.join(
            d, f"fp16_{hp}x{wp}_i{iters}_d{max_disp}_trt{_trt_version()}.engine")

    def _export_onnx(self, hp: int, wp: int, iters: int, max_disp: int,
                     onnx_path: str) -> None:
        """Run the export in a SUBPROCESS that exits before the engine build.

        Two reboots'-worth of lessons in one method: the tracer's memory spike
        (>5 GB at rig sizes) must be (a) CPU-side — on the GPU it is
        nvmap-pinned and unswappable, which starved PID 1 until the Tegra
        watchdog hard-reset the box — and (b) in a process that TERMINATES,
        because even swappable spike memory leaves the system thrashing for
        minutes, and trtexec launched into that storm found the unified-memory
        GPU unable to hand out even 26 MB per tactic (Error 10 at the stem,
        2026-07-22 14:48). A dead process gives every byte back at once.
        """
        log_path = onnx_path + ".export.log"
        tick(self._progress, f"TRT: exporting ONNX at {wp}×{hp} in a CPU "
                             f"subprocess (one-time; takes minutes; "
                             f"log: {log_path})…")
        cmd = [sys.executable, "-m", "studio.backends.fast_fs_trt", "export",
               self.ckpt_path, str(hp), str(wp), str(iters), str(max_disp),
               onnx_path]
        rc = self._run_narrated(cmd, log_path, cwd=_FS_REPO, label="export",
                                rss_of_child=True)
        if rc != 0 or not os.path.isfile(onnx_path):
            raise RuntimeError(
                f"ONNX export subprocess failed — tail:\n{_log_tail(log_path)}")

    def _wait_for_memory(self, min_avail_mb: int = 3000,
                         timeout_s: int = 180) -> None:
        """Block until the post-export swap storm drains before trtexec."""
        t0 = time.time()
        avail = 0
        while time.time() - t0 < timeout_s:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        avail = int(line.split()[1]) // 1024
                        break
            if avail >= min_avail_mb:
                return
            tick(self._progress, f"TRT: waiting for memory to settle "
                                 f"({avail} MB free, want {min_avail_mb})…")
            time.sleep(3)
        # proceed regardless — trtexec's own error will be definitive

    def _build_engine(self, onnx_path: str, engine_path: str) -> None:
        exe = _find_trtexec()
        if exe is None:
            raise RuntimeError(
                "trtexec not found (looked in /usr/src/tensorrt/bin and PATH) — "
                "is the JetPack TensorRT package installed?")
        # workspace cap: on 8 GB unified memory an uncapped builder times
        # tactics against ALL free device memory and can OOM the box at rig
        # sizes; 3 GiB keeps builds safe and costs at most a marginal tactic.
        # BARE NUMBER = MiB. Writing "3072MiB" hands trtexec a 3072-BYTE pool
        # (its banner shows 'workspace: 0.00293 MiB') and every tactic dies
        # with 'insufficient memory' in seconds — two chain runs were lost to
        # that suffix before trtexec's config echo gave it away.
        cmd = [exe, f"--onnx={onnx_path}", f"--saveEngine={engine_path}",
               "--fp16", "--memPoolSize=workspace:3072"]
        # Rig-size builds run 1.5-2 h at TRT's default optimization level 3.
        # FS_TRT_OPT_LEVEL=2 (or 1) trades a few percent of engine speed for a
        # drastically shorter build — the right call for feasibility probes;
        # rebuild the winning config at full level by unsetting it (the engine
        # filename does not encode the level: delete the engine to rebuild).
        opt = os.environ.get("FS_TRT_OPT_LEVEL", "").strip()
        if opt:
            cmd.append(f"--builderOptimizationLevel={opt}")
        log_path = engine_path + ".build.log"
        tick(self._progress, f"TRT: building the FP16 engine (opt level "
                             f"{opt or 'default'}) — one-time for this size; "
                             f"live log: {log_path}")
        t0 = time.time()
        rc = self._run_narrated(cmd, log_path, cwd=None, label="build")
        if rc != 0 or not os.path.isfile(engine_path):
            raise RuntimeError(f"trtexec failed after {time.time() - t0:.0f}s — "
                               f"log tail:\n{_log_tail(log_path)}")
        tick(self._progress, f"TRT: engine built in {time.time() - t0:.0f}s "
                             f"(build log kept beside it).")

    def _run_narrated(self, cmd, log_path: str, cwd, label: str,
                      rss_of_child: bool = False) -> int:
        """Run a long subprocess with its output streamed to a tail-able file
        and a progress tick every minute — hours of silence are how you end up
        ssh-ing into a box wondering if anything is alive."""
        with open(log_path, "w") as lf:
            lf.write("$ " + " ".join(cmd) + "\n")
            lf.flush()
            p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                 cwd=cwd, text=True)
            t0 = time.time()
            next_beat = 60.0
            while p.poll() is None:
                time.sleep(5)
                el = time.time() - t0
                if el >= next_beat:
                    next_beat = el + 60.0
                    note = _log_last_line(log_path)
                    if rss_of_child:
                        note = f"rss {_proc_rss_mb(p.pid)} MB · {note}"
                    tick(self._progress,
                         f"TRT {label}: {el / 60:.0f} min — {note}")
            return p.returncode

    def _ensure_runner(self, hp: int, wp: int, iters: int, max_disp: int) -> None:
        key = (hp, wp, iters, max_disp)
        if self._runner is not None and self._runner_key == key:
            return
        engine_path = self._engine_path(hp, wp, iters, max_disp)
        if not os.path.isfile(engine_path):
            os.makedirs(os.path.dirname(engine_path), exist_ok=True)
            onnx_path = engine_path.replace(".engine", ".onnx")
            try:
                # a leftover .onnx from a failed build is valid (its filename
                # pins the config) — reuse it instead of re-paying the
                # multi-minute export on every build retry
                if not os.path.isfile(onnx_path):
                    self._export_onnx(hp, wp, iters, max_disp, onnx_path)
                self._wait_for_memory()
                self._build_engine(onnx_path, engine_path)
                # success: the .onnx was only scaffolding for trtexec
                os.remove(onnx_path)
            finally:
                # never leave a half-written engine that would mask the error
                if os.path.isfile(engine_path) and os.path.getsize(engine_path) == 0:
                    os.remove(engine_path)
        # one engine resident at a time — each holds weights + workspace on an
        # 8 GB unified-memory device
        self._runner = None
        self._torch.cuda.empty_cache()
        tick(self._progress, "TRT: loading engine…")
        self._runner = _EngineRunner(engine_path, self._torch)
        self._runner_key = key

    # ------------------------------------------------------------- disparity
    def disparity(self, img0: np.ndarray, img1: np.ndarray,
                  params: StereoParams) -> DisparityResult:
        torch = self._torch
        mp = params.model_params
        iters = int(mp.get("valid_iters", 8))
        max_disp = int(mp.get("max_disp", 192))

        H, W = img0.shape[:2]
        t0 = torch.as_tensor((img0[..., :3].astype(np.float32) - _MEAN) / _STD)
        t1 = torch.as_tensor((img1[..., :3].astype(np.float32) - _MEAN) / _STD)
        t0 = t0.to("cuda")[None].permute(0, 3, 1, 2)
        t1 = t1.to("cuda")[None].permute(0, 3, 1, 2)
        padder = self._padder_cls(t0.shape, divis_by=32, force_square=False)
        t0, t1 = padder.pad(t0, t1)
        hp, wp = int(t0.shape[-2]), int(t0.shape[-1])

        self._ensure_runner(hp, wp, iters, max_disp)
        out = self._runner.infer({"left_image": t0.contiguous(),
                                  "right_image": t1.contiguous()})
        d = padder.unpad(out["disparity"].float())
        torch.cuda.synchronize()
        # clip(0) for the same reason as the torch adapter: small negative
        # disparities skew the colour range / exported .npy
        disp = d.data.cpu().numpy().reshape(H, W).clip(0, None).astype(np.float32)
        return DisparityResult(disp=disp)

    # TRT allocates the engine's activation memory OUTSIDE torch's allocator,
    # so the inherited reserved-bytes figure alone would under-report what the
    # device really gave up — add what the engine declares it holds.
    def peak_vram_gb(self) -> float:
        base = super().peak_vram_gb()
        if self._runner is not None:
            base += float(getattr(self._runner.engine, "device_memory_size", 0)) / 2 ** 30
        return base


def make() -> StereoBackend:
    return FastFsTrtBackend()


def _do_export(ckpt_path: str, hp: int, wp: int, iters: int, max_disp: int,
               onnx_path: str) -> None:
    """The actual export — runs in the throwaway subprocess (see
    _export_onnx). CPU-only by design; must never touch CUDA."""
    import torch
    if FAST_REPO not in sys.path:
        sys.path.insert(0, FAST_REPO)
    import core.foundation_stereo as _fs_module
    from scripts.make_single_onnx import (
        FastFoundationStereoSingleOnnx, _build_concat_volume_onnx,
        _build_gwc_volume_onnx)

    torch.autograd.set_grad_enabled(False)
    model = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    backfill_pickled_args(model)              # HF drop predates args.normalize
    model.args.max_disp = int(max_disp)
    model.args.valid_iters = int(iters)
    model.args.mixed_precision = False
    model = model.cpu().eval()
    wrapper = FastFoundationStereoSingleOnnx(model).cpu().eval()

    # the same monkey-patches upstream's __main__ applies before tracing
    _fs_module.normalize_image = lambda img: img
    _fs_module.build_gwc_volume_optimized_pytorch1 = _build_gwc_volume_onnx
    _fs_module.build_concat_volume_optimized_pytorch1 = _build_concat_volume_onnx

    left = torch.randn(1, 3, hp, wp, device="cpu")
    right = torch.randn(1, 3, hp, wp, device="cpu")
    kwargs = dict(opset_version=17,
                  input_names=["left_image", "right_image"],
                  output_names=["disparity"],
                  do_constant_folding=True)
    try:
        # torch ≥2.9 defaults to the dynamo exporter; upstream's export is
        # written for the tracer. Older torch lacks the kwarg — retry bare.
        torch.onnx.export(wrapper, (left, right), onnx_path,
                          dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(wrapper, (left, right), onnx_path, **kwargs)


if __name__ == "__main__":
    if len(sys.argv) == 8 and sys.argv[1] == "export":
        _do_export(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]),
                   int(sys.argv[5]), int(sys.argv[6]), sys.argv[7])
        sys.exit(0)
    print("usage: python -m studio.backends.fast_fs_trt export "
          "<ckpt> <hp> <wp> <iters> <max_disp> <onnx_path>", file=sys.stderr)
    sys.exit(2)
