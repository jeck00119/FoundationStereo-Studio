"""Engine-level Fast-FoundationStereo benchmark for the Orin Nano bring-up.

Loads the backend exactly the way the engine child does (registry.load_backend
→ adapter → run_inference), with no GUI in the process, and measures what the
device actually delivers per (Input scale, Max disparity) config:

    python tools/bench_orin.py warmup   # demo pair at the app-default config —
                                        # times the FIRST triton/inductor compile
                                        # and leaves the caches hot
    python tools/bench_orin.py sweep    # rig-sized pair (demo upscaled to
                                        # 2664×2304) across candidate configs —
                                        # cold + warm timings, GPU peaks, system
                                        # memory dip, disparity validity

Pure diagnostics: touches no QSettings, writes nothing outside stdout.
"""
import json
import os
import sys
import threading
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
sys.path.insert(0, REPO)

CKPT = os.path.join(os.path.dirname(REPO), "Fast-FoundationStereo",
                    "weights", "hf-c-release", "model_best_bp2_serialize.pth")


class MemWatch:
    """Track the MemAvailable minimum while a block runs."""

    def __init__(self):
        self.min_mb = float("inf")
        self._stop = threading.Event()
        self._thr = threading.Thread(target=self._run, daemon=True)

    def _read(self):
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024.0
        return float("inf")

    def _run(self):
        while not self._stop.is_set():
            self.min_mb = min(self.min_mb, self._read())
            time.sleep(0.2)

    def __enter__(self):
        self.start_mb = self._read()
        self.min_mb = self.start_mb
        self._thr.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thr.join(timeout=2)
        self.min_mb = min(self.min_mb, self._read())


def load_pair(rig_size=False):
    import cv2
    import imageio.v2 as iio
    left = iio.imread(os.path.join(REPO, "assets", "left.png"))[..., :3]
    right = iio.imread(os.path.join(REPO, "assets", "right.png"))[..., :3]
    if rig_size:
        # the Windows rig's working pair size — content is interpolated demo
        # scene, but pixels × disparity (the memory/time drivers) are exact
        left = cv2.resize(left, (2664, 2304), interpolation=cv2.INTER_CUBIC)
        right = cv2.resize(right, (2664, 2304), interpolation=cv2.INTER_CUBIC)
    return left, right


def one_run(backend, left, right, scale, max_disp, tag):
    import torch
    from studio.dtypes import StereoParams
    from studio.infer import run_inference

    params = StereoParams(scale=scale)
    params.model_params = {"valid_iters": 8, "max_disp": int(max_disp),
                           "hierarchical": False, "low_memory": False}
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    out = {"tag": tag, "scale": scale, "max_disp": int(max_disp)}
    try:
        with MemWatch() as mw:
            t0 = time.time()
            res = run_inference(backend, left, right, params)
            out["s"] = round(time.time() - t0, 2)
        disp = res.disp
        out["work_res"] = f"{disp.shape[1]}x{disp.shape[0]}"
        out["valid_pct"] = round(float((disp > 0).mean() * 100), 1)
        out["disp_p99"] = round(float(np.percentile(disp[disp > 0], 99)), 1)
        out["gpu_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        out["gpu_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 2**30, 2)
        out["sys_dip_mb"] = round(mw.start_mb - mw.min_mb)
        out["sys_min_avail_mb"] = round(mw.min_mb)
    except Exception as e:
        import traceback
        out["error"] = str(e).splitlines()[0][:160]
        out["trace_tail"] = " | ".join(traceback.format_exc().splitlines()[-6:])
    print("RESULT " + json.dumps(out), flush=True)
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "warmup"
    from studio.backends.registry import load_backend

    t0 = time.time()
    backend = load_backend("fast_foundation_stereo")
    backend.load(CKPT)
    print(f"LOAD {time.time() - t0:.1f}s  ckpt={os.path.basename(os.path.dirname(CKPT))}",
          flush=True)

    if mode == "one":       # one rig-sized config: bench_orin.py one <scale> <max_disp>
        left, right = load_pair(rig_size=True)
        sc, md = float(sys.argv[2]), int(sys.argv[3])
        one_run(backend, left, right, sc, md, "one · cold")
        one_run(backend, left, right, sc, md, "one · warm")
    elif mode == "warmup":
        left, right = load_pair(rig_size=False)      # 960×540 demo, native
        # app-default config == what verify_full_process.py will run
        one_run(backend, left, right, 0.5, 192, "compile(cold)")
        one_run(backend, left, right, 0.5, 192, "warm")
        one_run(backend, left, right, 0.5, 192, "warm2")
    else:
        left, right = load_pair(rig_size=True)       # 2664×2304 rig-sized
        configs = [
            (0.25, 160, "candidate"),
            (0.30, 192, "candidate"),
            (0.35, 192, "x-ref 3060 (932x806@192: 1.49GB/0.35s there)"),
            (0.40, 256, "candidate"),
            (0.50, 416, "windows-settled profile"),
        ]
        for scale, md, tag in configs:
            r1 = one_run(backend, left, right, scale, md, tag + " · cold")
            if "error" in r1:
                continue                    # OOM at this size — larger ones only get worse,
                                            # but keep going: max_disp differs per config
            one_run(backend, left, right, scale, md, tag + " · warm")
    print("BENCH DONE", flush=True)


if __name__ == "__main__":
    main()
