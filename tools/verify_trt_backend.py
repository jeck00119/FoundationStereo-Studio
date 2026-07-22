"""Accuracy + timing gate for the TensorRT backend vs the PyTorch backend.

Runs the SAME pair at the SAME config through both backends and compares
disparities. torch and trt run in separate subprocesses on purpose: the TRT
adapter disables dynamo process-wide for its ONNX export, which would silently
demote the torch backend to eager in a shared process.

    .venv/bin/python tools/verify_trt_backend.py all [--scale 0.5]
        [--max_disp 192] [--iters 8] [--rig]

Gate (fp16-typical): median |Δ| < 0.1 px AND p95 |Δ| < 0.5 px on valid
pixels, no NaN/Inf, valid%% within 1 point of each other. Exit 0 = pass.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
sys.path.insert(0, REPO)

CKPT = os.path.join(os.path.dirname(REPO), "Fast-FoundationStereo",
                    "weights", "hf-c-release", "model_best_bp2_serialize.pth")
OUT = os.path.join(os.environ.get("TMPDIR", "/tmp"), "fs_trt_verify")


def load_pair(rig: bool):
    import cv2
    import imageio.v2 as iio
    left = iio.imread(os.path.join(REPO, "assets", "left.png"))[..., :3]
    right = iio.imread(os.path.join(REPO, "assets", "right.png"))[..., :3]
    if rig:
        left = cv2.resize(left, (2664, 2304), interpolation=cv2.INTER_CUBIC)
        right = cv2.resize(right, (2664, 2304), interpolation=cv2.INTER_CUBIC)
    return left, right


def run_backend(kind: str, args) -> None:
    from studio.backends.registry import load_backend
    from studio.dtypes import StereoParams
    from studio.infer import run_inference

    key = {"torch": "fast_foundation_stereo", "trt": "fast_foundation_stereo_trt"}[kind]
    backend = load_backend(key)
    t0 = time.time()
    backend.load(CKPT, progress=lambda m: print(f"  [{kind}] {m}", flush=True))
    load_s = time.time() - t0

    left, right = load_pair(args.rig)
    params = StereoParams(scale=args.scale)
    params.model_params = {"valid_iters": args.iters, "max_disp": args.max_disp,
                           "hierarchical": False, "low_memory": False}
    times = []
    for i in range(3):                       # first run may compile/build
        t0 = time.time()
        res = run_inference(backend, left, right, params)
        times.append(time.time() - t0)
    np.save(os.path.join(OUT, f"disp_{kind}.npy"), res.disp)
    meta = {"kind": kind, "load_s": round(load_s, 2),
            "first_s": round(times[0], 2),
            "warm_s": round(min(times[1:]), 2),
            "shape": list(res.disp.shape)}
    with open(os.path.join(OUT, f"meta_{kind}.json"), "w") as f:
        json.dump(meta, f)
    print("META " + json.dumps(meta), flush=True)


def compare() -> int:
    a = np.load(os.path.join(OUT, "disp_torch.npy"))
    b = np.load(os.path.join(OUT, "disp_trt.npy"))
    if a.shape != b.shape:
        print(f"FAIL shape mismatch {a.shape} vs {b.shape}")
        return 1
    bad = int(np.isnan(b).sum() + np.isinf(b).sum())
    va, vb = a > 0, b > 0
    both = va & vb
    d = np.abs(a[both] - b[both])
    med, p95, mx = (float(np.median(d)), float(np.percentile(d, 95)),
                    float(d.max()))
    valid_gap = abs(float(va.mean()) - float(vb.mean())) * 100
    print(f"valid: torch {va.mean():.1%}  trt {vb.mean():.1%}  (gap {valid_gap:.2f} pt)")
    print(f"|Δdisp| on {both.sum()} px: median {med:.4f}  p95 {p95:.4f}  max {mx:.3f}")
    print(f"NaN/Inf in trt: {bad}")
    ok = med < 0.1 and p95 < 0.5 and bad == 0 and valid_gap < 1.0
    for m in (json.load(open(os.path.join(OUT, f"meta_{k}.json"))) for k in ("torch", "trt")):
        print(f"{m['kind']:>5}: load {m['load_s']}s  first {m['first_s']}s  warm {m['warm_s']}s")
    print("GATE " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["torch", "trt", "compare", "all"])
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--max_disp", type=int, default=192)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--rig", action="store_true",
                    help="upscale the demo pair to the 2664x2304 rig size")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    if args.mode in ("torch", "trt"):
        run_backend(args.mode, args)
    elif args.mode == "compare":
        sys.exit(compare())
    else:                                    # all: two clean subprocesses, then compare
        base = [sys.executable, os.path.abspath(__file__)]
        extra = [f"--scale={args.scale}", f"--max_disp={args.max_disp}",
                 f"--iters={args.iters}"] + (["--rig"] if args.rig else [])
        for kind in ("torch", "trt"):
            print(f"=== {kind} pass", flush=True)
            r = subprocess.run(base + [kind] + extra)
            if r.returncode != 0:
                print(f"{kind} pass failed rc={r.returncode}")
                sys.exit(r.returncode)
        sys.exit(compare())
