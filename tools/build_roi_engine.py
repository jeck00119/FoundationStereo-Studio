"""Build (and smoke-test) a TensorRT engine for an ROI-sized run.

Engines are keyed by (padded height, padded width, iters, max_disp), so the ROI
workflow needs one built once per crop size. This drives the real backend — the
same path the app takes on its first run at a new size — so whatever it produces
is exactly what the app will load.

    .venv/bin/python tools/build_roi_engine.py --w 1024 --h 1024 [--max-disp 64]
                                               [--iters 8] [--opt-level 2]

--opt-level 2 trades a few percent of engine speed for a much shorter build:
right for a feasibility probe, wrong for the engine a study will actually run on
(leave it unset for that). Pure diagnostics: touches no QSettings, writes only
the engine + its logs beside the checkpoint.
"""
import argparse
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
sys.path.insert(0, REPO)

CKPT = os.path.join(os.path.dirname(REPO), "Fast-FoundationStereo",
                    "weights", "hf-c-release", "model_best_bp2_serialize.pth")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a TRT engine for an ROI size")
    ap.add_argument("--w", type=int, default=1024, help="ROI width (px, /32)")
    ap.add_argument("--h", type=int, default=1024, help="ROI height (px, /32)")
    ap.add_argument("--max-disp", type=int, default=64, dest="max_disp")
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--opt-level", default="", dest="opt",
                    help="FS_TRT_OPT_LEVEL (empty = TensorRT's default, best engine)")
    ap.add_argument("--ckpt", default=CKPT)
    args = ap.parse_args()

    if args.opt:
        os.environ["FS_TRT_OPT_LEVEL"] = str(args.opt)
    if args.w % 32 or args.h % 32:
        print(f"warning: {args.w}x{args.h} is not a multiple of 32 — the network "
              f"pads up, so the engine will be built for the padded size instead")

    from studio.backends import load_backend
    from studio.dtypes import StereoParams

    t0 = time.time()
    def say(msg):
        print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

    say(f"engine target: {args.w}x{args.h}  iters {args.iters}  "
        f"max_disp {args.max_disp}  opt {args.opt or 'default'}")
    # NB: no "fail"/"error" words in the progress prose — these lines are what a
    # log watcher greps to decide whether the build succeeded, and a reference
    # figure phrased as a failure reads as one.
    say(f"cost volume ~{args.h*args.w*args.max_disp/1e6:.0f}M "
        f"(reference: 108M builds on this device, 179M does not)")

    backend = load_backend("fast_foundation_stereo_trt")
    backend.load(args.ckpt, None, progress=say)

    params = StereoParams(scale=1.0)
    params.model_params = {"valid_iters": args.iters, "max_disp": args.max_disp}
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (args.h, args.w, 3), dtype=np.uint8)

    say("first pass — builds the engine if it is not cached…")
    res = backend.disparity(img, img.copy(), params)      # triggers the build
    say(f"engine ready. disparity {res.disp.shape} "
        f"valid {100.0*float((res.disp>0).mean()):.1f}%")

    for i in range(3):                                     # warm timing
        t = time.time()
        backend.disparity(img, img.copy(), params)
        say(f"warm pass {i+1}: {time.time()-t:.3f} s")

    path = backend._engine_path(args.h, args.w, args.iters, args.max_disp)
    say(f"engine: {path}  ({os.path.getsize(path)/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
