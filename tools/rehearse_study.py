"""Dress rehearsal for a repeatability study — N real pairs, frozen geometry.

Runs the SAME chain the app's batch runs (rectify+crop -> TRT -> cloud ->
measure boxes) over a sample of real captures, with the ROI, Δ and boxes frozen
exactly as a batch freezes them. Reports per-pair readings and the spread that
IS the repeatability, plus a projected wall-clock for the full study.

    .venv/bin/python tools/rehearse_study.py --n 20 [--denoise] [--every 50]

Pure diagnostics: reads captures, writes nothing but stdout (+ optional --csv).
"""
import argparse
import glob
import os
import re
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
sys.path.insert(0, REPO)

CKPT = os.path.join(os.path.dirname(REPO), "Fast-FoundationStereo",
                    "weights", "hf-c-release", "model_best_bp2_serialize.pth")
PAIRS = "data/captures/pin_repeat"
CALIB = "data/calib/calib_provisional.json"

# the lead column, both leads, 524k px — under the device's proven ceiling
ROI = (2560, 1024, 512, 1024)
# boxes on the two leads, generous in XY so ~0.34 mm of CNC drift cannot slide
# a pin out of its own box over 1000 captures
BOXES = [("Lead A", 9.300, -3.862, 212.514),
         ("Lead B", 9.434, +4.589, 212.585)]
BOX_XY, BOX_Z = 2.5, 3.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="how many pairs to sample")
    ap.add_argument("--denoise", action="store_true", help="run with denoise ON")
    ap.add_argument("--trim", type=float, default=2.0)
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    import cv2
    cv2.setNumThreads(4)
    from studio.backends import load_backend
    from studio.cloud import build_cloud
    from studio.dtypes import StereoParams
    from studio.infer import run_inference
    from studio.measure import MeasureBox, measure_box
    from studio.rectify import (Rectifier, StereoCalibration,
                                find_disparity_shift, roi_rects)

    lefts = sorted(glob.glob(os.path.join(PAIRS, "left", "*.jpg")))
    if not lefts:
        print(f"no pairs under {PAIRS}", file=sys.stderr)
        return 2
    idx = np.unique(np.linspace(0, len(lefts) - 1, args.n).astype(int))
    sample = [lefts[i] for i in idx]
    print(f"{len(lefts)} pairs available; sampling {len(sample)} across the run")

    cal = StereoCalibration.load(CALIB)
    rec = Rectifier(cal, (4024, 3036))

    # Δ measured ONCE on the first sampled pair and then FROZEN, exactly as a
    # batch freezes it — re-measuring per pair would fold the finder's own
    # variation into the study's spread.
    l0 = cv2.cvtColor(cv2.imread(sample[0]), cv2.COLOR_BGR2RGB)
    r0 = cv2.cvtColor(cv2.imread(sample[0].replace("/left/", "/right/")), cv2.COLOR_BGR2RGB)
    f = find_disparity_shift(rec.rectify(l0, "L"), rec.rectify(r0, "R"), ROI)
    if not f["ok"]:
        print(f"could not measure Δ (score {f['score']:.2f})", file=sys.stderr)
        return 2
    shift = max(0.0, f["shift"] - 24.0)
    print(f"Δ frozen at {shift:.0f} px (raw {f['shift']:.0f}, match {f['score']:.3f}, "
          f"dy {f['dy']:+d})   denoise {'ON' if args.denoise else 'OFF'}")

    p = StereoParams(scale=1.0, fx=rec.fx, fy=rec.fy, cx=rec.cx, cy=rec.cy,
                     baseline=rec.baseline, roi=ROI, disp_shift=shift,
                     z_near=195.0, z_far=235.0, remove_invisible=True,
                     denoise=args.denoise)
    p.model_params = {"valid_iters": 8, "max_disp": 64}
    (lx, ly, lw, lh), (rx, ry, _, _) = roi_rects(p, 4024, 3036)
    boxes = [(n, MeasureBox(cx=x, cy=y, cz=z, sx=BOX_XY, sy=BOX_XY, sz=BOX_Z))
             for n, x, y, z in BOXES]

    backend = load_backend("fast_foundation_stereo_trt")
    backend.load(CKPT, None)

    rows, t_all = [], time.time()
    for k, lp in enumerate(sample):
        label = re.sub(r"\.jpg$", "", os.path.basename(lp))
        rp = lp.replace("/left/", "/right/")
        t0 = time.time()
        L = cv2.cvtColor(cv2.imread(lp), cv2.COLOR_BGR2RGB)
        R = cv2.cvtColor(cv2.imread(rp), cv2.COLOR_BGR2RGB)
        li = rec.rectify_roi(L, "L", lx, ly, lw, lh)
        ri = rec.rectify_roi(R, "R", rx, ry, lw, lh)
        res = run_inference(backend, li, ri, p)
        cloud = build_cloud(res, p)
        row = {"label": label, "s": time.time() - t0, "n": cloud.n,
               "valid": 100.0 * float((res.disp > 0).mean())}
        for name, box in boxes:
            m = measure_box(cloud.points, box, trim_pct=args.trim)
            row[name + "_h"] = m["h_span_t"] if m else None
            row[name + "_z"] = m["z_med"] if m else None
            row[name + "_n"] = m["n"] if m else 0
        rows.append(row)
        print(f"  [{k+1:2d}/{len(sample)}] {label:9s} {row['s']:5.2f}s "
              f"{row['n']:7,} pts  valid {row['valid']:5.1f}%  " +
              "  ".join(f"{n}: z {row[n+'_z']:.4f} h {row[n+'_h']:.4f}"
                        if row[n + "_z"] is not None else f"{n}: EMPTY"
                        for n, _ in boxes), flush=True)

    el = time.time() - t_all
    print(f"\n{len(rows)} pairs in {el:.1f}s = {el/len(rows):.2f}s/pair "
          f"-> 1000 pairs ≈ {el/len(rows)*1000/60:.0f} min")
    print(f"\n{'pin':8s} {'N':>3s} {'mean z':>10s} {'σ z (µm)':>10s} {'range z':>9s} "
          f"{'mean h':>9s} {'σ h (µm)':>10s}")
    for name, _ in boxes:
        z = np.array([r[name + "_z"] for r in rows if r[name + "_z"] is not None])
        h = np.array([r[name + "_h"] for r in rows if r[name + "_h"] is not None])
        if len(z) < 2:
            print(f"  {name:8s} too few readings"); continue
        print(f"{name:8s} {len(z):3d} {z.mean():10.4f} {z.std(ddof=1)*1000:10.1f} "
              f"{(z.max()-z.min())*1000:8.1f}µ {h.mean():9.4f} {h.std(ddof=1)*1000:10.1f}")
    if args.csv:
        import csv as _csv
        with open(args.csv, "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
