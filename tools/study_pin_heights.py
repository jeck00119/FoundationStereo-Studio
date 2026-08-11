"""Pin-height repeatability over a capture run — tracked, locally referenced.

Two things make this immune to what wrecked the naive version:

  * Each pin is TRACKED per capture (template-matched against a reference patch)
    rather than measured at a fixed pixel or a fixed world box. The rig's frame
    drifts ~15 px in x and ~34 px in y over 1000 loops; a fixed window slowly
    slides off a 1.6 mm feature, a tracked one does not.

  * Height is referenced to the LOCAL BOARD IN THE SAME CAPTURE, never to an
    absolute depth. The CNC's step (the stereo baseline) repeats to ~0.5 %, which
    moves absolute depth by >1 mm but cancels almost entirely in a difference
    taken within one frame — measured: absolute σ 1307 µm vs referenced 247 µm.

So no capture has to be excluded: settling transients move both the pin and its
reference together.

    .venv/bin/python tools/study_pin_heights.py [--n 0] [--csv out.csv]
"""
import argparse
import csv
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
PAIRS, CALIB = "data/captures/pin_repeat", "data/calib/calib_provisional.json"
ROI = (2560, 1024, 512, 1024)          # 524k px — under this device's build ceiling
REF_LOOP = "loop500"                   # template source: a settled capture
# pin sites in ROI pixels, with the local board reference for each (same row,
# on the green PCB ~1.5 mm away) — a pin is only meaningful against ITS board
# The reference must be TEXTURED or the network has nothing to match there and
# its depth is a guess. Measured mean local contrast (std of grey): the green PCB
# is 3.4 and the metal bar 3.6 — featureless; the component body is 23.8 and the
# leads 57-69. Referencing to the green strip gave sigma 630 um, to the body
# 605 um; to the featureless bar it was worse still. Body it is.
SITES = [("Lead A", (247, 115), (400, 300)),
         ("Lead B", (260, 954), (400, 800))]
PIN_R, BOARD_R, TRACK_R = 40, 40, 60    # sample radii, and the tracking margin


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="sample N pairs (0 = all)")
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    import cv2
    cv2.setNumThreads(4)
    from studio.backends import load_backend
    from studio.dtypes import StereoParams
    from studio.infer import run_inference
    from studio.rectify import (Rectifier, StereoCalibration,
                                find_disparity_shift, roi_rects)

    lefts = sorted(glob.glob(os.path.join(PAIRS, "left", "*.jpg")))
    if args.n:
        lefts = [lefts[i] for i in np.unique(np.linspace(0, len(lefts) - 1, args.n).astype(int))]
    print(f"{len(lefts)} pairs")

    cal = StereoCalibration.load(CALIB)
    rec = Rectifier(cal, (4024, 3036))
    ref_path = next((p for p in sorted(glob.glob(os.path.join(PAIRS, "left", "*.jpg")))
                     if REF_LOOP in p), lefts[len(lefts) // 2])
    rl = cv2.cvtColor(cv2.imread(ref_path), cv2.COLOR_BGR2RGB)
    rr = cv2.cvtColor(cv2.imread(ref_path.replace("/left/", "/right/")), cv2.COLOR_BGR2RGB)
    f = find_disparity_shift(rec.rectify(rl, "L"), rec.rectify(rr, "R"), ROI)
    shift = max(0.0, f["shift"] - 30.0)
    print(f"reference {os.path.basename(ref_path)}  Δ {shift:.0f} px (raw {f['shift']:.0f}, "
          f"match {f['score']:.2f})")

    p = StereoParams(scale=1.0, fx=rec.fx, fy=rec.fy, cx=rec.cx, cy=rec.cy,
                     baseline=rec.baseline, roi=ROI, disp_shift=shift,
                     z_near=180.0, z_far=260.0, remove_invisible=True, denoise=False)
    p.model_params = {"valid_iters": 8, "max_disp": 64}
    (lx, ly, lw, lh), (rx, ry, _, _) = roi_rects(p, 4024, 3036)

    ref_crop = cv2.cvtColor(rec.rectify_roi(rl, "L", lx, ly, lw, lh), cv2.COLOR_RGB2GRAY)
    templates = {}
    for name, (pu, pv), _ in SITES:
        templates[name] = ref_crop[pv - TRACK_R:pv + TRACK_R, pu - TRACK_R:pu + TRACK_R]

    backend = load_backend("fast_foundation_stereo_trt")
    backend.load(CKPT, None)

    def med(a, u, v, r):
        m = a[max(0, v - r):v + r, max(0, u - r):u + r]
        m = m[m > 0]
        return float(np.median(m)) if m.size >= 20 else np.nan

    rows, t0 = [], time.time()
    for i, lp in enumerate(lefts):
        label = re.sub(r"\.jpg$", "", os.path.basename(lp))
        L = cv2.cvtColor(cv2.imread(lp), cv2.COLOR_BGR2RGB)
        R = cv2.cvtColor(cv2.imread(lp.replace("/left/", "/right/")), cv2.COLOR_BGR2RGB)
        li = rec.rectify_roi(L, "L", lx, ly, lw, lh)
        ri = rec.rectify_roi(R, "R", rx, ry, lw, lh)
        res = run_inference(backend, li, ri, p)
        gray = cv2.cvtColor(li, cv2.COLOR_RGB2GRAY)
        row = {"label": label, "valid": 100.0 * float((res.disp > 0).mean())}
        for name, (pu, pv), (bu, bv) in SITES:
            # track the pin: search a window around its reference position
            s = 90
            y0, x0 = max(0, pv - TRACK_R - s), max(0, pu - TRACK_R - s)
            band = gray[y0:pv + TRACK_R + s, x0:pu + TRACK_R + s]
            t = templates[name]
            if band.shape[0] < t.shape[0] or band.shape[1] < t.shape[1]:
                row[name] = np.nan; continue
            r_ = cv2.matchTemplate(band, t, cv2.TM_CCOEFF_NORMED)
            _, sc, _, loc = cv2.minMaxLoc(r_)
            du, dv = (x0 + loc[0] + TRACK_R) - pu, (y0 + loc[1] + TRACK_R) - pv
            zp = med(res.depth, pu + du, pv + dv, PIN_R)
            zb = med(res.depth, bu + du, bv + dv, BOARD_R)
            row[name] = zb - zp                       # +ve = pin stands proud
            row[name + "_track"] = f"{du:+d},{dv:+d}"
            row[name + "_score"] = round(float(sc), 3)
            row[name + "_zpin"] = zp
        rows.append(row)
        if i < 5 or (i + 1) % 100 == 0 or i == len(lefts) - 1:
            print(f"  [{i+1:4d}/{len(lefts)}] {label:9s} " + "  ".join(
                f"{n}: h {row[n]:+.4f} trk {row[n+'_track']} m{row[n+'_score']:.2f}"
                if np.isfinite(row.get(n, np.nan)) else f"{n}: —" for n, _, _ in SITES),
                flush=True)

    el = time.time() - t0
    print(f"\n{len(rows)} pairs in {el/60:.1f} min = {el/len(rows):.2f} s/pair")
    print(f"\n{'pin':8s} {'N':>5s} {'mean mm':>10s} {'σ (µm)':>9s} {'range (µm)':>11s} {'p95-p5':>9s}")
    for name, _, _ in SITES:
        v = np.array([r[name] for r in rows], float)
        v = v[np.isfinite(v)]
        if len(v) < 2:
            print(f"{name:8s} too few"); continue
        print(f"{name:8s} {len(v):5d} {v.mean():10.4f} {v.std(ddof=1)*1000:9.1f} "
              f"{(v.max()-v.min())*1000:11.1f} {(np.percentile(v,95)-np.percentile(v,5))*1000:8.1f}µ")
    if args.csv:
        keys = list(rows[0])
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
