"""Pin-height repeatability over a capture run — headless, same code as the GUI.

Runs the app's OWN measurement (studio.sites_measure) over the sites you marked
in the GUI, reading them from the same QSettings the app writes. It is therefore
a true dress rehearsal for the Batch button rather than a second implementation
that can drift from it — an earlier version of this tool duplicated the tracking
and median logic, which is exactly how a rehearsal stops predicting the thing it
rehearses.

Why the measurement looks the way it does — both learned the hard way on 1000
real captures:

  * every site is TRACKED per capture, because the frame drifts ~15 px in x and
    ~34 px in y over a run and a fixed window slides off a 1.6 mm feature
  * a height is only ever pin-minus-reference WITHIN one capture, because the
    CNC step IS the stereo baseline and repeats to ~0.5 %, walking ABSOLUTE
    depth by >1 mm while nothing has moved (σ 2200 µm absolute vs ~600 µm
    differential)

    .venv/bin/python tools/study_pin_heights.py [--n 0] [--csv out.csv]
                                                [--ref loop500] [--sites f.json]
"""
import argparse
import csv
import glob
import json
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


def _settings() -> dict:
    """The ROI, Δ and marked sites the GUI saved — the same keys it writes."""
    from PySide6.QtCore import QSettings

    s = QSettings("FSStudio", "FoundationStereoStudio")

    def blob(k, d):
        try:
            return json.loads(s.value(k, "") or json.dumps(d))
        except (ValueError, TypeError):
            return d

    roi = blob("roi", {})
    scene = blob("scene", {})
    mp = blob("model_params", {}).get("fast_foundation_stereo_trt", {})
    return {"sites": blob("study_sites", []),
            "roi": tuple(roi["roi"]) if roi.get("roi") else None,
            "shift": float(roi.get("shift", 0.0)),
            "scale": float(scene.get("scale", 1.0)),
            "z_near": float(scene.get("z_near", 180.0)),
            "z_far": float(scene.get("z_far", 260.0)),
            "denoise": bool(scene.get("denoise", False)),
            "iters": int(mp.get("valid_iters", 8)),
            "max_disp": int(mp.get("max_disp", 64))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="sample N pairs (0 = all)")
    ap.add_argument("--csv", default="")
    ap.add_argument("--ref", default="", help="capture to cut templates from "
                                              "(default: the middle of the run)")
    ap.add_argument("--sites", default="", help="sites JSON instead of QSettings")
    args = ap.parse_args()

    import cv2
    cv2.setNumThreads(4)
    from studio.backends import load_backend
    from studio.dtypes import StereoParams
    from studio.infer import run_inference
    from studio.rectify import Rectifier, StereoCalibration, roi_rects
    from studio.sites_measure import (make_templates, measure_sites,
                                      pair_sites, untemplatable)

    cfg = _settings()
    if args.sites:
        cfg["sites"] = json.load(open(args.sites))
    pins = [s for s in cfg["sites"] if s.get("kind") == "pin"]
    if not pins:
        print("No pins marked. In the app: Input tab -> 'Mark: pin', click each "
              "pin, then 'Mark: reference' near them.", file=sys.stderr)
        return 2
    if cfg["roi"] is None:
        print("No ROI set. Tick ROI on the Input tab and drag it over the pins.",
              file=sys.stderr)
        return 2
    for p, r in pair_sites(cfg["sites"]):
        print(f"  {p['name']:10s} -> {r['name'] if r else 'NO REFERENCE'}")
    print(f"  ROI {cfg['roi']}  Δ {cfg['shift']:.0f}  scale {cfg['scale']}  "
          f"max_disp {cfg['max_disp']}  denoise {cfg['denoise']}")

    lefts = sorted(glob.glob(os.path.join(PAIRS, "left", "*.jpg")))
    if not lefts:
        print(f"no pairs under {PAIRS}", file=sys.stderr)
        return 2
    every = lefts
    if args.n:
        every = [lefts[i] for i in np.unique(np.linspace(0, len(lefts) - 1, args.n).astype(int))]
    print(f"  {len(every)} of {len(lefts)} pairs")

    cal = StereoCalibration.load(CALIB)
    rec = Rectifier(cal, (4024, 3036))
    p = StereoParams(scale=cfg["scale"], fx=rec.fx, fy=rec.fy, cx=rec.cx, cy=rec.cy,
                     baseline=rec.baseline, roi=cfg["roi"], disp_shift=cfg["shift"],
                     z_near=cfg["z_near"], z_far=cfg["z_far"],
                     remove_invisible=True, denoise=cfg["denoise"])
    p.model_params = {"valid_iters": cfg["iters"], "max_disp": cfg["max_disp"]}
    (lx, ly, lw, lh), (rx, ry, _, _) = roi_rects(p, 4024, 3036)

    def crop(path, side, X, Y):
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        return rec.rectify_roi(img, side, X, Y, lw, lh)

    ref_path = next((q for q in lefts if args.ref and args.ref in q),
                    every[len(every) // 2])
    print(f"  templates from {os.path.basename(ref_path)}")
    backend = load_backend("fast_foundation_stereo_trt")
    backend.load(CKPT, None)
    ref_res = run_inference(backend, crop(ref_path, "L", lx, ly),
                            crop(ref_path.replace("/left/", "/right/"), "R", rx, ry), p)
    bad = untemplatable(ref_res.rgb, cfg["sites"], p.roi, p.scale)
    if bad:
        print("\n  These sites cannot be tracked in this run's crop:", file=sys.stderr)
        for name, why in bad:
            print(f"    {name}: {why}", file=sys.stderr)
        print("  Fix the ROI or the Input scale in the app, then re-run.\n",
              file=sys.stderr)
        if len(bad) == len(cfg["sites"]):
            return 2          # nothing measurable — say so instead of "0 readings"
    templates = make_templates(ref_res.rgb, cfg["sites"], p.roi, p.scale)

    rows, t0 = [], time.time()
    for i, lp in enumerate(every):
        label = re.sub(r"\.jpg$", "", os.path.basename(lp))
        res = run_inference(backend, crop(lp, "L", lx, ly),
                            crop(lp.replace("/left/", "/right/"), "R", rx, ry), p)
        got = measure_sites(res.rgb, res.depth, cfg["sites"], templates, p.roi, p.scale)
        row = {"label": label, "valid": round(100.0 * float((res.disp > 0).mean()), 2)}
        for name, rec_ in got.items():
            row[name] = rec_["height"]
            row[name + "_score"] = round(rec_["score"], 3)
            row[name + "_track"] = "%+d,%+d" % rec_["track"]
        rows.append(row)
        if i < 3 or (i + 1) % 100 == 0 or i == len(every) - 1:
            print(f"  [{i+1:4d}/{len(every)}] {label:9s} " + "  ".join(
                f"{n}: {row[n]:+.4f} m{row[n+'_score']:.2f}"
                if np.isfinite(row[n]) else f"{n}: —" for n in got), flush=True)

    el = time.time() - t0
    print(f"\n{len(rows)} pairs in {el/60:.1f} min = {el/len(rows):.2f} s/pair")
    print(f"\n{'pin':10s} {'N':>5s} {'mean mm':>10s} {'σ (µm)':>9s} {'range (µm)':>11s}")
    for name in [x["name"] for x in pins]:
        v = np.array([r.get(name, np.nan) for r in rows], float)
        v = v[np.isfinite(v)]
        if len(v) < 2:
            print(f"{name:10s} {len(v):5d}   too few readings"); continue
        print(f"{name:10s} {len(v):5d} {v.mean():10.4f} {v.std(ddof=1)*1000:9.1f} "
              f"{(v.max()-v.min())*1000:11.1f}")
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]), extrasaction="ignore")
            w.writeheader(); w.writerows(rows)
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
