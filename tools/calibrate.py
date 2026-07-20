"""One-shot camera calibration for the single-camera CNC stereo rig.

Point it at a folder of CHECKERBOARD stereo pairs (the same left/right naming the
app's batch understands) and it writes calib.json (K, D, R, T, image_size) ready
to load in the app's "Raw — rectify with calibration" mode. Optionally also writes
k_rectified.txt (for the "already rectified" mode, if you ever rectify offline).

Run it with the app's venv from the repo root:
    .venv\\Scripts\\python.exe tools\\calibrate.py <checkerboard_folder> --cols 9 --rows 6 --square 20

--cols / --rows = number of INNER corners (a board of 10x7 SQUARES has 9x6 inner
                  corners — count the inner crossings, not the squares).
--square        = the physical size of ONE square, measured with calipers, in --unit.
--unit          = mm (default) or m. This sets the baseline unit you pick in the app.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

# tools/ lives one level below the repo root, which is what holds the studio package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    import cv2

    # the app's exact pairing + loading rules, from the Qt-free module — this CLI
    # no longer drags the whole PySide6 GUI stack in just to match filenames
    from studio.pairs import find_pairs, load_rgb

    ap = argparse.ArgumentParser(description="Single-camera stereo calibration -> calib.json")
    ap.add_argument("folder", help="folder of checkerboard stereo pairs")
    ap.add_argument("--cols", type=int, required=True, help="inner corners across")
    ap.add_argument("--rows", type=int, required=True, help="inner corners down")
    ap.add_argument("--square", type=float, required=True, help="one square's physical size")
    ap.add_argument("--unit", default="mm", choices=["mm", "m"], help="unit of --square (default mm)")
    ap.add_argument("--out", default="calib.json", help="output calibration file")
    ap.add_argument("--krect", action="store_true", help="also write k_rectified.txt")
    ap.add_argument("--alpha", type=float, default=0.0, help="stereoRectify alpha (0=crop, 1=keep all)")
    ap.add_argument("--force", action="store_true",
                    help="write outputs even if the reprojection-RMS quality gates fail")
    args = ap.parse_args()

    scan = find_pairs(args.folder)
    if not scan.pairs:
        print(f"No stereo pairs found in {args.folder} ({scan.method}).")
        sys.exit(1)
    print(f"Found {len(scan.pairs)} checkerboard pairs ({scan.method}).")

    pattern = (args.cols, args.rows)
    objp = np.zeros((args.rows * args.cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)
    objp *= float(args.square)                       # object points in the chosen unit

    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4)
    find_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    objpoints, ptsL, ptsR = [], [], []
    size = None
    for label, lp, rp in scan.pairs:
        gl = cv2.cvtColor(load_rgb(lp), cv2.COLOR_RGB2GRAY)
        gr = cv2.cvtColor(load_rgb(rp), cv2.COLOR_RGB2GRAY)
        if size is None:
            size = (gl.shape[1], gl.shape[0])        # (w, h)
        elif ((gl.shape[1], gl.shape[0]) != size or (gr.shape[1], gr.shape[0]) != size):
            # all corners are solved against ONE imageSize — a stray odd-sized
            # pair would silently corrupt the whole calibration
            print(f"  SKIP  {label} — {gl.shape[1]}×{gl.shape[0]} differs from "
                  f"{size[0]}×{size[1]} (every pair must share one resolution)")
            continue
        okl, cl = cv2.findChessboardCorners(gl, pattern, find_flags)
        okr, cr = cv2.findChessboardCorners(gr, pattern, find_flags)
        if okl and okr:
            cl = cv2.cornerSubPix(gl, cl, (11, 11), (-1, -1), crit)
            cr = cv2.cornerSubPix(gr, cr, (11, 11), (-1, -1), crit)
            objpoints.append(objp)
            ptsL.append(cl)
            ptsR.append(cr)
            print(f"  found: {label}")
        else:
            miss = "both" if not (okl or okr) else ("left" if not okl else "right")
            print(f"  SKIP  {label} — no corners in {miss}")

    if len(objpoints) < 6:
        print(f"\nOnly {len(objpoints)} usable pairs — need ~10+ covering the frame + tilts. "
              "Shoot more checkerboard poses.")
        sys.exit(1)

    # single-camera intrinsics from ALL views (left + right are the same camera)
    rms, K, D, _, _ = cv2.calibrateCamera(objpoints + objpoints, ptsL + ptsR, size, None, None)
    print(f"\nIntrinsics reproj RMS: {rms:.3f} px  (aim < ~0.5)")

    # stereo extrinsics R, T between the two shots, with intrinsics held fixed
    res = cv2.stereoCalibrate(objpoints, ptsL, ptsR, K, D, K, D, size,
                              criteria=crit, flags=cv2.CALIB_FIX_INTRINSIC)
    srms, R, T = float(res[0]), np.asarray(res[5]), np.asarray(res[6])
    print(f"Stereo    reproj RMS: {srms:.3f} px  (aim < ~1.0)")
    print(f"Baseline |T| = {float(np.linalg.norm(T)):.4g} {args.unit}  "
          "(should be ~ your CNC step)")

    # Quality gate — the "aim" numbers used to be print-only, so a garbage
    # solution (bad corners, too little coverage) still wrote calib.json and
    # quietly poisoned every later measurement. Gates are looser than the aims
    # so a decent-but-imperfect solve still writes; --force overrides.
    bad = []
    if rms > 0.8:
        bad.append(f"intrinsics RMS {rms:.3f} px (aim < ~0.5)")
    if srms > 1.5:
        bad.append(f"stereo RMS {srms:.3f} px (aim < ~1.0)")
    if bad and not args.force:
        print("\nNOT writing outputs — calibration quality failed:\n  - "
              + "\n  - ".join(bad)
              + "\nShoot better coverage (fill the frame, vary tilt/distance), "
                "or re-run with --force to write anyway.")
        sys.exit(2)

    out = {
        "K": K.tolist(), "D": np.ravel(D).tolist(),
        "R": R.tolist(), "T": np.ravel(T).tolist(),
        "image_width": int(size[0]), "image_height": int(size[1]),
        "baseline_unit": args.unit,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}  ->  load it in the app's 'Raw — rectify' mode "
          f"(set Baseline unit: {args.unit}).")

    if args.krect:
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            K, D, K, D, size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=args.alpha)
        fx, fy, cx, cy = P1[0, 0], P1[1, 1], P1[0, 2], P1[1, 2]
        base = abs(float(P2[0, 3] / P1[0, 0]))
        base_m = base / 1000.0 if args.unit == "mm" else base   # K.txt is in metres
        # next to --out, not the CWD — running from another directory used to
        # scatter this file wherever the shell happened to be
        krect = os.path.join(os.path.dirname(os.path.abspath(args.out)), "k_rectified.txt")
        with open(krect, "w", encoding="utf-8") as f:
            f.write(f"{fx} 0 {cx} 0 {fy} {cy} 0 0 1\n{base_m}\n")
        print(f"Wrote {krect} (rectified K + baseline {base_m:.6g} m). "
              "You must also rectify the images offline to use it.")


if __name__ == "__main__":
    main()
