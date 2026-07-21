"""One-shot camera calibration for the single-camera CNC stereo rig.

Point it at a folder of calibration-target stereo pairs (the same left/right
naming the app's batch understands) and it writes calib.json (K, D, R, T,
image_size) ready to load in the app's "Raw — rectify with calibration" mode.
Optionally also writes k_rectified.txt (for the "already rectified" mode).

Two target types:

CHECKERBOARD (classic — the whole board must be visible in every image):
    .venv\\Scripts\\python.exe tools\\calibrate.py <folder> --cols 9 --rows 6 --square 20
    --cols / --rows = INNER corners (a 10x7-SQUARE board has 9x6 inner corners).

CHARUCO (recommended — PARTIAL views count, so it works when the board is
bigger than the field of view, and corners can reach the frame edges):
    .venv\\Scripts\\python.exe tools\\calibrate.py <folder> --charuco 11x8 --square 6 --marker 4
    --charuco = the board's SQUARES, columns x rows (an "8x11" print held
                landscape is 11x8 — if you get 0 corners, the order is wrong).
    --marker  = the ArUco marker's physical size; --dict its dictionary (4X4_50
                default). Both sizes in --unit.

--square is the physical size of ONE square, MEASURED WITH CALIPERS across as
many squares as possible (printers rescale by up to ~1%) — never the nominal
value. --unit (mm default) sets the baseline unit you pick in the app.
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
    ap.add_argument("folder", help="folder of calibration-target stereo pairs")
    ap.add_argument("--cols", type=int, help="checkerboard: inner corners across")
    ap.add_argument("--rows", type=int, help="checkerboard: inner corners down")
    ap.add_argument("--charuco", metavar="CXxRY",
                    help="ChArUco board SQUARES, columns x rows (e.g. 11x8). "
                         "Partial views are fine — corners are matched by marker ID.")
    ap.add_argument("--marker", type=float,
                    help="ChArUco: the ArUco marker's physical size (same unit as --square)")
    ap.add_argument("--dict", default="4X4_50", dest="adict",
                    help="ChArUco: ArUco dictionary (default 4X4_50)")
    ap.add_argument("--square", type=float, required=True,
                    help="one square's physical size — caliper-MEASURED, not nominal")
    ap.add_argument("--unit", default="mm", choices=["mm", "m"], help="unit of --square (default mm)")
    ap.add_argument("--out", default="calib.json", help="output calibration file")
    ap.add_argument("--krect", action="store_true", help="also write k_rectified.txt")
    ap.add_argument("--alpha", type=float, default=0.0, help="stereoRectify alpha (0=crop, 1=keep all)")
    ap.add_argument("--force", action="store_true",
                    help="write outputs even if the reprojection-RMS quality gates fail")
    ap.add_argument("--simple-lens", action="store_true",
                    help="constrain the model: principal point at the image centre, zero "
                         "distortion. For near-distortion-free machine-vision lenses "
                         "(image circle >> sensor) this removes the degenerate directions "
                         "a free model dumps noise into (a wandering cx/cy + spurious k1 "
                         "can render stereoRectify unusable) at <0.1 px model cost.")
    args = ap.parse_args()

    charuco = None
    if args.charuco:
        try:
            sx, sy = (int(v) for v in args.charuco.lower().split("x"))
        except ValueError:
            ap.error("--charuco must look like 11x8 (squares across x squares down)")
        if not args.marker:
            ap.error("--charuco needs --marker (the ArUco marker's physical size)")
        dname = "DICT_" + args.adict.upper().removeprefix("DICT_")
        if not hasattr(cv2.aruco, dname):
            ap.error(f"unknown ArUco dictionary {args.adict!r}")
        board = cv2.aruco.CharucoBoard(
            (sx, sy), float(args.square), float(args.marker),
            cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dname)))
        charuco = (board, cv2.aruco.CharucoDetector(board),
                   board.getChessboardCorners().astype(np.float32))
    elif not (args.cols and args.rows):
        ap.error("give either --cols/--rows (checkerboard) or --charuco (ChArUco)")

    scan = find_pairs(args.folder)
    if not scan.pairs:
        print(f"No stereo pairs found in {args.folder} ({scan.method}).")
        sys.exit(1)
    print(f"Found {len(scan.pairs)} target pairs ({scan.method}).")

    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4)
    MIN_CORNERS = 6            # per view (intrinsics) and per pair (stereo, common IDs)
    intr_obj, intr_img = [], []              # every usable single VIEW (left + right)
    st_obj, st_L, st_R = [], [], []          # per-PAIR matched correspondences
    size = None

    if charuco is None:
        pattern = (args.cols, args.rows)
        objp = np.zeros((args.rows * args.cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)
        objp *= float(args.square)           # object points in the chosen unit
        find_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE

    def _detect(gray):
        """(corner_ids, image_points) for one view — full board for checker,
        whatever is visible for ChArUco."""
        if charuco is None:
            ok, c = cv2.findChessboardCorners(gray, pattern, find_flags)
            if not ok:
                return None, None
            c = cv2.cornerSubPix(gray, c, (11, 11), (-1, -1), crit)
            return np.arange(len(objp)), c.reshape(-1, 2)
        ch_c, ch_ids, _mc, _mi = charuco[1].detectBoard(gray)
        if ch_ids is None or len(ch_ids) < MIN_CORNERS:
            return None, None
        return ch_ids.ravel(), ch_c.reshape(-1, 2)

    def _obj_for(ids):
        return (objp if charuco is None else charuco[2])[ids]

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
        idl, cl = _detect(gl)
        idr, cr = _detect(gr)
        for ids, c in ((idl, cl), (idr, cr)):    # intrinsics: single camera, both shots
            if ids is not None:
                intr_obj.append(_obj_for(ids))
                intr_img.append(c.astype(np.float32))
        if idl is None or idr is None:
            miss = "both" if idl is None and idr is None else ("left" if idl is None else "right")
            print(f"  SKIP  {label} — no target in {miss}")
            continue
        common = np.intersect1d(idl, idr)        # stereo: corners BOTH shots identified
        if len(common) < MIN_CORNERS:
            print(f"  SKIP  {label} — only {len(common)} corners seen by both shots")
            continue
        li = {i: k for k, i in enumerate(idl)}
        ri = {i: k for k, i in enumerate(idr)}
        st_obj.append(_obj_for(common))
        st_L.append(cl[[li[i] for i in common]].astype(np.float32))
        st_R.append(cr[[ri[i] for i in common]].astype(np.float32))
        print(f"  found: {label}  ({len(common)} shared corners)")

    if len(st_obj) < 6:
        print(f"\nOnly {len(st_obj)} usable pairs — need ~10+ covering the frame + tilts. "
              "Shoot more target poses.")
        sys.exit(1)

    # single-camera intrinsics from ALL views (left + right are the same camera)
    if args.simple_lens:
        flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT
                 | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2
                 | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5
                 | cv2.CALIB_FIX_K6)
        K0 = np.array([[1000.0, 0, size[0] / 2], [0, 1000.0, size[1] / 2], [0, 0, 1]])
        rms, K, D, _, _ = cv2.calibrateCamera(intr_obj, intr_img, size, K0,
                                              np.zeros(5), flags=flags)
    else:
        rms, K, D, _, _ = cv2.calibrateCamera(intr_obj, intr_img, size, None, None)
    print(f"\nIntrinsics reproj RMS: {rms:.3f} px  (aim < ~0.5)  "
          f"[{len(intr_obj)} views]"
          + ("  [simple-lens model]" if args.simple_lens else ""))
    print(f"fx={K[0,0]:.1f}  fy={K[1,1]:.1f}  cx={K[0,2]:.1f}  cy={K[1,2]:.1f}")

    # stereo extrinsics R, T between the two shots, with intrinsics held fixed
    res = cv2.stereoCalibrate(st_obj, st_L, st_R, K, D, K, D, size,
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
