"""Auto-pair a CNC ChArUco capture session for tools/calibrate.py.

Capture rhythm per board pose: shoot at CNC position A, jog the measurement
step (+X), shoot at position B, then reposition the board and repeat. Files
just need to sort in capture order (timestamp names are fine). This tool
verifies each consecutive couple by its OPTICAL signature — a true CNC pair is
a pure horizontal shift (image rows preserved); a board reposition is not —
decides which side is left (the camera that moved +X sees features at SMALLER
x, so it is the RIGHT eye), and copies verified pairs into <folder>/paired/ as
poseNN_left.jpg / poseNN_right.jpg, ready for:

    .venv/Scripts/python.exe tools/calibrate.py <folder>/paired --charuco 11x8 --square <MEASURED> --marker <MEASURED*2/3>
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys

import numpy as np


def main() -> None:
    import cv2

    ap = argparse.ArgumentParser(description="Verify + name CNC ChArUco capture pairs")
    ap.add_argument("folder", help="folder of raw captures, sorted = capture order")
    ap.add_argument("--charuco", default="11x8", metavar="CXxRY",
                    help="board squares, columns x rows (default 11x8)")
    ap.add_argument("--square", type=float, default=6.0, help="square size (default 6)")
    ap.add_argument("--marker", type=float, default=4.0, help="marker size (default 4)")
    ap.add_argument("--dict", default="4X4_50", dest="adict", help="ArUco dictionary")
    args = ap.parse_args()

    sx, sy = (int(v) for v in args.charuco.lower().split("x"))
    board = cv2.aruco.CharucoBoard(
        (sx, sy), args.square, args.marker,
        cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, "DICT_" + args.adict.upper())))
    det = cv2.aruco.CharucoDetector(board)

    files = sorted(p for p in glob.glob(os.path.join(args.folder, "*"))
                   if os.path.splitext(p)[1].lower() in
                   (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
    views = []
    for f in files:
        g = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY)
        c, ids, _, _ = det.detectBoard(g)
        views.append((f, None if ids is None else ids.ravel(),
                      None if ids is None else c.reshape(-1, 2)))
        n = 0 if ids is None else len(ids)
        print(f"  {os.path.basename(f)}: {n} corners")

    outdir = os.path.join(args.folder, "paired")
    os.makedirs(outdir, exist_ok=True)
    pairs, skipped, i = 0, [], 0
    while i < len(views) - 1:
        (fa, ida, pa), (fb, idb, pb) = views[i], views[i + 1]
        if ida is None or idb is None:
            skipped.append(os.path.basename(fa if ida is None else fb))
            i += 1
            continue
        common, ia, ib = np.intersect1d(ida, idb, return_indices=True)
        if len(common) >= 6:
            d = pb[ib] - pa[ia]
            mdx, mdy = float(d[:, 0].mean()), float(d[:, 1].mean())
            dy_std = float(d[:, 1].std())
            # a pure X translation preserves rows regardless of depth structure
            # (dx varies with depth — that is the disparity — but dy must not)
            if abs(mdx) > 50 and abs(mdy) < 15 and dy_std < 6:
                pairs += 1
                # camera at +X sees features at smaller x → that shot is the RIGHT eye
                lf, rf = (fa, fb) if mdx < 0 else (fb, fa)
                shutil.copy(lf, os.path.join(outdir, f"pose{pairs:02d}_left.jpg"))
                shutil.copy(rf, os.path.join(outdir, f"pose{pairs:02d}_right.jpg"))
                print(f"PAIR {pairs:02d}: {os.path.basename(lf)} + {os.path.basename(rf)}"
                      f"   shift {abs(mdx):.0f} px, row drift {mdy:+.1f}±{dy_std:.1f} px")
                i += 2
                continue
        skipped.append(os.path.basename(fa))
        i += 1
    if i == len(views) - 1:
        skipped.append(os.path.basename(views[-1][0]))

    print(f"\n{pairs} verified pairs -> {outdir}")
    if skipped:
        print(f"not paired ({len(skipped)}): {', '.join(skipped[:10])}"
              + (" …" if len(skipped) > 10 else ""))
    if pairs < 6:
        print("Fewer than 6 pairs — capture more poses (A shot, CNC step, B shot, "
              "then move the board).")
        sys.exit(1)


if __name__ == "__main__":
    main()
