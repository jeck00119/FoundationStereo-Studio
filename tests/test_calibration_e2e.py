"""Calibration workflow ground truth — renders checkerboard stereo pairs
through a KNOWN camera (K, distortion, 5 mm pure-translation baseline), runs
the real tools/calibrate.py on the folder, then loads its calib.json through
the app's own StereoCalibration/Rectifier and rectifies a pair.

What this pins, with known answers: corner detection on our own renders, the
single-camera intrinsics trick, stereoCalibrate's R/T, the RMS quality gates
passing on clean data, the calib.json ↔ rectify.py contract, the derived
rectified fx/baseline, and the acid test — epipolar lines actually horizontal
after rectification.
"""
import json
import os
import runpy
import sys

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from studio.rectify import Rectifier, StereoCalibration

W, H = 640, 480
K_TRUE = np.array([[800.0, 0, 320.0], [0, 800.0, 240.0], [0, 0, 1]])
DIST_TRUE = np.array([-0.06, 0.02, 0.0, 0.0, 0.0])
T_TRUE = np.array([5.0, 0.0, 0.0])          # right camera 5 mm along +X (CNC step)
COLS, ROWS = 9, 6                            # inner corners
SQUARE = 20.0                                # mm


def _board_world():
    """The 10×7-square board's black quads + its inner-corner grid, in mm,
    centred on the origin of the board frame."""
    quads = []
    for r in range(ROWS + 1):
        for c in range(COLS + 1):
            if (r + c) % 2 == 0:
                x0, y0 = (c - (COLS + 1) / 2) * SQUARE, (r - (ROWS + 1) / 2) * SQUARE
                quads.append(np.array([[x0, y0, 0], [x0 + SQUARE, y0, 0],
                                       [x0 + SQUARE, y0 + SQUARE, 0], [x0, y0 + SQUARE, 0]]))
    return quads


def _render(rvec, tvec):
    """One camera view of the board: white canvas, projected black squares
    (through the TRUE intrinsics + distortion). Sub-pixel corners via the
    fixed-point `shift` — integer-rounded quads quantize every corner the
    calibration then measures, biasing the recovered camera by ~2 %."""
    img = np.full((H, W), 255, np.uint8)
    SHIFT = 4                                          # 1/16-px fixed point
    for quad in _board_world():
        pts, _ = cv2.projectPoints(quad, rvec, tvec, K_TRUE, DIST_TRUE)
        ipts = (pts.reshape(-1, 2) * (1 << SHIFT)).round().astype(np.int32)
        cv2.fillConvexPoly(img, ipts, 0, lineType=cv2.LINE_AA, shift=SHIFT)
    return img


def _poses():
    """A dozen poses covering the frame with tilt/distance variety."""
    out = []
    for i, (rx, ry, dx, dy, z) in enumerate([
            (0.00, 0.00, 0, 0, 420), (0.20, 0.00, 30, 10, 400),
            (-0.20, 0.10, -30, 20, 440), (0.10, -0.25, 40, -25, 380),
            (-0.15, -0.15, -45, -20, 460), (0.25, 0.15, 15, 35, 500),
            (0.00, 0.30, -20, 30, 420), (-0.30, 0.00, 25, -35, 480),
            (0.15, 0.20, -40, -10, 360), (-0.10, -0.30, 35, 25, 520),
            (0.30, -0.10, 0, -30, 440), (-0.25, 0.25, -15, 15, 400)]):
        out.append((np.array([rx, ry, 0.05 * ((i % 3) - 1)]),
                    np.array([float(dx), float(dy), float(z)])))
    return out


@pytest.fixture(scope="module")
def calib_run(tmp_path_factory):
    """Render the pairs, run the REAL tools/calibrate.py on them, return paths."""
    folder = tmp_path_factory.mktemp("checker")
    for i, (rvec, tvec) in enumerate(_poses()):
        cv2.imwrite(str(folder / f"cap{i:02d}_left.png"), _render(rvec, tvec))
        # right camera sits at +T in the left frame → X_right = X_left − T
        cv2.imwrite(str(folder / f"cap{i:02d}_right.png"), _render(rvec, tvec - T_TRUE))
    out = folder / "calib.json"
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    argv, sys.argv = sys.argv, ["calibrate.py", str(folder), "--cols", str(COLS),
                                "--rows", str(ROWS), "--square", str(SQUARE),
                                "--unit", "mm", "--out", str(out), "--krect"]
    try:
        runpy.run_path(os.path.join(repo, "tools", "calibrate.py"), run_name="__main__")
    except SystemExit as e:                       # argparse/main exit codes
        assert not e.code, f"calibrate.py exited with {e.code}"
    finally:
        sys.argv = argv
    return folder, out


def test_calibrate_recovers_the_true_camera(calib_run):
    _folder, out = calib_run
    blob = json.loads(out.read_text())
    K = np.array(blob["K"])
    T = np.array(blob["T"])
    assert abs(K[0, 0] - K_TRUE[0, 0]) / K_TRUE[0, 0] < 0.01      # fx within 1 %
    assert abs(K[0, 2] - K_TRUE[0, 2]) < 6 and abs(K[1, 2] - K_TRUE[1, 2]) < 6
    assert abs(np.linalg.norm(T) - 5.0) < 0.05                     # the CNC step, in mm
    assert abs(blob["D"][0] - DIST_TRUE[0]) < 0.02                 # k1 recovered
    assert blob["image_width"] == W and blob["image_height"] == H
    assert blob["baseline_unit"] == "mm"


def test_krect_matches_repo_convention(calib_run):
    folder, out = calib_run
    krect = out.parent / "k_rectified.txt"                         # written beside --out
    assert krect.is_file()
    lines = krect.read_text().strip().splitlines()
    vals = list(map(float, lines[0].split()))
    assert len(vals) == 9 and vals[0] > 0
    base_m = float(lines[1])
    assert abs(base_m - 0.005) < 5e-5                              # metres, per K.txt convention


def test_rectifier_derives_truth_and_flattens_epipolar_lines(calib_run):
    """The acid test: load calib.json through the APP's loader, rectify a real
    pair, and check the same physical corners land on the same image rows."""
    folder, out = calib_run
    calib = StereoCalibration.load(str(out))
    assert calib.image_size == (W, H)
    rect = Rectifier(calib, (W, H))
    assert abs(rect.baseline - 5.0) < 0.05                         # derived, in mm
    assert abs(rect.fx - K_TRUE[0, 0]) / K_TRUE[0, 0] < 0.05

    li = cv2.imread(str(folder / "cap01_left.png"), cv2.IMREAD_GRAYSCALE)
    ri = cv2.imread(str(folder / "cap01_right.png"), cv2.IMREAD_GRAYSCALE)
    rl = rect.rectify(li, "L")
    rr = rect.rectify(ri, "R")
    okl, cl = cv2.findChessboardCorners(rl, (COLS, ROWS))
    okr, cr = cv2.findChessboardCorners(rr, (COLS, ROWS))
    assert okl and okr
    dy = np.abs(cl.reshape(-1, 2)[:, 1] - cr.reshape(-1, 2)[:, 1])
    assert float(dy.mean()) < 0.5 and float(dy.max()) < 1.5        # rows aligned (px)
    # …and disparity from the rectified pair reproduces the true depth:
    # d = fx·B/Z for each corner (board ~at its pose depth)
    dx = cl.reshape(-1, 2)[:, 0] - cr.reshape(-1, 2)[:, 0]
    z_est = rect.fx * rect.baseline / np.maximum(dx, 1e-6)
    assert 300 < float(np.median(z_est)) < 520                     # plausible pose depth (mm)


# ===================================================== ChArUco path (partial views)
CH_SX, CH_SY = 11, 8                 # the user's board: 11 squares wide x 8 tall
CH_SQ, CH_MK = 6.0, 4.0              # mm
PPMM = 20                            # board raster: px per mm


def _charuco_board():
    return cv2.aruco.CharucoBoard(
        (CH_SX, CH_SY), CH_SQ, CH_MK,
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))


def _charuco_view(bimg, rvec, tvec, W=640, H=480):
    """Project the rasterized board (Z=0 plane, PPMM px/mm) through the TRUE
    zero-distortion camera: H = K [r1 r2 t] composed with the mm->raster scale."""
    R, _ = cv2.Rodrigues(rvec)
    Hmm = K_TRUE @ np.column_stack([R[:, 0], R[:, 1], tvec])
    Hpx = Hmm @ np.diag([1.0 / PPMM, 1.0 / PPMM, 1.0])
    return cv2.warpPerspective(bimg, Hpx, (W, H), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def _charuco_poses():
    """Close-range poses like the real rig: the board OVERFILLS the frame in the
    nearer ones, so partial views are genuinely exercised."""
    out = []
    for i, (rx, ry, ox, oy, z) in enumerate([
            (0.00, 0.00, 0, 0, 72), (0.15, 0.00, 8, 4, 64),
            (-0.15, 0.10, -8, 6, 80), (0.10, -0.20, 10, -6, 60),
            (-0.12, -0.12, -12, -5, 84), (0.20, 0.12, 5, 8, 76),
            (0.00, 0.22, -6, 9, 68), (-0.22, 0.00, 7, -9, 82),
            (0.12, 0.16, -10, -3, 62), (-0.08, -0.22, 9, 6, 86),
            (0.22, -0.08, 0, -8, 70), (-0.18, 0.18, -4, 4, 78)]):
        rvec = np.array([rx, ry, 0.04 * ((i % 3) - 1)])
        # board local coords run 0..66 x 0..48 mm; centre it near the axis
        tvec = np.array([-CH_SX * CH_SQ / 2 + ox, -CH_SY * CH_SQ / 2 + oy, float(z)])
        out.append((rvec, tvec))
    return out


@pytest.fixture(scope="module")
def charuco_run(tmp_path_factory):
    board = _charuco_board()
    bimg = board.generateImage((int(CH_SX * CH_SQ * PPMM), int(CH_SY * CH_SQ * PPMM)),
                               marginSize=0, borderBits=1)
    folder = tmp_path_factory.mktemp("charuco")
    for i, (rvec, tvec) in enumerate(_charuco_poses()):
        cv2.imwrite(str(folder / f"cap{i:02d}_left.png"), _charuco_view(bimg, rvec, tvec))
        cv2.imwrite(str(folder / f"cap{i:02d}_right.png"),
                    _charuco_view(bimg, rvec, tvec - T_TRUE))
    out = folder / "calib.json"
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    argv, sys.argv = sys.argv, ["calibrate.py", str(folder), "--charuco",
                                f"{CH_SX}x{CH_SY}", "--square", str(CH_SQ),
                                "--marker", str(CH_MK), "--dict", "4X4_50",
                                "--out", str(out)]
    try:
        runpy.run_path(os.path.join(repo, "tools", "calibrate.py"), run_name="__main__")
    except SystemExit as e:
        assert not e.code, f"calibrate.py (charuco) exited with {e.code}"
    finally:
        sys.argv = argv
    return folder, out


def test_charuco_recovers_the_true_camera(charuco_run):
    """Partial ChArUco views through the real tool must recover the camera the
    views were rendered with (zero distortion, so the solve is razor-sharp)."""
    _folder, out = charuco_run
    blob = json.loads(out.read_text())
    K = np.array(blob["K"])
    T = np.array(blob["T"])
    assert abs(K[0, 0] - K_TRUE[0, 0]) / K_TRUE[0, 0] < 0.01
    assert abs(K[0, 2] - K_TRUE[0, 2]) < 5 and abs(K[1, 2] - K_TRUE[1, 2]) < 5
    assert abs(np.linalg.norm(T) - 5.0) < 0.05       # the CNC step
    calib = StereoCalibration.load(str(out))
    rect = Rectifier(calib, (W, H))
    assert abs(rect.baseline - 5.0) < 0.05
