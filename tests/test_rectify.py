"""studio.rectify — derived rectified intrinsics + baseline from a calibration."""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from studio.rectify import CalibrationError, Rectifier, StereoCalibration


def _pure_translation_calib(baseline=5.0, size=(640, 480)):
    K = [[800.0, 0, 320.0], [0, 800.0, 240.0], [0, 0, 1]]
    return StereoCalibration(K=K, D=[0, 0, 0, 0, 0], R=np.eye(3),
                             T=[baseline, 0, 0], image_size=size)


def test_baseline_survives_rectification():
    calib = _pure_translation_calib(baseline=5.0)
    r = Rectifier(calib, (640, 480))
    assert abs(r.baseline - 5.0) < 1e-6
    assert r.fx > 0 and r.fy > 0


def test_k_scaling_for_other_resolution():
    """Calibration solved at 640×480, image at 1280×960 — fx/cx must scale ×2."""
    calib = _pure_translation_calib()
    r1 = Rectifier(calib, (640, 480))
    r2 = Rectifier(calib, (1280, 960))
    assert abs(r2.fx / r1.fx - 2.0) < 1e-3
    assert abs(r2.cx / r1.cx - 2.0) < 0.05
    assert abs(r2.baseline - r1.baseline) < 1e-6    # baseline is physical, unchanged


def test_missing_fields_raise_calibration_error(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text('{"K": [[1,0,0],[0,1,0],[0,0,1]], "D": [0,0,0,0,0]}')
    with pytest.raises(CalibrationError):
        StereoCalibration.load(str(p))


def test_json_roundtrip_like_calibrate_output(tmp_path):
    """The exact key set tools/calibrate.py writes must load."""
    import json
    p = tmp_path / "calib.json"
    p.write_text(json.dumps({
        "K": [[800, 0, 320], [0, 800, 240], [0, 0, 1]],
        "D": [0.01, -0.02, 0, 0, 0],
        "R": np.eye(3).tolist(),
        "T": [5.0, 0.01, -0.02],
        "image_width": 640, "image_height": 480,
        "baseline_unit": "mm",
    }))
    c = StereoCalibration.load(str(p))
    assert c.image_size == (640, 480)
    assert abs(c.baseline_raw - np.linalg.norm([5.0, 0.01, -0.02])) < 1e-9


# ------------------------------------------------- ROI-only rectification
def _rgb(w=640, h=480, seed=5):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def test_rectify_roi_is_identical_to_cropping_a_full_rectify():
    """The fast path must be EXACT, not merely close — it replaces the slow one.
    (Measured on the rig's 4024x3036 pair: 104 ms -> 6 ms per image.)"""
    calib = _pure_translation_calib()
    r = Rectifier(calib, (640, 480))
    img = _rgb()
    for side, (x0, y0, w, h) in (("L", (100, 60, 128, 96)),
                                 ("R", (0, 0, 64, 64)),
                                 ("L", (500, 400, 128, 96))):   # runs past the edge
        full = r.rectify(img, side)
        roi = r.rectify_roi(img, side, x0, y0, w, h)
        ww, hh = roi.shape[1], roi.shape[0]
        assert np.array_equal(roi, full[y0:y0 + hh, x0:x0 + ww])


def test_rectify_roi_rejects_a_negative_origin():
    """numpy would WRAP a negative slice start silently, remapping the opposite
    edge of the frame with no error — the one failure mode worth an exception."""
    r = Rectifier(_pure_translation_calib(), (640, 480))
    with pytest.raises(ValueError, match="negative origin"):
        r.rectify_roi(_rgb(), "R", -216, 0, 128, 96)
    with pytest.raises(ValueError, match="outside"):
        r.rectify_roi(_rgb(), "L", 700, 0, 128, 96)


def test_rectify_roi_clamps_a_window_that_overruns_the_frame():
    r = Rectifier(_pure_translation_calib(), (640, 480))
    out = r.rectify_roi(_rgb(), "L", 600, 440, 256, 256)
    assert out.shape[:2] == (480 - 440, 640 - 600)


# --------------------------------------------- measuring the pre-shift Δ
def _textured(w=900, h=300, seed=11):
    """A textured frame — matchTemplate needs something to lock onto."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def test_find_disparity_shift_recovers_a_known_offset():
    from studio.rectify import find_disparity_shift

    left = _textured()
    d = 240
    right = np.zeros_like(left)          # right[v] shows what left has at v+d
    right[:, :left.shape[1] - d] = left[:, d:]
    r = find_disparity_shift(left, right, (400, 60, 256, 128))
    assert r["ok"] is True
    assert r["shift"] == pytest.approx(d)
    assert r["dy"] == 0
    assert r["score"] > 0.9


def test_find_disparity_shift_reports_a_row_offset():
    """A pair that is NOT row-aligned is a calibration problem — say so rather
    than quietly matching a few rows down."""
    from studio.rectify import find_disparity_shift

    left = _textured()
    d, dy = 200, 5
    right = np.zeros_like(left)
    right[:left.shape[0] - dy, :left.shape[1] - d] = left[dy:, d:]
    r = find_disparity_shift(left, right, (400, 60, 256, 128))
    assert r["shift"] == pytest.approx(d)
    assert r["dy"] == -dy


def test_find_disparity_shift_refuses_a_textureless_roi():
    """A flat patch matches everywhere; a confident-looking Δ from it would
    silently mis-crop every pair of the study."""
    from studio.rectify import find_disparity_shift

    flat = np.full((300, 900, 3), 128, np.uint8)
    r = find_disparity_shift(flat, flat.copy(), (400, 60, 256, 128))
    assert r["ok"] is False
    assert r["texture"] < 3.0
    # the point of the texture gate: the CORRELATION still looks perfect here
    assert r["score"] > 0.9

    # near-flat (sensor noise only) must also be refused
    noisy = flat + np.random.default_rng(2).integers(0, 2, flat.shape, dtype=np.uint8)
    assert find_disparity_shift(noisy, noisy.copy(), (400, 60, 256, 128))["ok"] is False


def test_find_disparity_shift_survives_an_oversized_roi():
    from studio.rectify import find_disparity_shift

    left = _textured()
    r = find_disparity_shift(left, left.copy(), (10, 10, 5000, 5000))
    assert r["ok"] is False          # nothing left of column 10 to search
