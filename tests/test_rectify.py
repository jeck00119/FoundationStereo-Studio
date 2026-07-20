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
