"""studio.cloud — back-projection, LR-consistency, unit-agnostic depth math."""
import numpy as np

from studio.cloud import build_cloud, lrc_mask
from studio.dtypes import InferResult, StereoParams


def _flat_scene(H=8, W=10, disp_val=5.0, fx=100.0, baseline=2.0):
    """A synthetic fronto-parallel plane: constant disparity everywhere."""
    disp = np.full((H, W), disp_val, np.float32)
    depth = np.full((H, W), fx * baseline / disp_val, np.float32)
    rgb = np.full((H, W, 3), 128, np.uint8)
    K = np.array([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]], np.float32)
    r = InferResult(disp=disp, depth=depth, rgb=rgb, H=H, W=W, scale=1.0,
                    timing={}, K=K, baseline=baseline)
    p = StereoParams(fx=fx, fy=fx, cx=W / 2, cy=H / 2, baseline=baseline,
                     z_far=1e9, remove_invisible=False, denoise=False)
    return r, p


def test_build_cloud_depth_matches_fx_b_over_d():
    r, p = _flat_scene()
    c = build_cloud(r, p)
    assert c is not None and c.n == 8 * 10
    np.testing.assert_allclose(c.points[:, 2], 100.0 * 2.0 / 5.0, rtol=1e-5)


def test_build_cloud_z_clip():
    r, p = _flat_scene()          # all points at z = 40
    p.z_near, p.z_far = 41.0, 100.0
    assert build_cloud(r, p).n == 0
    p.z_near, p.z_far = 1.0, 39.0
    assert build_cloud(r, p).n == 0
    p.z_near, p.z_far = 1.0, 41.0
    assert build_cloud(r, p).n == 8 * 10


def test_build_cloud_none_without_calibration():
    r, p = _flat_scene()
    r.depth = None
    assert build_cloud(r, p) is None


def test_lrc_mask_consistent_flat_field():
    # constant disparity: left pixel u matches right pixel u-d with the same d
    H, W, d = 4, 20, 6.0
    dl = np.full((H, W), d, np.float32)
    dr = np.full((H, W), d, np.float32)
    m = lrc_mask(dl, dr, sign=-1)
    assert m[:, 6:].all()          # everything with an in-frame match agrees
    assert not m[:, :6].any()      # matches falling off the frame are rejected


def test_lrc_mask_detects_disagreement():
    H, W = 4, 20
    dl = np.full((H, W), 6.0, np.float32)
    dr = np.full((H, W), 9.0, np.float32)   # inconsistent by 3 px (> 1 px tol)
    assert not lrc_mask(dl, dr, sign=-1).any()


def test_unit_agnostic_rescale_equivalence():
    """mm and m descriptions of the same physical scene give the same cloud
    up to the unit factor — the invariant every unit switch relies on."""
    r_mm, p_mm = _flat_scene(baseline=5.0)          # 5 mm
    r_m, p_m = _flat_scene(baseline=0.005)          # same rig in metres
    c_mm = build_cloud(r_mm, p_mm)
    c_m = build_cloud(r_m, p_m)
    np.testing.assert_allclose(c_mm.points, c_m.points * 1000.0, rtol=1e-4)
