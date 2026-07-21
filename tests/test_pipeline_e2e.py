"""End-to-end pipeline ground truth — the exact chain the engine child runs
(run_inference → build_cloud), driven by a fake backend that emits the
analytically known disparity of a synthetic scene. No GPU, no Qt.

What this pins: depth = fx·B/disp at ANY input scale, K scaling, the
dual-reference flip trick, the left/right merge geometry, remove-invisible,
z-clipping and denoise — all through the real code path, not re-derived math.
"""
import numpy as np
import pytest

from studio.backends.base import StereoBackend
from studio.cloud import build_cloud
from studio.dtypes import DisparityResult, StereoParams
from studio.infer import run_inference

FULL_W, FULL_H = 200, 120
FX_FULL = 1000.0
BASELINE = 5.0          # mm
Z_TRUE = 50.0           # mm — fronto-parallel plane


class FlatSceneBackend(StereoBackend):
    """Pretends to be a stereo network looking at a flat plane at Z_TRUE:
    disparity = fx·B/Z everywhere, at whatever working scale it is fed."""

    def __init__(self, z_map=None):
        self._z_map = z_map          # optional (FULL_H, FULL_W) depth override

    def load(self, ckpt_path, params=None, progress=None):
        pass

    @property
    def is_loaded(self):
        return True

    def disparity(self, img0, img1, params):
        H, W = img0.shape[:2]
        scale = float(params.scale)
        fx = FX_FULL * scale
        if self._z_map is None:
            d = np.full((H, W), fx * BASELINE / Z_TRUE, np.float32)
        else:
            # nearest-sample the full-res depth map to the working grid
            ys = (np.arange(H) / scale).astype(int).clip(0, FULL_H - 1)
            xs = (np.arange(W) / scale).astype(int).clip(0, FULL_W - 1)
            z = self._z_map[np.ix_(ys, xs)]
            d = (fx * BASELINE / z).astype(np.float32)
        return DisparityResult(disp=d)


def _params(scale=1.0, **kw):
    p = StereoParams(scale=scale, fx=FX_FULL, fy=FX_FULL,
                     cx=FULL_W / 2, cy=FULL_H / 2, baseline=BASELINE,
                     z_near=0.0, z_far=1e9, remove_invisible=False, denoise=False)
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def _pair():
    rgb = np.full((FULL_H, FULL_W, 3), 128, np.uint8)
    return rgb, rgb.copy()


def test_depth_exact_at_full_scale():
    r = run_inference(FlatSceneBackend(), *_pair(), _params(scale=1.0))
    assert r.disp.shape == (FULL_H, FULL_W)
    valid = r.depth > 0
    assert valid.all()
    np.testing.assert_allclose(r.depth, Z_TRUE, rtol=1e-6)
    assert float(r.K[0, 0]) == FX_FULL and r.baseline == BASELINE


def test_depth_exact_at_half_scale():
    """The invariant every run relies on: scaling the images scales fx and the
    disparity together, so metric depth is UNCHANGED."""
    r = run_inference(FlatSceneBackend(), *_pair(), _params(scale=0.5))
    assert r.disp.shape == (FULL_H // 2, FULL_W // 2)
    np.testing.assert_allclose(r.depth[r.depth > 0], Z_TRUE, rtol=1e-6)
    assert float(r.K[0, 0]) == FX_FULL * 0.5      # K scaled with the working grid


def test_cloud_lands_on_the_true_plane():
    p = _params(scale=1.0)
    r = run_inference(FlatSceneBackend(), *_pair(), p)
    c = build_cloud(r, p)
    assert c.n == FULL_H * FULL_W
    np.testing.assert_allclose(c.points[:, 2], Z_TRUE, rtol=1e-6)
    # X spans what the camera geometry says it should: (u-cx)·Z/fx
    assert abs(c.points[:, 0].min() - (0 - FULL_W / 2) * Z_TRUE / FX_FULL) < 0.01
    assert abs(c.points[:, 0].max() - (FULL_W - 1 - FULL_W / 2) * Z_TRUE / FX_FULL) < 0.01


def test_remove_invisible_drops_unmatchable_columns():
    """Pixels whose match u−d falls off the right image can't be verified —
    remove_invisible must drop exactly those columns."""
    p = _params(scale=1.0, remove_invisible=True)
    r = run_inference(FlatSceneBackend(), *_pair(), p)
    d = int(round(FX_FULL * BASELINE / Z_TRUE))          # 100 px
    c = build_cloud(r, p)
    assert c.n == FULL_H * (FULL_W - d)
    u_min = (d - FULL_W / 2) * Z_TRUE / FX_FULL          # first surviving column's X
    assert abs(c.points[:, 0].min() - u_min) < 0.01


def test_dual_reference_merges_both_eyes_consistently():
    p = _params(scale=1.0, dual_reference=True)
    r = run_inference(FlatSceneBackend(), *_pair(), p)
    assert r.disp_right is not None and r.rgb_right is not None
    np.testing.assert_allclose(r.disp_right, r.disp, rtol=1e-6)   # flat scene: same d
    c = build_cloud(r, p)
    assert c.origin is not None and set(np.unique(c.origin)) == {0, 1}
    np.testing.assert_allclose(c.points[:, 2], Z_TRUE, rtol=1e-6)  # both eyes on the plane
    # right-eye points are translated +B into the left frame: their X range
    # extends B beyond the left eye's
    xl = c.points[c.origin == 0][:, 0]
    xr = c.points[c.origin == 1][:, 0]
    assert abs(xr.max() - (xl.max() + BASELINE)) < 0.02
    # a constant-disparity pair is perfectly left-right consistent where in-frame
    assert c.reliable is not None and c.reliable.mean() > 0.4


def test_z_clip_separates_a_two_level_scene():
    z = np.full((FULL_H, FULL_W), Z_TRUE, np.float64)
    z[:, FULL_W // 2:] = 2 * Z_TRUE                       # a step: 50 mm and 100 mm
    p = _params(scale=1.0)
    r = run_inference(FlatSceneBackend(z_map=z), *_pair(), p)
    c_all = build_cloud(r, p)
    assert c_all.n == FULL_H * FULL_W
    p.z_far = 1.5 * Z_TRUE                                # keep only the near level
    c_near = build_cloud(r, p)
    assert c_near.n == FULL_H * (FULL_W // 2)
    assert c_near.points[:, 2].max() < 1.5 * Z_TRUE


def test_denoise_removes_a_planted_flyer():
    pytest.importorskip("open3d")
    z = np.full((FULL_H, FULL_W), Z_TRUE, np.float64)
    z[FULL_H // 2, FULL_W // 2] = Z_TRUE / 5              # one absurdly near point
    p = _params(scale=1.0, denoise=True, denoise_nb_points=20, denoise_std=2.0)
    r = run_inference(FlatSceneBackend(z_map=z), *_pair(), p)
    c = build_cloud(r, p)
    assert c.n < FULL_H * FULL_W                          # something was removed…
    assert c.points[:, 2].min() > Z_TRUE * 0.9            # …and it was the flyer


def test_no_calibration_means_no_cloud():
    p = _params(scale=1.0, fx=0.0, baseline=0.0)
    r = run_inference(FlatSceneBackend(), *_pair(), p)
    assert r.depth is None and r.K is None
    assert build_cloud(r, p) is None


def test_denoise_degrades_without_open3d(monkeypatch):
    """A missing open3d (Jetson aarch64 wheel gaps) must skip the denoise step,
    not kill the run — the cloud comes back unfiltered."""
    import studio.cloud as cloud_mod

    def _no_o3d(*a, **k):
        raise ImportError("No module named 'open3d'")

    monkeypatch.setattr(cloud_mod, "denoise_mask", _no_o3d)
    p = _params(scale=1.0, denoise=True)
    r = run_inference(FlatSceneBackend(), *_pair(), p)
    c = build_cloud(r, p)
    assert c is not None and c.n == FULL_H * FULL_W    # everything kept, nothing crashed
