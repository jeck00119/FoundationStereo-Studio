"""studio.analyze — the Phase-2 fixes: pin verticality by blob shape, true RMS."""
import numpy as np

from studio.analyze import pin_analysis, region_flatness

BOARD_N = np.array([0.0, 0.0, -1.0])     # board normal toward the camera (-Z)
BOARD_C = np.array([0.0, 0.0, 10.0])     # board plane at z = 10


def _rng():
    return np.random.default_rng(3)


def test_vertical_shaft_reads_near_zero_tilt():
    """An elongated cylinder along the board normal — the classic shaft case."""
    rng = _rng()
    th = rng.uniform(0, 2 * np.pi, 3000)
    z = rng.uniform(4.0, 10.0, 3000)                 # heights 0..6 above the board
    pts = np.stack([0.3 * np.cos(th), 0.3 * np.sin(th), z], 1)
    r = pin_analysis(pts, BOARD_N, BOARD_C)
    assert r is not None
    assert abs(r["height"] - 6.0) < 0.1
    assert r["verticality"] is not None and r["verticality"] < 3.0


def test_flat_topped_pin_disc_reads_near_zero_tilt():
    """The Phase-2 fix: a stereo cloud of a flat-topped pin is mostly its top
    disc. The first PCA axis lies IN the disc, so the old code read a perfectly
    vertical pin as ~90° tilted; the disc normal must be used instead."""
    rng = _rng()
    n = 3000
    rad = np.sqrt(rng.uniform(0, 1, n)) * 2.0        # filled disc, radius 2
    th = rng.uniform(0, 2 * np.pi, n)
    z = np.full(n, 4.0) + rng.normal(0, 0.01, n)     # flat top, 6 above the board
    pts = np.stack([rad * np.cos(th), rad * np.sin(th), z], 1)
    r = pin_analysis(pts, BOARD_N, BOARD_C)
    assert r is not None
    assert abs(r["height"] - 6.0) < 0.1
    assert r["verticality"] is not None and r["verticality"] < 3.0


def test_blob_returns_no_verticality():
    """An isotropic blob has no defensible axis — must answer None, not a number."""
    rng = _rng()
    pts = rng.normal(0, 1.0, (3000, 3)) + [0, 0, 5.0]
    r = pin_analysis(pts, BOARD_N, BOARD_C)
    assert r is not None
    assert r["verticality"] is None


def test_pin_too_sparse_returns_none():
    assert pin_analysis(np.zeros((5, 3)), BOARD_N, BOARD_C) is None


def test_region_rms_is_true_rms_of_gaussian_noise():
    """A flat patch with N(0, σ) roughness must report rms ≈ σ."""
    rng = _rng()
    sigma = 0.05
    xy = rng.uniform(-5, 5, (20000, 2))
    z = np.full(len(xy), 10.0) + rng.normal(0, sigma, len(xy))
    pts = np.column_stack([xy, z])
    A = np.array([-4.0, -4.0, 10.0])
    B = np.array([4.0, 4.0, 10.0])
    r = region_flatness(pts, A, B, BOARD_N, BOARD_C)
    assert r is not None
    assert abs(r["rms"] - sigma) / sigma < 0.15
    assert abs(r["z_mean"]) < 0.01           # the patch sits ON the board plane


def test_surface_profile_recovers_known_ramp_angle():
    """A plane tilted by a known angle about X: the profile's slope angle and
    rise must match, and the plotted t/h arrays must span the picked line."""
    from studio.analyze import surface_profile

    rng = _rng()
    ang = np.deg2rad(5.0)
    xy = rng.uniform(-10, 10, (40000, 2))
    # height above the board grows with y at tan(5°); board plane at z=10, n=-Z:
    # a point's z = 10 - h  (h measured along -Z toward the camera)
    h = np.tan(ang) * xy[:, 1] + rng.normal(0, 0.01, len(xy))
    pts = np.column_stack([xy[:, 0], xy[:, 1], 10.0 - h])
    A = np.array([0.0, -8.0, 10.0 - np.tan(ang) * -8.0])
    B = np.array([0.0, 8.0, 10.0 - np.tan(ang) * 8.0])
    r = surface_profile(pts, A, B, BOARD_N, BOARD_C)
    assert r is not None
    assert abs(abs(r["angle"]) - 5.0) < 0.3
    assert abs(abs(r["d_height"]) - 16.0 * np.tan(ang)) < 0.05
    assert len(r["t"]) == len(r["h"]) >= 2
    assert 0.0 <= r["t"][0] and r["t"][-1] <= 16.0 + 1e-6   # bins live on the picked line
    assert r["poly"].shape[1] == 3                           # 3D polyline for the view
