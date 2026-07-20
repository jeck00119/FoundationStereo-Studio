"""studio.measure — OBB membership + measurement in the box frame."""
import numpy as np

from studio.measure import MeasureBox, measure_box, points_in_box


def _pin_cloud(h=6.0, r=0.5, n=4000, seed=7):
    """A vertical cylinder shell (a 'pin') from z=0..h, centred on the origin."""
    rng = np.random.default_rng(seed)
    th = rng.uniform(0, 2 * np.pi, n)
    z = rng.uniform(0, h, n)
    return np.stack([r * np.cos(th), r * np.sin(th), z], 1).astype(np.float32)


def test_points_in_box_axis_aligned():
    pts = _pin_cloud()
    box = MeasureBox(cx=0, cy=0, cz=3.0, sx=2.0, sy=2.0, sz=6.0)
    assert points_in_box(pts, box).all()
    tight = MeasureBox(cx=0, cy=0, cz=1.0, sx=2.0, sy=2.0, sz=2.0)
    m = points_in_box(pts, tight)
    assert 0 < m.sum() < len(pts)
    assert (pts[m][:, 2] <= 2.0 + 1e-6).all()


def test_measure_box_height_along_box_axis():
    pts = _pin_cloud(h=6.0)
    box = MeasureBox(cx=0, cy=0, cz=3.0, sx=2.0, sy=2.0, sz=8.0)
    m = measure_box(pts, box, trim_pct=0.0)
    assert m is not None
    assert abs(m["h_span"] - 6.0) < 0.05


def test_rotated_box_measures_rotated_pin_identically():
    """Rotate pin AND box together by the same quaternion — height must match
    the axis-aligned measurement (the whole point of the OBB)."""
    pts = _pin_cloud(h=6.0)
    ang = np.deg2rad(30.0)
    qx, qw = np.sin(ang / 2), np.cos(ang / 2)      # rotation about X by 30°
    R = np.array([[1, 0, 0],
                  [0, np.cos(ang), -np.sin(ang)],
                  [0, np.sin(ang), np.cos(ang)]])
    rpts = (pts @ R.T).astype(np.float32)
    c = R @ np.array([0, 0, 3.0])
    box = MeasureBox(cx=c[0], cy=c[1], cz=c[2], sx=2.0, sy=2.0, sz=8.0,
                     qx=qx, qy=0, qz=0, qw=qw)
    m = measure_box(rpts, box, trim_pct=0.0)
    assert m is not None
    assert abs(m["h_span"] - 6.0) < 0.05


def test_measure_box_empty_returns_none():
    pts = _pin_cloud()
    far = MeasureBox(cx=100, cy=100, cz=100, sx=1, sy=1, sz=1)
    assert measure_box(pts, far, trim_pct=0.0) is None


def test_scaled_box_tracks_unit_switch():
    box = MeasureBox(cx=1, cy=-2, cz=3, sx=4, sy=5, sz=6, qx=0.1, qy=0.2, qz=0.3, qw=0.9)
    s = box.scaled(1000.0)
    assert (s.cx, s.cy, s.cz, s.sx, s.sy, s.sz) == (1000, -2000, 3000, 4000, 5000, 6000)
    assert (s.qx, s.qy, s.qz, s.qw) == (0.1, 0.2, 0.3, 0.9)   # rotation is unitless


def test_trim_cleans_a_flyer():
    """Raw span chases the single most extreme point; the trimmed span is the
    repeatability-grade number — the diagnostic the readout is built around."""
    pts = _pin_cloud(h=6.0, n=5000)
    pts = np.vstack([pts, [[0.0, 0.0, 60.0]]]).astype(np.float32)   # one flyer far above
    box = MeasureBox(cx=0, cy=0, cz=30.0, sx=2.0, sy=2.0, sz=62.0)
    m = measure_box(pts, box, trim_pct=2.0)
    assert m["h_span"] > 50.0                    # raw span measures the flyer
    assert abs(m["h_span_t"] - 6.0) < 0.6        # trimmed span measures the pin
    m0 = measure_box(pts, box, trim_pct=0.0)
    assert abs(m0["h_span_t"] - m0["h_span"]) < 1e-9   # trim 0 == raw


def test_fit_plane_latches_board_not_pins():
    """A board plus tall pins: the two-pass fit must return the BOARD's normal."""
    from studio.measure import fit_plane

    rng = np.random.default_rng(11)
    board = np.column_stack([rng.uniform(-20, 20, 20000),
                             rng.uniform(-20, 20, 20000),
                             rng.normal(0, 0.02, 20000)])
    th = rng.uniform(0, 2 * np.pi, 2000)
    pins = np.column_stack([np.cos(th), np.sin(th), rng.uniform(0, 8.0, 2000)])
    n, c = fit_plane(np.vstack([board, pins]))
    assert abs(abs(n[2]) - 1.0) < 1e-3           # normal is ±Z (the board)
    assert abs(c[2]) < 0.5                       # centroid ON the board, not up the pins


def test_rotation_to_axis_maps_normal_onto_target():
    from studio.measure import rotation_to_axis

    n = np.array([0.2, -0.3, -0.9]); n /= np.linalg.norm(n)
    R = rotation_to_axis(n, (0.0, 0.0, -1.0))
    np.testing.assert_allclose(R @ n, [0.0, 0.0, -1.0], atol=1e-9)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)      # proper rotation
    assert abs(np.linalg.det(R) - 1.0) < 1e-9
    np.testing.assert_allclose(rotation_to_axis((0, 0, -1), (0, 0, -1)),
                               np.eye(3), atol=1e-12)              # parallel → identity
