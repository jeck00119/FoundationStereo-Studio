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
