"""Volume-box measurement over a point cloud — pure numpy: no Qt, no torch, no GPU.

Kept out of the viewers so the arithmetic can be tested on its own and reused by a
batch pass later. Nothing here goes near the engine child either, so editing a box
is instant however big the cloud is.

The box is an ORIENTED box (an OBB): it carries a rotation, so it can be tilted to
lie along a pin on a board that isn't perfectly square to the camera. Every spatial
number is measured in the box's OWN frame — in particular the pin "height" is the
extent along the box's local Z axis, which is the true tip-to-base distance once
you have aligned the box to the pin, not a world-Z projection that would fold in the
board's tilt. When the box is left un-rotated this collapses exactly to the plain
axis-aligned behaviour, so nothing regresses.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Ceiling on the voxel grid one measurement may address. Occupancy is counted by
# hashing each point's (i,j,k) cell to a single int64, so the cost is O(points)
# and a fine grid is cheap — but that hash is (i*ny + j)*nz + k, and the product
# has to stay inside int64. 1e15 keeps four orders of magnitude of headroom under
# int64's 9.2e18 and still allows a 100 mm box at a 1 µm voxel, which is already
# far finer than anything this rig can resolve.
_MAX_CELLS = 1e15

# unit cube corners in [-1,1]^3, ordered so _EDGES wires them into 12 segments
_UNIT_CORNERS = np.array([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
], np.float64)


@dataclass
class MeasureBox:
    """An oriented box in WORLD coordinates.

    World = the frame of ``CloudResult.points`` and the Depth tab: X right, Y
    down, Z = distance from the camera, in whatever unit ``baseline`` was given in
    (mm by default). The 3D view negates Y and Z so the cloud looks upright — that
    flip is a DISPLAY detail and stops at the draw call; every number here is world.

    Orientation is a quaternion (x,y,z,w), identity by default. It is a quaternion
    rather than Euler angles specifically so the box the user rotates with the
    three.js gizmo and the box measured here are the SAME pose — the gizmo speaks
    quaternions, and going through an Euler triple in between would invite an
    axis-order mismatch that silently rotates the measurement off the pin.
    """

    cx: float = 0.0
    cy: float = 0.0
    cz: float = 0.0
    sx: float = 5.0
    sy: float = 5.0
    sz: float = 5.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0

    # ------------------------------------------------------------------ pose
    @property
    def center(self) -> np.ndarray:
        return np.array([self.cx, self.cy, self.cz], np.float64)

    @property
    def half(self) -> np.ndarray:
        return np.array([self.sx, self.sy, self.sz], np.float64) / 2.0

    @property
    def is_axis_aligned(self) -> bool:
        return (abs(self.qx) < 1e-9 and abs(self.qy) < 1e-9
                and abs(self.qz) < 1e-9 and abs(abs(self.qw) - 1.0) < 1e-9)

    def rotation_matrix(self) -> np.ndarray:
        """R mapping LOCAL axes → WORLD (its columns are the box's axes in world).

        So ``(p - center) @ R`` projects a world point onto the box axes, i.e.
        gives its LOCAL coordinates — that is the whole containment/height test.
        """
        x, y, z, w = self.qx, self.qy, self.qz, self.qw
        n = x * x + y * y + z * z + w * w
        if n < 1e-12:
            return np.eye(3)
        s = 2.0 / n                      # handles a non-unit quaternion too
        xx, yy, zz = x * x * s, y * y * s, z * z * s
        xy, xz, yz = x * y * s, x * z * s, y * z * s
        wx, wy, wz = w * x * s, w * y * s, w * z * s
        return np.array([
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ], np.float64)

    def to_local(self, points) -> np.ndarray:
        """World points → the box's local frame (centre at 0, axes = box axes)."""
        return (np.asarray(points, np.float64) - self.center) @ self.rotation_matrix()

    # --------------------------------------------------------------- geometry
    def corners(self) -> np.ndarray:
        """The 8 world corners of the (possibly rotated) box, ordered for _EDGES."""
        u = _UNIT_CORNERS * self.half                       # local corners at ±half
        return (self.center + u @ self.rotation_matrix().T).astype(np.float32)

    @property
    def lo(self) -> np.ndarray:
        """World AABB min — the axis-aligned bound of the oriented box (== centre−half
        when un-rotated). Used as a cheap prefilter and by any axis-aligned drawing."""
        return self.corners().min(0).astype(np.float64)

    @property
    def hi(self) -> np.ndarray:
        return self.corners().max(0).astype(np.float64)

    @property
    def volume(self) -> float:
        return float(self.sx) * float(self.sy) * float(self.sz)

    def scaled(self, f: float) -> "MeasureBox":
        """A copy with every length multiplied by `f` — a mm⇄m switch. Orientation
        is unit-free, so the quaternion rides along unchanged.

        The box is a physical volume sitting on physical points. A unit switch
        rescales the points, so it has to move the box by exactly the same factor
        or the box silently ends up measuring somewhere else.
        """
        return MeasureBox(self.cx * f, self.cy * f, self.cz * f,
                          self.sx * f, self.sy * f, self.sz * f,
                          self.qx, self.qy, self.qz, self.qw)


# (the wireframe the 3D view draws is built by the web page itself — the old
# _EDGES/edges_from_bounds/box_wireframe helpers had no callers left and are gone)


def points_in_box(points, box: MeasureBox) -> np.ndarray:
    """Bool mask over `points` of what lies inside `box` (bounds inclusive).

    Axis-aligned boxes take the plain min/max path — the common case. A rotated
    box needs the exact local-frame test, but running that
    matmul over the whole cloud every drag would be wasteful, so it is gated behind
    a cheap world-AABB prefilter: only points inside the oriented box's bounding
    box get projected onto the box axes.
    """
    if points is None or len(points) == 0:
        return np.zeros(0, bool)
    p = np.asarray(points)
    half = box.half
    # dtype-match the bounds to the cloud so a float64 bound doesn't silently
    # upcast (and thus fully copy) a float32 cloud on every comparison.
    dt = p.dtype if p.dtype.kind == "f" else np.float64
    if box.is_axis_aligned:
        lo = (box.center - half).astype(dt)
        hi = (box.center + half).astype(dt)
        return np.all((p >= lo) & (p <= hi), axis=1)
    pre = np.all((p >= box.lo.astype(dt)) & (p <= box.hi.astype(dt)), axis=1)
    out = np.zeros(len(p), bool)
    if not pre.any():
        return out
    local = (p[pre].astype(np.float64, copy=False) - box.center) @ box.rotation_matrix()
    inside = np.all(np.abs(local) <= half, axis=1)
    out[np.nonzero(pre)[0][inside]] = True
    return out


def _occupied_local(local: np.ndarray, half: np.ndarray, voxel: float):
    """(occupied cell count, the volume they take up INSIDE the box), computed in
    the box's LOCAL frame so it works identically for a rotated box.

    Be clear about what this measures on a STEREO cloud: the reconstruction is one
    visible surface, not a solid, so the occupied cells form a SHELL over whatever
    the camera can see. It answers "how much of this box has anything in it" — it
    is not the material volume of the object, and one viewpoint cannot give you
    that. It also moves with the voxel size (a shell's cell count scales with its
    area / voxel²), so compare these numbers only at the same voxel.

    Cells are CLIPPED to the box: the grid tiles from −half in whole cubes, so the
    last cell on each axis usually overhangs the far face; counting it as a full
    cube let the filled volume exceed the box (a 1.6 mm box at a 0.5 mm voxel tiles
    2.0 mm → 122% full). The final cell contributes only its remainder, so a solid
    box reads exactly 100%.
    """
    if voxel <= 0:
        return None
    span = 2.0 * half
    ncell = np.maximum(np.ceil(span / voxel).astype(np.int64), 1)
    if float(ncell[0]) * float(ncell[1]) * float(ncell[2]) > _MAX_CELLS:
        return None
    idx = np.floor((local + half) / voxel).astype(np.int64)   # local runs −half..+half
    idx = np.clip(idx, 0, ncell - 1)
    key = (idx[:, 0] * ncell[1] + idx[:, 1]) * ncell[2] + idx[:, 2]
    u = np.unique(key)
    w = []
    for a in range(3):
        wa = np.full(int(ncell[a]), float(voxel))
        wa[-1] = float(span[a]) - (int(ncell[a]) - 1) * float(voxel)
        w.append(wa)
    ka = u % ncell[2]
    ja = (u // ncell[2]) % ncell[1]
    ia = u // (ncell[2] * ncell[1])
    return int(len(u)), float((w[0][ia] * w[1][ja] * w[2][ka]).sum())


def measure_box(points, box: MeasureBox, trim_pct: float = 2.0,
                voxel: float = 0.0) -> dict | None:
    """Everything the box has to say about the points inside it — None if it
    caught nothing.

    Spatial stats are in the box's LOCAL frame: ``h_*`` is the height along the
    box's own Z axis (the pin height once the box is aligned to the pin), and
    ``sec_x/sec_y`` are the section across it. ``z_*`` stay in WORLD z — the depth
    range of the caught points — so the depth readout keeps meaning regardless of
    how the box is turned (and collapses onto ``h_*`` when the box is un-rotated).

    Every extreme is reported raw AND trimmed, because raw min/max are one point
    each and this rig's cloud already scatters ~0.6-1 mm about a flat surface — so
    the single most extreme point in a box is a flyer that ``max − min`` chases and
    that cannot repeat between two captures. The trimmed pair drops `trim_pct` off
    each end and is the one a repeatability study should use; the GAP between raw
    and trimmed is itself the "is this box full of noise?" diagnostic.
    """
    if points is None or len(points) == 0:
        return None
    p = np.asarray(points)
    m = points_in_box(p, box)
    n = int(m.sum())
    if n == 0:
        return None
    # Upcast the SURVIVORS, not the cloud: a few thousand points, where float64 is
    # free and keeps the percentile interpolation exact.
    inside = p[m].astype(np.float64, copy=False)
    local = (inside - box.center) @ box.rotation_matrix()     # box frame
    z = inside[:, 2]                                          # world depth
    lz = local[:, 2]                                          # height axis
    t = float(np.clip(trim_pct, 0.0, 49.0))
    h_lo_t, h_hi_t = (float(v) for v in np.percentile(lz, [t, 100.0 - t]))
    z_lo_t, z_hi_t = (float(v) for v in np.percentile(z, [t, 100.0 - t]))
    lmn, lmx = local.min(0), local.max(0)
    out = {
        "n": n,
        # height along the box's own Z axis — the rotation-aware measurement
        "h_min": float(lmn[2]), "h_max": float(lmx[2]), "h_span": float(lmx[2] - lmn[2]),
        "h_min_t": h_lo_t, "h_max_t": h_hi_t, "h_span_t": h_hi_t - h_lo_t,
        "sec_x": float(lmx[0] - lmn[0]), "sec_y": float(lmx[1] - lmn[1]),
        # world depth of the caught points (identity-box back-compat: == h_* offset)
        "z_min": float(z.min()), "z_max": float(z.max()), "z_span": float(z.max() - z.min()),
        "z_min_t": z_lo_t, "z_max_t": z_hi_t, "z_span_t": z_hi_t - z_lo_t,
        "z_med": float(np.median(z)),
        "ext": (float(lmx[0] - lmn[0]), float(lmx[1] - lmn[1]), float(lmx[2] - lmn[2])),
        "box_vol": box.volume, "trim_pct": t,
        "occ_vol": None, "occ_cells": 0, "voxel": float(voxel), "fill_pct": None,
    }
    occ = _occupied_local(local, box.half, float(voxel))
    if occ is not None:
        cells, vol = occ
        out["occ_cells"] = cells
        out["occ_vol"] = vol
        if box.volume > 0:
            out["fill_pct"] = 100.0 * vol / box.volume
    return out


# ------------------------------------------------------------ plane levelling
def fit_plane(points):
    """Robust dominant-plane fit → (unit normal, centroid float64).

    Two-pass PCA: fit a plane to everything, drop the points farthest from it
    (the pins standing off the board, edges, flyers), then refit on the flat
    majority — so it latches onto the BOARD surface, not the pins on top of it.
    Deterministic (SVD + quantile, no RANSAC)."""
    p = np.asarray(points, np.float64)
    if len(p) > 200_000:                         # deterministic subsample — keep it cheap
        p = p[np.linspace(0, len(p) - 1, 200_000).astype(np.int64)]
    c = p.mean(0)
    _, _, vt = np.linalg.svd(p - c, full_matrices=False)
    n = vt[-1]
    d = np.abs((p - c) @ n)
    inl = p[d <= np.quantile(d, 0.75)]           # keep the flattest 75% (the board)
    if len(inl) >= 3:
        c = inl.mean(0)
        _, _, vt = np.linalg.svd(inl - c, full_matrices=False)
        n = vt[-1]
    return n / (np.linalg.norm(n) + 1e-12), c


def rotation_to_axis(n, target=(0.0, 0.0, 1.0)):
    """3x3 rotation matrix mapping unit vector ``n`` onto ``target`` by the minimal
    rotation (Rodrigues). ``target`` should be the nearest axis so it's a small turn."""
    n = np.asarray(n, np.float64); n = n / (np.linalg.norm(n) + 1e-12)
    t = np.asarray(target, np.float64); t = t / (np.linalg.norm(t) + 1e-12)
    v = np.cross(n, t)
    s = float(np.linalg.norm(v))
    c = float(np.dot(n, t))
    if s < 1e-9:                                  # already parallel (target = nearest axis)
        return np.eye(3)
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))
