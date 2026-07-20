"""Point-cloud analysis tools — pure numpy over a cloud's points (display unit).

Everything references the fitted BOARD PLANE (measure.fit_plane): "height" is signed
distance along the plane normal, and angles are relative to the plane — so on a
levelled board these read as true stand-off and slope. The window collects picked
points from the 3D view and calls these; nothing here touches Qt or the engine, so
it is instant and headless-testable.
"""
from __future__ import annotations

import numpy as np

from .measure import fit_plane

__all__ = ["board_plane", "plane_basis", "surface_profile", "point_distance",
           "region_flatness", "deviation", "pin_analysis"]


def board_plane(points):
    """(normal, centroid) of the dominant plane, normal oriented toward the camera
    (−Z), so 'height' along it is positive AWAY from the board (toward the parts)."""
    n, c = fit_plane(points)
    if n[2] > 0:
        n = -n
    return n, np.asarray(c, np.float64)


def plane_basis(n):
    """An orthonormal in-plane basis (u, v) for unit normal n."""
    n = np.asarray(n, np.float64); n = n / np.linalg.norm(n)
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    u = a - (a @ n) * n; u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v


def _project(points, c, u, v, n):
    """(in-plane U, in-plane V, height H) coords of points in the plane frame."""
    d = np.asarray(points, np.float64) - np.asarray(c, np.float64)
    return d @ u, d @ v, d @ n


def _layer_mask(h, seed):
    """Boolean mask of the height-CONNECTED layer that contains `seed`. A selection that
    spans several Z levels (e.g. floating pin heads ABOVE the board) keeps only the level
    you clicked on, not the whole Z column; a single continuous surface (one layer, or a
    pin joined to its board) stays whole, because 'connected' = no EMPTY height gap.

    Walks out from the seed through the SORTED heights and stops at the first GAP wider
    than a level break — defined as a big fraction of the robust height spread (18 % of the
    2–98 percentile range). A separate level (floating heads well above the board) leaves a
    gap that big; a single surface's own sparse TAIL, a sparse WALL down into a pit/defect,
    or a small/tilted patch have only small gaps, so the level — and its extremes and any
    connected defect — stays WHOLE (trimming them would drop the very max−min a flatness
    read exists to catch, a false pass). The percentile spread ignores stray flyers so one
    outlier can't blow up the threshold. Returns all-True when there's nothing to isolate."""
    h = np.asarray(h, np.float64)
    if h.size < 40:                                     # too few to judge levels — keep all
        return np.ones(h.size, bool)
    rng = float(np.percentile(h, 98) - np.percentile(h, 2))
    if rng < 1e-9:
        return np.ones(h.size, bool)
    thresh = 0.18 * rng                                 # a level air-gap is a big fraction of the spread
    order = np.argsort(h)
    hs = h[order]
    k = int(np.clip(np.searchsorted(hs, seed), 0, hs.size - 1))
    lo = k
    while lo > 0 and hs[lo] - hs[lo - 1] <= thresh:
        lo -= 1
    hi = k
    while hi < hs.size - 1 and hs[hi + 1] - hs[hi] <= thresh:
        hi += 1
    return (h >= hs[lo]) & (h <= hs[hi])


def surface_profile(points, A, B, n, c, corridor=None, nbins=60, isolate=False):
    """Sample the surface HEIGHT along A→B (median in a thin corridor) — a cross
    section that follows the surface. Returns a dict with the profile (t,h), the
    surface-following 3D polyline, the slope angle vs the plane (robust best-fit),
    the height along it, the straight 3D distance and the in-plane length. None if
    too few points fall in the corridor. With ``isolate=True`` the corridor keeps only
    the height-connected level under the picked ends (so the profile follows the pin
    heads, not the board glimpsed between them)."""
    p = np.asarray(points, np.float64)
    A = np.asarray(A, np.float64); B = np.asarray(B, np.float64)
    n = np.asarray(n, np.float64); n = n / np.linalg.norm(n); c = np.asarray(c, np.float64)
    u, v = plane_basis(n)

    au, av = (A - c) @ u, (A - c) @ v
    bu, bv = (B - c) @ u, (B - c) @ v
    du, dv = bu - au, bv - av
    L = float(np.hypot(du, dv))
    if L < 1e-9:
        return None
    dirx, diry = du / L, dv / L                       # along the line (in-plane)
    perpx, perpy = -diry, dirx

    pu, pv, ph = _project(p, c, u, v, n)
    ru, rv = pu - au, pv - av
    t = ru * dirx + rv * diry                         # position along the line
    s = ru * perpx + rv * perpy                        # perpendicular offset
    if corridor is None:
        corridor = max(0.03 * L, 1e-6)
    inb = (t >= -0.02 * L) & (t <= 1.02 * L) & (np.abs(s) <= corridor)
    if int(inb.sum()) < 5:
        return None
    tb, hb, pc = t[inb], ph[inb], p[inb]               # pc = corridor points (to highlight)
    if isolate:                                        # keep only the picked level (first end's)
        hseed = float((A - c) @ n)
        lm = _layer_mask(hb, hseed)
        tb, hb, pc = tb[lm], hb[lm], pc[lm]
        if tb.size < 5:
            return None

    edges = np.linspace(0.0, L, nbins + 1)
    which = np.clip(np.digitize(tb, edges) - 1, 0, nbins - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    pt, pht = [], []
    for k in range(nbins):
        m = which == k
        if m.any():
            pt.append(centers[k]); pht.append(float(np.median(hb[m])))
    if len(pt) < 2:
        return None
    pt = np.asarray(pt); pht = np.asarray(pht)

    slope = float(np.polyfit(pt, pht, 1)[0])           # robust slope of the profile
    angle = float(np.degrees(np.arctan(slope)))
    # surface-following polyline in 3D (plane frame → world)
    px = au + pt * dirx
    py = av + pt * diry
    poly = (c[None, :] + px[:, None] * u[None, :]
            + py[:, None] * v[None, :] + pht[:, None] * n[None, :])
    return {
        "t": pt, "h": pht, "poly": poly.astype(np.float32),
        "angle": angle, "d_height": float(pht[-1] - pht[0]),
        "dist": float(np.linalg.norm(B - A)),
        "n_pts": int(pc.shape[0]),
        "used": pc.astype(np.float32),                 # corridor points measured (to highlight)
    }


def point_distance(A, B):
    A = np.asarray(A, np.float64); B = np.asarray(B, np.float64)
    d = B - A
    return {"dist": float(np.linalg.norm(d)),
            "dx": float(d[0]), "dy": float(d[1]), "dz": float(d[2])}


def region_flatness(points, A, B, n, c, isolate=False):
    """Flatness of the cloud inside the axis-aligned (in-plane) rectangle spanned by
    corners A and B: RMS roughness about the PATCH's OWN plane (so a flat-but-tilted
    patch reads flat) + that plane's tilt vs the board, PLUS the height about the board
    plane over the patch — its mean (z_mean) and full max−min spread (z_range). None if
    the rectangle is empty.

    With ``isolate=True`` only the height-connected LEVEL under the picked corners is
    kept (see _layer_mask) — so a rectangle drawn over floating pin heads measures the
    heads alone, not the board showing through the gaps between them."""
    n = np.asarray(n, np.float64); n = n / np.linalg.norm(n); c = np.asarray(c, np.float64)
    u, v = plane_basis(n)
    pu, pv, ph = _project(points, c, u, v, n)
    au, av = (np.asarray(A) - c) @ u, (np.asarray(A) - c) @ v
    bu, bv = (np.asarray(B) - c) @ u, (np.asarray(B) - c) @ v
    lo_u, hi_u = min(au, bu), max(au, bu)
    lo_v, hi_v = min(av, bv), max(av, bv)
    inr = (pu >= lo_u) & (pu <= hi_u) & (pv >= lo_v) & (pv <= hi_v)
    if int(inr.sum()) < 20:
        return None
    reg = np.asarray(points, np.float64)[inr]
    dev = ph[inr]                                       # height about the BOARD plane
    if isolate:                                          # keep only the picked level
        # seed on the FIRST corner's level — never the mean of the two, since if the
        # corners sit on different levels the mean can land on a THIRD level neither touched
        hseed = float((np.asarray(A, np.float64) - c) @ n)
        lm = _layer_mask(dev, hseed)
        if int(lm.sum()) < 20:
            return None
        reg, dev = reg[lm], dev[lm]
    ln, lc = fit_plane(reg)                             # the patch's OWN plane
    if ln[2] > 0:
        ln = -ln
    tilt = float(np.degrees(np.arccos(np.clip(abs(ln @ n), 0, 1))))
    resid = (reg - lc) @ ln                             # roughness about the LOCAL plane
    # the 4 rectangle corners (closed loop) at the region's median height, so the
    # analyzed patch is DRAWN in 3D — otherwise the user only sees the A→B line
    hm = float(np.median(dev))
    def corner(cu, cv):
        return (c + cu * u + cv * v + hm * n).astype(np.float32).tolist()
    corners = [corner(lo_u, lo_v), corner(hi_u, lo_v), corner(hi_u, hi_v),
               corner(lo_u, hi_v), corner(lo_u, lo_v)]
    return {"n_pts": int(dev.size),                         # points actually measured
            # true RMS, as the readout claims — resid is about the fit's INLIER
            # plane, so its mean over the whole patch is nonzero and std would
            # under-report whenever outliers are one-sided
            "rms": float(np.sqrt(np.mean(resid ** 2))),
            "z_mean": float(np.mean(dev)),                  # mean height above the board
            "z_range": float(dev.max() - dev.min()),        # max−min height in the patch
            "local_tilt": tilt, "corners": corners,
            "used": reg.astype(np.float32),                 # the exact points measured (to highlight)
            "size_u": hi_u - lo_u, "size_v": hi_v - lo_v}


def deviation(points, n, c):
    """Signed distance of every point from the board plane (for a height/deviation
    heatmap) + a symmetric colour range (±2σ, robust). NaN-safe."""
    n = np.asarray(n, np.float64); n = n / np.linalg.norm(n)
    d = (np.asarray(points, np.float64) - np.asarray(c, np.float64)) @ n
    s = float(np.nanstd(d))
    rng = 2.0 * s if np.isfinite(s) and s > 0 else 1.0
    return np.nan_to_num(d, nan=0.0).astype(np.float32), rng


def pin_analysis(points_in_box, n, c):
    """A pin's height above its local board and verticality (axis angle vs the
    board normal). `points_in_box` are the cloud points inside a measure box on
    the pin. None if too sparse."""
    p = np.asarray(points_in_box, np.float64)
    if len(p) < 30:
        return None
    n = np.asarray(n, np.float64); n = n / np.linalg.norm(n); c = np.asarray(c, np.float64)
    h = (p - c) @ n
    # Height = the tip's stand-off from the BOARD PLANE. The plane IS the board
    # reference, so this is robust to a TIGHT box that excludes the board — a q05
    # "board" estimate would land on the shaft and under-report the height badly.
    tip = float(np.quantile(h, 0.99))                   # robust pin tip (drop top flyers)
    if tip <= 0:
        return None
    height = tip
    pin = p[h > 0.4 * tip]                              # the pin body (upper part)
    vert = None
    if len(pin) >= 10:
        pc = pin.mean(0)
        _, sv, vt = np.linalg.svd(pin - pc, full_matrices=False)
        # A stereo cloud is a VISIBLE-SURFACE shell: a flat-topped pin is mostly
        # its top disc, whose first principal axis lies IN the disc — trusting it
        # unconditionally read a perfectly vertical pin as ~90° tilted. Pick the
        # axis by the blob's actual shape:
        if sv[0] >= 2.0 * max(sv[1], 1e-12):
            axis = vt[0]                # clearly elongated — the shaft direction
        elif sv[2] <= 0.5 * max(sv[1], 1e-12):
            axis = vt[-1]               # clearly flat — the top disc's normal
        else:
            axis = None                 # blob-like: no defensible axis → show "—"
        if axis is not None:
            axis = axis / np.linalg.norm(axis)
            vert = float(np.degrees(np.arccos(np.clip(abs(axis @ n), 0, 1))))
    return {"height": height, "verticality": vert, "n_pts": len(p)}
