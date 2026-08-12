"""Measuring marked sites — tracked per capture, referenced within the frame.

The alternative, world-space measure boxes, does not survive a real run on this
rig: boxes seeded from one capture were EMPTY on 19 of 20 later ones. Two things
defeat them, and both are handled here instead.

  * The frame drifts (~15 px in x, ~34 px in y over 1000 captures), so a fixed
    window slides off a 1.6 mm feature. Every site is therefore TRACKED — matched
    against a template cut from the reference capture — before it is sampled.

  * The CNC's step IS the stereo baseline, and it repeats to ~0.5 %. That walks
    ABSOLUTE depth by >1 mm while the parts have not moved at all. So a height is
    only ever reported as pin-minus-reference WITHIN ONE CAPTURE, where the error
    is common to both terms and cancels: measured σ 2200 µm absolute vs ~600 µm
    differential.

Sites are stored in FULL-FRAME RECTIFIED pixels (that is what the Input tab
shows and what the user clicked). A run reconstructs a cropped, scaled window of
that frame, so every coordinate has to be mapped through the ROI — see
``site_pixel``. Pure numpy/cv2: no Qt, no engine, so it is testable headless.
"""
from __future__ import annotations

import numpy as np

#: radius (working-scale px) of the patch sampled for a depth, and of the
#: template used to track it. The template is larger than the sample so it
#: locks onto surrounding structure rather than the feature's own flat top.
SAMPLE_R = 40
TEMPLATE_R = 60
#: how far around its nominal position a site is searched for
SEARCH_R = 90
#: a track weaker than this means the template no longer matches anything —
#: better to report nothing than to sample wherever the best guess landed
MIN_TRACK_SCORE = 0.5
#: reserved key: the (roi, scale) the templates were cut under
_GEOM = "__geom__"


def site_pixel(site, roi, scale: float):
    """A site's (u, v) in the WORKING-SCALE crop a run actually produced.

    Sites are marked on the full rectified frame; the run sees roi cropped out of
    it and then scaled. Getting this wrong puts every sample somewhere else on
    the board, which looks like a plausible measurement rather than an error.
    """
    x0, y0 = (0, 0) if roi is None else (int(roi[0]), int(roi[1]))
    return (int(round((int(site["x"]) - x0) * scale)),
            int(round((int(site["y"]) - y0) * scale)))


def pair_sites(sites) -> list:
    """[(pin, ref)] — each pin paired with its NEAREST reference.

    Nearest rather than explicit pairing because a reference is only valid if it
    is close: it stands in for "the surface this pin sits near", and the further
    away it is, the more of the board's own tilt leaks into the difference.
    Pins with no reference at all come back with ref None so the caller can say
    so rather than silently measuring an absolute depth.
    """
    pins = [s for s in sites if s.get("kind") == "pin"]
    refs = [s for s in sites if s.get("kind") == "ref"]
    out = []
    for p in pins:
        if not refs:
            out.append((p, None))
            continue
        r = min(refs, key=lambda q: (q["x"] - p["x"]) ** 2 + (q["y"] - p["y"]) ** 2)
        out.append((p, r))
    return out


def make_templates(rgb, sites, roi, scale: float, r: int = TEMPLATE_R) -> dict:
    """Cut a tracking template per site from the reference capture's crop.

    The result carries the (roi, scale) it was cut under. Templates and samples
    are positioned by two separate calls to ``site_pixel``, and if those ever
    disagree every reading lands somewhere else on the board and still looks
    plausible — so ``measure_sites`` checks rather than assumes.
    """
    import cv2

    gray = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY)
    H, W = gray.shape[:2]
    out = {}
    for s in sites:
        u, v = site_pixel(s, roi, scale)
        if not (r <= u < W - r and r <= v < H - r):
            continue                      # site is outside the crop — not measurable
        out[s["name"]] = gray[v - r:v + r, u - r:u + r].copy()
    out[_GEOM] = (None if roi is None else tuple(int(v) for v in roi), float(scale))
    return out


def _track(gray, template, u, v, search=SEARCH_R):
    """Where this site sits NOW: (du, dv, score) relative to its nominal spot."""
    import cv2

    r = template.shape[0] // 2
    H, W = gray.shape[:2]
    y0, x0 = max(0, v - r - search), max(0, u - r - search)
    y1, x1 = min(H, v + r + search), min(W, u + r + search)
    band = gray[y0:y1, x0:x1]
    if band.shape[0] < template.shape[0] or band.shape[1] < template.shape[1]:
        return 0, 0, 0.0
    res = cv2.matchTemplate(band, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    return (x0 + loc[0] + r) - u, (y0 + loc[1] + r) - v, float(score)


def _median_depth(depth, u, v, r=SAMPLE_R, need=20):
    H, W = depth.shape[:2]
    m = depth[max(0, v - r):min(H, v + r), max(0, u - r):min(W, u + r)]
    m = m[m > 0]
    return float(np.median(m)) if m.size >= need else np.nan


def measure_sites(rgb, depth, sites, templates, roi, scale: float) -> dict:
    """{pin name: {"height", "track", "score", "z_pin", "z_ref"}} for one capture.

    ``height`` is reference-depth minus pin-depth, so POSITIVE means the pin
    stands proud of its reference. NaN where the pin could not be tracked, had no
    reference, or landed on too few valid depth pixels — all of which the caller
    must report as a missing reading rather than a zero.
    """
    import cv2

    want = (None if roi is None else tuple(int(v) for v in roi), float(scale))
    have = templates.get(_GEOM)
    if have is not None and have != want:
        raise ValueError(
            f"templates were cut for roi/scale {have} but the run is {want} — "
            "every reading would be sampled at the wrong place and still look "
            "plausible. Re-cut the templates for this run.")
    gray = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY)
    out = {}
    for pin, ref in pair_sites(sites):
        name = pin["name"]
        rec = {"height": np.nan, "track": (0, 0), "score": 0.0,
               "z_pin": np.nan, "z_ref": np.nan, "ref": ref["name"] if ref else None}
        t = templates.get(name)
        if t is None or ref is None:
            out[name] = rec
            continue
        pu, pv = site_pixel(pin, roi, scale)
        du, dv, score = _track(gray, t, pu, pv)
        rec["track"], rec["score"] = (du, dv), score
        if score < MIN_TRACK_SCORE:
            out[name] = rec
            continue
        ru, rv = site_pixel(ref, roi, scale)
        # the reference moves with the pin: one rigid scene, one shift
        z_pin = _median_depth(depth, pu + du, pv + dv)
        z_ref = _median_depth(depth, ru + du, rv + dv)
        rec["z_pin"], rec["z_ref"] = z_pin, z_ref
        rec["height"] = z_ref - z_pin
        out[name] = rec
    return out
