"""Site measurement — tracked, differenced inside one capture.

Ground truth is synthetic and exact: a depth map with a known pin/reference step
and a textured image that can be SHIFTED by a known amount, so tracking either
follows it or the test fails. Written because the alternative (world-space
measure boxes) passed every unit test it had and then came back empty on 19 of
20 real captures — the failure was in what the measurement assumed about the
rig, not in its arithmetic.
"""
import numpy as np
import pytest

from studio.sites_measure import (make_templates, measure_sites, pair_sites,
                                  site_pixel)

ROI = (2560, 1024, 512, 1024)
PIN = {"kind": "pin", "x": 2807, "y": 1139, "name": "Pin 1"}
REF = {"kind": "ref", "x": 2960, "y": 1300, "name": "Ref 1"}


def _scene(shift=(0, 0), z_ref=220.0, step=1.5, H=1024, W=512, seed=0):
    """Textured RGB + a depth map where the pin stands `step` proud of the ref.

    `shift` moves the WHOLE scene, image and depth together — a rigid drift, the
    thing tracking exists to absorb.
    """
    rng = np.random.default_rng(seed)
    rgb = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
    depth = np.full((H, W), z_ref, np.float32)
    pu, pv = site_pixel(PIN, ROI, 1.0)
    ru, rv = site_pixel(REF, ROI, 1.0)
    depth[pv - 60:pv + 60, pu - 60:pu + 60] = z_ref - step      # nearer = proud
    du, dv = shift
    if du or dv:
        rgb = np.roll(np.roll(rgb, dv, axis=0), du, axis=1)
        depth = np.roll(np.roll(depth, dv, axis=0), du, axis=1)
    return rgb, depth, (pu, pv), (ru, rv)


# ------------------------------------------------------------------ mapping
def test_site_pixel_maps_through_the_roi_and_scale():
    assert site_pixel(PIN, ROI, 1.0) == (2807 - 2560, 1139 - 1024)
    assert site_pixel(PIN, ROI, 0.5) == (round(247 * 0.5), round(115 * 0.5))
    assert site_pixel(PIN, None, 1.0) == (2807, 1139)      # no crop = full frame


def test_pair_sites_takes_the_nearest_reference():
    far = {"kind": "ref", "x": 3500, "y": 2500, "name": "Far"}
    pairs = pair_sites([PIN, far, REF])
    assert len(pairs) == 1 and pairs[0][1]["name"] == "Ref 1"


def test_pin_without_a_reference_is_reported_not_guessed():
    """An unreferenced pin must NOT silently fall back to absolute depth — that
    is the measurement whose σ was 2200 µm."""
    (pin, ref), = pair_sites([PIN])
    assert ref is None
    rgb, depth, _, _ = _scene()
    t = make_templates(rgb, [PIN], ROI, 1.0)
    got = measure_sites(rgb, depth, [PIN], t, ROI, 1.0)
    assert not np.isfinite(got["Pin 1"]["height"])
    assert got["Pin 1"]["ref"] is None


# --------------------------------------------------------------- measuring
def test_height_is_reference_minus_pin():
    rgb, depth, _, _ = _scene(step=1.5)
    sites = [PIN, REF]
    got = measure_sites(rgb, depth, sites, make_templates(rgb, sites, ROI, 1.0), ROI, 1.0)
    assert got["Pin 1"]["height"] == pytest.approx(1.5, abs=1e-3)
    assert got["Pin 1"]["ref"] == "Ref 1"
    assert got["Pin 1"]["score"] > 0.9


def test_tracking_follows_a_drifting_frame():
    """The rig drifts ~15 px x / ~34 px y over a run. A fixed window slides off a
    1.6 mm feature; a tracked one must not."""
    ref_rgb, _, _, _ = _scene()
    sites = [PIN, REF]
    templates = make_templates(ref_rgb, sites, ROI, 1.0)
    for du, dv in [(0, 0), (-15, 34), (20, -25)]:
        rgb, depth, _, _ = _scene(shift=(du, dv), step=1.5)
        got = measure_sites(rgb, depth, sites, templates, ROI, 1.0)
        assert got["Pin 1"]["track"] == (du, dv), (du, dv)
        assert got["Pin 1"]["height"] == pytest.approx(1.5, abs=1e-3)


def test_height_is_immune_to_a_baseline_scale_error():
    """The CNC step IS the baseline and repeats to ~0.5 %, which walks absolute
    depth by >1 mm. Differencing inside one capture is what cancels it."""
    sites = [PIN, REF]
    rgb, depth, _, _ = _scene(z_ref=220.0, step=1.5)
    t = make_templates(rgb, sites, ROI, 1.0)
    h0 = measure_sites(rgb, depth, sites, t, ROI, 1.0)["Pin 1"]["height"]
    # a 0.5 % baseline error scales EVERY depth in the capture
    rgb2, depth2, _, _ = _scene(z_ref=220.0, step=1.5)
    depth2 = (depth2 * 1.005).astype(np.float32)
    h1 = measure_sites(rgb2, depth2, sites, t, ROI, 1.0)["Pin 1"]["height"]
    assert abs(h1 - h0) < 0.01           # residual is 0.5 % OF THE STEP, not of Z
    assert abs(h1 - h0) < 0.005 * 220.0 / 10


def test_untrackable_site_reports_nothing_rather_than_a_guess():
    rgb, depth, _, _ = _scene()
    sites = [PIN, REF]
    t = make_templates(rgb, sites, ROI, 1.0)
    flat = np.full_like(rgb, 128)        # nothing to lock onto
    got = measure_sites(flat, depth, sites, t, ROI, 1.0)
    assert not np.isfinite(got["Pin 1"]["height"])


def test_site_outside_the_crop_gets_no_template():
    outside = {"kind": "pin", "x": 3900, "y": 2900, "name": "Outside"}
    rgb, _, _, _ = _scene()
    t = make_templates(rgb, [outside, REF], ROI, 1.0)
    assert "Outside" not in t


def test_no_valid_depth_reports_nothing():
    rgb, depth, _, _ = _scene()
    sites = [PIN, REF]
    t = make_templates(rgb, sites, ROI, 1.0)
    got = measure_sites(rgb, np.zeros_like(depth), sites, t, ROI, 1.0)
    assert not np.isfinite(got["Pin 1"]["height"])


def test_templates_and_run_must_agree_on_roi_and_scale():
    """Templates and samples are positioned by two separate site_pixel calls. If
    they disagree, every reading lands elsewhere on the board and still looks
    plausible — the failure mode this whole session kept hitting."""
    rgb, depth, _, _ = _scene()
    sites = [PIN, REF]
    t = make_templates(rgb, sites, ROI, 1.0)
    with pytest.raises(ValueError, match="wrong place"):
        measure_sites(rgb, depth, sites, t, ROI, 0.5)      # scale mismatch
    with pytest.raises(ValueError, match="wrong place"):
        measure_sites(rgb, depth, sites, t, (0, 0, 512, 1024), 1.0)   # roi mismatch
    measure_sites(rgb, depth, sites, t, ROI, 1.0)          # matching: fine


def test_geom_key_is_not_mistaken_for_a_site():
    rgb, _, _, _ = _scene()
    t = make_templates(rgb, [PIN, REF], ROI, 1.0)
    assert "__geom__" in t and t["__geom__"] == (ROI, 1.0)
    assert set(t) - {"__geom__"} == {"Pin 1", "Ref 1"}


def test_untemplatable_says_why_instead_of_reporting_zero_readings():
    """A site with no template silently produces no reading, and "0 readings" is
    the least useful thing a study can report. The usual cause is scale: a
    512x1024 ROI at scale 0.25 is 128x256 working px and a 120x120 template
    cannot fit inside it at all."""
    from studio.sites_measure import untemplatable

    rgb, _, _, _ = _scene()                     # 1024x512, scale 1.0
    assert untemplatable(rgb, [PIN, REF], ROI, 1.0) == []

    small = np.zeros((256, 128, 3), np.uint8)   # the same ROI at scale 0.25
    bad = untemplatable(small, [PIN, REF], ROI, 0.25)
    assert len(bad) == 2
    for name, why in bad:
        assert "cannot fit" in why or "outside" in why
        assert "128×256" in why or "128" in why

    outside = {"kind": "pin", "x": 3900, "y": 2900, "name": "Outside"}
    bad = untemplatable(rgb, [outside], ROI, 1.0)
    assert len(bad) == 1 and "outside" in bad[0][1]
