"""Widget-level regression tests for the Phase-2 fixes that live inside Qt
classes: the NaN-safe export, the incremental repeatability table, and the
ToggleSwitch state sync. One shared QApplication (conftest.qapp)."""
import csv

import numpy as np


# --------------------------------------------------------- render_rgb / NaN
def test_render_rgb_survives_nan(qapp):
    from studio.viewers import ImageView2D

    v = ImageView2D(scalar=True, unit="px")
    arr = np.linspace(1, 50, 25, dtype=np.float32).reshape(5, 5)
    arr[2, 2] = np.nan
    arr[1, 1] = np.inf
    arr[3, 3] = 0.0
    v.set_image(arr)
    rgb = v.render_rgb()               # used to raise IndexError on the NaN
    assert rgb is not None and rgb.shape == (5, 5, 3) and rgb.dtype == np.uint8
    assert (rgb[2, 2] == 0).all()      # NaN exports black
    assert (rgb[3, 3] == 0).all()      # invalid (<=0) exports black


def test_render_rgb_all_invalid(qapp):
    from studio.viewers import ImageView2D

    v = ImageView2D(scalar=True, unit="px")
    v.set_image(np.zeros((4, 4), np.float32))
    rgb = v.render_rgb()
    assert rgb is not None and (rgb == 0).all()


# ------------------------------------------------- repeatability incremental
def _table_text(t):
    return [[t.item(r, c).text() if t.item(r, c) else ""
             for c in range(t.columnCount())] for r in range(t.rowCount())]


def test_repeat_incremental_matches_full_rebuild(qapp):
    from studio.repeat import RepeatabilityView

    a = RepeatabilityView()
    b = RepeatabilityView()
    records = [("cap1", {"p1": 0.005, "p2": 0.006}),
               ("cap2", {"p1": 0.0052, "p2": None}),
               ("cap3", {"p1": 0.0051, "p2": 0.0059, "p3": 0.001}),  # new pin
               ("cap4", {"p1": 0.0049, "p3": 0.0011})]
    for label, vals in records:
        a.add_record(label, vals)      # incremental path
        b.add_record(label, vals)
    b._refresh()                       # force the full-rebuild path on b
    assert _table_text(a.log_table) == _table_text(b.log_table)
    assert _table_text(a.stat_table) == _table_text(b.stat_table)
    assert a.count() == 4


def test_repeat_csv_matches_table_precision(qapp, tmp_path):
    from studio.repeat import RepeatabilityView

    v = RepeatabilityView()
    v.add_record("c1", {"p1": 0.0051234})
    v.add_record("c2", {"p1": 0.0049876})
    path = tmp_path / "out.csv"
    # drive _export's writer directly through the same code path minus the dialog
    from unittest.mock import patch
    with patch("studio.repeat.QFileDialog.getSaveFileName",
               return_value=(str(path), "CSV (*.csv)")):
        v._export()
    rows = list(csv.reader(path.open()))
    assert rows[0] == ["capture", "p1 (mm)"]          # unit on the pin column
    summary_idx = rows.index(["summary", "N", "mean (mm)", "sigma (mm)",
                              "min (mm)", "max (mm)", "range (mm)"])
    srow = rows[summary_idx + 1]
    # mean rounded to table precision (mm -> 4 decimals), sigma to 5
    assert srow[0] == "p1" and srow[1] == "2"
    assert len(srow[2].split(".")[-1]) <= 4
    assert len(srow[3].split(".")[-1]) <= 5


def test_repeat_lock_disables_buttons(qapp):
    from studio.repeat import RepeatabilityView

    v = RepeatabilityView()
    v.set_locked(True)
    assert not v.log_btn.isEnabled()
    assert not v.clear_btn.isEnabled()
    assert not v.export_btn.isEnabled()
    v.set_locked(False)
    assert v.log_btn.isEnabled()


# ------------------------------------------------------------ ToggleSwitch
def test_toggle_switch_blocked_setchecked_paints_right_state(qapp):
    from studio.widgets import ToggleSwitch

    sw = ToggleSwitch(False)
    assert sw._offset == 0.0
    sw.blockSignals(True)
    sw.setChecked(True)                # toggled never fires — old code froze at 0.0
    sw.blockSignals(False)
    assert sw.isChecked() and sw._offset == 1.0
    sw.blockSignals(True)
    sw.setChecked(False)
    sw.blockSignals(False)
    assert not sw.isChecked() and sw._offset == 0.0


# --------------------------------------------------- panel gating / context
def test_collapsible_gate_hint(qapp):
    from studio.widgets import CollapsibleSection
    from PySide6.QtWidgets import QLabel

    sec = CollapsibleSection("Test", expanded=True)
    inner = QLabel("content")
    sec.add(inner)
    assert not sec.is_gated()
    sec.set_gate_hint("locked until X")
    assert sec.is_gated()
    assert not sec.body.isVisibleTo(sec)      # body swapped out...
    assert sec._gate.isVisibleTo(sec)         # ...for the hint line
    sec.set_gate_hint(None)
    assert not sec.is_gated()
    assert sec.body.isVisibleTo(sec)
    sec.set_expanded(False)                   # collapsed: neither shows
    sec.set_gate_hint("locked")
    assert not sec.body.isVisibleTo(sec) and not sec._gate.isVisibleTo(sec)


def test_parampanel_starts_gated_and_ungates(qapp):
    from studio.panels import ParamPanel

    p = ParamPanel()
    assert p.sec_cloud.is_gated()             # no calibration yet
    assert p.sec_measure.is_gated()           # no cloud yet
    assert p.sec_analyze.is_gated()
    p.set_calibration_ready(True)
    p.set_cloud_ready(True)
    assert not p.sec_cloud.is_gated()
    assert not p.sec_measure.is_gated()
    assert not p.sec_analyze.is_gated()
    # gating must not disturb values/boxes
    p.set_cloud_ready(False)
    assert p.measure_on is False
    assert isinstance(p.values(), dict)


def test_analyze_tool_contextual_visibility(qapp):
    from studio.panels import ParamPanel

    p = ParamPanel()
    p.set_cloud_ready(True)
    assert not p.isolate_btn.isVisibleTo(p)   # tool Off
    assert not p.ref_btn.isVisibleTo(p)
    p.analyze_combo.setCurrentIndex(1)        # profile
    assert p.isolate_btn.isVisibleTo(p)
    assert p.analyze_plot.isVisibleTo(p)
    assert not p.ref_btn.isVisibleTo(p)       # flat-ref is a Region concept
    p.analyze_combo.setCurrentIndex(2)        # distance
    assert not p.isolate_btn.isVisibleTo(p)
    p.analyze_combo.setCurrentIndex(3)        # region
    assert p.isolate_btn.isVisibleTo(p)
    assert p.ref_btn.isVisibleTo(p)
    p.analyze_combo.setCurrentIndex(0)        # off again
    assert not p.isolate_btn.isVisibleTo(p) and not p.ref_btn.isVisibleTo(p)
    # an APPLIED reference stays visible whatever the tool, so it can be removed
    p.set_flat_ref_checked(True)
    assert p.ref_btn.isVisibleTo(p)
    p.set_flat_ref_checked(False)
    assert not p.ref_btn.isVisibleTo(p)


def test_analyze_card_hides_when_idle(qapp):
    from studio.panels import ParamPanel

    p = ParamPanel()
    p.set_cloud_ready(True)                   # ungate the section around the card
    assert not p.analyze_card.isVisibleTo(p)  # idle at construction
    p.set_analyze_out("Click two points on the cloud.")
    assert p.analyze_card.isVisibleTo(p)
    p.set_analyze_out("")
    assert not p.analyze_card.isVisibleTo(p)
    p.set_analyze_result("Distance", "1.234", "mm")
    assert p.analyze_card.isVisibleTo(p)


def test_level_and_pin_buttons_new_homes(qapp):
    """Level lives in Analyze (usable without the Volume box); pin analysis
    lives in the Measure body next to the box it acts on."""
    from studio.panels import ParamPanel

    p = ParamPanel()
    assert p.level_btn.parent() is p.sec_analyze.body
    assert p.measure_body.isAncestorOf(p.pin_btn)
    # signals still wired
    fired = []
    p.levelRequested.connect(lambda on: fired.append(("level", on)))
    p.pinAnalyzeRequested.connect(lambda: fired.append(("pin", None)))
    p.level_btn.setChecked(True)
    p.pin_btn.click()
    assert ("level", True) in fired and ("pin", None) in fired


def test_flat_ref_disabled_until_region_exists(qapp):
    """The zero button must not be pressable before there is a Region to zero
    to — but an APPLIED reference always stays clickable so it can be removed."""
    from studio.panels import ParamPanel

    p = ParamPanel()
    p.set_cloud_ready(True)
    p.analyze_combo.setCurrentIndex(3)        # Region tool: button becomes visible...
    assert p.ref_btn.isVisibleTo(p)
    assert not p.ref_btn.isEnabled()          # ...but disabled - nothing measured yet
    assert "Measure a Region first" in p.ref_btn.toolTip()
    p.set_flat_ref_available(True)            # a region measurement landed
    assert p.ref_btn.isEnabled()
    p.set_flat_ref_checked(True)              # reference applied
    p.set_flat_ref_available(False)           # picks reset - applied ref stays removable
    assert p.ref_btn.isEnabled()
    p.set_flat_ref_checked(False)             # removed, and no region left
    p.set_flat_ref_available(False)
    assert not p.ref_btn.isEnabled()


def test_unrectified_pair_warning(qapp, tmp_path):
    """A raw pair (constant vertical misalignment) loaded in 'already rectified'
    mode must warn; a row-aligned pair must not."""
    import imageio.v2 as imageio
    from studio.panels import InputPanel

    rng = np.random.default_rng(4)
    base = (rng.random((768, 1024)) * 255).astype(np.uint8)
    import cv2
    base = cv2.blur(base, (7, 7))                      # blobs ORB can latch onto
    rgb = np.dstack([base] * 3)
    misaligned = np.roll(np.roll(rgb, -120, axis=1), -19, axis=0)   # dx −120, dy −19
    aligned = np.roll(rgb, -120, axis=1)                             # dx only

    notices = []
    for name, right in (("bad", misaligned), ("good", aligned)):
        p = InputPanel()
        p.notice.connect(notices.append)
        lp, rp = tmp_path / f"{name}_l.png", tmp_path / f"{name}_r.png"
        imageio.imwrite(str(lp), rgb)
        imageio.imwrite(str(rp), right)
        got = []
        p.notice.connect(got.append)
        p.load_image(str(lp), "left")
        p.load_image(str(rp), "right")
        warned = any("does not look rectified" in n for n in got)
        assert warned == (name == "bad"), (name, got)


def test_disparity_saturation_detector(qapp):
    """Pixels pinned at the top of the search range must be detected — that is
    the smeared-cloud signature when max_disp is too small for the scene."""
    from studio.main_window import MainWindow

    d = np.full((100, 100), 120.0, np.float32)
    assert MainWindow._disparity_saturation(d, 192) == 0.0     # healthy
    d[:, :30] = 191.0                                          # 30% pinned at the cap
    sat = MainWindow._disparity_saturation(d, 192)
    assert 0.28 < sat < 0.32
    assert MainWindow._disparity_saturation(d, None) == 0.0    # model without the knob
    assert MainWindow._disparity_saturation(np.zeros((4, 4), np.float32), 192) == 0.0
