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
