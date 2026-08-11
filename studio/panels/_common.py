"""Shared panel furniture: the per-unit widget tables, small widget factories,
and the dynamic per-model parameter widgets.

Split out of the old single panels.py so the two panels (input, params) can be
read on their own — these pieces are used by both and by nothing else.
"""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel,
                               QVBoxLayout, QWidget)

from ..widgets import StatSlider, ToggleSwitch, no_wheel, set_tip


# per-unit widget config (mm is the default — best for close/PCB work)
# baseline spin:  (min, max, decimals, suffix). The baseline LINEARLY scales every
# depth (z = fx·baseline/disp) and setDecimals ROUNDS the stored value, so it carries
# extra digits — a derived stereo-calibration baseline is full-precision and must not
# be quantised on the way in.
# All three rows span the SAME physical range (0–10 m), like the z/box tables
# below — the mm/µm rows used to cap at 2 m while m allowed 100 m, so cycling
# units silently CLAMPED any baseline over 2 m (a depth-scaling corruption).
_BASELINE_CFG = {
    "mm": (0.0, 10_000.0, 6, " mm"),
    "m":  (0.0, 10.0, 9, " m"),
    "µm": (0.0, 10_000_000.0, 3, " µm"),     # 1 nm precision, like mm's 6-dec
}
# z-near / z-far sliders:  (min, max, default, step, fmt, suffix). Every row is the mm
# row scaled by the exact unit factor (range, default AND step): µm = mm×1000, m = mm÷1000.
# Keeping them consistent means a value survives ANY unit switch losslessly — critical now
# that the units button CYCLES mm→µm→m→mm, so "put it back to mm" routes through metres.
# (A coarser/mismatched metres row used to snap the clip plane, e.g. 250 mm → 200 mm.)
_ZNEAR_CFG = {
    "mm": (0.0, 500.0, 0.0, 0.5, "{:.1f}", " mm"),
    "m":  (0.0, 0.5, 0.0, 0.0005, "{:.4f}", " m"),
    "µm": (0.0, 500_000.0, 0.0, 500.0, "{:.0f}", " µm"),
}
_ZFAR_CFG = {
    "mm": (1.0, 2000.0, 250.0, 1.0, "{:.0f}", " mm"),
    "m":  (0.001, 2.0, 0.25, 0.001, "{:.3f}", " m"),
    "µm": (1000.0, 2_000_000.0, 250_000.0, 1000.0, "{:.0f}", " µm"),
}
# measure box:  (min, max, decimals, suffix). The CENTRE goes negative — X is
# measured from the optical axis, so anything left of it is negative, and so is
# anything above it in Y. Only Z (depth) is always positive.
_BOXC_CFG = {
    "mm": (-5000.0, 5000.0, 4, " mm"),
    "m":  (-5.0, 5.0, 7, " m"),
    "µm": (-5_000_000.0, 5_000_000.0, 1, " µm"),
}
_BOXS_CFG = {
    "mm": (0.0001, 1000.0, 4, " mm"),
    "m":  (0.0000001, 1.0, 7, " m"),
    "µm": (0.1, 1_000_000.0, 1, " µm"),
}
_BOX_DEFAULT_MM = 5.0     # a 5 mm cube — visible on a PCB without swallowing it

_REF_TIP = (
    "Measure a Region on a zone you KNOW is flat, then press this to ZERO to it: "
    "its average height becomes an offset subtracted from every pin / point / region "
    "height (correcting the cloud's systematic bow), and its max−min is reported as "
    "the flatness uncertainty. Press again to remove the correction.")


def np_to_qpixmap(arr: np.ndarray) -> QPixmap:
    a = np.ascontiguousarray(arr)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    a = a[..., :3].astype(np.uint8)
    h, w = a.shape[:2]
    img = QImage(a.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(img.copy())


def make_spin(lo, hi, dec, suffix="", tip="") -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(dec)
    s.setButtonSymbols(QDoubleSpinBox.NoButtons)
    if suffix:
        s.setSuffix(suffix)
    if tip:
        set_tip(s, tip)
    no_wheel(s)   # scrolling the panel must not change the value under the cursor
    return s


def field_row(name: str, widget: QWidget, width: int = 58) -> QHBoxLayout:
    row = QHBoxLayout()
    lbl = QLabel(name)
    lbl.setProperty("role", "muted")
    lbl.setFixedWidth(width)
    row.addWidget(lbl)
    row.addWidget(widget, 1)
    return row


def _toggle_row(name: str, checked: bool, tip: str = "") -> tuple[QWidget, ToggleSwitch]:
    row = QWidget()
    lay = QHBoxLayout(row)
    lay.setContentsMargins(0, 0, 0, 0)
    lbl = QLabel(name)
    lbl.setProperty("role", "muted")
    sw = ToggleSwitch(checked)
    lay.addWidget(lbl)
    lay.addStretch(1)
    lay.addWidget(sw)
    if tip:
        set_tip(row, tip)
    return row, sw


# --------------------------------------------------- per-model param widgets
# A backend declares its knobs as ParamSpecs; these functions are the only code
# that turns those into widgets and back, so adding a ParamSpec to a backend is
# all it takes to appear in the parameter panel.

def build_param_widgets(specs: list, lay: QVBoxLayout, on_change=None,
                        values: dict | None = None) -> dict:
    """Render `specs` into `lay`, one row each. Returns {key: (kind, widget)}.

    `values` seeds the widgets (a model's remembered settings); anything missing
    falls back to the ParamSpec's own default. Each widget is CONSTRUCTED at its
    value and only then connected to `on_change` — so restoring settings can't
    fire a spurious 'you changed something, re-run' signal.
    """
    values = values or {}
    out: dict = {}
    for ps in specs:
        val = values.get(ps.key, ps.default)
        if ps.kind == "toggle":
            row, w = _toggle_row(ps.label, bool(val), ps.tooltip)
            if on_change is not None:
                w.toggled.connect(lambda *_: on_change())
            lay.addWidget(row)
        elif ps.kind == "choice":
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            lbl = QLabel(ps.label)
            lbl.setProperty("role", "muted")
            w = no_wheel(QComboBox())
            for v, label in ps.options:
                w.addItem(label, v)
            i = w.findData(val)
            if i < 0:
                i = w.findData(ps.default)
            if i >= 0:
                w.setCurrentIndex(i)
            if on_change is not None:
                w.currentIndexChanged.connect(lambda *_: on_change())
            rl.addWidget(lbl)
            rl.addStretch(1)
            rl.addWidget(w)
            if ps.tooltip:
                set_tip(row, ps.tooltip)
            lay.addWidget(row)
        else:   # slider
            # isfinite, not just float(): json.loads happily returns NaN/Infinity
            # for the bare literals json.dumps emits, float("nan") PASSES this
            # guard, and StatSlider then does int(round(nan)) -> ValueError. That
            # escaped set_backend -> _restore_settings -> __init__, so one such
            # value in the saved settings stopped the app from starting at all,
            # with no way to fix it from the UI.
            try:
                val = float(val)
                if not math.isfinite(val):
                    raise ValueError("not finite")
            except (TypeError, ValueError):
                val = ps.default
            w = StatSlider(ps.label, ps.minv, ps.maxv, val, ps.step,
                           ps.fmt, ps.suffix, tip=ps.tooltip)
            if on_change is not None:
                w.valueChanged.connect(lambda *_: on_change())
            lay.addWidget(w)
        out[ps.key] = (ps.kind if ps.kind in ("toggle", "choice") else "slider", w)
    return out


def read_param_widgets(widgets: dict) -> dict:
    """Current values, keyed by ParamSpec.key — this is what becomes
    StereoParams.model_params. Sliders yield floats; every adapter casts with
    int() where it needs an int, which is the long-standing contract."""
    out = {}
    for key, (kind, w) in widgets.items():
        if kind == "toggle":
            out[key] = w.isChecked()
        elif kind == "choice":
            out[key] = w.currentData()
        else:
            out[key] = w.value()
    return out


def set_param_widgets(widgets: dict, values: dict) -> None:
    """Push values in (a defaults reset, or restoring saved settings).

    Deliberately forgiving: unknown keys are ignored and each widget clamps its
    own value, so a stale saved blob from an older build — or one naming a knob a
    backend has since dropped — can't wedge the panel."""
    for key, val in (values or {}).items():
        entry = widgets.get(key)
        if entry is None:
            continue
        kind, w = entry
        if kind == "toggle":
            w.setChecked(bool(val))
        elif kind == "choice":
            i = w.findData(val)
            if i >= 0:
                w.setCurrentIndex(i)
        else:
            try:
                w.setValue(float(val))
            except (TypeError, ValueError):
                pass


def sanitize_params(spec, values: dict) -> dict:
    """Coerce a remembered blob into something the panel could actually produce.

    The widgets clamp, quantise and type-check every knob, so anything read back
    off them is in-spec by construction. Values for a model that is NOT on screen
    never touch a widget — they go from QSettings straight to the engine — so a
    blob from an older build, a hand-edited INI, or (the real one) another model's
    settings under this key would be sent verbatim. Nothing would crash: every
    adapter reads its knobs with model_params.get(key, default), so a foreign key
    is ignored and a foreign VALUE is applied — FoundationStereo and Fast-FS share
    `valid_iters` with different ranges, so a plausible wrong answer comes back
    instead of an error. Drop what the spec doesn't declare; clamp what it does.
    """
    if spec is None:
        return {}
    out = {}
    for ps in spec.params:
        v = (values or {}).get(ps.key, ps.default)
        if ps.kind == "toggle":
            out[ps.key] = bool(v)
        elif ps.kind == "choice":
            allowed = [val for val, _ in ps.options]
            out[ps.key] = v if v in allowed else (
                ps.default if ps.default in allowed else (allowed[0] if allowed else ps.default))
        else:
            try:
                v = float(v)
                if not math.isfinite(v):
                    raise ValueError("not finite")
            except (TypeError, ValueError):
                v = float(ps.default)
            out[ps.key] = min(max(v, ps.minv), ps.maxv)   # the slider's own range
    return out


