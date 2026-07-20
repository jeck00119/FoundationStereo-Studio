"""Left (input + calibration) and right (parameters) panels."""
from __future__ import annotations

import math
import os

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFileDialog, QFrame,
                               QHBoxLayout, QLabel, QPushButton, QVBoxLayout,
                               QWidget)

from .backends import BACKENDS, get_spec
from .dtypes import UNIT_PER_M, StereoParams
from .measure import MeasureBox
from .pairs import IMG_FILTER, load_rgb
from .rectify import Rectifier, StereoCalibration
from .widgets import (Collapsible, CollapsibleSection, ImageDrop, MetricCard,
                      StatSlider, ToggleSwitch, no_wheel, set_tip)

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


class InputPanel(QWidget):
    imagesChanged = Signal()
    calibrationChanged = Signal()
    notice = Signal(str)             # something to tell the user (shown in the status bar)
    modelChanged = Signal(str)       # the selected backend key changed
    checkpointChanged = Signal()     # the selected checkpoint changed (same backend)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("LeftPane")
        self.setFixedWidth(232)
        self._units = "mm"          # display/entry unit for baseline (mm default)
        self._saved_ckpt: dict = {}   # backend key -> the checkpoint last used for it
        self.left_path = self.right_path = None
        self.left_rgb: np.ndarray | None = None
        self.right_rgb: np.ndarray | None = None
        # raw = the originally loaded pixels; left_rgb/right_rgb are what the pipeline
        # sees (== raw when already-rectified, or the rectified pair in raw mode)
        self.left_raw: np.ndarray | None = None
        self.right_raw: np.ndarray | None = None
        self._rect_mode_on = False        # False = images already rectified (default)
        self._calib = None                # StereoCalibration (raw mode)
        self._calib_path = ""
        self._rectifier = None            # Rectifier built from _calib + image size

        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(11)

        # --- stereo pair --- (foldable category)
        self.sec_pair = CollapsibleSection("Stereo pair", body_spacing=10)
        lay.addWidget(self.sec_pair)
        self.drop_l = ImageDrop("Left image")
        self.drop_r = ImageDrop("Right image")
        set_tip(self.drop_l, "The LEFT photo of a stereo pair — two cameras side by side "
                "shooting the same scene at the same instant. Click or drop an image.")
        set_tip(self.drop_r, "The RIGHT photo of the same stereo pair. Must be rectified "
                "with the left (cameras level and horizontal).")
        self.drop_l.clicked.connect(lambda: self._pick("left"))
        self.drop_r.clicked.connect(lambda: self._pick("right"))
        self.drop_l.fileDropped.connect(lambda p: self._load(p, "left"))
        self.drop_r.fileDropped.connect(lambda p: self._load(p, "right"))
        self.sec_pair.add(self.drop_l)
        self.sec_pair.add(self.drop_r)

        # --- calibration --- (foldable category)
        self.sec_calib = CollapsibleSection("Calibration", "Needs run", "rerun", body_spacing=10)
        lay.addWidget(self.sec_calib)
        # --- input mode: already-rectified (default) vs raw + rectify ---
        self.rect_mode = no_wheel(QComboBox())
        self.rect_mode.addItems(["Images already rectified",
                                 "Raw — rectify with calibration"])
        self.rect_mode.setToolTip(
            "Already rectified: feed the pair as-is (optionally load a K.txt for depth).\n\n"
            "Raw — rectify: load a stereo calibration and the app undistorts + row-aligns "
            "every pair, deriving the depth calibration for you.")
        self.sec_calib.add(self.rect_mode)
        # clarify (in 'already rectified' mode) that the fx/baseline fields below are
        # the DEPTH calibration — separate from rectification, and optional
        self.mode_hint = QLabel(
            "The fields below are the depth calibration (fx + baseline) — needed for "
            "metric depth (mm) & 3D. Load K.txt, or leave blank for disparity only.")
        self.mode_hint.setProperty("role", "muted")
        self.mode_hint.setStyleSheet("font-size:11px;")
        self.mode_hint.setWordWrap(True)
        self.sec_calib.add(self.mode_hint)
        # raw-mode calibration loader (hidden until 'Raw' is chosen)
        self._rect_box = QWidget()
        _rb = QVBoxLayout(self._rect_box)
        _rb.setContentsMargins(0, 0, 0, 0)
        _rb.setSpacing(6)
        self.load_calib = QPushButton("Load calibration…")
        self.load_calib.setToolTip("Load a single-camera stereo calibration (K, distortion, "
                                   "R, T) from .npz / .json / OpenCV .yml.")
        self.load_calib.clicked.connect(self._load_calibration)
        # the calibration's translation (baseline) unit — cv2.stereoCalibrate uses
        # the checkerboard square's unit, OFTEN metres. Getting this wrong makes
        # depth 1000x off, so it's an explicit choice, not a silent mm assumption.
        _urow = QHBoxLayout()
        _urow.setContentsMargins(0, 0, 0, 0)
        _urow.setSpacing(6)
        _ulbl = QLabel("Baseline unit:")
        _ulbl.setProperty("role", "muted")
        _ulbl.setStyleSheet("font-size:11px;")
        self.calib_unit = no_wheel(QComboBox())
        self.calib_unit.addItems(["mm", "m"])
        self.calib_unit.setFixedWidth(58)
        self.calib_unit.setToolTip("The unit your calibration's translation (baseline) is in. "
                                   "cv2.stereoCalibrate uses the checkerboard square's unit — "
                                   "often metres. If depth comes out ~1000x off, switch this.")
        self.calib_unit.currentIndexChanged.connect(self._on_calib_unit)
        _urow.addWidget(_ulbl)
        _urow.addWidget(self.calib_unit)
        _urow.addStretch(1)
        self.calib_status = QLabel("no calibration loaded")
        self.calib_status.setProperty("role", "muted")
        self.calib_status.setStyleSheet("font-size:11px;")
        self.calib_status.setWordWrap(True)
        _rb.addWidget(self.load_calib)
        _rb.addLayout(_urow)
        _rb.addWidget(self.calib_status)
        self.sec_calib.add(self._rect_box)
        self._rect_box.setVisible(False)
        self.fx = make_spin(0, 1e6, 5, tip="Horizontal focal length in pixels, from your "
                "camera calibration. Needed to turn disparity into real-world depth.")
        self.fy = make_spin(0, 1e6, 5, tip="Vertical focal length in pixels — usually equal to fx.")
        self.cx = make_spin(0, 1e6, 5, tip="Optical-center X in pixels — normally near image width ÷ 2.")
        self.cy = make_spin(0, 1e6, 5, tip="Optical-center Y in pixels — normally near image height ÷ 2.")
        _blo, _bhi, _bdec, _bsuf = _BASELINE_CFG[self._units]
        self.baseline = make_spin(_blo, _bhi, _bdec, suffix=_bsuf, tip="Distance between the two "
                "camera positions. With focal length this converts disparity into metric depth. "
                "Required for the Depth map and 3D cloud. A K.txt baseline is in metres and is "
                "converted to the current unit automatically.")
        grid = QVBoxLayout()
        grid.setSpacing(6)
        for a, b in (("fx", self.fx), ("fy", self.fy), ("cx", self.cx), ("cy", self.cy)):
            grid.addLayout(field_row(a, b))
        grid.addLayout(field_row("baseline", self.baseline))
        self.sec_calib.add_layout(grid)
        self.load_k = QPushButton("Load K.txt…")
        self.load_k.setToolTip("Load fx/fy/cx/cy and baseline from a K.txt file. A K.txt sitting "
                               "next to your left image is loaded automatically.")
        self.load_k.clicked.connect(self._load_ktxt)
        self.sec_calib.add(self.load_k)
        self.calib_hint = QLabel("no calibration → disparity only")
        self.calib_hint.setProperty("role", "muted")
        self.calib_hint.setStyleSheet("font-size:11px;")
        self.sec_calib.add(self.calib_hint)

        for s in (self.fx, self.fy, self.cx, self.cy, self.baseline):
            s.valueChanged.connect(self._calib_changed)
        self.rect_mode.currentIndexChanged.connect(self._on_rect_mode)

        # --- model --- (foldable category)
        self.sec_model = CollapsibleSection("Model", "Needs run", "rerun", body_spacing=8)
        lay.addWidget(self.sec_model)
        self.model_combo = no_wheel(QComboBox())
        self.model_combo.setToolTip("Stereo model. Switching reloads the engine in a fresh "
                                    "process — a few seconds for the small models, longer for "
                                    "FoundationStereo's 3 GB weights. More models can be added "
                                    "under studio/backends/.")
        for spec in BACKENDS.values():
            ok, reason = spec.availability()
            self.model_combo.addItem(spec.display_name if ok else f"{spec.display_name}  ⚠", spec.key)
            idx = self.model_combo.count() - 1
            self.model_combo.setItemData(idx, spec.description if ok else reason, Qt.ToolTipRole)
            if not ok:
                item = self.model_combo.model().item(idx)
                if item is not None:
                    item.setEnabled(False)
        self.ckpt_combo = no_wheel(QComboBox())
        self.ckpt_combo.setToolTip("Model weights (checkpoint).")
        self.sec_model.add(self.model_combo)
        self.sec_model.add(self.ckpt_combo)
        self._populate_checkpoints()
        self.model_combo.currentIndexChanged.connect(self._on_model_combo)
        self.ckpt_combo.currentIndexChanged.connect(self._on_ckpt_combo)

        self.device_lbl = QLabel("device: —")
        self.device_lbl.setProperty("role", "muted")
        self.device_lbl.setStyleSheet("font-size:11px;")
        self.sec_model.add(self.device_lbl)
        lay.addStretch(1)

    # ---------------------------------------------------------- helpers
    def load_image(self, path: str, which: str) -> None:
        """Public programmatic load ('left' / 'right') — the same path a drop or
        file-pick takes. Exists so callers (the demo autoload) don't reach into
        the private _load."""
        self._load(path, which)

    def _pick(self, which: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, f"Choose {which} image", "", IMG_FILTER)
        if path:
            self._load(path, which)

    def _load(self, path: str, which: str) -> None:
        try:
            # the SAME loader the batch uses (studio.pairs), so a hand-dropped
            # image and a batched one can never be converted differently
            arr = load_rgb(path)
        except Exception as exc:  # noqa: BLE001
            # was a bare `return`: dropping an unreadable/unsupported file did
            # NOTHING AT ALL on screen — no tile, no message, no clue why.
            self.notice.emit(f"Couldn't open {os.path.basename(path)}: {exc}")
            return
        if which == "left":
            self.left_path, self.left_raw = path, arr
            if not self._rect_mode_on:
                self._autoload_ktxt(path)   # a K.txt only applies to a rectified pair
        else:
            self.right_path, self.right_raw = path, arr
        self._ensure_rectifier()            # the image size is known now
        self._process_side(which)           # sets left_rgb/right_rgb + the thumbnail
        self.imagesChanged.emit()

    def _autoload_ktxt(self, img_path: str) -> None:
        # A K.txt beside the image is authoritative for THIS pair — always load
        # it, even over prior calibration, so switching pairs can never silently
        # reuse the wrong intrinsics. (If none sits beside it, keep what's there.)
        k = os.path.join(os.path.dirname(img_path), "K.txt")
        if not os.path.isfile(k):
            return
        # …authoritative only if it plausibly BELONGS to this image. The repo's
        # demo assets/K.txt (960×540 camera, cx≈489) sitting beside a user's
        # 2664×2304 pair used to be applied silently — ~3.4× wrong fx and 12.6×
        # wrong baseline, i.e. confidently wrong metric depth with no warning.
        # Any real (rectified) camera's principal point sits near the image
        # centre, so a cx/cy outside the middle half of THIS frame means the
        # file describes a different camera/resolution.
        if self.left_raw is not None:
            try:
                with open(k) as f:
                    vals = [float(x) for x in f.read().split()[:6]]
                cx, cy = vals[2], vals[5]
                h, w = self.left_raw.shape[:2]
                if not (0.25 * w <= cx <= 0.75 * w and 0.25 * h <= cy <= 0.75 * h):
                    self.notice.emit(
                        f"Ignored {os.path.basename(k)} next to this image — its principal "
                        f"point ({cx:.0f}, {cy:.0f}) doesn't fit a {w}×{h} image (it looks "
                        "like another camera's calibration). Load the right file manually.")
                    return
            except Exception:  # noqa: BLE001 — unreadable/short: let _parse_ktxt report it
                pass
        self._parse_ktxt(k)

    def _load_ktxt(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load K.txt", "", "Calib (*.txt)")
        if path:
            self._parse_ktxt(path)

    def _parse_ktxt(self, path: str) -> bool:
        """Load intrinsics (row-major 3×3 K on line 1) + baseline (line 2).
        Parses and validates FIRST, then applies all-or-nothing, so a malformed
        file can never leave a half-set, inconsistent calibration. The K.txt
        baseline is in METRES (FoundationStereo convention) and is converted to
        the panel's current display unit."""
        try:
            with open(path) as f:
                lines = [ln for ln in f.read().strip().splitlines() if ln.strip()]
            if not lines:
                raise ValueError("file is empty")
            vals = list(map(float, lines[0].split()))
            if len(vals) < 6:
                raise ValueError("first line must hold the row-major K matrix "
                                 "(9 numbers; the first 6 — fx 0 cx 0 fy cy — suffice)")
            baseline_m = float(lines[1].split()[0]) if len(lines) > 1 else 0.0
            # float("nan") parses fine — without this a NaN fx lands in the spin
            # (Qt clamps it arbitrarily) while a NaN baseline fails `> 0` and
            # silently KEEPS the previous pair's baseline: a mixed calibration
            if not all(math.isfinite(v) for v in vals[:6]) or not math.isfinite(baseline_m):
                raise ValueError("contains a non-finite number (NaN/Inf)")
        except Exception as exc:  # noqa: BLE001
            self.notice.emit(f"Couldn't read {os.path.basename(path)}: {exc}")
            return False
        self.fx.setValue(vals[0]); self.fy.setValue(vals[4])
        self.cx.setValue(vals[2]); self.cy.setValue(vals[5])
        if baseline_m > 0:
            self.baseline.setValue(baseline_m * UNIT_PER_M[self._units])
        return True

    def _calib_changed(self, *_) -> None:
        ok = self.has_calibration
        self.calib_hint.setText(
            "metric depth + 3D enabled" if ok else "no calibration → disparity only"
        )
        self.calibrationChanged.emit()

    # ------------------------------------------------------- rectification
    def _on_rect_mode(self, idx: int) -> None:
        """Switch between 'already rectified' (feed as-is) and 'raw — rectify'."""
        raw = (idx == 1)
        self._rect_mode_on = raw
        self._rect_box.setVisible(raw)
        self.mode_hint.setVisible(not raw)   # the raw-mode loader explains itself
        self.load_k.setVisible(not raw)
        self._set_fields_readonly(raw)
        if raw:
            if self._calib is None:
                # Entering raw mode with NO calibration loaded: the fields still
                # held the previous manual/K.txt intrinsics, now presented
                # read-only as if derived — and has_calibration stayed True, so a
                # Run computed metric depth for UNRECTIFIED images with them.
                # Blank the fields until a calibration actually derives values.
                self._clear_derived_calibration()
            self._ensure_rectifier()
        else:
            self._rectifier = None
        self._reprocess_images()
        self._refresh_calib_status()
        self._calib_changed()          # depth availability / hint may have changed
        self.imagesChanged.emit()      # the effective pair changed (rectified vs raw)

    def _set_fields_readonly(self, ro: bool) -> None:
        """In raw mode the intrinsics are DERIVED from rectification, so the fields
        become a read-only readout rather than an input. Button symbols are left
        alone: make_spin builds every field with NoButtons, and restoring
        UpDownArrows here grew arrow buttons the fields never had."""
        for s in (self.fx, self.fy, self.cx, self.cy, self.baseline):
            s.setReadOnly(ro)

    def _load_calibration(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load stereo calibration", "",
            "Calibration (*.npz *.json *.yml *.yaml *.xml);;All files (*)")
        if not path:
            return
        try:
            self._calib = StereoCalibration.load(path)
        except Exception as exc:  # noqa: BLE001
            self.notice.emit(f"Couldn't read calibration: {exc}")
            return
        self._calib_path = path
        self._rectifier = None            # force a rebuild at the current image size
        self._ensure_rectifier()
        self._reprocess_images()
        self._refresh_calib_status()
        self._calib_changed()
        self.imagesChanged.emit()

    def _ensure_rectifier(self) -> None:
        """Build the rectifier once we have both a calibration and an image size
        (from the file, else the loaded pair). No-op outside raw mode; cheap to call
        repeatedly (it rebuilds only when the size actually changes)."""
        if not self._rect_mode_on or self._calib is None:
            return
        # Prefer the LOADED image's size — that's the resolution we actually rectify;
        # the Rectifier scales K if the calibration was solved at another resolution.
        # Fall back to the file's image_size only to derive K before any image loads.
        ref = self.left_raw if self.left_raw is not None else self.right_raw
        if ref is not None:
            size = (ref.shape[1], ref.shape[0])
        elif self._calib.image_size is not None:
            size = self._calib.image_size
        else:
            return                        # wait until an image tells us the size
        if self._rectifier is not None and self._rectifier.size == tuple(size):
            return
        try:
            self._rectifier = Rectifier(self._calib, size)
        except Exception as exc:  # noqa: BLE001
            self._rectifier = None
            self._clear_derived_calibration()   # don't keep stale K from a prior calib
            self.notice.emit(f"Rectification failed: {exc}")
            return
        self._apply_derived_calibration()

    def _calib_src_unit(self) -> str:
        """The unit the loaded calibration's baseline (T) is expressed in."""
        return self.calib_unit.currentText() if hasattr(self, "calib_unit") else "mm"

    def _on_calib_unit(self, *_) -> None:
        """The calibration's baseline unit changed — re-derive + restate."""
        self._apply_derived_calibration()
        self._refresh_calib_status()
        self._calib_changed()

    def _apply_derived_calibration(self) -> None:
        """Fill fx/fy/cx/cy/baseline (read-only) from the rectifier. The baseline is
        in the calibration's OWN unit (the Baseline-unit selector) → converted to the
        display unit. Signals blocked so it doesn't fire a spurious re-run cue."""
        r = self._rectifier
        if r is None:
            return
        b = r.baseline * (UNIT_PER_M[self._units] / UNIT_PER_M[self._calib_src_unit()])
        for s, v in ((self.fx, r.fx), (self.fy, r.fy), (self.cx, r.cx),
                     (self.cy, r.cy), (self.baseline, b)):
            s.blockSignals(True)
            s.setValue(v)
            s.blockSignals(False)
        self.calib_hint.setText("metric depth + 3D enabled")

    def _clear_derived_calibration(self) -> None:
        """Zero the derived fields — used when a calibration fails to build, so a
        broken calibration can't leave stale intrinsics that a Run would use."""
        for s in (self.fx, self.fy, self.cx, self.cy, self.baseline):
            s.blockSignals(True)
            s.setValue(0.0)
            s.blockSignals(False)
        self.calib_hint.setText("no calibration → disparity only")

    def _process_side(self, which: str) -> None:
        """Compute left_rgb/right_rgb from the raw image — rectified in raw mode
        (once a rectifier is ready), else a straight passthrough — and refresh the
        drop thumbnail to show what the pipeline will actually see."""
        raw = self.left_raw if which == "left" else self.right_raw
        if raw is None:
            return
        rsz = (raw.shape[1], raw.shape[0])
        if self._rect_mode_on and self._rectifier is not None and self._rectifier.size == rsz:
            rgb = self._rectifier.rectify(raw, "L" if which == "left" else "R")
        else:
            if self._rect_mode_on and self._rectifier is not None:
                # size mismatch (a stray odd-sized image) — never remap through
                # wrong-sized maps (that silently samples garbage); pass through + warn
                self.notice.emit(
                    f"{which} image is {rsz[0]}×{rsz[1]} but the calibration is for "
                    f"{self._rectifier.size[0]}×{self._rectifier.size[1]} — not rectified.")
            rgb = raw
        h, w = rgb.shape[:2]
        pm = np_to_qpixmap(rgb)
        if which == "left":
            self.left_rgb = rgb
            self.drop_l.set_image(self.left_path or "", pm, w, h)
        else:
            self.right_rgb = rgb
            self.drop_r.set_image(self.right_path or "", pm, w, h)

    def _reprocess_images(self) -> None:
        if self.left_raw is not None:
            self._process_side("left")
        if self.right_raw is not None:
            self._process_side("right")

    def _refresh_calib_status(self) -> None:
        if self._calib is None:
            self.calib_status.setText("no calibration loaded")
            return
        name = os.path.basename(self._calib_path)
        u = self._calib_src_unit()
        if self._rectifier is not None:
            self.calib_status.setText(f"{name}  ·  baseline {self._rectifier.baseline:.7g} {u}")
        else:
            self.calib_status.setText(f"{name}  ·  load a pair to rectify")

    def process_pair(self, left_raw, right_raw):
        """Apply the active rectification to a RAW pair (passthrough when not in raw
        mode / no rectifier). Used by the folder batch so batched pairs are fed
        exactly like a hand-dropped one. Raises on a size mismatch so a folder whose
        resolution differs from the reference pair fails LOUDLY per-image (the batch
        banks it as failed) instead of silently remapping through wrong-sized maps."""
        self._ensure_rectifier()
        if self._rect_mode_on and self._rectifier is not None:
            size = (left_raw.shape[1], left_raw.shape[0])
            if self._rectifier.size != size:
                raise ValueError(
                    f"image is {size[0]}×{size[1]} but the calibration/reference pair is "
                    f"{self._rectifier.size[0]}×{self._rectifier.size[1]}; batch images must "
                    "match the reference resolution")
            return (self._rectifier.rectify(left_raw, "L"),
                    self._rectifier.rectify(right_raw, "R"))
        return left_raw, right_raw

    def rect_state(self) -> dict:
        return {"raw": self._rect_mode_on, "calib_path": self._calib_path,
                "calib_unit": self._calib_src_unit()}

    def restore_rect_state(self, blob) -> None:
        if not isinstance(blob, dict):
            return
        u = blob.get("calib_unit")
        if u in ("mm", "m"):
            self.calib_unit.blockSignals(True)
            self.calib_unit.setCurrentText(u)
            self.calib_unit.blockSignals(False)
        path = blob.get("calib_path") or ""
        if not blob.get("raw"):
            return
        if path and os.path.isfile(path):
            try:
                self._calib = StereoCalibration.load(path)
                self._calib_path = path
            except Exception as exc:  # noqa: BLE001 — a broken calib must not wedge startup
                self._calib, self._calib_path = None, ""
                # parity with the missing-file branch below: a PRESENT but
                # unreadable file was swallowed silently, leaving raw mode with
                # only the small "no calibration loaded" label as explanation
                self.notice.emit(
                    f"Saved calibration couldn't be read ({os.path.basename(path)}: "
                    f"{str(exc).splitlines()[0][:120]}) — re-load it to rectify.")
            self.rect_mode.setCurrentIndex(1)   # fires _on_rect_mode → sets up the UI
            self._refresh_calib_status()
        elif path:
            # was set up in raw mode last session, but the calibration file is gone —
            # say so instead of silently coming up in 'already rectified'
            self.notice.emit(f"Saved calibration not found ({os.path.basename(path)}) — "
                             "starting in 'already rectified'; re-load it to rectify.")

    # ---------------------------------------------------------- accessors
    @property
    def has_calibration(self) -> bool:
        return self.fx.value() > 0 and self.baseline.value() > 0

    @property
    def ready(self) -> bool:
        return self.left_rgb is not None and self.right_rgb is not None

    def calibration(self) -> dict:
        return dict(
            fx=self.fx.value(), fy=self.fy.value(),
            cx=self.cx.value(), cy=self.cy.value(),
            baseline=self.baseline.value(),
        )

    def set_units(self, unit: str) -> None:
        """Switch the baseline field between mm and m, rescaling its value so the
        physical calibration is preserved. Emits nothing (signals blocked) — the
        depth/cloud are rescaled by the caller instead of re-run."""
        if unit == self._units or unit not in _BASELINE_CFG:
            return
        factor = UNIT_PER_M[unit] / UNIT_PER_M[self._units]
        self._units = unit
        lo, hi, dec, suf = _BASELINE_CFG[unit]
        v = self.baseline.value() * factor
        self.baseline.blockSignals(True)
        self.baseline.setDecimals(dec)
        self.baseline.setRange(lo, hi)
        self.baseline.setSuffix(suf)
        self.baseline.setValue(v)
        self.baseline.blockSignals(False)

    # ---------------------------------------------------------- model select
    def _populate_checkpoints(self) -> None:
        """Rebuild the checkpoint list for the selected model, re-selecting the
        one you last used for THAT model (its default the first time) — the
        checkpoint is part of a model's settings, so it is remembered per model
        just like its knobs."""
        spec = self.current_spec()
        self.ckpt_combo.blockSignals(True)
        self.ckpt_combo.clear()
        if spec is not None:
            for ck in spec.checkpoints:
                self.ckpt_combo.addItem(ck.label, ck.path)
                idx = self.ckpt_combo.count() - 1
                if not ck.available():
                    item = self.ckpt_combo.model().item(idx)
                    if item is not None:
                        item.setEnabled(False)
                    self.ckpt_combo.setItemData(idx, "weights not found", Qt.ToolTipRole)
            # saved_ckpt(), not _saved_ckpt.get(): a remembered path whose file is
            # gone is still TRUTHY, so it used to short-circuit the default
            # fallback, leave the combo on a disabled dead item, and hand that dead
            # path to the engine while a perfectly good checkpoint sat one row down.
            want = self.saved_ckpt(spec.key)
            i = self.ckpt_combo.findData(want) if want else -1
            item = self.ckpt_combo.model().item(i) if i >= 0 else None
            if i >= 0 and item is not None and item.isEnabled():
                self.ckpt_combo.setCurrentIndex(i)
        self.ckpt_combo.blockSignals(False)

    def _on_ckpt_combo(self, *_) -> None:
        key = self.current_backend_key()
        if key:
            self._saved_ckpt[key] = self.current_checkpoint_path()
        self.checkpointChanged.emit()

    def saved_ckpt(self, key: str) -> str:
        """The checkpoint `key` will run with — remembered, else its default.

        The remembered path is re-checked against disk every time: it can come
        from a previous session's settings, and weights get moved or deleted
        between runs. Without this a stale path would be handed to the engine and
        fail the model for no reason while a perfectly good default sat next to it.
        """
        spec = get_spec(key)
        got = self._saved_ckpt.get(key)
        if got and any(c.path == got and c.available() for c in
                       (spec.checkpoints if spec is not None else [])):
            return got
        d = spec.default_checkpoint() if spec is not None else None
        return d.path if d is not None and d.available() else ""

    def saved_ckpts(self) -> dict:
        return dict(self._saved_ckpt)

    def restore_ckpts(self, blob) -> None:
        if isinstance(blob, dict):
            self._saved_ckpt.update({k: v for k, v in blob.items() if isinstance(v, str)})

    def section_states(self) -> dict:
        """{key: expanded?} for the foldable category headers — persisted so the
        panel reopens folded the way you left it."""
        return {"pair": self.sec_pair.is_expanded(),
                "calibration": self.sec_calib.is_expanded(),
                "model": self.sec_model.is_expanded()}

    def restore_section_states(self, blob) -> None:
        if not isinstance(blob, dict):
            return
        for key, sec in (("pair", self.sec_pair), ("calibration", self.sec_calib),
                         ("model", self.sec_model)):
            if key in blob:
                sec.set_expanded(bool(blob[key]))

    def _on_model_combo(self, *_) -> None:
        self._populate_checkpoints()
        key = self.current_backend_key()
        if key:
            self.modelChanged.emit(key)

    def current_backend_key(self) -> str:
        return self.model_combo.currentData() or ""

    def current_spec(self):
        return get_spec(self.current_backend_key())

    def current_checkpoint_path(self) -> str:
        return self.ckpt_combo.currentData() or ""

    def restore_selection(self, backend_key: str, ckpt_path: str) -> None:
        """Select a saved model + checkpoint WITHOUT emitting modelChanged /
        checkpointChanged — used on startup, where the window performs the one
        real model load itself after reading the restored selection."""
        self.model_combo.blockSignals(True)
        for i in range(self.model_combo.count()):
            item = self.model_combo.model().item(i)
            if (self.model_combo.itemData(i) == backend_key
                    and item is not None and item.isEnabled()):
                self.model_combo.setCurrentIndex(i)
                break
        self.model_combo.blockSignals(False)
        self._populate_checkpoints()          # match the (possibly changed) model
        if ckpt_path:
            self.ckpt_combo.blockSignals(True)
            for i in range(self.ckpt_combo.count()):
                item = self.ckpt_combo.model().item(i)
                if (self.ckpt_combo.itemData(i) == ckpt_path
                        and item is not None and item.isEnabled()):
                    self.ckpt_combo.setCurrentIndex(i)
                    # an explicit pick is this model's checkpoint from now on
                    key = self.current_backend_key()
                    if key:
                        self._saved_ckpt[key] = ckpt_path
                    break
            self.ckpt_combo.blockSignals(False)


class ParamPanel(QWidget):
    cloudParamsChanged = Signal()
    inferenceParamsChanged = Signal()   # a 'needs run' setting changed
    pointSizeChanged = Signal(float)
    measureChanged = Signal()           # the measure box moved / resized / toggled
    boxesChanged = Signal()             # a box was added/removed — window redraws + measures + persists
    boxSelectionChanged = Signal()      # a different box became the active one
    addBoxRequested = Signal()          # "+ Add" — only the window knows the cloud to place it on
    logRequested = Signal()             # "Log reading" — snapshot the pin heights for repeatability
    levelRequested = Signal(bool)       # "Level to plane" toggled — window fits + rotates the cloud
    analyzeToolChanged = Signal(str)    # picking tool: '' | profile | distance | region | point
    deviationToggled = Signal(bool)     # deviation-from-plane heatmap on/off
    isolateLayerToggled = Signal(bool)  # region/profile: measure only the picked Z level
    flatRefToggled = Signal(bool)       # zero board-referenced heights to a flat region
    pinAnalyzeRequested = Signal()      # analyze the selected box's pin

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("RightPane")
        self.setFixedWidth(238)
        self._units = "mm"          # display unit for z-near / z-far (mm default)
        self._saved: dict = {}      # backend key -> that model's inference settings
        self._current_key = None    # whose knobs are on screen right now
        # measure-box orientation (x,y,z,w). Driven by the 3D gizmo, not by spins —
        # a rotation has no natural spin-box form — so the panel just carries it and
        # folds it into every box it builds. Identity = axis-aligned.
        self._box_quat = (0.0, 0.0, 0.0, 1.0)
        # the measure boxes — each a dict {name, c[3], s[3], q[4], trim} in
        # CANONICAL METRES so they survive a mm⇄m switch. The spins + _box_quat +
        # Trim are the live editor for _boxes[_sel]; the others sit static.
        self._boxes: list = []
        self._sel: int = -1

        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(11)

        # --- inference --- (foldable category)
        self.sec_inference = CollapsibleSection("Inference", "Needs run", "rerun")
        lay.addWidget(self.sec_inference)
        self.scale = StatSlider("Input scale", 0.25, 1.0, 0.5, 0.05, "{:.2f}", "×",
            tip="Processing resolution — GPU memory scales with the pixel count, so "
                "this is the biggest lever you have.\n\n"
                "0.50× by default because 1.00× does not fit a normal card: the "
                "bundled 2664×2304 pair at 0.50× already reserves ~10 GB of a 12 GB "
                "3060, and 1.00× needs roughly 4× that.\n\n"
                "Going over doesn't fail cleanly — the Windows driver backs the "
                "overflow with system RAM over PCIe instead of erroring, so the run "
                "silently takes minutes instead of seconds. Raise it only for small "
                "images or a big card, and watch peak VRAM on the Compare cards.")
        self.sec_inference.add(self.scale)
        # per-model inference params — rebuilt from the active backend's schema
        self._dyn = QVBoxLayout()
        self._dyn.setContentsMargins(0, 0, 0, 0)
        self._dyn.setSpacing(13)
        self.sec_inference.add_layout(self._dyn)
        self._dyn_widgets: dict = {}   # ParamSpec.key -> (kind, widget)
        row, self.dual_ref = _toggle_row("Both eyes (2×)", False,
            "Also run the RIGHT image as reference and merge both point clouds — fills in "
            "surfaces only one camera can see (silhouette edges) for a denser result. Takes "
            "about twice as long. Then use “Color” in the 3D view to see each eye or "
            "reliable-vs-occluded points.")
        self.sec_inference.add(row)

        # --- point cloud --- (foldable category)
        self.sec_cloud = CollapsibleSection("Point cloud", "Live", "live")
        lay.addWidget(self.sec_cloud)
        _zn = _ZNEAR_CFG[self._units]
        _zf = _ZFAR_CFG[self._units]
        self.z_near = StatSlider("z-near", _zn[0], _zn[1], _zn[2], _zn[3], _zn[4], _zn[5],
            tip="Hide points CLOSER than this distance. Leave at 0 for normal scenes; "
                "raise it for close-up / macro work to trim foreground clutter.")
        self.z_far = StatSlider("z-far", _zf[0], _zf[1], _zf[2], _zf[3], _zf[4], _zf[5],
            tip="Hide points FARTHER than this distance. Lower it to trim the background "
                "and focus on your subject (for a close PCB, a few hundred mm is typical).")
        self.sec_cloud.add(self.z_near)
        self.sec_cloud.add(self.z_far)
        row, self.remove_invisible = _toggle_row("Remove invisible", True,
            "Drop points only one camera could see (occluded edges) — these are unreliable. "
            "Recommended ON.")
        self.sec_cloud.add(row)
        row, self.denoise = _toggle_row("Denoise cloud", True,
            "Remove stray floating outlier points for a cleaner 3D cloud.")
        self.sec_cloud.add(row)
        # the two Denoise knobs only matter with Denoise on — fold them away when off
        self._denoise_sub = QWidget()
        _ds = QVBoxLayout(self._denoise_sub)
        _ds.setContentsMargins(0, 0, 0, 0)
        _ds.setSpacing(13)
        self.denoise_std = StatSlider("Denoise strength", 0.5, 3.0, 2.0, 0.1, "{:.1f}", " σ",
            tip="How aggressively to remove outliers (in standard deviations). Lower = removes "
                "more points. 2.0 is a good balance.")
        self.denoise_nb = StatSlider("Denoise neighbors", 5, 60, 20, 1, "{:.0f}",
            tip="How many neighboring points to examine per point. Higher = smoother but a bit slower.")
        _ds.addWidget(self.denoise_std)
        _ds.addWidget(self.denoise_nb)
        self.sec_cloud.add(self._denoise_sub)
        self.point_size = StatSlider("Point size", 1, 6, 2, 0.5, "{:.1f}",
            tip="On-screen size of each point in the 3D view. Cosmetic only — doesn't change the data.")
        self.sec_cloud.add(self.point_size)

        # --- measure --- (foldable category)
        self.sec_measure = CollapsibleSection("Measure", "Live", "live")
        lay.addWidget(self.sec_measure)
        row, self.measure_sw = _toggle_row("Volume box", False,
            "Put a box on the cloud and report what is inside it: how many points, the "
            "nearest and farthest depth, the span between them, and how much of the box "
            "the points actually fill.\n\n"
            "With this on, CLICKING the Depth or Disparity tab drops the box on that "
            "pixel — far quicker than typing a centre in.")
        self.sec_measure.add(row)

        # Everything below only means anything with the tool ON, so it lives in a body
        # that folds away when the box is off — the section is then just the toggle.
        # (It is shown/hidden via _sync_measure_body, driven from BOTH the toggle and
        # the box ops, because switching on by adding a box blocks the toggle's signal.)
        self.measure_body = QWidget()
        mb = QVBoxLayout(self.measure_body)
        mb.setContentsMargins(0, 0, 0, 0)
        mb.setSpacing(11)
        self.sec_measure.add(self.measure_body)

        self.measure_hint = QLabel("click the Depth / Disparity map to place it")
        self.measure_hint.setProperty("role", "muted")
        self.measure_hint.setStyleSheet("font-size:11px;")
        mb.addWidget(self.measure_hint)

        # the box list: every box draws + measures at once, each its own colour; the
        # picked one takes the 3D handles + the knobs below. Remembered across restarts.
        # "+"/"−" glyph buttons — the old "+ Add"/"− Del" clipped their own text against
        # the theme's button padding, so both read as a dash.
        self.box_combo = no_wheel(QComboBox())
        self.box_combo.setToolTip(
            "The measure boxes. All of them draw and measure at once, each in its own "
            "colour; pick one here (or click it in the 3D view) to move/resize it. Its "
            "position, size, tilt and Trim are all remembered across restarts.")
        self.box_add = QPushButton("+")
        self.box_add.setFixedWidth(34)
        self.box_add.setToolTip("Add a new box, dropped on the middle of the cloud.")
        self.box_del = QPushButton("−")
        self.box_del.setFixedWidth(34)
        self.box_del.setToolTip("Delete the selected box.")
        box_lbl = QLabel("Box")
        box_lbl.setProperty("role", "muted")
        box_lbl.setFixedWidth(26)
        prow = QHBoxLayout()
        prow.setContentsMargins(0, 0, 0, 0)
        prow.setSpacing(5)
        prow.addWidget(box_lbl)
        prow.addWidget(self.box_combo, 1)
        prow.addWidget(self.box_add)
        prow.addWidget(self.box_del)
        mb.addLayout(prow)
        self._rebuild_box_combo()

        # Trim — the height-accuracy knob you reach for while reading a box.
        self.trim = StatSlider("Trim", 0.0, 10.0, 2.0, 0.5, "{:.1f}", " %",
            tip="How much to shave off each end of the depth range before reporting "
                "z-min / z-max.\n\n"
                "Raw min and max are ONE point each, and this rig's cloud already scatters "
                "~0.6–1 mm about a flat surface — so the most extreme point in a box is a "
                "flyer, and the raw span measures it rather than your part. The readout "
                "shows both: when they disagree badly, the box is full of noise.")
        mb.addWidget(self.trim)

        # pin analysis acts on the SELECTED box, so it lives here next to the box
        # list — it used to sit in Analyze, a section away from the box it needs
        self.pin_btn = QPushButton("Analyze selected pin")
        self.pin_btn.setToolTip("Analyze the SELECTED measure box's pin: height above its "
                                "local board and how vertical the pin is.")
        self.pin_btn.clicked.connect(self.pinAnalyzeRequested)
        mb.addWidget(self.pin_btn)

        self.box_log = QPushButton("＋  Log reading")
        self.box_log.setToolTip("Record the current pin heights (every box) as one capture "
                                "in the Repeatability tab — build up a mean · σ · range per pin.")
        self.box_log.clicked.connect(self.logRequested)
        mb.addWidget(self.box_log)

        # Precise numeric centre / size — folded away by default. The box is normally
        # placed with the 3D handles or by clicking the Depth tab, so these are the
        # occasional exact-entry path, not the primary control. (Rotation is reset from
        # the "Reset rot" button in the 3D view, next to the rotate handle.)
        self.box_precise = Collapsible("Position & size", expanded=False)
        set_tip(self.box_precise.header, "Type an exact centre and size. Usually easier "
                "to place the box with the 3D handles or by clicking the Depth tab.")
        pbody = self.box_precise.body_layout()
        _bc, _bs = _BOXC_CFG[self._units], _BOXS_CFG[self._units]
        _ctip = ("Centre of the box, in the same frame as the Depth tab: X right of the "
                 "optical axis, Y down from it, Z = distance from the camera. Click the "
                 "Depth tab to set all three at once.")
        _stip = "Size of the box along each axis."
        self.box_cx = make_spin(_bc[0], _bc[1], _bc[2], _bc[3], _ctip)
        self.box_cy = make_spin(_bc[0], _bc[1], _bc[2], _bc[3], _ctip)
        self.box_cz = make_spin(_bc[0], _bc[1], _bc[2], _bc[3], _ctip)
        self.box_sx = make_spin(_bs[0], _bs[1], _bs[2], _bs[3], _stip)
        self.box_sy = make_spin(_bs[0], _bs[1], _bs[2], _bs[3], _stip)
        self.box_sz = make_spin(_bs[0], _bs[1], _bs[2], _bs[3], _stip)
        _sz0 = _BOX_DEFAULT_MM * UNIT_PER_M[self._units] / UNIT_PER_M["mm"]
        for w in (self.box_sx, self.box_sy, self.box_sz):
            w.setValue(_sz0)
        for title, trio in (("centre", (self.box_cx, self.box_cy, self.box_cz)),
                            ("size", (self.box_sx, self.box_sy, self.box_sz))):
            cap = QLabel(title)
            cap.setProperty("role", "muted")
            cap.setStyleSheet("font-size:11px;")
            pbody.addWidget(cap)
            for axis, w in zip("xyz", trio):
                pbody.addLayout(field_row(f"   {axis}", w, width=26))
        mb.addWidget(self.box_precise)
        self.measure_body.setVisible(False)   # tool starts off → body folded

        # --- analyze (point-cloud analysis tools) ---
        self.sec_analyze = CollapsibleSection("Analyze", "Live", "live")
        lay.addWidget(self.sec_analyze)
        # Level heads the section: every analyze height references the board
        # plane, so this is the master "make heights board-true" switch. It used
        # to live inside the Measure body — reachable ONLY with the Volume box
        # switched on, though levelling matters just as much without it.
        self.level_btn = QPushButton("Level to board plane")
        self.level_btn.setObjectName("Toggle")   # fills accent when ON (checked)
        self.level_btn.setCheckable(True)
        self.level_btn.setToolTip(
            "Fit the flat PCB surface and rotate the whole cloud so the board sits "
            "straight (facing you). Pin heights are then measured perpendicular to the "
            "BOARD, not the camera — so a tilted mount can't bias them. Applies to every "
            "cloud, including a batch.")
        self.level_btn.toggled.connect(self.levelRequested)
        self.sec_analyze.add(self.level_btn)
        self._ANALYZE_TOOLS = ["", "profile", "distance", "region", "point"]
        self.analyze_combo = no_wheel(QComboBox())
        self.analyze_combo.addItems(["Off", "Surface angle · pick 2", "Distance · pick 2",
                                     "Region flatness · pick 2", "Point info · pick 1"])
        self.analyze_combo.setToolTip(
            "Pick points on the 3D cloud to measure:\n"
            "• Surface angle — 2 points → the slope vs the board + a cross-section profile\n"
            "• Distance — 2 points → 3D distance + ΔX/ΔY/ΔZ\n"
            "• Region flatness — 2 corners → flatness (RMS / peak-to-peak) of that patch\n"
            "• Point info — 1 point → its coordinates + height above the board")
        self.analyze_combo.currentIndexChanged.connect(self._on_analyze_combo)
        self.sec_analyze.add(self.analyze_combo)

        # isolate the picked Z LEVEL (e.g. floating pin heads) from the whole depth column
        self.isolate_btn = QPushButton("Isolate layer")
        self.isolate_btn.setObjectName("Toggle")
        self.isolate_btn.setCheckable(True)
        self.isolate_btn.setToolTip(
            "When ON, a Region or Surface-angle selection measures only the connected Z "
            "LEVEL you clicked on — e.g. floating pin heads — not the whole depth column "
            "(the board glimpsed between them). OFF = every point inside the pick.")
        self.isolate_btn.toggled.connect(self.isolateLayerToggled)
        self.sec_analyze.add(self.isolate_btn)

        # the shared readout for the pick tools AND the pin button — a tidy card
        self.analyze_card = MetricCard()
        self.sec_analyze.add(self.analyze_card)

        import pyqtgraph as pg
        self.analyze_plot = pg.PlotWidget()
        self.analyze_plot.setFixedHeight(130)
        self.analyze_plot.setBackground(None)
        self.analyze_plot.setMenuEnabled(False)
        self.analyze_plot.showGrid(x=True, y=True, alpha=0.2)
        self.analyze_plot.setLabel("left", "height")
        self.analyze_plot.setLabel("bottom", "along line")
        self._profile_curve = self.analyze_plot.plot(pen=pg.mkPen("#f4883f", width=2))
        self.analyze_plot.hide()
        self.sec_analyze.add(self.analyze_plot)

        # flat-reference "zero": measure a KNOWN-flat zone, then subtract its average
        # (the cloud's systematic error there) from every board-referenced height, and
        # show its max−min as the flatness uncertainty
        self.ref_btn = QPushButton("Set flat reference (zero)")
        self.ref_btn.setObjectName("Toggle")
        self.ref_btn.setCheckable(True)
        self.ref_btn.setToolTip(
            "Measure a Region on a zone you KNOW is flat, then press this to ZERO to it: "
            "its average height becomes an offset subtracted from every pin / point / region "
            "height (correcting the cloud's systematic bow), and its max−min is reported as "
            "the flatness uncertainty. Press again to remove the correction.")
        self.ref_btn.toggled.connect(self.flatRefToggled)
        self.ref_btn.toggled.connect(lambda *_: self._sync_ref_visibility())
        self.sec_analyze.add(self.ref_btn)
        self.ref_lbl = QLabel("")
        self.ref_lbl.setProperty("role", "muted")
        self.ref_lbl.setStyleSheet("font-size:11px;")
        self.ref_lbl.setWordWrap(True)
        self.sec_analyze.add(self.ref_lbl)

        # whole-cloud action, set off below a divider
        rule = QFrame(); rule.setObjectName("InfoRule"); rule.setFixedHeight(1)
        self.sec_analyze.add(rule)
        self.dev_btn = QPushButton("Deviation heatmap")
        self.dev_btn.setObjectName("Toggle")
        self.dev_btn.setCheckable(True)
        self.dev_btn.setToolTip("Recolour the cloud by distance from the board plane "
                                "(blue low → red high), so a bow or high/low spot pops out.")
        self.dev_btn.toggled.connect(self.deviationToggled)
        self.sec_analyze.add(self.dev_btn)

        lay.addStretch(1)

        # tool-contextual visibility: Isolate only affects region/profile picks,
        # the flat-reference pair only means anything around a Region measurement
        # (or while a reference is actively applied), and an idle result card is
        # just noise — start everything in its "tool off" state.
        self.isolate_btn.setVisible(False)
        self._sync_ref_visibility("")
        self.analyze_card.hide()
        # gated OFF until the window reports calibration / a cloud (it re-syncs
        # these on every state change via _sync_panel_gates)
        self.set_calibration_ready(False)
        self.set_cloud_ready(False)

        for w in (self.box_cx, self.box_cy, self.box_cz,
                  self.box_sx, self.box_sy, self.box_sz):
            # Emit on Enter / arrows / focus-out, never per keystroke. Each emission
            # re-measures the whole cloud (~85 ms on 4M points, and again per model
            # while overlaying), and every half-typed value is a box nobody meant:
            # typing "57" passes through "5", which is somewhere else entirely.
            w.setKeyboardTracking(False)
            w.valueChanged.connect(lambda *_: self.measureChanged.emit())
        self.trim.valueChanged.connect(lambda *_: self.measureChanged.emit())
        self.measure_sw.toggled.connect(lambda *_: self.measureChanged.emit())
        self.measure_sw.toggled.connect(self._sync_measure_body)
        self.box_combo.activated.connect(self.select_box)
        self.box_add.clicked.connect(self.addBoxRequested)
        self.box_del.clicked.connect(self.remove_box)

        for w in (self.z_near, self.z_far, self.denoise_std, self.denoise_nb):
            w.valueChanged.connect(lambda *_: self.cloudParamsChanged.emit())
        for t in (self.remove_invisible, self.denoise):
            t.toggled.connect(lambda *_: self.cloudParamsChanged.emit())
        self.denoise.toggled.connect(self._denoise_sub.setVisible)   # fold sub-knobs when off
        self._denoise_sub.setVisible(self.denoise.isChecked())
        self.point_size.valueChanged.connect(self.pointSizeChanged)

        # 'needs run' params — changing these makes the shown result stale
        # (per-model dynamic widgets are wired in set_backend)
        self.scale.valueChanged.connect(lambda *_: self.inferenceParamsChanged.emit())
        self.dual_ref.toggled.connect(lambda *_: self.inferenceParamsChanged.emit())

    def values(self) -> dict:
        return dict(
            scale=self.scale.value(),
            dual_reference=self.dual_ref.isChecked(),
            z_near=self.z_near.value(),
            z_far=self.z_far.value(),
            remove_invisible=self.remove_invisible.isChecked(),
            denoise=self.denoise.isChecked(),
            denoise_std=self.denoise_std.value(),
            denoise_nb_points=int(self.denoise_nb.value()),
            model_params=self.model_params(),
        )

    def build_params(self, calibration: dict) -> StereoParams:
        return StereoParams(**self.values(), **calibration)

    def model_params(self) -> dict:
        """Current values of the per-model dynamic widgets (keyed by ParamSpec.key)."""
        return read_param_widgets(self._dyn_widgets)

    def set_backend(self, spec) -> None:
        """Rebuild the per-model knobs from the backend's ParamSpec schema,
        REMEMBERING each model's settings.

        The outgoing model's values are stashed under its key and the incoming
        model's are restored (its documented defaults the first time). Two reasons:
        switching model and back used to silently reset your tuning, and a
        comparison needs to run every model with the values you set for IT — which
        means those values have to survive the panel being rebuilt for the others.
        """
        outgoing = self._current_key
        if outgoing is not None and self._dyn_widgets:
            self._saved[outgoing] = read_param_widgets(self._dyn_widgets)
        while self._dyn.count():
            item = self._dyn.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._dyn_widgets = {}
        self._current_key = spec.key if spec is not None else None
        if spec is None:
            return
        self._dyn_widgets = build_param_widgets(
            spec.params, self._dyn, lambda: self.inferenceParamsChanged.emit(),
            values=self._saved.get(spec.key))

    @property
    def current_key(self) -> str | None:
        """Which model the knobs on screen belong to — the last selection that
        actually stuck, which is not the same as the loaded model."""
        return self._current_key

    def saved_params(self, key: str) -> dict:
        """What `key` will run with: its live widgets if it is the model on
        screen (so edits you have not navigated away from still count), else its
        remembered values, else its documented defaults.

        Both the summary a Compare card shows and the params its run actually
        sends come through here, so they agree by construction — and sanitising
        here covers both at once.
        """
        if key and key == self._current_key and self._dyn_widgets:
            return read_param_widgets(self._dyn_widgets)   # in-spec by construction
        spec = get_spec(key)
        if key in self._saved:
            return sanitize_params(spec, self._saved[key])
        return spec.param_defaults() if spec is not None else {}

    def reset_model(self, key: str) -> None:
        """Put one model back to its documented defaults."""
        spec = get_spec(key)
        if spec is None:
            return
        self._saved.pop(key, None)
        if key == self._current_key:
            set_param_widgets(self._dyn_widgets, spec.param_defaults())

    def saved_all(self) -> dict:
        """Every model's settings, for persistence. Includes the on-screen model,
        whose live widgets are the truth."""
        out = {k: dict(v) for k, v in self._saved.items()}
        if self._current_key is not None and self._dyn_widgets:
            out[self._current_key] = read_param_widgets(self._dyn_widgets)
        return out

    def restore_all(self, blob) -> None:
        """Reload remembered settings (from QSettings). Forgiving by design: a
        stale blob naming a knob a backend has since dropped is ignored rather
        than wedging the panel."""
        if not isinstance(blob, dict):
            return
        for k, v in blob.items():
            if isinstance(v, dict):
                self._saved[k] = dict(v)
        if self._current_key in self._saved and self._dyn_widgets:
            set_param_widgets(self._dyn_widgets, self._saved[self._current_key])

    # ---- foldable categories ------------------------------------------------
    def section_states(self) -> dict:
        """{key: expanded?} for the foldable category headers — persisted so the
        panel reopens folded the way you left it."""
        return {"inference": self.sec_inference.is_expanded(),
                "cloud": self.sec_cloud.is_expanded(),
                "measure": self.sec_measure.is_expanded(),
                "analyze": self.sec_analyze.is_expanded()}

    def restore_section_states(self, blob) -> None:
        if not isinstance(blob, dict):
            return
        for key, sec in (("inference", self.sec_inference),
                         ("cloud", self.sec_cloud), ("measure", self.sec_measure),
                         ("analyze", self.sec_analyze)):
            if key in blob:
                sec.set_expanded(bool(blob[key]))

    # ------------------------------------------------------------ measure box
    @property
    def measure_on(self) -> bool:
        return self.measure_sw.isChecked()

    def measure_box(self) -> MeasureBox:
        """The box as the panel currently describes it — world frame, current unit,
        carrying the orientation the 3D gizmo last set."""
        q = self._box_quat
        return MeasureBox(
            cx=self.box_cx.value(), cy=self.box_cy.value(), cz=self.box_cz.value(),
            sx=self.box_sx.value(), sy=self.box_sy.value(), sz=self.box_sz.value(),
            qx=q[0], qy=q[1], qz=q[2], qw=q[3])

    def _sync_measure_body(self, *_) -> None:
        """Fold the measure controls away when the tool is off, unfold them when on.
        Called from the toggle AND from the box ops, because switching the tool on by
        adding a box goes through a signal-blocked setChecked (main_window._on_add_box)
        that the toggled connection never sees."""
        self.measure_body.setVisible(self.measure_sw.isChecked())

    def measure_opts(self) -> float:
        """Trim % — the knob measure_box() takes."""
        return self.trim.value()

    # ---- the box list -----------------------------------------------------
    def has_boxes(self) -> bool:
        return bool(self._boxes)

    def selected_index(self) -> int:
        return self._sel

    def _rebuild_box_combo(self, select_index: int | None = None) -> None:
        self.box_combo.blockSignals(True)
        self.box_combo.clear()
        for p in self._boxes:
            self.box_combo.addItem(p["name"])
        if select_index is not None and 0 <= select_index < self.box_combo.count():
            self.box_combo.setCurrentIndex(select_index)
        self.box_del.setEnabled(bool(self._boxes))
        self.box_combo.blockSignals(False)

    def _sync_selected(self) -> None:
        """Write the live spins/quat/Trim into _boxes[_sel] (canonical metres)."""
        if not (0 <= self._sel < len(self._boxes)):
            return
        box = self.measure_box()
        inv = 1.0 / UNIT_PER_M[self._units]
        p = self._boxes[self._sel]
        p["c"] = [box.cx * inv, box.cy * inv, box.cz * inv]
        p["s"] = [box.sx * inv, box.sy * inv, box.sz * inv]
        p["q"] = [box.qx, box.qy, box.qz, box.qw]
        p["trim"] = self.measure_opts()

    def _load_into_spins(self, p: dict) -> None:
        """Load a box dict (metres) into the spins/quat/Trim (current unit), quietly."""
        f = UNIT_PER_M[self._units]
        self._box_quat = (p["q"][0], p["q"][1], p["q"][2], p["q"][3])
        self._quiet_set((
            (self.box_cx, p["c"][0]*f), (self.box_cy, p["c"][1]*f), (self.box_cz, p["c"][2]*f),
            (self.box_sx, p["s"][0]*f), (self.box_sy, p["s"][1]*f), (self.box_sz, p["s"][2]*f)))
        self.trim.setValue(p["trim"])                    # StatSlider.setValue is emit-free

    def box_specs(self) -> list:
        """(name, MeasureBox, trim) for EVERY box, in the current unit — the selected
        box first synced from the live spins."""
        self._sync_selected()
        f = UNIT_PER_M[self._units]
        out = []
        for p in self._boxes:
            box = MeasureBox(
                cx=p["c"][0]*f, cy=p["c"][1]*f, cz=p["c"][2]*f,
                sx=p["s"][0]*f, sy=p["s"][1]*f, sz=p["s"][2]*f,
                qx=p["q"][0], qy=p["q"][1], qz=p["q"][2], qw=p["q"][3])
            out.append((p["name"], box, p["trim"]))
        return out

    def select_box(self, index: int) -> None:
        """Make box `index` the active one (gizmo + spins) — from the combo or a click."""
        if not (0 <= index < len(self._boxes)) or index == self._sel:
            if 0 <= index < len(self._boxes):
                self._rebuild_box_combo(select_index=index)
            return
        self._sync_selected()
        self._sel = index
        self._load_into_spins(self._boxes[index])
        self._rebuild_box_combo(select_index=index)
        self.boxSelectionChanged.emit()

    def _next_name(self) -> str:
        used = {p["name"] for p in self._boxes}
        i = 1
        while f"Pin {i}" in used:
            i += 1
        return f"Pin {i}"

    def append_box(self, box: MeasureBox, trim: float,
                   name: str | None = None) -> None:
        """Add a box (given in the CURRENT unit), select it, announce the change."""
        self._sync_selected()
        inv = 1.0 / UNIT_PER_M[self._units]
        self._boxes.append({
            "name": name or self._next_name(),
            "c": [box.cx*inv, box.cy*inv, box.cz*inv],
            "s": [box.sx*inv, box.sy*inv, box.sz*inv],
            "q": [box.qx, box.qy, box.qz, box.qw],
            "trim": trim})
        self._sel = len(self._boxes) - 1
        self._load_into_spins(self._boxes[self._sel])
        self._rebuild_box_combo(select_index=self._sel)
        self._sync_measure_body()   # adding the first box switches the tool on quietly
        self.boxesChanged.emit()

    def remove_box(self) -> None:
        if not (0 <= self._sel < len(self._boxes)):
            return
        del self._boxes[self._sel]
        if self._boxes:
            self._sel = min(self._sel, len(self._boxes) - 1)
            self._load_into_spins(self._boxes[self._sel])
        else:
            self._sel = -1
        self._rebuild_box_combo(select_index=self._sel if self._sel >= 0 else None)
        self.boxesChanged.emit()

    def transform_boxes(self, R, center_m, inverse: bool = False) -> None:
        """Rigidly move every box by the levelling rotation about ``center_m`` (metres)
        and reset each to axis-aligned. After levelling, the board and its pins align
        to the world Z, so an axis-aligned box measures the true perpendicular pin
        height — any prior manual box tilt is no longer needed."""
        self._sync_selected()
        Rot = np.asarray(R, np.float64)
        if inverse:
            Rot = Rot.T
        c = np.asarray(center_m, np.float64)
        for p in self._boxes:
            cc = Rot @ (np.asarray(p["c"], np.float64) - c) + c
            p["c"] = [float(cc[0]), float(cc[1]), float(cc[2])]
            p["q"] = [0.0, 0.0, 0.0, 1.0]         # axis-aligned in the levelled frame
        if 0 <= self._sel < len(self._boxes):
            self._load_into_spins(self._boxes[self._sel])

    def set_level_checked(self, on: bool) -> None:
        """Sync the Level button without emitting (the window drives the levelling)."""
        self.level_btn.blockSignals(True)
        self.level_btn.setChecked(bool(on))
        self.level_btn.blockSignals(False)

    # ------------------------------------------------------------- analyze
    def _current_tool(self) -> str:
        idx = self.analyze_combo.currentIndex()
        return self._ANALYZE_TOOLS[idx] if 0 <= idx < len(self._ANALYZE_TOOLS) else ""

    def _on_analyze_combo(self, idx: int) -> None:
        tool = self._ANALYZE_TOOLS[idx] if 0 <= idx < len(self._ANALYZE_TOOLS) else ""
        self.analyze_plot.setVisible(tool == "profile")
        self.isolate_btn.setVisible(tool in ("region", "profile"))
        self._sync_ref_visibility(tool)
        self.analyzeToolChanged.emit(tool)

    def _sync_ref_visibility(self, tool: str | None = None) -> None:
        """Show the flat-reference pair only where it can act: around a Region
        measurement, or while a reference is APPLIED (so it can always be
        removed). Anywhere else the two rows were permanent dead weight."""
        tool = self._current_tool() if tool is None else tool
        show = tool == "region" or self.ref_btn.isChecked()
        self.ref_btn.setVisible(show)
        self.ref_lbl.setVisible(show and bool(self.ref_lbl.text()))

    def set_analyze_out(self, text: str) -> None:
        """Hint / empty / error state — a single muted line in the card. Empty
        text means NOTHING to say: the card hides entirely instead of parking a
        placeholder hint in the panel."""
        if not text:
            self.analyze_card.hide()
            return
        self.analyze_card.show_message(text)
        self.analyze_card.show()

    def set_analyze_result(self, eyebrow, value, unit="", rows=None, caption="") -> None:
        """A measurement — big headline value + an aligned key/value table."""
        self.analyze_card.show_result(eyebrow, value, unit, rows, caption)
        self.analyze_card.show()

    def set_flat_ref_text(self, text: str) -> None:
        self.ref_lbl.setText(text or "")
        self._sync_ref_visibility()

    def set_flat_ref_checked(self, on: bool) -> None:
        self.ref_btn.blockSignals(True)
        self.ref_btn.setChecked(bool(on))
        self.ref_btn.blockSignals(False)
        self._sync_ref_visibility()

    # ------------------------------------------------------- state gating
    def set_calibration_ready(self, on: bool) -> None:
        """Cloud settings only do anything once metric depth is possible."""
        self.sec_cloud.set_gate_hint(
            None if on else "Needs calibration — load a K.txt or enter fx + baseline "
                            "in the left panel.")

    def set_cloud_ready(self, on: bool) -> None:
        """Measure/Analyze act on the 3D cloud — swapped for a one-line hint
        until one exists, so the panel isn't a wall of controls that do nothing.
        Values/boxes are kept; only visibility changes."""
        hint = None if on else "Run a pair (with calibration) to build the 3D cloud first."
        self.sec_measure.set_gate_hint(hint)
        self.sec_analyze.set_gate_hint(hint)

    def set_deviation_checked(self, on: bool) -> None:
        """Sync the Deviation button without emitting (the window drives the heatmap)."""
        self.dev_btn.blockSignals(True)
        self.dev_btn.setChecked(bool(on))
        self.dev_btn.blockSignals(False)

    def set_profile(self, t, h) -> None:
        if t is None or len(t) < 2:
            self._profile_curve.setData([], [])
        else:
            self._profile_curve.setData(list(t), list(h))

    def boxes_blob(self) -> dict:
        """The whole box set as plain JSON data (for QSettings)."""
        self._sync_selected()
        return {"boxes": [dict(p) for p in self._boxes], "sel": self._sel}

    def restore_boxes(self, blob) -> None:
        # accepts the new {"boxes":[...], "sel":i} OR an old bare list (presets)
        if isinstance(blob, dict):
            raw, sel = blob.get("boxes", []), blob.get("sel", 0)
        elif isinstance(blob, list):
            raw, sel = blob, 0
        else:
            raw, sel = [], -1
        clean = []
        for p in raw if isinstance(raw, list) else []:
            try:
                d = {"name": str(p["name"]),
                     "c": [float(x) for x in p["c"]][:3],
                     "s": [float(x) for x in p["s"]][:3],
                     "q": [float(x) for x in p["q"]][:4],
                     "trim": float(p["trim"])}
                # json round-trips NaN/Infinity, and a non-finite trim reaches
                # StatSlider.setValue → int(round(nan)) → ValueError INSIDE
                # _restore_settings — the exact "one bad settings value stops the
                # app from starting" class the param sliders already guard against
                if not all(math.isfinite(v)
                           for v in d["c"] + d["s"] + d["q"] + [d["trim"]]):
                    continue
                clean.append(d)
            except (KeyError, TypeError, ValueError, IndexError):
                continue
        self._boxes = [p for p in clean if len(p["c"]) == 3 and len(p["s"]) == 3
                       and len(p["q"]) == 4]
        if self._boxes:
            self._sel = sel if isinstance(sel, int) and 0 <= sel < len(self._boxes) else 0
            self._load_into_spins(self._boxes[self._sel])
        else:
            self._sel = -1
        self._rebuild_box_combo(select_index=self._sel if self._sel >= 0 else None)

    def set_box_center(self, x: float, y: float, z: float) -> None:
        """Move the box to a point, firing measureChanged exactly once."""
        self._quiet_set(((self.box_cx, x), (self.box_cy, y), (self.box_cz, z)))
        self.measureChanged.emit()

    def set_box_quiet(self, box: MeasureBox) -> None:
        """Mirror a whole box (centre + size + orientation) into the panel WITHOUT
        emitting.

        The in-viewport gizmo is the one moving here; it redraws itself live, so an
        emit per drag-tick would just fire a redundant measure of a box already on
        screen. The rotation has no spin box, so it is stashed on _box_quat and
        rides back out through measure_box().
        """
        self._box_quat = (box.qx, box.qy, box.qz, box.qw)
        self._quiet_set((
            (self.box_cx, box.cx), (self.box_cy, box.cy), (self.box_cz, box.cz),
            (self.box_sx, box.sx), (self.box_sy, box.sy), (self.box_sz, box.sz)))

    @staticmethod
    def _quiet_set(pairs) -> None:
        """Push values into spins without each one emitting.

        Not tidiness: every emission re-measures the whole cloud, and the
        intermediate ones would be measuring a box that is only part-moved — so
        setting a centre would fire three measurements, two of them of a box that
        never existed. The caller emits once, for the box it actually meant.
        """
        for w, v in pairs:
            w.blockSignals(True)
            w.setValue(float(v))
            w.blockSignals(False)

    @staticmethod
    def _reconf_spin(w: QDoubleSpinBox, cfg, value: float) -> None:
        """Re-range a spin for a new unit and give it the converted value, quietly.

        Decimals FIRST, then range, then the value. QDoubleSpinBox rounds whatever
        it is handed to its current `decimals`, so setting 0.057660 m into a spin
        still on mm's 3 decimals would land as 0.058 — a 0.34 mm move of the box
        that nothing would report. (The range clamps the old value in passing;
        that is harmless because the real value is written straight after.)
        """
        w.blockSignals(True)
        w.setDecimals(cfg[2])
        w.setRange(cfg[0], cfg[1])
        w.setSuffix(cfg[3])
        w.setValue(float(value))
        w.blockSignals(False)

    def set_units(self, unit: str) -> None:
        """Switch z-near / z-far and the measure box between mm and m, rescaling
        their current values so the physical clip planes — and the physical box —
        are preserved. Signals stay blocked throughout, so no spurious cloud
        rebuild or re-measure fires; the window re-applies both itself."""
        if unit == self._units or unit not in _ZNEAR_CFG:
            return
        factor = UNIT_PER_M[unit] / UNIT_PER_M[self._units]
        self._units = unit
        zn = self.z_near.value() * factor
        zf = self.z_far.value() * factor
        c = _ZNEAR_CFG[unit]
        self.z_near.reconfigure(c[0], c[1], zn, c[3], c[4], c[5])
        c = _ZFAR_CFG[unit]
        self.z_far.reconfigure(c[0], c[1], zf, c[3], c[4], c[5])
        # The box is a physical volume sitting on physical points, and the window
        # rescales the points on a unit switch — so the box has to move by exactly
        # the same factor, or it quietly ends up measuring a different piece of
        # the board while still reading like a valid measurement.
        box = self.measure_box().scaled(factor)
        bc, bs = _BOXC_CFG[unit], _BOXS_CFG[unit]
        for w, v in ((self.box_cx, box.cx), (self.box_cy, box.cy), (self.box_cz, box.cz)):
            self._reconf_spin(w, bc, v)
        for w, v in ((self.box_sx, box.sx), (self.box_sy, box.sy), (self.box_sz, box.sz)):
            self._reconf_spin(w, bs, v)
        # Trim is a percentage — unit-free, so it is deliberately NOT rescaled.
