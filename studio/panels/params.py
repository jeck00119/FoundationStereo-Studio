"""ParamPanel — the right panel: inference settings, point-cloud settings, the
measure boxes and the analyze tools."""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFrame, QHBoxLayout,
                               QLabel, QPushButton, QVBoxLayout, QWidget)

from ..backends import get_spec
from ..dtypes import UNIT_PER_M, StereoParams
from ..measure import MeasureBox
from ..widgets import (Collapsible, CollapsibleSection, MetricCard,
                       StatSlider, no_wheel, set_tip)
from ._common import _ZNEAR_CFG, _ZFAR_CFG, _BOXC_CFG, _BOXS_CFG, _BOX_DEFAULT_MM, _REF_TIP, make_spin, field_row, _toggle_row, build_param_widgets, read_param_widgets, set_param_widgets, sanitize_params


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
            "Put a box on the cloud and report what is inside it: how many points it "
            "caught, the height along the box's own axis (raw and Trim-cleaned — the "
            "pin height once the box is aligned), its cross-section, and the world "
            "depth range.\n\n"
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
            tip="How much to shave off each end of the box's HEIGHT range (along its "
                "own axis) before reporting the trimmed height.\n\n"
                "Raw min and max are ONE point each, and this rig's cloud already scatters "
                "~0.6–1 mm about a flat surface — so the most extreme point in a box is a "
                "flyer, and the raw span measures it rather than your part. The readout "
                "shows both: when they disagree badly, the box is full of noise.")
        mb.addWidget(self.trim)

        # pin analysis acts on the SELECTED box, so it lives here next to the box
        # list — it used to sit in Analyze, a section away from the box it needs
        self.pin_btn = QPushButton("Analyze selected pin")
        self.pin_btn.setToolTip("Analyze the SELECTED measure box's pin: height above the "
                                "board plane (the levelled plane when Level is on) and "
                                "how vertical the pin is.")
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
        self.analyze_plot.setLabel("left", f"height ({self._units})")
        self.analyze_plot.setLabel("bottom", f"along line ({self._units})")
        self._profile_curve = self.analyze_plot.plot(pen=pg.mkPen("#f4883f", width=2))
        self.analyze_plot.hide()
        self.sec_analyze.add(self.analyze_plot)

        # flat-reference "zero": measure a KNOWN-flat zone, then subtract its average
        # (the cloud's systematic error there) from every board-referenced height, and
        # show its max−min as the flatness uncertainty
        self.ref_btn = QPushButton("Set flat reference (zero)")
        self.ref_btn.setObjectName("Toggle")
        self.ref_btn.setCheckable(True)
        self.ref_btn.setToolTip(_REF_TIP)
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
        self.set_flat_ref_available(False)   # nothing measured yet to zero to
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

    def set_flat_ref_available(self, on: bool) -> None:
        """The zero button can only ACT when a measured Region exists to zero to
        — pressing it without one just bounced off with a status line. Disabled
        (with the unlock step in the tooltip) until the window reports a region;
        an APPLIED reference always stays clickable, so it can be removed."""
        en = bool(on) or self.ref_btn.isChecked()
        self.ref_btn.setEnabled(en)
        self.ref_btn.setToolTip(_REF_TIP if en else
            "Measure a Region first (Analyze → Region flatness · pick 2, on a "
            "zone you KNOW is flat) — this button then zeroes every height to it.")

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
        if t is None or len(t) < 2 or not np.isfinite(np.asarray(h, float)).any():
            self._profile_curve.setData([], [])
        else:
            # connect='finite': bins with no points are NaN, and the curve must
            # BREAK there — bridging occluded stretches drew surface that isn't
            self._profile_curve.setData(np.asarray(t, float), np.asarray(h, float),
                                        connect="finite")

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
        # The profile plot's DATA is cleared by the window on a unit switch (the
        # picks die with the old frame) — only its axis labels carry the unit.
        self.analyze_plot.setLabel("left", f"height ({unit})")
        self.analyze_plot.setLabel("bottom", f"along line ({unit})")
