"""Main window — assembles panels + viewers, owns the worker thread, wires
Run / export / live-probe / settings."""
from __future__ import annotations

import copy
import json
import os

import numpy as np
from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel,
                               QMainWindow, QMenu, QMessageBox, QProgressBar,
                               QPushButton, QScrollArea, QSizePolicy, QToolBar,
                               QVBoxLayout, QWidget)

from .backends import DEFAULT_BACKEND, get_spec
from .batch import BatchDialog, load_cloud, load_rgb
from .compare import PLANE_TIP
from .dtypes import (ANGLE_DECIMALS, UNIT_DECIMALS, UNIT_PER_M, CloudResult,
                     StereoParams)
from .engine import REPO_ROOT
from .analyze import (board_plane, deviation, pin_analysis, point_distance,
                      region_flatness, surface_profile)
from .measure import (MeasureBox, fit_plane, measure_box, points_in_box,
                      rotation_to_axis)
from .panels import InputPanel, ParamPanel
from .theme import apply_theme
from .viewers import ViewerStack
from .worker import EngineClient


class MainWindow(QMainWindow):
    def __init__(self, theme: str = "dark") -> None:
        super().__init__()
        self.setWindowTitle("FoundationStereo Studio")
        self.resize(1320, 820)
        self.theme = theme
        self.result = None          # the SHOWN result (a pointer into `results` in compare mode)
        self.cloud = None
        # --- model comparison: results cached per backend key, blinked between ---
        self.results: dict = {}     # key -> InferResult
        self.clouds: dict = {}      # key -> CloudResult
        self.mstats: dict = {}      # key -> derived stats (time, VRAM, valid%, RMS…)
        self._shown_model = None    # which cached result the views are displaying
        self._overlay_on = False    # 3D tab is drawing every model's cloud at once
        self._overlay_queue: list = []   # models still to rebuild for the overlay
        self._comparing = False
        self._compare_queue: list = []
        self._compare_total = 0
        self._compare_failed: dict = {}
        self._compare_params_snapshot = None   # the scene, frozen for the whole sweep
        # Which model's result the CHILD currently has cached. Not the same thing as
        # the loaded model: engine_process adopts whatever result a "rebuild" ships
        # it, so a foreign rebuild changes this without any load happening.
        self._child_result_key = None
        self._busy = False
        self._model_ready = False
        self._pending_run = False   # a "Load & Run" is waiting on the load
        # What the engine ACTUALLY holds. Initialised HERE, before _build_ui, because
        # restoring settings flips Compare's checkboxes, which emits changed ->
        # _update_run_enabled part-way through __init__ — anything reachable from
        # there must already exist.
        self._loaded_backend_key, self._loaded_ckpt = None, ""
        self._autodemo = bool(os.environ.get("FS_STUDIO_DEMO"))
        self._reset_cloud_view = True   # frame the camera on the next cloud
        self._cloud_pending = False     # a cloud rebuild is queued behind a busy one
        self._stale = False             # a 'needs run' setting changed since last run
        self._pair_version = 0          # bumped whenever the image pair changes
        self._run_pair_version = -1     # the pair a dispatched run belongs to
        self._units = "mm"              # display unit for depth/cloud (mm default)
        self._box_placed = False        # has the measure box ever been put somewhere?
        # --- level-to-plane: a fixed rotation that flattens the board (removes the
        # camera-vs-board tilt). Applied to EVERY cloud, so the whole fixed-fixture
        # batch reconstructs straight and pin heights read perpendicular to the board.
        self._level_R = None            # 3x3 rotation, or None (off)
        self._level_c_m = None          # rotation centre, canonical metres
        # (each cloud's un-levelled points ride on the CloudResult itself as
        # .raw_points — a single window-level slot desynced from self.cloud on
        # every blink/batch and spliced one cloud's points into another's colors)
        # --- analyze tools: pick points on the cloud to measure surface/pins ---
        self._analyze_tool = ""         # '' | profile | distance | region | point
        self._picked: list = []         # points clicked for the current measurement
        self._dev_on = False            # deviation-from-plane heatmap active
        self._analyze_isolate = False   # region/profile: keep only the picked Z level
        self._z_offset_m = None         # flat-reference zero offset (canonical metres)
        self._z_ref_pp_m = 0.0          # the reference zone's max−min (flatness uncertainty)
        self._last_region = None        # last region_flatness result (to zero from)
        self._analyze_last = None       # what the analyze card shows now: tool name | 'pin' | None
        # --- folder batch: a third reply-consumer beside compare/overlay. Runs each
        # pair through the loaded model + frozen boxes, logging one row per capture. ---
        self._batching = False
        self._batch_kind = "pairs"         # "pairs" (run model) | "clouds" (measure files)
        self._batch_dialog = None
        self._batch_cancel = False
        self._batch_queue: list = []       # pairs: [(label,l,r)] · clouds: [(label,path)]
        self._batch_label = ""             # the item currently in flight
        self._batch_specs: list = []       # boxes frozen at batch start
        self._batch_params = None          # the scene (calibration+cloud), frozen
        self._batch_file_factor = 1.0      # clouds: file-unit -> working-unit scale
        self._cloud_shown = False          # clouds: shown one cloud for confirmation yet
        self._batch_total = 0
        self._batch_done = 0
        self._batch_logged = 0
        self._batch_empty = 0              # captures where no box caught any points
        self._batch_failed: list = []      # [(label, reason)]

        self._build_ui()
        self._start_worker()

        # These timers MUST be created before _restore_settings(): restoring a saved
        # non-mm unit runs _set_units → _apply_measure → self._measure_timer.start(),
        # so if the timer didn't exist yet the app would crash on every launch after
        # the user last closed it in µm or m (AttributeError: no '_measure_timer').
        self._vram_timer = QTimer(self)
        self._vram_timer.timeout.connect(self._poll_vram)
        # debounce cloud-param edits: collapse a slider drag into ONE rebuild
        self._cloud_timer = QTimer(self)
        self._cloud_timer.setSingleShot(True)
        self._cloud_timer.setInterval(250)
        self._cloud_timer.timeout.connect(self._rebuild_cloud)
        # debounce the box MEASURE. The Trim slider emits continuously and
        # each measure_box is a full scan of up to millions of points, so a slider
        # drag would fire one ~85 ms numpy pass per tick. _apply_measure pushes the
        # box to the view immediately; only this heavier read waits to settle.
        self._measure_timer = QTimer(self)
        self._measure_timer.setSingleShot(True)
        self._measure_timer.setInterval(120)
        self._measure_timer.timeout.connect(self._remeasure)

        self._restore_settings()   # may restore a non-mm unit → needs the timers above
        self._sync_panel_gates()   # initial gate state (no cloud yet; calib per restore)

        # NOTHING is loaded at startup. Opening the app used to spend ~20 s and up
        # to 3 GB of VRAM on whichever model you happened to close with — before
        # you'd even chosen an image, and again every time you opened it to look at
        # something. So `_loaded_backend_key` stays None, `_model_ready` stays False,
        # `_needs_load()` is therefore True, and Run reads "▶ Load & Run": pressing
        # it loads what the panel shows and then runs it, which is the exact path
        # Compare already uses for each of its models. The engine CHILD does start
        # (see _start_worker) — it holds no model and no VRAM, it just means the
        # first Run isn't also paying for torch's import.
        self.overlay.hide()
        self._set_status("Drop a stereo pair, then Run.")
        # Run was built reading "▶ Run" and _on_model_loaded used to be what
        # corrected it. Nothing loads now, so nothing fires — say what the button
        # will actually DO before it's ever pressed.
        self._sync_run_button()
        self._update_run_enabled()
        self._maybe_autodemo()   # moved off _on_model_loaded, which no longer fires

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        tb = QToolBar()
        tb.setMovable(False)
        self.addToolBar(tb)
        brand = QLabel("  FoundationStereo Studio")
        brand.setObjectName("Brand")
        tb.addWidget(brand)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        tb.addWidget(spacer)

        self.batch_btn = QPushButton("Batch…")
        self.batch_btn.setToolTip(
            "Run a whole folder of stereo pairs through the loaded model and your "
            "measure boxes, logging one row per capture into the Repeatability tab.\n\n"
            "Set up first: Run one pair, place a box on each pin, then Batch.")
        self.batch_btn.clicked.connect(self._open_batch)
        tb.addWidget(self.batch_btn)

        self.compare_btn = QPushButton("Compare…")
        self.compare_btn.setToolTip(
            "Open the Compare tab — run several models on this pair, each with the "
            "settings you choose, and flip between the results.")
        # Opens the tab rather than running: the settings a comparison uses are
        # there to be seen and edited, so the button leads to them.
        self.compare_btn.clicked.connect(
            lambda: self.viewer.setCurrentWidget(self.viewer.compare_view))
        tb.addWidget(self.compare_btn)

        self.export_btn = QPushButton("Export ▾")
        self.export_btn.setEnabled(False)
        self.export_btn.setToolTip("Save results — disparity/depth images (PNG), raw arrays "
                                   "(NPY), or the 3D point cloud (PLY).")
        self._build_export_menu()
        tb.addWidget(self.export_btn)

        self.units_btn = QPushButton("mm")
        self.units_btn.setFixedWidth(42)
        self.units_btn.setToolTip("Measurement unit for depth and the 3D cloud.\n"
                                  "Click to cycle mm → µm → m. µm is best for small pins; "
                                  "mm for general close-up / PCB work.")
        self.units_btn.clicked.connect(self._toggle_units)
        tb.addWidget(self.units_btn)

        self.theme_btn = QPushButton("☾" if self.theme == "dark" else "☀")
        self.theme_btn.setFixedWidth(38)
        self.theme_btn.setToolTip("Switch between dark and light theme.")
        self.theme_btn.clicked.connect(self._toggle_theme)
        tb.addWidget(self.theme_btn)

        self.run_btn = QPushButton("▶  Run")
        self.run_btn.setObjectName("Accent")
        self.run_btn.setShortcut("Ctrl+Return")
        self.run_btn.setEnabled(False)
        self.run_btn.setToolTip("Run the selected model on the loaded pair (Ctrl+Enter). Turns "
                                "amber when a setting changed and you need to re-run.")
        self.run_btn.clicked.connect(self._run)
        tb.addWidget(self.run_btn)

        # panels + viewer
        self.input_panel = InputPanel()
        self.param_panel = ParamPanel()
        self.viewer = ViewerStack()

        self.input_panel.imagesChanged.connect(self._on_images)
        self.input_panel.calibrationChanged.connect(self._update_run_enabled)
        self.input_panel.calibrationChanged.connect(self._mark_stale)
        self.input_panel.calibrationChanged.connect(self._sync_panel_gates)
        self.input_panel.notice.connect(self._set_status)
        self.input_panel.modelChanged.connect(self._on_model_changed)
        self.input_panel.checkpointChanged.connect(self._on_checkpoint_changed)
        self.param_panel.cloudParamsChanged.connect(self._schedule_rebuild)
        self.param_panel.inferenceParamsChanged.connect(self._mark_stale)
        self.param_panel.pointSizeChanged.connect(self.viewer.cloud_view.set_point_size)
        self.param_panel.measureChanged.connect(self._apply_measure)
        self.param_panel.boxesChanged.connect(self._apply_measure)
        self.param_panel.boxSelectionChanged.connect(self._apply_measure)
        self.param_panel.addBoxRequested.connect(self._on_add_box)
        self.param_panel.logRequested.connect(self._log_reading)
        self.param_panel.levelRequested.connect(self._on_level_toggled)
        self.param_panel.analyzeToolChanged.connect(self._on_analyze_tool)
        self.param_panel.deviationToggled.connect(self._on_deviation)
        self.param_panel.isolateLayerToggled.connect(self._on_isolate_layer)
        self.param_panel.flatRefToggled.connect(self._on_flat_ref)
        self.param_panel.pinAnalyzeRequested.connect(self._on_pin_analyze)
        self.param_panel.measure_sw.toggled.connect(self._on_measure_toggled)
        self.viewer.repeat_view.logRequested.connect(self._log_reading)
        self.viewer.hovered.connect(self._on_hover)
        self.viewer.pixelClicked.connect(self._on_pixel_clicked)
        self.viewer.cloud_view.boxEdited.connect(self._on_box_edited)
        self.viewer.cloud_view.boxSelected.connect(self._on_box_selected)
        self.viewer.cloud_view.pointPicked.connect(self._on_point_picked)
        self.viewer.cloud_view.overlayToggled.connect(self._on_overlay_toggled)
        self.viewer.compare_view.runRequested.connect(self._start_compare)
        self.viewer.compare_view.showRequested.connect(self._show_model)
        self.viewer.compare_view.editRequested.connect(self._edit_model_settings)
        self.viewer.compare_view.changed.connect(self._update_run_enabled)
        # a knob moved in the Inference panel -> restate what its model will run with
        self.param_panel.inferenceParamsChanged.connect(self._refresh_compare_cards)
        self.input_panel.checkpointChanged.connect(self._refresh_compare_cards)
        # the three blink bars mirror one another — flipping model on any tab
        # flips them all, so Disparity/Depth/3D always show the SAME model
        for bar in self._model_bars():
            bar.picked.connect(self._show_model)

        # build the per-model inference widgets for the initial model selection
        self.param_panel.set_backend(self.input_panel.current_spec())

        # Caption for the RESULT VIEWERS: "the picture above came from this model,
        # and here is how it scored". Mounted under the whole tab stack, so it has
        # to be told which tabs it can honestly caption — see _update_compare_strip.
        self.compare_lbl = QLabel("")
        self.compare_lbl.setObjectName("CompareStrip")
        self.compare_lbl.setTextFormat(Qt.PlainText)
        self.compare_lbl.setStyleSheet(
            'font-family:"Cascadia Mono","Consolas",monospace; font-size:11px;'
            "padding:4px 8px;")
        self.compare_lbl.hide()

        center = QWidget()
        cvl = QVBoxLayout(center)
        cvl.setContentsMargins(0, 0, 0, 0)
        cvl.setSpacing(0)
        cvl.addWidget(self.viewer, 1)
        cvl.addWidget(self.compare_lbl)
        # The strip captions the tab you are ON, so it has to re-decide on every tab
        # change, not only when a Show button re-points it. Connected here rather
        # than beside the other viewer signals: those are wired before compare_lbl
        # exists, and this slot touches it.
        self.viewer.currentChanged.connect(lambda *_: self._update_compare_strip())

        central = QWidget()
        central.setObjectName("Pane")
        lay = QHBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self._scroll(self.input_panel))
        lay.addWidget(center, 1)
        lay.addWidget(self._scroll(self.param_panel))
        self.setCentralWidget(central)

        # Loading scrim. Hidden at startup (nothing loads until Run) — every SHOW
        # goes through _show_load_overlay, which sets the real model name + weight
        # size first, so this placeholder text is never actually displayed.
        self.overlay = QLabel("◆  Loading model…", central)
        self.overlay.setObjectName("Overlay")
        self.overlay.setAlignment(Qt.AlignCenter)
        self.overlay.setStyleSheet(
            "background: rgba(10,13,20,0.90); color:#E7ECF4;"
            'font-family:"Cascadia Mono","Consolas",monospace; font-size:15px;')
        self.overlay.setGeometry(central.rect())
        self.overlay.raise_()

        # status bar
        sb = self.statusBar()
        self.status_lbl = QLabel("Starting…")
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setFixedWidth(120)
        self.progress.hide()
        sb.addWidget(self.status_lbl, 1)
        sb.addWidget(self.progress)
        self.probe_lbl = QLabel("")
        self.points_lbl = QLabel("")
        self.timing_lbl = QLabel("")
        self.vram_lbl = QLabel("VRAM —")
        for w in (self.probe_lbl, self.points_lbl, self.timing_lbl, self.vram_lbl):
            sb.addPermanentWidget(w)

    def _scroll(self, w: QWidget) -> QScrollArea:
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setWidget(w)
        sa.setFixedWidth(w.width() + 16)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        return sa

    def _build_export_menu(self) -> None:
        m = QMenu(self)
        acts = [
            ("Disparity image (PNG)…", lambda: self._export("disp_png")),
            ("Depth image (PNG)…", lambda: self._export("depth_png")),
            ("Disparity — raw (.npy)…", lambda: self._export("disp_npy")),
            ("Depth — raw (.npy)…", lambda: self._export("depth_npy")),
            ("Point cloud (.ply)…", lambda: self._export("ply")),
            ("Everything → folder…", lambda: self._export("all")),
        ]
        for label, fn in acts:
            a = QAction(label, self)
            a.triggered.connect(fn)
            m.addAction(a)
        self.export_btn.setMenu(m)

    # -------------------------------------------------------------- worker
    def _start_worker(self) -> None:
        self.worker = EngineClient()
        self.worker.progress.connect(self._set_status)
        self.worker.busy.connect(self._on_busy)
        self.worker.modelLoaded.connect(self._on_model_loaded)
        self.worker.inferenceDone.connect(self._on_inference)
        self.worker.cloudDone.connect(self._on_cloud)
        self.worker.error.connect(self._on_error)
        self.worker.vram.connect(self._on_vram)
        self.worker.start()   # spawns the engine child process

    # --------------------------------------------------------------- slots
    def _set_status(self, msg: str) -> None:
        self.status_lbl.setText(msg)

    def _on_busy(self, busy: bool) -> None:
        self._busy = busy
        # keep the bar up BETWEEN pairs of a batch (busy flickers false between them)
        self.progress.setVisible(busy or self._batching)
        if self._batching:
            # the batch owns the toolbar's enabled-state (everything locked) and the
            # progress bar's range/value — don't let per-pair busy toggles fight it
            self._update_run_enabled()
            return
        # `not _comparing` too: busy flickers FALSE between compare legs (cloudDone
        # fires before the next leg re-raises busy), and a unit/model click landing
        # in that gap corrupted the sweep — the frozen snapshot kept the old unit
        # while results banked into new-unit caches, or a model change stranded the
        # queue. The sweep ends re-enable these explicitly.
        idle = not busy and not self._comparing
        self.units_btn.setEnabled(idle)       # no mm⇄m switch while a run is in flight
        # no model/checkpoint switch mid-run — it would tear down the busy child
        self.input_panel.model_combo.setEnabled(idle)
        self.input_panel.ckpt_combo.setEnabled(idle)
        self._update_run_enabled()

    def _on_model_loaded(self, device: str) -> None:
        self._model_ready = True
        self.overlay.hide()
        self._vram_timer.start(1500)
        self.input_panel.device_lbl.setText(f"device: {device}")
        self._sync_run_button()
        self._update_run_enabled()
        if self._comparing:
            # next leg of the comparison — run this model straight away
            QTimer.singleShot(0, self._compare_run_current)
            return
        if self._pending_run:
            # this load was "Load & Run" — finish what the user asked for
            self._pending_run = False
            QTimer.singleShot(0, self._run)
            return
        self._set_status("Model ready.")

    # ------------------------------------------------------- model switching
    def _load_overlay_text(self, spec, ckpt: str, note: str = "one-time") -> str:
        """Overlay caption for a load: the REAL model name + the REAL weight size
        (read off the file), so it can never claim the wrong model or a stale
        '~20 s · 3 GB' — Fast-FoundationStereo is 68 MB and loads in ~4 s."""
        name = spec.display_name if spec else "model"
        try:
            mb = os.path.getsize(ckpt) / (1024 * 1024)
            size = f"{mb / 1024:.1f} GB" if mb >= 1024 else f"{mb:.0f} MB"
            return f"◆  Loading {name}…\n\n{size} · {note}"
        except OSError:
            return f"◆  Loading {name}…"

    # ------------------------------------------------------ model comparison
    def _model_bars(self) -> tuple:
        return (self.viewer.disp_view.model_bar,
                self.viewer.depth_view.model_bar,
                self.viewer.cloud_view.model_bar)

    @staticmethod
    def _plane_rms(cloud) -> float | None:
        """Robust spread of the cloud about its best-fit plane — a noise proxy for
        FLAT subjects (the user's PCBs).

        Two-pass on purpose. A plain least-squares fit over the whole cloud measures
        mostly the REAL structure — components standing off the board, edges,
        occlusion outliers — and RMS is very outlier-sensitive, so it would rank
        models by their worst few thousand pixels rather than their surface noise.
        So: fit, drop the worst 20% of residuals, refit on the inner 80% and report
        that spread. Deterministic (SVD + quantile, no RANSAC — an earlier
        investigation showed RANSAC's randomness alone moved a plane fit by ~0.8°,
        swamping the thing being measured).

        Still only meaningful when the subject really is flat; the strip's tooltip
        says so, and it is a comparison between models on the SAME scene, not an
        absolute figure.
        """
        if cloud is None or getattr(cloud, "n", 0) < 500:
            return None
        p = np.asarray(cloud.points, np.float64)
        if len(p) > 200_000:                      # deterministic subsample — keep it cheap
            p = p[np.linspace(0, len(p) - 1, 200_000).astype(np.int64)]
        c = p.mean(0)
        _, _, vt = np.linalg.svd(p - c, full_matrices=False)
        d = np.abs((p - c) @ vt[-1])              # vt[-1] = plane normal
        inner = p[d <= np.quantile(d, 0.8)]
        if len(inner) < 100:
            return None
        c2 = inner.mean(0)
        _, _, vt2 = np.linalg.svd(inner - c2, full_matrices=False)
        return float(np.sqrt(np.mean(((inner - c2) @ vt2[-1]) ** 2)))

    def _model_stats(self, result, cloud) -> dict:
        disp = result.disp
        valid = np.isfinite(disp) & (disp > 0)
        med = None
        if result.depth is not None and np.any(result.depth > 0):
            med = float(np.median(result.depth[result.depth > 0]))
        return {
            "net_s": float(result.timing.get("net_s", 0.0)),
            "peak_gb": float(result.timing.get("peak_vram_gb", 0.0)),
            "valid_pct": 100.0 * float(valid.sum()) / disp.size,
            "med_depth": med,
            "n_pts": int(getattr(cloud, "n", 0) or 0),
            "plane_rms": self._plane_rms(cloud),
        }

    def _compare_params(self, key: str) -> StereoParams:
        """The scene is shared, the model's own settings are not.

        Scale + calibration + cloud settings come from the SNAPSHOT taken when the
        sweep started, so every model provably solves the same problem — reading
        the live panels per leg let a mid-sweep nudge of the scale slider or the
        baseline field change the scene between models, while the stats strip went
        on claiming they all ran on the same one. The per-model inference knobs are
        whatever the Inference panel remembers for THAT model."""
        p = self._compare_params_snapshot
        p = copy.copy(p) if p is not None else self._current_params()
        p.model_params = self.param_panel.saved_params(key)
        return p

    # --------------------------------------------- editing a model's settings
    def _edit_model_settings(self, key: str) -> None:
        """Point the Inference panel at `key` WITHOUT loading it.

        Setting up a comparison means touching every model, and loading each one
        just to move a slider would cost ~20 s a piece (FoundationStereo is 3 GB).
        So this only re-points the panel; the engine keeps whatever it holds and
        the models get loaded one by one when you run.

        The dropdown follows (via restore_selection, which blocks signals so no
        switch fires), which means the selection may now LEAD the engine — `_run`
        is what reconciles that, and `_sync_run_button` keeps it visible.
        """
        spec = get_spec(key)
        if spec is None:
            return
        ok, reason = spec.availability()
        if not ok:
            QMessageBox.warning(self, "No weights",
                                f"{spec.display_name} can't be set up — {reason}.")
            return
        self.input_panel.restore_selection(key, self.input_panel.saved_ckpt(key))
        self.param_panel.set_backend(spec)      # restores THIS model's settings
        self._refresh_compare_cards()           # highlights this card, names its weights
        self._sync_run_button()
        self._update_run_enabled()
        if key == self._loaded_backend_key:
            self._set_status(f"Editing {spec.display_name} — it's the loaded model, "
                             "so Run uses it directly.")
        else:
            self._set_status(
                f"Editing {spec.display_name} — nothing loaded. Run comparison to "
                "run every ticked model, or Load & Run to try this one alone.")

    def _refresh_compare_cards(self) -> None:
        """Name the weights each column will run with.

        The columns used to restate every knob too. They don't any more: clicking a
        card puts that model in the Inference panel, so the settings are one click
        away in the place that owns them — and the numbers, which are why you're on
        this tab, aren't buried under four lines about them. Weights stay because
        that's identity, not a setting, and it's nowhere else on the card."""
        cv = self.viewer.compare_view
        # With the Edit button gone, the highlight is the only thing saying which
        # model the panel is editing — and that is ALWAYS the selected one, so it
        # belongs here rather than being re-derived at five call sites that could
        # each get it wrong.
        cv.set_editing(self.input_panel.current_backend_key())
        for key in cv.cards:
            spec = get_spec(key)
            ckpt = self.input_panel.saved_ckpt(key)
            label = ""
            if spec is not None and ckpt:
                label = next((c.label for c in spec.checkpoints if c.path == ckpt),
                             os.path.basename(ckpt))
            cv.set_ckpt(key, label)

    def _start_compare(self) -> None:
        if self._busy or self._comparing or self._batching:
            return
        if not self.input_panel.ready:
            QMessageBox.information(self, "Load a pair",
                                    "Load both left and right images first.")
            return
        keys = self.viewer.compare_view.selected_keys()
        if len(keys) < 2:
            QMessageBox.information(
                self, "Compare",
                "Tick at least two models in the Compare tab.\n\nA model is only "
                "selectable once its weights are on disk.")
            return
        self._pair_version += 1        # supersede anything still in flight
        self._clear_results()          # also drops any previous comparison
        self._comparing = True
        self._compare_queue = list(keys)
        self._compare_total = len(keys)
        self._compare_failed = {}
        # Freeze the scene now. Every leg reuses this, so the comparison is
        # apples-to-apples even if the scale/calibration/cloud controls are touched
        # while it runs (they stay live — only the cards are locked).
        self._compare_params_snapshot = self._current_params()
        self.viewer.compare_view.set_running(True)
        for k in keys:
            self.viewer.compare_view.set_status(k, "queued")
        self._update_run_enabled()
        self._compare_next()

    def _compare_next(self) -> None:
        if not self._comparing:
            return
        if not self._compare_queue:
            self._finish_compare()
            return
        key = self._compare_queue[0]
        spec = get_spec(key)
        ckpt = self.input_panel.saved_ckpt(key)
        done = self._compare_total - len(self._compare_queue) + 1
        self._set_status(f"Comparing {done}/{self._compare_total} — {spec.display_name}…")
        self.viewer.compare_view.set_status(key, "loading…")
        self._show_load_overlay(
            self._load_overlay_text(spec, ckpt, f"comparing {done}/{self._compare_total}"))
        # keep the dropdowns honest about what the engine is really holding
        # (signals blocked inside, so this can't re-enter _on_model_changed)
        self.input_panel.restore_selection(key, ckpt)
        self.param_panel.set_backend(spec)
        self._loaded_backend_key, self._loaded_ckpt = key, ckpt
        self._switch_engine(spec.python_exe, key, ckpt)

    def _compare_run_current(self) -> None:
        if not self._comparing or not self._compare_queue:
            return
        key = self._compare_queue[0]
        self.viewer.compare_view.set_status(key, "running…")
        self._run_pair_version = self._pair_version
        self.worker.runInference(self.input_panel.left_rgb, self.input_panel.right_rgb,
                                 self._compare_params(key))

    def _abort_compare(self, reason: str) -> None:
        """Stop an in-flight sweep and hand the app back to the user.

        The sweep is driven entirely by replies landing in _on_cloud/_on_error. Any
        event that makes those replies stale — loading a new pair is the reachable
        one, since the image drops stay live during a comparison — would otherwise
        see them dropped by the pair-version gate BEFORE the queue is popped, so
        the sweep would never advance and never finish: `_comparing` stuck True,
        Run and Run-comparison disabled forever, every card locked, and no error
        anywhere. Restart-only. So: unwind explicitly instead.
        """
        if not self._comparing:
            return
        self._comparing = False
        self._compare_queue = []
        self._compare_params_snapshot = None
        self.viewer.compare_view.set_running(False)
        for k in self.viewer.compare_view.cards:
            self.viewer.compare_view.set_status(k, "not run yet")
        self._restore_scene_controls()
        self._update_run_enabled()
        self._set_status(reason)

    def _restore_scene_controls(self) -> None:
        """Re-enable the unit/model controls a compare sweep held disabled across
        its busy gaps (see _on_busy — its `idle` test keeps them off mid-sweep, so
        the sweep's two exits have to hand them back)."""
        idle = not self._busy
        self.units_btn.setEnabled(idle)
        self.input_panel.model_combo.setEnabled(idle)
        self.input_panel.ckpt_combo.setEnabled(idle)

    def _finish_compare(self) -> None:
        self._comparing = False
        self._compare_params_snapshot = None
        self.viewer.compare_view.set_running(False)
        self._restore_scene_controls()
        models = [(k, get_spec(k).display_name.split("·")[0].strip(), get_spec(k).display_name)
                  for k in self.results]
        for bar in self._model_bars():
            bar.set_models(models)
        # only offer the overlay once there is more than one cloud to stack
        self.viewer.cloud_view.set_overlay_available(len(self._overlay_keys()) > 1)
        if self.results:
            self._show_model(next(iter(self.results)))
        self._update_run_enabled()
        n = len(self.results)
        msg = (f"Compared {n} models — use the Model buttons to flip between them."
               if n else "Comparison produced no results.")
        if self._compare_failed:
            msg += f"  {len(self._compare_failed)} failed."
        self._set_status(msg)
        if self._compare_failed:
            QMessageBox.warning(
                self, "Compare — some models failed",
                "\n\n".join(f"{get_spec(k).display_name}:\n{v}"
                            for k, v in self._compare_failed.items()))

    # ------------------------------------------------------- 3D model overlay
    def _overlay_keys(self) -> list:
        """Models with a cloud to draw, in registry order — so a model keeps the
        same colour for the whole session however the sweep happened to order it."""
        return [k for k in self.viewer.compare_view.cards
                if self.clouds.get(k) is not None]

    def _short_name(self, key: str) -> str:
        spec = get_spec(key)
        return spec.display_name.split("·")[0].strip() if spec else key

    def _show_overlay(self) -> None:
        """Every model's cloud at once, each point tagged with who made it.

        Concatenating and tagging rather than drawing N scatters: the view already
        colours points from a per-point category array (that is what "Camera (L·R)"
        is), so a model index is the same mechanism and inherits the same filtering,
        grid fitting and readout for free.

        origin/reliable are dropped here on purpose — "which eye" and "is this
        occluded" are questions about ONE model's reconstruction, and answering them
        across a pile of different models' points would be meaningless.
        """
        keys = self._overlay_keys()
        if len(keys) < 2:
            return
        pts, cols, idx = [], [], []
        for i, k in enumerate(keys):
            c = self.clouds[k]
            pts.append(c.points)
            cols.append(c.colors)
            idx.append(np.full(len(c.points), i, np.uint8))
        self._overlay_on = True
        cv = self.viewer.cloud_view
        cv.set_cloud(np.concatenate(pts), np.concatenate(cols),
                     model=np.concatenate(idx),
                     model_names=[self._short_name(k) for k in keys],
                     reset_view=False)
        cv.color_combo.setCurrentText("Model")   # photo colour would be an unreadable mush
        self._apply_measure()    # set_cloud left the box alone; the targets just changed
        self._set_status(f"Overlaying {len(keys)} models — "
                         "untick one in the key to take it out.")

    def _on_overlay_toggled(self, on: bool) -> None:
        self._overlay_on = bool(on)
        if on:
            self._show_overlay()
        elif self._shown_model is not None:
            self._show_model(self._shown_model)
        else:
            self.viewer.show_cloud(self.cloud, reset_view=False)
            self._apply_measure()

    # ---------------------------------------------------------- measure box
    @staticmethod
    def _has_points(c) -> bool:
        return c is not None and getattr(c, "n", 0) > 0

    def _measure_targets(self) -> list:
        """[(label, cloud)] the box should report on.

        While overlaying that is every model on screen — one box, one set of
        points, one line each, which is the comparison the overlay exists for.
        Otherwise it is just the cloud you are looking at.
        """
        if self._overlay_on:
            return [(self._short_name(k), self.clouds[k]) for k in self._overlay_keys()
                    if self._has_points(self.clouds[k])]
        if not self._has_points(self.cloud):
            return []
        name = self._short_name(self._shown_model) if self._shown_model else ""
        return [(name, self.cloud)]

    def _sync_panel_gates(self) -> None:
        """Tell the parameter panel which sections currently CAN do anything:
        cloud settings need calibration, Measure/Analyze need a cloud on screen.
        Called from the two places display state changes funnel through
        (_apply_measure runs on every cloud change; _clear_results on every
        clear) plus calibration edits — cheap and idempotent."""
        self.param_panel.set_calibration_ready(self.input_panel.has_calibration)
        self.param_panel.set_cloud_ready(self._has_points(self.cloud))

    def _apply_measure(self) -> None:
        """Push the whole set of boxes into the 3D view (drawing them all + the
        selected one's drag handles). The authoritative path — every non-drag trigger
        (spins, a new cloud, a unit switch, add/remove/select, the toggle) comes here.

        Pure numpy over the cached cloud — no engine round-trip, so it stays instant
        while you drag a spin, and it works with the child busy.
        """
        self._sync_panel_gates()
        cv = self.viewer.cloud_view
        on = self.param_panel.measure_on
        if on:
            specs = self.param_panel.box_specs()          # syncs the selected box
            # non-editable while a batch runs — the study's boxes are frozen, so the
            # gizmo mustn't be draggable (it would diverge the drawn box from the
            # logged one and persist the nudge) — and non-editable while an Analyze
            # tool is armed, or the drag gizmo swallows the click meant to PICK a point
            editable = not self._batching and not self._analyze_tool
            cv.set_boxes([b for _n, b, _t in specs],
                         self.param_panel.selected_index(), editable)
            self.viewer.repeat_view.set_pins([n for n, _b, _t in specs])
        else:
            cv.set_boxes([], -1, False)                   # off: take every box + handle down
        self._measure_timer.start()   # debounced: coalesce trim slider ticks

    def _remeasure(self) -> None:
        """Measure the boxes and push the readout. Split from _apply_measure only so
        the geometry push and the (heavier) measurement read as two steps. Persists
        the box set here too (debounced) so edits survive even an unclean exit."""
        self._save_boxes()
        cv = self.viewer.cloud_view
        if not self.param_panel.measure_on:
            cv.set_measurement("")
            return
        specs = self.param_panel.box_specs()
        if not specs:
            cv.set_measurement("")
            return
        sel = self.param_panel.selected_index()
        if len(specs) == 1:
            # a single box keeps the full detail (+ the per-model overlay breakdown)
            _n, box, trim = specs[0]
            targets = self._measure_targets()
            if not targets:
                cv.set_measurement("")
                return
            stats = [(tn, measure_box(c.points, box, trim_pct=trim))
                     for tn, c in targets]
            cv.set_measurement(self._measure_text(stats, trim))
            band = next((m for _, m in stats if m is not None), None)
        else:
            # several boxes: one line each, against the shown cloud
            cloud = self.cloud if self._has_points(self.cloud) else None
            rows, band = [], None
            for i, (name, box, trim) in enumerate(specs):
                m = (measure_box(cloud.points, box, trim_pct=trim)
                     if cloud is not None else None)
                rows.append((name, m, i == sel))
                if i == sel:
                    band = m
            cv.set_measurement(self._boxes_text(rows))
        # the Trim highlight follows the SELECTED box's trimmed band
        if band is not None:
            cv.set_box_scalars(band["h_min_t"], band["h_max_t"])

    def _boxes_text(self, rows: list) -> str:
        """The multi-box readout: one line per box, ▸ marking the selected one, then a
        rule and the max−min spread of the pin heights (how coplanar the pin tips are)."""
        u = self._units
        dec = UNIT_DECIMALS.get(u, 2)
        width = max((len(n) for n, _, _ in rows), default=4)
        lines = ["▣ Boxes"]
        heights = []
        for name, m, is_sel in rows:
            mark = "▸" if is_sel else " "
            if m is None:
                lines.append(f" {mark} {name:<{width}}   empty")
                continue
            heights.append(m["h_span_t"])
            lines.append(f" {mark} {name:<{width}}   {m['n']:>7,} pts"
                         f"   h {m['h_span_t']:.{dec}f} {u}")
        if len(heights) >= 2:
            spread = max(heights) - min(heights)
            lines.append("─" * max(len(ln) for ln in lines[1:]))
            lines.append(f"   h max − min   {spread:.{dec}f} {u}")
        return "\n".join(lines)

    def _on_box_edited(self, idx, box, final: bool) -> None:
        """The 3D gizmo moved a box (idx — always the selected one). Mirror it into
        the panel spins and, once the drag settles, re-measure everything.

        Mid-drag we deliberately do NOT re-measure in Python: the web view shows its
        own cheap live readout for the dragged box straight from the points it holds,
        so hammering measure_box every mouse-move would only add lag."""
        if self._batching:
            return                        # boxes are frozen during a batch
        if idx == self.param_panel.selected_index():
            self.param_panel.set_box_quiet(box)
        self._box_placed = True
        if final:
            self._apply_measure()

    def _on_box_selected(self, idx: int) -> None:
        """A box was clicked in the 3D view — make it the active one."""
        self.param_panel.select_box(idx)   # emits boxSelectionChanged → _apply_measure

    def _log_reading(self) -> None:
        """Snapshot the current per-pin heights into the Repeatability tab as one
        capture. The height logged is the TRIMMED box height (``h_span_t``) — the
        repeatability-grade number — measured against the current cloud and stored in
        canonical metres so a mm⇄m switch just re-labels the table."""
        rv = self.viewer.repeat_view
        if not self.param_panel.measure_on:
            self._set_status("Turn the Volume box on and put a box on each pin first.")
            return
        if not self._has_points(self.cloud):
            self._set_status("No cloud to log yet — run a pair to build one.")
            return
        specs = self.param_panel.box_specs()
        if not specs:
            self._set_status("No measure boxes to log.")
            return
        inv = 1.0 / UNIT_PER_M[self._units]              # current unit -> metres
        vals = {}
        for name, box, trim in specs:
            m = measure_box(self.cloud.points, box, trim_pct=trim)
            vals[name] = m["h_span_t"] * inv if m is not None else None
        rv.add_record(self._capture_label(), vals)
        got = sum(v is not None for v in vals.values())
        self._set_status(f"Logged reading {rv.count()} · {got}/{len(vals)} pins caught points")

    def _capture_label(self) -> str:
        """A name for the logged capture — the left image's filename, else a counter."""
        p = self.input_panel.left_path
        if p:
            return os.path.splitext(os.path.basename(p))[0]
        return f"capture {self.viewer.repeat_view.count() + 1}"

    # ------------------------------------------------------ level to plane
    def _all_clouds(self) -> list:
        """Every distinct CloudResult the session holds — the compare cache plus the
        shown one (identity check, not ``in``: dataclass __eq__ compares arrays)."""
        clouds = [c for c in self.clouds.values() if c is not None]
        if self.cloud is not None and not any(c is self.cloud for c in clouds):
            clouds.append(self.cloud)
        return clouds

    def _ingest_level(self, cloud):
        """Record the raw points ON the cloud and apply the active level rotation, so
        every cloud (single run, compare, or batch) reconstructs in the levelled
        frame. With level off, raw_points is the same array as points (no copy)."""
        if self._has_points(cloud):
            cloud.raw_points = cloud.points
            if self._level_R is not None:
                cloud.points = self._apply_level(cloud.points)
        return cloud

    def _apply_level(self, points):
        """Rotate points into the levelled frame about the stored centre. The centre
        is canonical metres; scale it to the display unit the points are in."""
        if self._level_R is None:
            return points
        c = np.asarray(self._level_c_m, np.float64) * UNIT_PER_M[self._units]
        return (np.asarray(points) - c) @ self._level_R.T + c

    def _relevel_current(self) -> None:
        """Re-derive EVERY cached cloud from its own raw points under the current
        level state, then re-show. All clouds, not just the shown one: the overlay
        and the multi-target measure read the compare caches directly, so leaving
        them in the old frame mixed levelled and unlevelled points in one readout."""
        for c in self._all_clouds():
            raw = getattr(c, "raw_points", None)
            if raw is not None:
                c.points = self._apply_level(raw)   # level off -> returns raw itself
        if not self._has_points(self.cloud):
            return
        if self._overlay_on:
            self._show_overlay()        # re-stacks the re-levelled caches (+ measures)
        else:
            self.viewer.show_cloud(self.cloud, reset_view=True)
            self._apply_measure()
        self._reset_analyze_overlay()   # picks were in the pre-level frame
        self._reapply_deviation()       # heatmap must reference the re-levelled plane

    def _level_state(self) -> dict:
        if self._level_R is None:
            return {}
        return {"R": self._level_R.tolist(), "c": np.asarray(self._level_c_m).tolist()}

    def _on_level_toggled(self, on: bool) -> None:
        """Level button: fit the board plane, rotate the cloud (and the boxes) so the
        board is flat, and remember the rotation for every subsequent cloud."""
        if on:
            raw = getattr(self.cloud, "raw_points", None) if self.cloud is not None else None
            if raw is None or len(raw) < 500:
                self._set_status("No cloud to level yet — run a pair first.")
                self.param_panel.set_level_checked(False)
                return
            n, c = fit_plane(raw)
            if n[2] > 0:                    # point the board normal toward the camera (−Z)
                n = -n
            R = rotation_to_axis(n, (0.0, 0.0, -1.0))
            tilt = float(np.degrees(np.arccos(np.clip(-n[2], -1.0, 1.0))))
            self._level_c_m = np.asarray(c, np.float64) / UNIT_PER_M[self._units]
            self._level_R = R
            self.param_panel.transform_boxes(R, self._level_c_m, inverse=False)
            self._relevel_current()
            self._set_status(f"Levelled to the board plane — removed {tilt:.{ANGLE_DECIMALS}f}° of tilt.")
        else:
            if self._level_R is not None:
                self.param_panel.transform_boxes(self._level_R, self._level_c_m, inverse=True)
            self._level_R = self._level_c_m = None
            self._relevel_current()
            self._set_status("Levelling off — showing the raw camera-frame cloud.")

    # ------------------------------------------------------ analyze tools
    def _board_plane(self):
        """(normal, centroid) reference plane for the analyze tools — the levelling
        plane if levelling is on (board normal = −Z), else a fresh fit."""
        if self._level_R is not None:
            c = np.asarray(self._level_c_m, np.float64) * UNIT_PER_M[self._units]
            return np.array([0.0, 0.0, -1.0]), c
        return board_plane(self.cloud.points)

    def _on_analyze_tool(self, tool: str) -> None:
        self._analyze_tool = tool or ""
        self._picked = []
        cv = self.viewer.cloud_view
        cv.set_analyze_tool(tool or None)
        cv.clear_analyze()
        self.param_panel.set_profile(None, None)
        self.param_panel.set_analyze_out(
            "Click two points on the cloud." if tool in ("profile", "distance", "region")
            else "Click a point on the cloud." if tool == "point" else "")
        self._apply_measure()   # re-push box editability: freeze the gizmo while armed

    def _reset_analyze_overlay(self) -> None:
        """Drop the picked points, the 3D overlay, the profile plot and the readout —
        called on ANY change to the cloud's frame/scale (unit, level) or identity
        (new pair, model switch/blink), so a stale pick can't pair with a fresh one in
        a mismatched frame and old markers can't float over a new cloud."""
        self._picked = []
        self._last_region = None        # its points are gone — can't zero from it anymore
        self.param_panel.set_flat_ref_available(False)   # (an APPLIED ref stays removable)
        self._analyze_last = None       # nothing shown in the card now
        self.viewer.cloud_view.clear_analyze()
        self.param_panel.set_profile(None, None)
        self.param_panel.set_analyze_out("")

    def _reapply_deviation(self) -> None:
        """Re-paint the deviation heatmap after a cloud repaint. It's pushed as the
        cloud's colours, so every photo repaint (rebuild, level, unit, blink) wipes
        it; re-applying here keeps it live instead of silently reverting."""
        if self._overlay_on:
            return   # recoloring here would tint the whole overlay by one model's plane
        if self._dev_on and self._has_points(self.cloud):
            n, c = self._board_plane()
            d, rng = deviation(self.cloud.points, n, c)
            # colors-only push: the full set_cloud re-serialized the entire cloud
            # (n×15 bytes + JS geometry rebuild) TWICE per repaint — once for the
            # photo repaint, once more just to change these colors
            self.viewer.cloud_view.set_cloud_colors(self._turbo(d, -rng, rng))

    def _on_point_picked(self, x: float, y: float, z: float) -> None:
        if not self._analyze_tool or not self._has_points(self.cloud):
            return
        need = 1 if self._analyze_tool == "point" else 2
        p = np.array([x, y, z], np.float64)
        self._picked = [p] if len(self._picked) >= need else self._picked + [p]
        self.viewer.cloud_view.set_analyze_geom(
            markers=[list(q) for q in self._picked], line=None)
        if len(self._picked) >= need:
            self._compute_analyze()

    def _compute_analyze(self) -> None:
        # no up-front float64 copy of the whole cloud: point/distance never touch
        # it (the copy was ~2× cloud memory per click for nothing), and profile/
        # region convert internally
        pts = self.cloud.points
        n, c = self._board_plane()
        u, dec = self._units, UNIT_DECIMALS.get(self._units, 2)
        off = self._z_off()                       # flat-reference correction (0 if none)
        zed = "  (zeroed)" if off else ""
        cv, tool = self.viewer.cloud_view, self._analyze_tool
        self._analyze_last = tool                 # remember what's shown (to re-run on offset/rebuild)
        try:
            if tool == "point":
                P = self._picked[0]
                self.param_panel.set_analyze_result(
                    "Point · height", f"{float((P - c) @ n) - off:.{dec}f}", u,
                    rows=[("x", f"{P[0]:.{dec}f} {u}"),
                          ("y", f"{P[1]:.{dec}f} {u}"),
                          ("z", f"{P[2]:.{dec}f} {u}")],
                    caption="height above the board plane" + zed)
            elif tool == "distance":
                A, B = self._picked
                d = point_distance(A, B)
                cv.set_analyze_geom(markers=[list(A), list(B)], line=[list(A), list(B)])
                self.param_panel.set_analyze_result(
                    "Distance", f"{d['dist']:.{dec}f}", u,
                    rows=[("Δx", f"{d['dx']:+.{dec}f} {u}"),
                          ("Δy", f"{d['dy']:+.{dec}f} {u}"),
                          ("Δz", f"{d['dz']:+.{dec}f} {u}")])
            elif tool == "profile":
                A, B = self._picked
                r = surface_profile(pts, A, B, n, c, isolate=self._analyze_isolate)
                if r is None:
                    self.param_panel.set_analyze_out("No surface between those points — pick two on the part.")
                    return
                cv.set_analyze_geom(markers=[list(A), list(B)],
                                    line=[list(q) for q in r["poly"]])
                self._highlight_used(r.get("used"))
                self.param_panel.set_profile(r["t"], r["h"])
                self.param_panel.set_analyze_result(
                    "Surface angle", f"{r['angle']:+.{ANGLE_DECIMALS}f}°", "",
                    rows=[("rise", f"{r['d_height']:+.{dec}f} {u}"),
                          ("distance", f"{r['dist']:.{dec}f} {u}"),
                          ("samples", f"{r['n_pts']:,}")],
                    caption="slope vs the board plane")
            elif tool == "region":
                A, B = self._picked
                r = region_flatness(pts, A, B, n, c, isolate=self._analyze_isolate)
                if r is None:
                    self.param_panel.set_analyze_out("Empty region — pick two corners over the board.")
                    return
                cv.set_analyze_geom(markers=[list(A), list(B)], line=r["corners"])
                self._highlight_used(r.get("used"))
                self._last_region = r          # the raw result — what a flat-reference zeroes from
                self.param_panel.set_flat_ref_available(True)   # now there IS something to zero to
                self.param_panel.set_analyze_result(
                    "Region flatness", f"{r['rms']:.{dec}f}", u,
                    rows=[("max − min", f"{r['z_range']:.{dec}f} {u}"),
                          ("avg Z", f"{r['z_mean'] - off:+.{dec}f} {u}"),
                          ("local tilt", f"{r['local_tilt']:.{ANGLE_DECIMALS}f}°"),
                          ("size u", f"{r['size_u']:.{dec}f} {u}"),
                          ("size v", f"{r['size_v']:.{dec}f} {u}"),
                          ("points", f"{r['n_pts']:,}")],
                    caption="RMS vs patch plane · Z above board" + zed)
        except Exception as exc:   # noqa: BLE001 — analysis must never crash the UI
            self.param_panel.set_analyze_out(f"couldn't measure: {exc}")

    def _z_off(self) -> float:
        """The active flat-reference offset in the DISPLAY unit (0 if no reference)."""
        if self._z_offset_m is None:
            return 0.0
        return float(self._z_offset_m) * UNIT_PER_M[self._units]

    def _on_flat_ref(self, on: bool) -> None:
        """Zero board-referenced heights to the last flat Region (on), or clear (off).
        The offset is the region's average height (the cloud's systematic error at a zone
        that should read 0); it's stored in canonical metres so it survives a unit switch
        and applies to every cloud of the same fixture."""
        if on:
            r = self._last_region
            if r is None:
                self.param_panel.set_flat_ref_checked(False)
                self._set_status("Measure a Region on a flat zone first, then zero to it.")
                return
            self._z_offset_m = float(r["z_mean"]) / UNIT_PER_M[self._units]
            self._z_ref_pp_m = float(r["z_range"]) / UNIT_PER_M[self._units]
            self._set_status("Flat reference set — board-referenced heights are now corrected.")
        else:
            self._z_offset_m = None
            # un-applying may leave nothing to re-apply to (the region was reset)
            self.param_panel.set_flat_ref_available(self._last_region is not None)
        self._update_ref_label()
        self._refresh_analyze()       # re-run whatever's shown (incl. a pin) with/without the correction

    def _refresh_analyze(self) -> None:
        """Re-run whatever the analyze card currently shows against the current cloud +
        settings (flat-ref offset, isolate) — so the readout never goes stale after the
        offset toggles or the cloud is rebuilt live. No-op if the card shows nothing."""
        if self._analyze_last == "pin":
            self._on_pin_analyze()
        elif self._analyze_tool and self._picked and \
                len(self._picked) >= (1 if self._analyze_tool == "point" else 2):
            self._compute_analyze()

    def _update_ref_label(self) -> None:
        u, dec = self._units, UNIT_DECIMALS.get(self._units, 2)
        if self._z_offset_m is None:
            self.param_panel.set_flat_ref_text("")
            return
        off = self._z_offset_m * UNIT_PER_M[u]
        pp = self._z_ref_pp_m * UNIT_PER_M[u]
        self.param_panel.set_flat_ref_text(
            f"correcting {-off:+.{dec}f} {u}   ·   flatness ±{pp / 2:.{dec}f} {u}")

    def _highlight_used(self, used) -> None:
        """Light up the exact points a region/profile measured, so the user can confirm
        the right zone/level is used. Uniformly subsampled — a verification cue, not a
        full re-render — so the runJavaScript payload stays small."""
        if used is None or len(used) == 0:
            self.viewer.cloud_view.set_analyze_highlight(None)
            return
        used = np.asarray(used)
        cap = 5000
        if len(used) > cap:
            used = used[np.linspace(0, len(used) - 1, cap).astype(np.int64)]
        self.viewer.cloud_view.set_analyze_highlight(used)

    def _on_isolate_layer(self, on: bool) -> None:
        """Toggle 'measure only the picked Z level' — re-run the live region/profile."""
        self._analyze_isolate = bool(on)
        if self._analyze_tool in ("region", "profile") and len(self._picked) >= 2:
            self._compute_analyze()

    def _on_deviation(self, on: bool) -> None:
        if on and self._overlay_on:
            # the heatmap is one model's distance to ONE board plane — over a stack
            # of different models' points it's meaningless, and pushing it would
            # silently collapse the overlay to the single shown cloud
            self.param_panel.set_deviation_checked(False)
            self._set_status("Deviation heatmap shows a single model — untick Overlay first.")
            return
        self._dev_on = bool(on)
        if not self._has_points(self.cloud):
            return
        if on:
            n, c = self._board_plane()
            d, rng = deviation(self.cloud.points, n, c)
            # colors-only: toggling the heatmap re-tints the cloud on screen —
            # re-shipping every position for that was the single biggest
            # avoidable transfer in the app (~60 MB at 4M points)
            self.viewer.cloud_view.set_cloud_colors(self._turbo(d, -rng, rng))
            dec = UNIT_DECIMALS.get(self._units, 4)
            self._set_status(f"Deviation heatmap — ±{rng:.{dec}f} {self._units} about the board plane.")
        else:
            self.viewer.cloud_view.set_cloud_colors(self.cloud.colors)

    @staticmethod
    def _turbo(vals, lo, hi):
        import cv2
        t = np.clip((np.asarray(vals) - lo) / (hi - lo + 1e-12), 0.0, 1.0)
        lut = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(-1, 1),
                                cv2.COLORMAP_TURBO).reshape(-1, 3)[:, ::-1]   # BGR→RGB
        return np.ascontiguousarray(lut[(t * 255).astype(np.uint8)], np.uint8)

    def _on_pin_analyze(self) -> None:
        if not self.param_panel.measure_on or not self._has_points(self.cloud):
            self._set_status("Turn on the Volume box and select a pin box first.")
            return
        specs = self.param_panel.box_specs()
        sel = self.param_panel.selected_index()
        if not specs or not (0 <= sel < len(specs)):
            self._set_status("Select a measure box on a pin.")
            return
        _name, box, _trim = specs[sel]
        mask = points_in_box(self.cloud.points, box)
        n, c = self._board_plane()
        r = pin_analysis(np.asarray(self.cloud.points)[mask], n, c)
        u, dec = self._units, UNIT_DECIMALS.get(self._units, 2)
        if r is None:
            self.param_panel.set_analyze_out("Pin box too sparse — place it tighter on the pin.")
            return
        off = self._z_off()            # flat-reference correction (0 if none)
        self._analyze_last = "pin"     # remember (so an offset toggle / rebuild re-runs it)
        vert = f"{r['verticality']:.{ANGLE_DECIMALS}f}°" if r["verticality"] is not None else "—"
        self.param_panel.set_analyze_result(
            "Pin height", f"{r['height'] - off:.{dec}f}", u,
            rows=[("verticality", vert), ("points", f"{r['n_pts']:,}")],
            caption="above the board plane" + ("  (zeroed)" if off else ""))

    def _default_box(self):
        """A new box: the current box's size, tilt-free, centred ON the cloud —
        on the actual point nearest the componentwise median. The bare median is
        a point in EMPTY SPACE on deep or sparse scenes (median x/y/z need not
        lie on any surface), and a box centred there caught nothing."""
        ref = self.param_panel.measure_box()
        if self._has_points(self.cloud):
            pts = np.asarray(self.cloud.points)
            sub = pts[::max(1, len(pts) // 200_000)]     # nearest-point snap, kept cheap
            med = np.median(sub, axis=0)
            near = sub[int(np.argmin(((sub - med) ** 2).sum(1)))]
            cx, cy, cz = float(near[0]), float(near[1]), float(near[2])
        else:
            cx, cy, cz = ref.cx, ref.cy, ref.cz
        return MeasureBox(cx=cx, cy=cy, cz=cz, sx=ref.sx, sy=ref.sy, sz=ref.sz,
                          qx=0.0, qy=0.0, qz=0.0, qw=1.0)

    def _on_add_box(self) -> None:
        """+ Add (or the first box when the tool is switched on): drop a box on the
        middle of the cloud and select it."""
        self._box_placed = True
        sw = self.param_panel.measure_sw
        if not sw.isChecked():        # make sure the tool is on so the box shows
            sw.blockSignals(True)
            sw.setChecked(True)
            sw.blockSignals(False)
        self.param_panel.append_box(self._default_box(), self.param_panel.measure_opts())

    def _save_boxes(self) -> None:
        """Persist the box set the instant it changes, so a hand-built pin layout
        survives even an unclean exit."""
        try:
            self.settings.setValue("box_presets", json.dumps(self.param_panel.boxes_blob()))
        except Exception:   # noqa: BLE001 — a settings write must never break the UI
            pass

    def _measure_text(self, stats: list, trim: float) -> str:
        """The box readout. One model gets the full picture; several (an overlay)
        get a line each, because three multi-line blocks is not something anyone reads.

        The headline number is HEIGHT along the box's own axis (``h_*``) — the pin
        height once the box is aligned to the pin — with the raw/trimmed pair so a
        noisy box shows itself. Depth stays in world z as a sanity anchor."""
        u = self._units
        dec = UNIT_DECIMALS.get(u, 2)
        if len(stats) == 1:
            name, m = stats[0]
            head = "▣ Volume box" + (f"  ·  {name}" if name else "")
            if m is None:
                return f"{head}\n   empty — no points inside it"
            lines = [
                head,
                f"   {m['n']:,} pts",
                f"   height    {m['h_span']:.{dec}f} {u}"
                f"   ·  trim {trim:g}% {m['h_span_t']:.{dec}f} {u}   (along box axis)",
                f"   section   {m['sec_x']:.{dec}f} × {m['sec_y']:.{dec}f} {u}",
                f"   depth     {m['z_min']:.{dec}f} → {m['z_max']:.{dec}f} {u}   (world)",
            ]
            return "\n".join(lines)
        head = f"▣ Volume box  ·  height (box axis) · trim {trim:g}%"
        width = max(len(n) for n, _ in stats)
        lines = [head]
        for name, m in stats:
            if m is None:
                lines.append(f"   {name:<{width}}   empty")
                continue
            lines.append(f"   {name:<{width}}   {m['n']:>7,} pts"
                         f"   height {m['h_span_t']:.{dec}f} {u}")
        return "\n".join(lines)

    def _on_measure_toggled(self, on: bool) -> None:
        """Switching the tool on with no boxes yet: drop a first one on the cloud, so
        it doesn't read as 'the toggle does nothing' (a box defaults to the origin =
        inside the camera)."""
        if on and not self.param_panel.has_boxes():
            self._on_add_box()   # adds a box at the cloud centre + selects it

    def _on_pixel_clicked(self, x: int, y: int) -> None:
        """Clicking the Disparity/Depth map drops the measure box on that spot.

        Only while the box is switched on — otherwise a stray click in the map
        would move a box the user isn't thinking about.

        Back-projected with result.K and result.depth, which are BOTH at the
        working scale (infer.py builds K as params.intrinsics(scale) and the depth
        map alongside it). Reaching for the calibration panel's raw fx/cx instead
        would place the box at twice its offset from the optical axis at the 0.50×
        default — close enough to the cloud to look plausible, and wrong.
        """
        if not self.param_panel.measure_on or self.result is None or self._batching:
            return
        r = self.result
        if r.depth is None or r.K is None:
            self._set_status("No depth — set calibration before placing the box.")
            return
        if not (0 <= y < r.depth.shape[0] and 0 <= x < r.depth.shape[1]):
            return
        z = float(r.depth[y, x])
        if z <= 0:
            self._set_status("No depth at that pixel — pick one the model matched.")
            return
        K = r.K
        bx = (x - float(K[0, 2])) * z / float(K[0, 0])
        by = (y - float(K[1, 2])) * z / float(K[1, 1])
        self._box_placed = True
        self.param_panel.set_box_center(bx, by, z)
        dec = UNIT_DECIMALS.get(self._units, 2)
        self._set_status(f"Box centred on pixel ({x}, {y}) — depth "
                         f"{z:.{dec}f} {self._units}.")

    def _show_model(self, key: str) -> None:
        """Blink to a cached model's result: same zoom, same colour levels, so
        what changes on screen is only what the models actually disagree about."""
        r = self.results.get(key)
        if r is None:
            return
        # Asking for ONE model is asking to leave the overlay. Untick without
        # re-emitting: the toggle's own handler calls straight back into here.
        self._overlay_on = False
        self.viewer.cloud_view.set_overlay_checked(False)
        self._shown_model = key
        self.result = r
        self.cloud = self.clouds.get(key)
        self.viewer.disp_view.set_image_blink(r.disp)
        # Clear, don't skip, when the target has none: guarding these on "is not
        # None" left the PREVIOUS model's depth and 3D cloud on screen under the
        # new model's name — the one thing a blink comparator must never do.
        if r.depth is not None:
            self.viewer.depth_view.set_image_blink(r.depth)
        else:
            self.viewer.depth_view.clear()
        self.viewer.show_cloud(self.cloud, reset_view=False)   # show_cloud(None) clears
        for bar in self._model_bars():
            bar.set_current(key)
        self.viewer.compare_view.set_shown(key)
        self.points_lbl.setText(f"{self.cloud.n:,} pts" if self.cloud else "")
        self.timing_lbl.setText(f"{r.timing.get('net_s', 0):.2f} s  ·  {r.W}×{r.H}")
        self.export_btn.setEnabled(True)
        self._apply_measure()   # same box, different points — re-measure and redraw
        self._reset_analyze_overlay()   # picks belonged to the previous model's cloud
        self._reapply_deviation()       # re-tint the blinked-in cloud if the heatmap's on
        self._update_compare_strip()

    def _captionable_tab(self) -> bool:
        """Is the current tab actually showing a model's result?

        The strip is pinned under the whole tab stack, so without this it followed
        you onto tabs with nothing to caption: onto Input, labelling your two source
        photos with some model's stats, and onto Compare, where it restated one
        column of the very table you were reading — while naming the model you'd
        pressed Show on, not the one the status bar said you were editing. Two lines
        answering different questions with nothing above them to tell them apart is
        just the app appearing to contradict itself.
        """
        return self.viewer.currentWidget() in (
            self.viewer.disp_view, self.viewer.depth_view, self.viewer.cloud_view)

    def _update_compare_strip(self) -> None:
        key = self._shown_model
        if key is None or key not in self.mstats or not self._captionable_tab():
            self.compare_lbl.hide()
            return
        s = self.mstats[key]
        dec = UNIT_DECIMALS.get(self._units, 2)
        # "Showing" because the status bar directly below says "Editing <other
        # model>", and both lines lead with a model name: say which question this
        # one answers.
        parts = [f"Showing  {get_spec(key).display_name}", f"{s['net_s']:.2f} s"]
        if s["peak_gb"]:
            parts.append(f"{s['peak_gb']:.1f} GB peak")
        parts.append(f"{s['valid_pct']:.1f}% valid")
        if s["med_depth"] is not None:
            parts.append(f"depth {s['med_depth']:.{dec}f} {self._units}")
        if s["plane_rms"] is not None:
            parts.append(f"plane σ {s['plane_rms']:.{dec}f} {self._units}")
        if s["n_pts"]:
            parts.append(f"{s['n_pts']:,} pts")
        self.compare_lbl.setText("   ·   ".join(parts))
        self.compare_lbl.setToolTip(
            "Stats for the model shown above — every model ran on the same pair, "
            "scale and calibration, each with the settings in its Compare-tab "
            "column.\n\nplane σ = " + PLANE_TIP)
        self.compare_lbl.show()

    def _discard_previous_model_output(self) -> None:
        """A different model/checkpoint is coming — throw away the outgoing one's
        disparity, depth and 3D cloud rather than leaving them on screen.

        They were only marked 'stale' before, which meant model B sat there
        showing model A's cloud until you pressed Run. Also bump the pair version:
        it is what `_on_inference`/`_on_cloud` gate on, so anything still in
        flight from the old engine is dropped instead of repainting the views we
        just cleared. (The fresh child starts with no cached result of its own, so
        live cloud rebuilds can't resurrect A's data either.)"""
        self._pair_version += 1
        self._clear_results()
        if self.input_panel.left_rgb is not None:
            self.viewer.setCurrentWidget(self.viewer.input_view)

    def _switch_engine(self, python_exe, key: str, ckpt: str) -> None:
        """Replace the engine child. The ONE place that does, so the fact that a
        fresh child starts with no cached result is recorded in exactly one spot."""
        self._child_result_key = None
        self.worker.switchBackend(python_exe, key, ckpt)

    def _show_load_overlay(self, text: str) -> None:
        self._model_ready = False
        self._update_run_enabled()
        self.overlay.setText(text)
        self.overlay.setGeometry(self.centralWidget().rect())
        self.overlay.show()
        self.overlay.raise_()

    def _on_model_changed(self, key: str) -> None:
        """User picked a different model — rebuild its parameter panel and reload
        the engine. A fresh child guarantees clean VRAM and the right interpreter
        (a per-model venv). The shown result is left in place but marked stale."""
        spec = get_spec(key)
        ckpt = self.input_panel.current_checkpoint_path()
        name = spec.display_name if spec else key
        if not ckpt:
            # Revert to the model the PANEL is on, and DON'T rebuild it — otherwise
            # the dropdown would advertise a model with no weights while the panel
            # (and therefore Run) still uses the old one.
            #
            # The panel, not _loaded_backend_key: nothing loads at startup, so that
            # stays None until your first Run — reverting to it did nothing at all
            # and left the dropdown sitting on the weightless model it had just
            # refused, with the knobs still showing a different one.
            self._set_status(f"{name}: no weights found — kept the current model.")
            QMessageBox.warning(
                self, "No weights",
                f"{name} has no checkpoint available. Download its weights into "
                "the model's folder, then select it again.")
            back = self.param_panel.current_key or DEFAULT_BACKEND
            self.input_panel.restore_selection(back, self.input_panel.saved_ckpt(back))
            return
        self.param_panel.set_backend(spec)   # restores THIS model's saved settings
        # The dropdown can now LEAD the engine (Compare's "Edit settings" re-points
        # it without loading), so selecting the model the engine is ALREADY holding
        # is a real index change — and would trigger a pointless reload of up to
        # 3 GB. Just re-point the panel instead.
        if (self._model_ready and key == self._loaded_backend_key
                and ckpt == self._loaded_ckpt):
            self._refresh_compare_cards()
            self._sync_run_button()
            self._set_status(f"{name} — already loaded.")
            return
        self._loaded_backend_key, self._loaded_ckpt = key, ckpt
        self._refresh_compare_cards()
        self._sync_run_button()
        self._show_load_overlay(self._load_overlay_text(spec, ckpt, "switching model · fresh engine"))
        self._set_status(f"Switching to {name}…")
        self._discard_previous_model_output()
        python_exe = spec.python_exe if spec else None
        self._switch_engine(python_exe, key, ckpt)

    def _on_checkpoint_changed(self) -> None:
        """User picked a different checkpoint — reload weights in a fresh engine.

        This used to assume the model was unchanged ("a different checkpoint for
        the SAME model") and only updated _loaded_ckpt. That assumption died with
        "Edit settings", which re-points the dropdown at another model without
        loading: picking a checkpoint then loads THAT model while
        _loaded_backend_key still named the old one — so Run saw sel == loaded,
        skipped the reload, and handed this model's knobs to the resident one.
        Both FoundationStereo and Fast-FS declare valid_iters/hierarchical/
        low_memory, so the foreign VALUES applied (FS's 32 iters to a model
        trained for 8) — no crash, no clue, just a wrong result.
        """
        key = self.input_panel.current_backend_key()
        spec = get_spec(key)
        ckpt = self.input_panel.current_checkpoint_path()
        if not ckpt:
            return
        self._loaded_backend_key, self._loaded_ckpt = key, ckpt
        self._refresh_compare_cards()
        self._sync_run_button()
        self._show_load_overlay(self._load_overlay_text(spec, ckpt, "switching checkpoint"))
        self._set_status("Loading checkpoint…")
        self._discard_previous_model_output()   # different weights = different output
        python_exe = spec.python_exe if spec else None
        self._switch_engine(python_exe, key, ckpt)

    def _maybe_autodemo(self) -> None:
        """FS_STUDIO_DEMO=1 → auto-load the bundled NVIDIA demo pair and Run it.

        The demo images ship with FoundationStereo in ``assets/`` alongside
        ``K.txt``, which the InputPanel auto-loads, so depth + 3D cloud work too.
        """
        if not self._autodemo:
            return
        assets = os.path.join(REPO_ROOT, "assets")
        left, right = os.path.join(assets, "left.png"), os.path.join(assets, "right.png")
        if not (os.path.isfile(left) and os.path.isfile(right)):
            self._set_status("Demo images not found in assets/.")
            return
        self.input_panel.load_image(left, "left")
        self.input_panel.load_image(right, "right")
        self._set_status("NVIDIA demo pair loaded — running…")
        QTimer.singleShot(150, self._run)

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        if self.overlay.isVisible():
            self.overlay.setGeometry(self.centralWidget().rect())

    def _on_inference(self, result) -> None:
        if self._pair_version != self._run_pair_version:
            return   # result belongs to a pair that has since been replaced — drop it
        if self._batching and self._batch_kind == "pairs":
            # disparity/depth land first; keep them for the views but wait for the
            # cloud (next reply) to measure. focus=False so the tab stays on
            # Repeatability, where the user is watching the table fill.
            self.result = result
            self.viewer.show_result(result, focus=False)
            return
        if self._comparing and self._compare_queue:
            self.results[self._compare_queue[0]] = result   # cache, and show it as it lands
            self._child_result_key = self._compare_queue[0]
        else:
            self._child_result_key = self._loaded_backend_key
        self.result = result
        self._clear_stale()             # result is now current (clear on success, not dispatch)
        self._reset_cloud_view = True   # a new inference reframes the 3D view
        # don't steal the tab mid-comparison — the user is watching the columns fill
        self.viewer.show_result(result, focus=not self._comparing)
        t = result.timing.get("net_s", 0)
        self.timing_lbl.setText(f"{t:.2f} s  ·  {result.W}×{result.H}")
        self.export_btn.setEnabled(True)
        if self._autodemo:
            self.viewer.setCurrentWidget(self.viewer.disp_view)

    def _on_cloud(self, cloud) -> None:
        if self._pair_version != self._run_pair_version:
            return   # stale cloud from a superseded pair (busy already cleared by worker)
        cloud = self._ingest_level(cloud)   # keep raw + apply the level rotation (if any)
        if self._batching and self._batch_kind == "pairs":
            self._batch_on_cloud(cloud)
            return
        if self._overlay_queue:
            key = self._overlay_queue.pop(0)
            self.clouds[key] = cloud
            r = self.results.get(key)
            if r is not None:
                self.mstats[key] = self._model_stats(r, cloud)
                self.viewer.compare_view.show_stats(key, self.mstats[key])
            if key == self._shown_model:
                self.cloud = cloud          # keep the single-model view in step too
            QTimer.singleShot(0, self._overlay_next)
            return
        if self._comparing and self._compare_queue:
            key = self._compare_queue.pop(0)      # this model is done — record + advance
            self.clouds[key] = cloud
            r = self.results.get(key)
            if r is not None:
                self.mstats[key] = self._model_stats(r, cloud)
                # fill this model's column now, so the comparison reads as it runs
                self.viewer.compare_view.show_stats(key, self.mstats[key])
                self.viewer.compare_view.set_status(key, "done")
            self.cloud = cloud
            self.viewer.show_cloud(cloud, reset_view=self._reset_cloud_view)
            self._reset_cloud_view = False
            self.points_lbl.setText(f"{cloud.n:,} pts" if cloud else "")
            self._apply_measure()
            self._reapply_deviation()
            if self._cloud_pending:   # a cloud param was nudged mid-sweep
                self._cloud_pending = False
                self._cloud_timer.start()
            QTimer.singleShot(0, self._compare_next)
            return
        self.cloud = cloud
        fresh = self._reset_cloud_view          # a new inference (Run/Re-run) — new points
        self.viewer.show_cloud(cloud, reset_view=self._reset_cloud_view)
        self._reset_cloud_view = False
        self.points_lbl.setText(f"{cloud.n:,} pts" if cloud else "")
        if (cloud is not None and cloud.n == 0 and self.result is not None
                and self.result.depth is not None and (self.result.depth > 0).any()):
            # Depth exists but the z-clip ate every point — on the PCB-tuned
            # default z-far (250 mm) ANY metre-scale scene lands here, showing
            # an inexplicably empty 3D tab with a cheerful "Done." Say why.
            self._set_status(
                f"Cloud is empty — every point fell outside the z-near/z-far range "
                f"(z-far {self.param_panel.z_far.value():g} {self._units}). Raise "
                "z-far in the Point cloud section; the cloud rebuilds live.")
        self._apply_measure()
        if fresh:
            self._reset_analyze_overlay()       # picks belonged to the old reconstruction
        else:
            self._refresh_analyze()             # live param rebuild — re-measure on the new points
        self._reapply_deviation()               # a live rebuild repaint wipes the heatmap
        if self._shown_model is not None:
            # keep the comparison cache in step with a live rebuild, or flipping
            # away and back would resurrect the pre-tweak cloud
            self.clouds[self._shown_model] = cloud
            if self.result is not None:
                self.mstats[self._shown_model] = self._model_stats(self.result, cloud)
                # …and the Compare column too: it keeps its OWN copy of the stats,
                # so updating only mstats left the column disagreeing with the strip
                self.viewer.compare_view.show_stats(
                    self._shown_model, self.mstats[self._shown_model])
            self._update_compare_strip()
        if self._cloud_pending:   # a param changed mid-rebuild — apply it now
            self._cloud_pending = False
            self._cloud_timer.start()

    def _report_error(self, msg: str) -> None:
        """Show an error that did NOT come from the engine (export, file I/O).

        Deliberately separate from _on_error: that one is the worker's error slot
        and advances the comparison state machine, so routing an export failure
        through it popped the queue and banked whichever model was mid-run as
        'failed' — attributing a disk error to a model and tearing down its child.
        """
        self._set_status(f"Error: {str(msg).splitlines()[0][:120]}")
        QMessageBox.critical(self, "FoundationStereo Studio", msg)

    def _on_error(self, msg: str) -> None:
        # A failed load/switch left the loading scrim up: without this the window
        # sits behind a permanent "◆ Loading …" forever (Run stays disabled too),
        # which reads as a frozen app rather than a failure.
        self.overlay.hide()
        # a load that failed must not silently fire the run it was queued for
        self._pending_run = False
        if self._batching and self._batch_kind == "pairs":
            if not self.worker.alive:
                # the engine crashed (usually OOM). Every later pair would fail the
                # same way — stop the whole batch rather than bank 900 as failed.
                self._finish_batch(aborted="the engine stopped (likely out of GPU memory)")
                return
            self._batch_failed.append((self._batch_label, str(msg).splitlines()[0][:200]))
            self._batch_done += 1
            self._update_batch_progress()
            QTimer.singleShot(0, self._batch_next)
            return
        if self._overlay_queue:
            # An overlay rebuild leg failed. Without this the queue never popped, so
            # the overlay wedged AND every later arriving cloud was mis-attributed to
            # the queue head (the overlay branch of _on_cloud claims whatever lands).
            # Drop the whole rebuild; the caches keep their previous clouds.
            self._overlay_queue.clear()
            if self._cloud_pending:        # a param edit was waiting on this rebuild
                self._cloud_pending = False
                self._cloud_timer.start()
            self._set_status("Overlay rebuild failed — showing the previous clouds.")
            QMessageBox.critical(self, "FoundationStereo Studio", msg)
            return
        if self._comparing and self._compare_queue:
            # one model failing (e.g. it OOMs at this scale) must not abandon the
            # whole comparison — bank the reason, carry on, summarise at the end
            key = self._compare_queue.pop(0)
            self._compare_failed[key] = str(msg).split("\n\n")[0][:300]
            # The child sends ("inference", …) BEFORE building the cloud, so a cloud
            # that then dies leaves this model's result already cached. Drop it:
            # otherwise it is both a result and a failure — it earns a blink button
            # from `results` but has no cloud or stats behind it, so selecting it
            # shows the PREVIOUS model's 3D cloud under its name.
            self.results.pop(key, None)
            self.clouds.pop(key, None)
            self.mstats.pop(key, None)
            self.viewer.compare_view.set_status(key, "failed — see the summary at the end")
            QTimer.singleShot(0, self._compare_next)
            return
        if not self.worker.alive:
            # The engine child is GONE (crash/OOM), not merely reporting an error:
            # the next Run must LOAD again, not dispatch into a dead pipe. Without
            # this, _model_ready stayed True so _needs_load() said False, and every
            # Run raised "Lost connection" forever — recoverable only by switching
            # to a different model or restarting the app. _model_ready=False makes
            # Run read "Load & Run", and pressing it respawns a fresh child.
            self._model_ready = False
            self._sync_run_button()
            self._update_run_enabled()
        self._set_status("Error — model not loaded." if not self._model_ready else "Error.")
        QMessageBox.critical(self, "FoundationStereo Studio", msg)

    def _clear_results(self) -> None:
        """Drop every model-DERIVED artifact: disparity, depth, the 3D cloud, their
        status readouts, any queued rebuild, and Export. The Input tab is left
        alone — the loaded pair is the user's data, not model output.

        Shared by the two events that invalidate a result: the pair changing, and
        the model/checkpoint changing (model B must never be shown A's output)."""
        self.result = None
        self.cloud = None
        self.results.clear()          # the comparison cache is model output too
        self.clouds.clear()
        self.mstats.clear()
        self._shown_model = None
        # The overlay is a view of `clouds`, which we just emptied — leave it armed
        # and the next rebuild would try to stack models that no longer exist.
        self._overlay_on = False
        self._overlay_queue.clear()
        self.viewer.cloud_view.set_overlay_available(False)   # also unticks it
        self._reset_cloud_view = True
        self._cloud_pending = False
        self._cloud_timer.stop()      # a debounced rebuild must not fire post-clear
        self.viewer.disp_view.clear()
        self.viewer.depth_view.clear()
        self.viewer.cloud_view.clear()   # each clear() also drops its Model bar
        self.viewer.compare_view.clear_results()   # …and the Compare columns' numbers
        self.export_btn.setEnabled(False)
        self._clear_stale()           # nothing shown -> "Run", not "Re-run"
        self.points_lbl.setText("")
        self.timing_lbl.setText("")
        self.probe_lbl.setText("")
        self._reset_analyze_overlay()   # picks/overlay belonged to the cleared cloud
        self._update_compare_strip()
        self._sync_panel_gates()        # no cloud any more — Measure/Analyze gate off

    def _on_images(self) -> None:
        # a changed pair invalidates the previous result — clear it so stale
        # disparity/depth/cloud (and Export) can't be shown against a new pair
        self._pair_version += 1   # any in-flight run's results are now superseded
        # …but a comparison is a state machine driven by those very replies, so
        # superseding them silently would strand it. Unwind it explicitly.
        self._abort_compare("New pair loaded — the comparison was cancelled. "
                            "Press Run comparison to start it on this pair.")
        self._clear_results()
        self._reset_analyze_overlay()   # picks belong to the old pair's cloud
        if self.input_panel.left_rgb is not None:
            self.viewer.show_input(self.input_panel.left_rgb, self.input_panel.right_rgb)
            self.viewer.setCurrentWidget(self.viewer.input_view)
        self._update_run_enabled()
        self._set_status("Loaded — press Run." if self.input_panel.ready
                         else "Load both left and right images.")

    def _on_hover(self, x: int, y: int) -> None:
        if self.result is None:
            return
        parts = [f"px ({x}, {y})"]
        d = self.result.disp
        if 0 <= y < d.shape[0] and 0 <= x < d.shape[1]:
            dv = float(d[y, x])
            parts.append(f"disp {dv:.3f} px" if dv > 0 else "disp —")   # sub-pixel
            if self.result.depth is not None:
                zv = float(self.result.depth[y, x])
                dec = UNIT_DECIMALS.get(self._units, 2)
                parts.append(f"depth {zv:.{dec}f} {self._units}" if zv > 0 else "depth —")
        self.probe_lbl.setText("   ".join(parts))

    # ----------------------------------------------------------------- run
    def _current_params(self) -> StereoParams:
        return self.param_panel.build_params(self.input_panel.calibration())

    def _update_run_enabled(self) -> None:
        # "Run" is possible whenever the engine is free and a pair is loaded: if the
        # selected model isn't resident, Run loads it first (see _needs_load). This
        # deliberately does NOT require _model_ready — a load that FAILED leaves it
        # False, and gating on it left Run dead with "Waiting for the engine…" while
        # nothing was coming. Now pressing Run simply retries the load.
        free = (self.input_panel.ready and not self._busy and not self._comparing
                and not self._batching and not self._overlay_queue)
        self.run_btn.setEnabled(free and (self._model_ready or self._needs_load()))
        self._sync_run_button()   # every path that changes enabled-ness can change the LABEL
        self.viewer.compare_view.set_run_state(*self._compare_run_state(free))

    def _compare_run_state(self, free: bool) -> tuple:
        """(can the Compare tab run, and if not — why not). Saying the reason beats
        a dead button the user has to guess about.

        A sweep loads every model itself, so it needs the engine FREE, not ready."""
        cv = self.viewer.compare_view
        if self._comparing:
            return False, "Comparing…"
        if len(cv.selected_keys()) < 2:
            return False, "Tick at least two models."
        if not self.input_panel.ready:
            return False, "Load a stereo pair first."
        if not free:
            return False, "Waiting for the engine…"
        if not self.input_panel.has_calibration:
            # legitimate — you still get disparity, time and VRAM, just no metric
            # depth / cloud / plane σ. Say so rather than blocking the run.
            return True, "No calibration — disparity only (no depth, cloud or plane σ)."
        return True, ""

    def _run(self) -> None:
        if self._busy or self._comparing or self._batching or self._overlay_queue:
            return   # a run/overlay-rebuild is in flight — guards programmatic callers too
        if not self.input_panel.ready:
            QMessageBox.information(self, "Load a pair", "Load both left and right images first.")
            return
        # The panel may be showing a model the engine does NOT hold (you pressed
        # "Edit settings" on the Compare tab). Running now would hand this model's
        # knobs to a different model — whose adapter reads its own keys with
        # .get(default), so it would silently run at ITS defaults and look fine.
        # Load what's on screen first; _on_model_loaded resumes the run.
        #
        # `not _model_ready` is part of the same test on purpose: _loaded_backend_key
        # is set OPTIMISTICALLY at dispatch, so after a load that FAILED it names a
        # model the engine does not actually have. Trusting it there would dispatch
        # inference at an engine holding nothing. The Run button is disabled in that
        # state, but a guard that relies on a widget being greyed out is not a
        # guarantee — this makes a forced/programmatic call re-load instead.
        sel = self.input_panel.current_backend_key()
        if self._needs_load():
            spec = get_spec(sel)
            ckpt = self.input_panel.current_checkpoint_path()
            if not ckpt:
                QMessageBox.warning(self, "No weights",
                                    f"{spec.display_name if spec else sel} has no checkpoint.")
                return
            self._pending_run = True
            self._loaded_backend_key, self._loaded_ckpt = sel, ckpt
            self._show_load_overlay(self._load_overlay_text(spec, ckpt, "loading to run"))
            self._set_status(f"Loading {spec.display_name if spec else sel} to run…")
            self._switch_engine(spec.python_exe if spec else None, sel, ckpt)
            return
        self._run_pair_version = self._pair_version   # tag which pair this run is for
        p = self._current_params()
        self.worker.runInference(self.input_panel.left_rgb, self.input_panel.right_rgb, p)

    # ---------------------------------------------------------- stale cue
    def _needs_load(self) -> bool:
        """Would running right now require loading something first?

        Compares the (model, checkpoint) PAIR, not just the model: those two can
        disagree independently, and a key-only test let a checkpoint switch on a
        previewed model leave the engine holding one model while the state said
        another. `not _model_ready` is in here too because _loaded_backend_key is
        set optimistically at dispatch, so after a failed load it names a model
        the engine does not actually have.
        """
        sel = self.input_panel.current_backend_key()
        if not sel:
            return False
        if not self._model_ready:
            return True
        return (sel != self._loaded_backend_key
                or self.input_panel.current_checkpoint_path() != self._loaded_ckpt)

    def _sync_run_button(self) -> None:
        """Run's label states what pressing it will actually DO.

        'Load & Run' when the settings panel is showing a model the engine isn't
        holding — a dead button with no explanation would be worse, and silently
        running the other model would be worse still."""
        sel = self.input_panel.current_backend_key()
        spec = get_spec(sel)
        name = spec.display_name if spec is not None else sel
        if self._needs_load():
            self.run_btn.setText("▶  Load & Run")
            self.run_btn.setToolTip(
                f"Load {name} and run it on this pair with the settings shown.\n\n"
                "The engine is holding a different model right now, so this loads "
                "first — that's the wait.")
        else:
            self.run_btn.setText("▶  Re-run" if self._stale else "▶  Run")
            self.run_btn.setToolTip(
                "Run the selected model on the loaded pair (Ctrl+Enter). Turns "
                "amber when a setting changed and you need to re-run.")

    def _mark_stale(self) -> None:
        """A 'Needs run' setting (inference param or calibration) changed after a
        result was produced — flag that the shown result is out of date."""
        if self.result is None or self._stale:
            return
        self._stale = True
        self._sync_run_button()
        self._set_stale_style(True)
        self._set_status("Settings changed — Run to apply.")

    def _clear_stale(self) -> None:
        if not self._stale:
            return
        self._stale = False
        self._sync_run_button()
        self._set_stale_style(False)

    def _set_stale_style(self, stale: bool) -> None:
        self.run_btn.setProperty("stale", True if stale else False)
        self.run_btn.style().unpolish(self.run_btn)
        self.run_btn.style().polish(self.run_btn)

    def _schedule_rebuild(self) -> None:
        """Debounce: a slider drag restarts this 250 ms timer, so many ticks
        collapse into a single rebuild once the user stops."""
        if self.result is None or self.result.depth is None:
            return
        self._cloud_timer.start()

    def _overlay_next(self) -> None:
        """Rebuild the overlay's clouds one at a time, then redraw it."""
        if not self._overlay_queue:
            self._show_overlay()
            if self._cloud_pending:
                # a cloud param was edited while the overlay rebuilt — without this
                # the flag sat consumed-by-nobody and the edit never applied
                self._cloud_pending = False
                self._cloud_timer.start()
            return
        key = self._overlay_queue[0]
        r = self.results.get(key)
        if r is None:                       # nothing to rebuild from — skip it
            self._overlay_queue.pop(0)
            QTimer.singleShot(0, self._overlay_next)
            return
        self._child_result_key = key        # the child adopts whatever we ship it
        self.worker.rebuildCloud(r, self._current_params())

    def _rebuild_cloud(self) -> None:
        if self.result is None or self.result.depth is None:
            return
        if self._busy or self._comparing or self._batching or self._overlay_queue:
            # A rebuild/inference is in flight, or another state machine (compare
            # leg, batch, overlay rebuild) owns the reply stream — remember to run
            # ONCE with the latest params instead of queueing many. Dispatching here
            # would land this rebuild's cloud in that machine's branch of _on_cloud
            # and be banked as ITS next reply (wrong model / bogus batch capture).
            self._cloud_pending = True
            return
        self._cloud_pending = False
        if self._overlay_on and not self._overlay_queue:
            # The cloud settings are shared, so they have to reach every model on
            # screen. Rebuilding just the "shown" one would drop the other models'
            # points back to the old z-range/denoise while the overlay claimed to be
            # comparing them at the same settings — and _on_cloud would have replaced
            # the whole overlay with that single model's cloud.
            self._overlay_queue = list(self._overlay_keys())
            self._overlay_next()
            return
        # Rebuilding from the child's own cache is only safe when that cache IS the
        # shown model's result. Test against what the child actually holds, NOT
        # against the loaded model: engine_process ADOPTS any result a rebuild ships
        # it (engine_process.py, "rebuild"), so one foreign rebuild leaves the child
        # caching a model it never ran. Keying off _loaded_backend_key then rebuilt
        # the last-swept model's cloud from a different model's depth — silently,
        # and it poisoned that model's cached cloud and stats for the session.
        foreign = (self._shown_model is not None
                   and self._shown_model != self._child_result_key)
        if foreign:
            self._child_result_key = self._shown_model   # the child adopts it
        self.worker.rebuildCloud(self.result if foreign else None,
                                 self._current_params())

    # -------------------------------------------------------------- batch
    def _batch_ready(self) -> tuple:
        """(can a batch start, and if not — the one thing to fix first). A batch
        reuses the loaded model, the current calibration and the placed boxes, so
        all three have to be in place — which is exactly the state you're in right
        after running one pair and dropping a box on each pin."""
        if self._busy or self._comparing or self._overlay_queue:
            return False, "The engine is busy — wait for the current run to finish."
        if not self._model_ready or self._needs_load():
            return False, ("Load a model first: drop a representative pair and press "
                           "Run once. The batch reuses that loaded model.")
        if not self.input_panel.has_calibration:
            return False, ("Batch needs calibration for depth. Load K.txt (or type "
                           "fx/baseline) so each pair builds a 3D cloud.")
        if not self.param_panel.measure_on or not self.param_panel.has_boxes():
            return False, ("Place your measure boxes first: turn on the Volume box and "
                           "drop one box on each pin. The batch measures those same "
                           "boxes on every capture.")
        return True, ""

    def _open_batch(self) -> None:
        if self._batching:                       # already running — surface the monitor
            if self._batch_dialog is not None:
                self._batch_dialog.raise_()
                self._batch_dialog.activateWindow()
            return
        if self._busy or self._comparing:
            QMessageBox.information(
                self, "Batch",
                "Wait for the current run/comparison to finish before batching.")
            return
        if self._batch_dialog is not None:       # close a leftover finished monitor first
            self._batch_dialog.close()
        # Both sources measure the SAME boxes, so a box on each pin is the one thing
        # needed to even open. Model + calibration are checked per-source at Run
        # (stereo pairs need them; saved clouds are already reconstructed, so don't).
        if not self.param_panel.measure_on or not self.param_panel.has_boxes():
            QMessageBox.information(
                self, "Batch — place your boxes first",
                "Turn on the Volume box and drop one box on each pin — the batch "
                "measures those same boxes on every capture.\n\nTo place boxes you "
                "need a cloud on screen, so run one pair first.")
            return
        dlg = BatchDialog(self._units, self)
        dlg._on_run_cb = self._start_batch
        dlg._on_run_clouds_cb = self._start_cloud_batch
        dlg._on_cancel_cb = self._cancel_batch
        dlg.finished.connect(lambda *_: self._on_batch_dialog_closed())
        self._batch_dialog = dlg
        dlg.show()

    def _start_batch(self, pairs: list) -> None:
        """Freeze the scene + boxes and start walking the pair queue. Loads each
        pair's RGB directly and calls runInference on it — the input panel is never
        touched, so _pair_version stays put and the reply-gate keeps passing."""
        if self._batching or not pairs:
            return
        ok, reason = self._batch_ready()
        if not ok:                                # state slipped between opening and Run
            QMessageBox.information(self, "Batch", reason)
            if self._batch_dialog is not None:
                self._batch_dialog.on_finished(f"Not started — {reason}")
            return
        # The batch replaces the shown scene with its own captures — drop the shown
        # result and the compare caches FIRST. Leaving them meant the first
        # post-batch live rebuild wrote a batch pair's cloud/stats into the
        # previously shown model's Compare column (cache poisoning), and blinking
        # back showed the compare pair's disparity over a batch pair's 3D cloud.
        self._clear_results()
        self._batching = True
        self._batch_kind = "pairs"
        self._batch_cancel = False
        self._batch_queue = list(pairs)
        self._batch_total = len(pairs)
        self._batch_done = self._batch_logged = self._batch_empty = 0
        self._batch_failed = []
        self._batch_params = self._current_params()        # scene frozen for the run
        self._batch_specs = self.param_panel.box_specs()   # boxes frozen for the run
        self._reset_cloud_view = True
        # lock the scene — no image/model/box/unit change mid-study. The study's
        # own buttons and Export lock too: Log would inject a mislabeled row,
        # Clear would wipe the accumulating study, and an Export dialog would let
        # the batch advance underneath its own snapshot.
        self.input_panel.setEnabled(False)
        self.param_panel.setEnabled(False)
        self.run_btn.setEnabled(False)
        self.compare_btn.setEnabled(False)
        self.units_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.viewer.repeat_view.set_locked(True)
        self.viewer.repeat_view.set_pins([n for n, _b, _t in self._batch_specs])
        self.viewer.setCurrentWidget(self.viewer.repeat_view)   # watch it fill
        self.progress.setRange(0, self._batch_total)
        self.progress.setValue(0)
        self.progress.show()
        self._batch_next()

    def _batch_next(self) -> None:
        if not self._batching:
            return
        if self._batch_cancel or not self._batch_queue:
            self._finish_batch()
            return
        label, lpath, rpath = self._batch_queue.pop(0)
        self._batch_label = label
        try:
            left, right = load_rgb(lpath), load_rgb(rpath)
            # rectify each pair exactly as a hand-dropped one (passthrough when the
            # input is already rectified) — the frozen _batch_params holds the same
            # derived calibration, so every capture is measured in one frame
            left, right = self.input_panel.process_pair(left, right)
        except Exception as exc:  # noqa: BLE001 — a bad image skips, never aborts
            self._batch_failed.append((label, f"couldn't read/rectify image: {exc}"))
            self._batch_done += 1
            self._update_batch_progress()
            QTimer.singleShot(0, self._batch_next)
            return
        self._update_batch_progress(label)        # show the one about to run
        self._run_pair_version = self._pair_version
        self.worker.runInference(left, right, self._batch_params)

    def _batch_on_cloud(self, cloud) -> None:
        """A batched pair's cloud arrived: show it, measure the frozen boxes, log a
        row, advance. Runs off the reply, so the loop is paced by the engine."""
        if not self._has_points(cloud):
            self._batch_failed.append((self._batch_label, "no cloud / no points"))
        else:
            self.cloud = cloud
            self.viewer.show_cloud(cloud, reset_view=self._reset_cloud_view)
            self._reset_cloud_view = False
            self.points_lbl.setText(f"{cloud.n:,} pts")
            self._apply_measure()                 # draw the boxes on this cloud
            self._batch_log(cloud)
        self._batch_done += 1
        self._update_batch_progress()
        QTimer.singleShot(0, self._batch_next)

    def _batch_log(self, cloud) -> None:
        """Measure every frozen box against `cloud` and log one capture (trimmed
        height per pin, canonical metres) into the Repeatability table."""
        inv = 1.0 / UNIT_PER_M[self._units]
        vals = {}
        for name, box, trim in self._batch_specs:
            m = measure_box(cloud.points, box, trim_pct=trim)
            vals[name] = m["h_span_t"] * inv if m is not None else None
        self.viewer.repeat_view.add_record(self._batch_label, vals)
        self._batch_logged += 1
        if not any(v is not None for v in vals.values()):
            self._batch_empty += 1

    # ------------------------------------------------- batch: saved cloud files
    def _start_cloud_batch(self, files: list, file_unit: str) -> None:
        """Re-measure saved cloud files through the frozen boxes — NO engine, pure
        numpy — so this drives its own QTimer loop (nothing to wait on but the next
        file read). Points are scaled from the files' unit to the working unit, so
        the boxes (which live in the working unit) actually catch them."""
        if self._batching or not files:
            return
        if self._busy or self._comparing:
            QMessageBox.information(
                self, "Batch",
                "The engine is busy — wait for the current run/comparison to finish.")
            if self._batch_dialog is not None:
                self._batch_dialog.on_finished("Not started — engine busy.")
            return
        if not self.param_panel.measure_on or not self.param_panel.has_boxes():
            QMessageBox.information(self, "Batch",
                                    "Turn on the Volume box and place a box on each pin first.")
            if self._batch_dialog is not None:
                self._batch_dialog.on_finished("Not started — no measure boxes.")
            return
        # same cache-poisoning defence as the pairs batch: the file clouds replace
        # the scene, so the previous result/compare caches must not survive them
        self._clear_results()
        self._batching = True
        self._batch_kind = "clouds"
        self._batch_cancel = False
        self._batch_queue = [(os.path.splitext(os.path.basename(f))[0], f) for f in files]
        self._batch_total = len(files)
        self._batch_done = self._batch_logged = self._batch_empty = 0
        self._batch_failed = []
        self._batch_specs = self.param_panel.box_specs()   # boxes frozen for the run
        self._batch_file_factor = (UNIT_PER_M[self._units]
                                   / UNIT_PER_M.get(file_unit, UNIT_PER_M[self._units]))
        self._cloud_shown = False
        self.input_panel.setEnabled(False)
        self.param_panel.setEnabled(False)
        self.run_btn.setEnabled(False)
        self.compare_btn.setEnabled(False)
        self.units_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.viewer.repeat_view.set_locked(True)
        self.viewer.repeat_view.set_pins([n for n, _b, _t in self._batch_specs])
        self.viewer.setCurrentWidget(self.viewer.repeat_view)
        self.progress.setRange(0, self._batch_total)
        self.progress.setValue(0)
        self.progress.show()
        self._cloud_next()

    def _cloud_next(self) -> None:
        if not self._batching:
            return
        if self._batch_cancel or not self._batch_queue:
            self._finish_batch()
            return
        label, path = self._batch_queue.pop(0)
        self._batch_label = label
        try:
            pts, cols = load_cloud(path)
        except Exception as exc:  # noqa: BLE001 — a bad file skips, never aborts
            self._batch_failed.append(
                (label, f"couldn't read cloud: {str(exc).splitlines()[0][:160]}"))
            self._batch_done += 1
            self._update_batch_progress(label)
            QTimer.singleShot(0, self._cloud_next)
            return
        if len(pts) == 0:                 # a valid-but-empty cloud: fail it (as the pairs path does)
            self._batch_failed.append((label, "empty cloud (no points)"))
            self._batch_done += 1
            self._update_batch_progress(label)
            QTimer.singleShot(0, self._cloud_next)
            return
        pts = (pts * self._batch_file_factor).astype(np.float32)  # file unit -> working unit
        if cols is None or len(cols) != len(pts):
            cols = np.full((len(pts), 3), 160, np.uint8)
        cloud = CloudResult(points=pts, colors=cols, n=len(pts))
        cloud = self._ingest_level(cloud)    # level saved clouds the same way as live ones
        if not self._cloud_shown:            # show ONE, with the boxes, as confirmation
            self._cloud_shown = True
            self.cloud = cloud
            self.viewer.show_cloud(cloud, reset_view=True)
            self.points_lbl.setText(f"{cloud.n:,} pts")
            self._apply_measure()
        self._batch_log(cloud)
        self._batch_done += 1
        self._update_batch_progress(label)
        QTimer.singleShot(0, self._cloud_next)

    def _update_batch_progress(self, label: str | None = None) -> None:
        self.progress.setValue(self._batch_done)
        shown = label or self._batch_label
        self._set_status(f"Batch {self._batch_done}/{self._batch_total}"
                         + (f" — {shown}" if shown else ""))
        if self._batch_dialog is not None:
            self._batch_dialog.on_progress(
                self._batch_done, self._batch_total, shown,
                self._batch_logged, self._batch_empty, len(self._batch_failed))

    def _cancel_batch(self) -> None:
        if self._batching:
            self._batch_cancel = True   # takes effect after the in-flight pair finishes
            self._set_status("Cancelling batch — finishing the current image…")

    def _finish_batch(self, aborted: str | None = None) -> None:
        self._batching = False
        self.input_panel.setEnabled(True)
        self.param_panel.setEnabled(True)
        self.compare_btn.setEnabled(True)
        self.units_btn.setEnabled(not self._busy)
        self.export_btn.setEnabled(self.result is not None)
        self.viewer.repeat_view.set_locked(False)
        self.progress.setRange(0, 0)      # back to the indeterminate busy spinner
        self.progress.setVisible(self._busy)
        self._update_run_enabled()
        n_fail = len(self._batch_failed)
        if aborted:
            head = f"Batch stopped — {aborted}."
        elif self._batch_cancel:
            head = "Batch cancelled."
        else:
            head = "Batch complete."
        summary = f"{head}  Logged {self._batch_logged}/{self._batch_total}."
        if self._batch_empty:
            summary += f"  {self._batch_empty} caught no points."
        if n_fail:
            summary += f"  {n_fail} failed."
        self._set_status(summary)
        self.viewer.setCurrentWidget(self.viewer.repeat_view)
        if self._batch_dialog is not None:
            self._batch_dialog.on_finished(summary, self._batch_failed)
        elif aborted or n_fail:
            QMessageBox.information(self, "Batch", summary)

    def _on_batch_dialog_closed(self) -> None:
        # The monitor was closed. If a batch is running, BatchDialog.closeEvent has
        # already requested a cancel (deliberate — otherwise Esc/the window-X would
        # strand a locked UI with no cancel affordance); here we just stop poking a
        # dead widget. _finish_batch pops a summary box when there's something to say.
        self._batch_dialog = None

    # -------------------------------------------------------------- export
    def _export(self, kind: str) -> None:
        if self.result is None:
            return
        import imageio.v2 as imageio

        from .engine import StereoEngine

        r = self.result
        if kind == "all":
            d = QFileDialog.getExistingDirectory(self, "Export everything to…")
            if not d:
                return
            try:
                base = os.path.join(d, "fs_output")
                imageio.imwrite(base + "_disparity.png", self.viewer.disp_view.render_rgb())
                np.save(base + "_disparity.npy", r.disp)
                if r.depth is not None:
                    imageio.imwrite(base + "_depth.png", self.viewer.depth_view.render_rgb())
                    np.save(base + f"_depth_{self._units}.npy", r.depth)
                if self.cloud is not None:
                    StereoEngine.save_cloud(base + "_cloud.ply", self.cloud)
                self._set_status(f"Exported to {d}")
            except Exception as exc:  # noqa: BLE001
                self._report_error(str(exc))
            return

        specs = {
            "disp_png": ("Save disparity image", "PNG (*.png)", ".png"),
            "depth_png": ("Save depth image", "PNG (*.png)", ".png"),
            "disp_npy": ("Save raw disparity", "NumPy (*.npy)", ".npy"),
            "depth_npy": (f"Save depth ({self._units})", "NumPy (*.npy)", ".npy"),
            "ply": ("Save point cloud", "PLY (*.ply)", ".ply"),
        }
        title, filt, ext = specs[kind]
        path, _ = QFileDialog.getSaveFileName(self, title, "", filt)
        if not path:
            return
        if not path.lower().endswith(ext):
            path += ext
        try:
            if kind == "disp_png":
                imageio.imwrite(path, self.viewer.disp_view.render_rgb())
            elif kind == "depth_png":
                if r.depth is None:
                    raise ValueError("No depth — set calibration first.")
                imageio.imwrite(path, self.viewer.depth_view.render_rgb())
            elif kind == "disp_npy":
                np.save(path, r.disp)
            elif kind == "depth_npy":
                if r.depth is None:
                    raise ValueError("No depth — set calibration first.")
                np.save(path, r.depth)
            elif kind == "ply":
                if self.cloud is None:
                    raise ValueError("No point cloud — set calibration and run.")
                StereoEngine.save_cloud(path, self.cloud)
            self._set_status(f"Saved {os.path.basename(path)}")
        except Exception as exc:  # noqa: BLE001
            self._report_error(str(exc))

    # -------------------------------------------------------------- units
    _UNIT_CYCLE = ("mm", "µm", "m")   # click order — mm→µm (small pins) →m→mm

    def _toggle_units(self) -> None:
        if self._busy or self._comparing or self._batching or self._overlay_queue:
            return   # never switch mid-run/sweep — cached results are in flight
        order = self._UNIT_CYCLE
        i = order.index(self._units) if self._units in order else 0
        self._set_units(order[(i + 1) % len(order)])

    def _set_units(self, unit: str) -> None:
        """Switch depth/cloud units (mm ⇄ m). The pipeline is unit-agnostic, so a
        switch is an exact rescale: calibration + z-planes are rescaled in the
        panels, any existing depth map / cloud are multiplied by the factor and
        re-shown, and the child's CACHED result is rescaled too (so a later live
        rebuild stays in the new unit). No re-run needed; nothing is marked stale."""
        if unit not in UNIT_PER_M or unit == self._units:
            return
        factor = UNIT_PER_M[unit] / UNIT_PER_M[self._units]
        self._units = unit
        self.units_btn.setText(unit)
        self.input_panel.set_units(unit)     # rescales baseline field
        self.param_panel.set_units(unit)     # rescales z-near / z-far
        self.viewer.set_units(unit)          # relabels depth readout + cloud grid
        # Rescale EVERY cached result, not just the shown one — otherwise flipping
        # to another compared model after a unit switch would show its old unit.
        # Identity checks, not `in`: InferResult is a dataclass whose __eq__ would
        # compare numpy arrays and raise "truth value is ambiguous".
        results = list(self.results.values())
        if self.result is not None and not any(r is self.result for r in results):
            results.append(self.result)
        for r in results:
            if r.depth is not None:
                r.depth = r.depth * factor
            r.baseline = float(r.baseline) * factor
        for c in self._all_clouds():
            # raw_points may BE the points array (level off) — rescale once and
            # re-share, or the second multiply would double-scale the shared array
            same = getattr(c, "raw_points", None) is c.points
            c.points = c.points * factor
            if getattr(c, "raw_points", None) is not None:
                c.raw_points = c.points if same else c.raw_points * factor
        for s in self.mstats.values():           # the strip's numbers are unit-bearing too
            for k in ("med_depth", "plane_rms"):
                if s.get(k) is not None:
                    s[k] = s[k] * factor
        self.viewer.compare_view.rescale_stats(factor)   # its own copies of the same
        if self.result is not None:
            if self.result.depth is not None:
                self.viewer.depth_view.set_image(self.result.depth)
            self.worker.rescaleResult(factor)   # keep the child's cached copy in step
        if self._overlay_on:
            # Every cached cloud was just rescaled above, so the overlay only needs
            # rebuilding from them. show_cloud() here would have quietly collapsed it
            # to the single shown model — a unit switch is not a request to stop
            # comparing.
            self._show_overlay()
        elif self.cloud is not None:
            self.viewer.show_cloud(self.cloud, reset_view=True)
        # param_panel.set_units rescaled the box itself (silently); this redraws it
        # against the rescaled points and restates its numbers in the new unit.
        self._apply_measure()
        self._update_compare_strip()
        self.probe_lbl.setText("")           # old readout was in the previous unit
        self._reset_analyze_overlay()        # picks/overlay were in the previous unit
        self._reapply_deviation()            # the rescale repaint wiped the heatmap
        self._update_ref_label()             # restate the flat-reference offset in the new unit
        self.settings.setValue("units", unit)

    # -------------------------------------------------------------- theme
    def _toggle_theme(self) -> None:
        self.theme = "light" if self.theme == "dark" else "dark"
        apply_theme(QApplication.instance(), self.theme)
        self.theme_btn.setText("☾" if self.theme == "dark" else "☀")

    # --------------------------------------------------------- vram + misc
    def _poll_vram(self) -> None:
        if self._busy:
            return   # the child can't answer mid-op; requests would just queue up
        self.worker.requestVram()

    def _on_vram(self, used: float, total: float) -> None:
        if total > 0:
            self.vram_lbl.setText(f"VRAM {used:.1f}/{total:.0f} GB")

    # ------------------------------------------------------------ settings
    def _restore_settings(self) -> None:
        self.settings = QSettings("FSStudio", "FoundationStereoStudio")
        s = self.settings
        # Per-model settings + checkpoints + which models are ticked to compare.
        # JSON blobs — nested dicts don't survive QSettings' native encoding
        # intact. Anything unreadable, or naming a knob a backend no longer has,
        # is ignored so a stale blob can't wedge the panel.
        def _blob(name):
            try:
                return json.loads(s.value(name, "") or "{}")
            except (ValueError, TypeError):
                return {}

        # ORDER MATTERS: the per-model checkpoint map has to exist before the
        # selection is restored (restore_selection populates the checkpoint combo
        # from it), and the per-model settings before the panel is built from them.
        self.input_panel.restore_ckpts(_blob("model_ckpts"))
        self.input_panel.restore_selection(
            s.value("backend", DEFAULT_BACKEND), s.value("ckpt", ""))
        self.param_panel.restore_all(_blob("model_params"))
        self.param_panel.set_backend(self.input_panel.current_spec())
        self.param_panel.restore_section_states(_blob("sections_param"))
        self.input_panel.restore_section_states(_blob("sections_input"))
        self.input_panel.restore_rect_state(_blob("rectify"))   # raw-mode + calib path
        lvl = _blob("level")
        if isinstance(lvl, dict) and lvl.get("R") and lvl.get("c"):
            try:    # a persisted level rotation applies to every cloud this session too
                self._level_R = np.array(lvl["R"], np.float64).reshape(3, 3)
                self._level_c_m = np.array(lvl["c"], np.float64).reshape(3)
                self.param_panel.set_level_checked(True)
            except Exception:   # noqa: BLE001 — a bad blob must not wedge startup
                self._level_R = self._level_c_m = None
        try:      # saved measure boxes (new {boxes,sel} dict, or an old bare list)
            self.param_panel.restore_boxes(json.loads(s.value("box_presets", "") or "[]"))
        except (ValueError, TypeError):
            self.param_panel.restore_boxes([])
        self.viewer.compare_view.restore(_blob("compare"))
        self._refresh_compare_cards()
        # NOTE: calibration is intentionally NOT restored — it is per-pair, and
        # reusing a previous session's intrinsics on a new pair silently produces
        # wrong metric depth. Each pair's K.txt (auto-loaded) is the source.
        geo = s.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        # restore the mm/m preference (no result exists yet → just reconfigures UI)
        saved_units = s.value("units", "mm")
        if saved_units in UNIT_PER_M and saved_units != self._units:
            self._set_units(saved_units)

    def closeEvent(self, e) -> None:
        # stop the periodic timers FIRST so no vram poll / cloud rebuild fires a
        # send at the engine while (or after) we're tearing the connection down
        self._vram_timer.stop()
        self._cloud_timer.stop()
        self._measure_timer.stop()
        # flush the box set NOW: _remeasure (which persists it) is debounced 120 ms,
        # so a box edit made just before closing would otherwise be lost
        self._save_boxes()
        s = self.settings
        s.setValue("theme", self.theme)
        s.setValue("geometry", self.saveGeometry())
        s.setValue("backend", self.input_panel.current_backend_key())
        s.setValue("ckpt", self.input_panel.current_checkpoint_path())
        # One try per blob, never one shared: these are independent, and sharing a
        # try meant an unserialisable value in the FIRST silently dropped the other
        # two — you'd lose your Compare selection because a model param went bad.
        for name, get in (("model_params", self.param_panel.saved_all),
                          ("model_ckpts", self.input_panel.saved_ckpts),
                          ("compare", self.viewer.compare_view.values),
                          ("sections_param", self.param_panel.section_states),
                          ("sections_input", self.input_panel.section_states),
                          ("rectify", self.input_panel.rect_state),
                          ("level", self._level_state)):
            try:   # never let a settings blob block the app from closing
                s.setValue(name, json.dumps(get()))
            except Exception:   # noqa: BLE001
                pass
        s.remove("calibration")   # drop any stale blob persisted by old versions
        self.worker.stop()
        super().closeEvent(e)
