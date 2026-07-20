"""WebCloudView — the 3D point-cloud view, backed by three.js in a QWebEngineView.

A drop-in for the old pyqtgraph CloudView3D: same public methods and signals, same
bottom-bar Qt controls (colour combo, overlay checkbox, legend, model bar). Only
the rendering surface changed — from a bare GL scatter to a real WebGL scene with a
proper move/rotate/scale gizmo (studio/web/cloud.html).

Data flows one way each direction:
  Python → JS  : window.api.* via page().runJavaScript (commands, buffered until the
                 page signals it is ready); the cloud itself goes as a packed binary
                 served over a localhost http server and fetched by the page.
  JS → Python  : a QWebChannel bridge — the gizmo reports the box pose back here, and
                 measure.py (in the window) turns it into the authoritative readout.
"""
from __future__ import annotations

import functools
import http.server
import math
import os
import shutil
import socketserver
import tempfile
import threading
import weakref

import numpy as np
from PySide6.QtCore import QObject, QUrl, Signal, Slot
from PySide6.QtGui import QColor
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QLabel,
                               QVBoxLayout, QWidget)

from .dtypes import UNIT_DECIMALS
from .measure import MeasureBox
from .widgets import CLOUD_COLOR_MODES, CloudLegend, FlowLayout, ModelBar, no_wheel

_HERE = os.path.dirname(os.path.abspath(__file__))
_WEB_SRC = os.path.join(_HERE, "web")
_MODE = {"Camera (L·R)": "camera", "Reliability": "reliability", "Model": "model"}
_HI_MODE = {"Highlight": "highlight", "Height map": "height", "Trim": "trim"}


def _js_num(v) -> str:
    """A finite value as a JS number literal. plain float() first: a numpy float64
    renders as 'np.float64(3.0)' under numpy 2, and repr(inf)/repr(nan) are bare
    identifiers — both invalid JS that would abort the whole setBox call, so map
    any non-finite to 0."""
    v = float(v)
    return repr(v) if math.isfinite(v) else "0"


def _fin(v) -> float:
    """A finite float for JSON — json.dumps emits invalid 'Infinity'/'NaN' otherwise."""
    v = float(v)
    return v if math.isfinite(v) else 0.0


def _ctl_group(*widgets) -> QWidget:
    """Bundle a label with its control(s) into one widget, so the reflowing control
    strip wraps them as a unit and never splits a label from its combo."""
    w = QWidget()
    h = QHBoxLayout(w)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(6)
    for x in widgets:
        h.addWidget(x)
    return w


def _webroot() -> str:
    """A private temp dir holding the page + vendored three.js + qwebchannel.js,
    served over http. Static files are copied in once; cloud binaries are written
    here per push. Kept out of the source tree so a running app never writes into
    studio/web/."""
    root = tempfile.mkdtemp(prefix="fs_cloud_")
    shutil.copy(os.path.join(_WEB_SRC, "cloud.html"), root)
    shutil.copytree(os.path.join(_WEB_SRC, "vendor"), os.path.join(root, "vendor"))
    # qwebchannel.js ships inside Qt as a resource; extract the version-matched copy
    # rather than vendoring one that could drift from the installed PySide6.
    import PySide6.QtWebChannel  # noqa: F401  — registers the :/qtwebchannel resource
    from PySide6.QtCore import QFile, QIODevice

    f = QFile(":/qtwebchannel/qwebchannel.js")
    if f.open(QIODevice.ReadOnly):
        with open(os.path.join(root, "qwebchannel.js"), "wb") as out:
            out.write(bytes(f.readAll().data()))
        f.close()
    return root


def _serve(directory: str) -> int:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=directory)
    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd.server_address[1]


class _Bridge(QObject):
    """QWebChannel endpoint the page talks to. Slots are what JS calls; the Qt
    signals fan those out to the view without the view having to know about the
    channel."""

    edited = Signal(int, float, float, float, float, float, float,
                    float, float, float, float, bool)   # (box index, pose…, final?)
    selected = Signal(int)                              # user clicked a box in 3D
    picked = Signal(float, float, float)                # a cloud point clicked in analyze mode
    ready = Signal()

    @Slot(int, float, float, float, float, float, float, float, float, float, float, bool)
    def boxChanged(self, idx, cx, cy, cz, sx, sy, sz, qx, qy, qz, qw, final):
        self.edited.emit(idx, cx, cy, cz, sx, sy, sz, qx, qy, qz, qw, final)

    @Slot(int)
    def boxSelected(self, idx):
        self.selected.emit(idx)

    @Slot(float, float, float)
    def pointPicked(self, x, y, z):
        self.picked.emit(x, y, z)

    @Slot()
    def pageReady(self):
        self.ready.emit()


class WebCloudView(QWidget):
    """3D point cloud + measure-box gizmo. Public surface matches CloudView3D."""

    overlayToggled = Signal(bool)
    boxEdited = Signal(int, object, bool)   # (box index, world MeasureBox, final?)
    boxSelected = Signal(int)               # user clicked a box in the 3D view
    pointPicked = Signal(float, float, float)   # a cloud point clicked in analyze mode

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._root = _webroot()
        self._port = _serve(self._root)
        # Delete the temp webroot when this view (or the interpreter) is finalised —
        # otherwise every launch leaks its vendored three.js copy plus the last
        # cloud binary into %TEMP%. The daemon http thread dies with the process.
        weakref.finalize(self, shutil.rmtree, self._root, True)
        self._ready = False
        self._queue: list = []
        self._cloud_seq = 0
        self._last_bin: str | None = None

        # cloud metadata kept for the label + the model overlay
        self._n = 0
        self._origin = None
        self._reliable = None
        self._model = None
        self._model_names: list = []
        self._has_right = False
        self._hidden: set = set()
        self._color_mode = "photo"
        self._unit = "mm"
        self._grid_step = 0.0

        self.view = QWebEngineView()
        self.view.page().setBackgroundColor(QColor("#0a0c12"))   # no white flash
        self._bridge = _Bridge()
        self._channel = QWebChannel()
        self._channel.registerObject("bridge", self._bridge)
        self.view.page().setWebChannel(self._channel)
        self._bridge.ready.connect(self._on_ready)
        self._bridge.edited.connect(self._on_box_edited)
        self._bridge.selected.connect(self.boxSelected)   # re-emit to the window
        self._bridge.picked.connect(self.pointPicked)     # re-emit picked points
        self.view.load(QUrl(f"http://127.0.0.1:{self._port}/cloud.html"))

        # ---- bottom control bar — the SAME Qt widgets the GL view used, so the
        # window wires to them unchanged; they just drive JS now ----
        self.model_bar = ModelBar()
        self.overlay_chk = QCheckBox("Overlay")
        self.overlay_chk.setEnabled(False)
        self.overlay_chk.setToolTip(
            "Draw every compared model's cloud at once, each in its own colour, "
            "instead of one at a time.\n\nWhere they overlap the colours mix — what "
            "you're looking for is where one colour stands alone. Untick a model in "
            "the key to take it out.")
        self.overlay_chk.toggled.connect(self.overlayToggled)
        self.legend = CloudLegend()
        self.legend.toggled.connect(self._on_model_visible)
        self.color_combo = no_wheel(QComboBox())
        self.color_combo.addItems(CLOUD_COLOR_MODES)
        self.color_combo.setFixedWidth(130)
        self.color_combo.setToolTip(
            "How to paint the points:\n"
            "• Photo — real image colors\n"
            "• Camera (L·R) — which eye each point came from (needs “Both eyes”)\n"
            "• Reliability — green where both views agree, red where occluded\n"
            "• Model — which model produced each point (needs Overlay)")
        self.color_combo.currentTextChanged.connect(self._on_color_mode)

        # in-box highlight: what to do with the points inside the volume box.
        # Disabled until there is a box to edit (set by set_boxes).
        self.hi_combo = no_wheel(QComboBox())
        self.hi_combo.addItems(["Off", "Highlight", "Height map", "Trim"])
        self.hi_combo.setFixedWidth(120)
        self.hi_combo.setEnabled(False)
        self.hi_combo.setToolTip(
            "Show the points INSIDE the volume box (needs the box on):\n"
            "• Highlight — inside points brighten and pop, outside dims\n"
            "• Height map — inside points colored by height along the box's Z axis\n"
            "   (blue = low → red = high), so you can see the pin heights\n"
            "• Trim — inside points green where kept, red where the Trim % cuts them")
        self.hi_combo.currentTextChanged.connect(self._on_highlight)
        self.isolate_chk = QCheckBox("Isolate")
        self.isolate_chk.setEnabled(False)
        self.isolate_chk.setToolTip(
            "Hide everything outside the box instead of dimming it, so only the "
            "box contents remain.")
        self.isolate_chk.toggled.connect(self._on_isolate)

        self.cloud_lbl = QLabel("")
        self.cloud_lbl.setProperty("role", "muted")

        # The control strip reflows onto a second row instead of overflowing when
        # several models overlay at once: model bar + colour key + colour + in-box +
        # point count is more than a narrow 3D view can fit on one line. Each label is
        # bundled with its control so a wrap never separates them; the model bar and
        # legend hide themselves in the single-model case and reserve no space.
        strip = QWidget()
        strip.setContentsMargins(2, 0, 2, 0)
        flow = FlowLayout(strip, margin=0, spacing=10)
        flow.addWidget(self.model_bar)
        flow.addWidget(self.overlay_chk)
        flow.addWidget(self.legend)
        flow.addWidget(_ctl_group(QLabel("Color"), self.color_combo))
        flow.addWidget(_ctl_group(QLabel("In box"), self.hi_combo, self.isolate_chk))
        flow.addWidget(self.cloud_lbl)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        lay.addWidget(self.view, 1)
        lay.addWidget(strip)

    # ------------------------------------------------------------ JS plumbing
    def _js(self, code: str) -> None:
        """Run a JS command now, or hold it until the page says it is ready. The
        page loads asynchronously and the first cloud can arrive before then, so
        buffering keeps the very first push from being dropped on the floor."""
        if self._ready:
            self.view.page().runJavaScript(code)
        else:
            self._queue.append(code)

    def _on_ready(self) -> None:
        self._ready = True
        for code in self._queue:
            self.view.page().runJavaScript(code)
        self._queue.clear()

    # ------------------------------------------------------------- data in
    def set_cloud(self, points, colors, origin=None, reliable=None,
                  reset_view: bool = False, model=None, model_names=None) -> None:
        if points is None or len(points) == 0:
            self.clear()
            return
        n = len(points)
        # pack: xyz float32 | rgb uint8 | [origin u8] | [reliable u8] | [model u8]
        # Streamed as contiguous chunks — f.write takes each ndarray's buffer
        # directly, so a 4M-point cloud never doubles up into a ~60 MB
        # concatenated `bytes` on the way to disk.
        flags = 0
        chunks = [np.ascontiguousarray(points, "<f4"),
                  np.ascontiguousarray(colors, "u1")]
        if origin is not None:
            chunks.append(np.ascontiguousarray(origin, "u1")); flags |= 1
        if reliable is not None:
            chunks.append(np.ascontiguousarray(np.asarray(reliable) != 0, "u1")); flags |= 2
        if model is not None:
            chunks.append(np.ascontiguousarray(model, "u1")); flags |= 4

        self._cloud_seq += 1
        name = f"cloud_{self._cloud_seq}.bin"
        with open(os.path.join(self._root, name), "wb") as f:
            for chunk in chunks:
                f.write(chunk)
        if self._last_bin:                      # keep only the latest on disk
            try:
                os.remove(os.path.join(self._root, self._last_bin))
            except OSError:
                pass
        self._last_bin = name

        # metadata for the label + overlay legend
        self._n = n
        self._origin = np.asarray(origin) if origin is not None else None
        self._reliable = np.asarray(reliable) if reliable is not None else None
        self._model = np.asarray(model) if model is not None else None
        self._has_right = self._origin is not None and bool(np.any(self._origin == 1))
        self._grid_step = self._nice_grid_step(points)
        if model is None:
            self._model_names = []
            self._hidden = set()
            self.legend.clear()
        elif model_names is not None and list(model_names) != self._model_names:
            self._model_names = list(model_names)
            self._hidden = set()
            self.legend.set_models(self._model_names)
            self._js("api.clearHidden()")

        self._js(f"api.loadCloud('{name}', {n}, {flags}, {1 if reset_view else 0})")
        self._update_cloud_lbl()

    def clear(self) -> None:
        self._js("api.clearCloud()")
        self._js("api.setBoxes([], -1, false)")   # no cloud — take the gizmo down too
        self._js("api.clearAnalyze()")            # …and drop any analyze markers/overlay
        self._n = 0
        self._origin = self._reliable = self._model = None
        self._model_names = []
        self._hidden = set()
        self._has_right = False
        self.legend.clear()
        self.cloud_lbl.setText("")

    # ------------------------------------------------------- colour / models
    def _on_color_mode(self, text: str) -> None:
        self._color_mode = _MODE.get(text, "photo")
        self._js(f"api.setColorMode('{self._color_mode}')")
        self._update_cloud_lbl()

    def _on_model_visible(self, idx: int, on: bool) -> None:
        self._hidden.discard(idx) if on else self._hidden.add(idx)
        self._js(f"api.setModelHidden({idx}, {'false' if on else 'true'})")
        self._update_cloud_lbl()

    def _on_highlight(self, text: str) -> None:
        self._js(f"api.setBoxHighlight('{_HI_MODE.get(text, 'off')}')")

    def _on_isolate(self, on: bool) -> None:
        self._js(f"api.setBoxIsolate({'true' if on else 'false'})")

    def set_point_size(self, s: float) -> None:
        self._js(f"api.setPointSize({float(s)})")

    def set_units(self, unit: str) -> None:
        self._unit = unit
        # keep the live drag readout in the same unit/precision as the panel
        self._js(f"api.setUnit('{unit}', {UNIT_DECIMALS.get(unit, 2)})")
        self._update_cloud_lbl()

    # ------------------------------------------------------------ overlay
    def set_overlay_available(self, on: bool) -> None:
        self.overlay_chk.setEnabled(on)
        if not on:
            self.set_overlay_checked(False)

    def set_overlay_checked(self, on: bool) -> None:
        self.overlay_chk.blockSignals(True)
        self.overlay_chk.setChecked(on)
        self.overlay_chk.blockSignals(False)

    # ------------------------------------------------------------ measure boxes
    def set_boxes(self, views, selected: int, editable: bool) -> None:
        """Draw the whole set of boxes. `views` is a list of MeasureBox (current
        unit); `selected` is the one the gizmo edits (-1 for none)."""
        import json
        arr = [{"c": [_fin(b.cx), _fin(b.cy), _fin(b.cz)],
                "s": [_fin(b.sx), _fin(b.sy), _fin(b.sz)],
                "q": [_fin(b.qx), _fin(b.qy), _fin(b.qz), _fin(b.qw)]} for b in views]
        self._js(f"api.setBoxes({json.dumps(arr)}, {int(selected)}, "
                 f"{'true' if editable else 'false'})")
        # the in-box highlight controls only mean anything with a box to edit
        active = editable and bool(views)
        self.hi_combo.setEnabled(active)
        self.isolate_chk.setEnabled(active)

    def set_measurement(self, text: str) -> None:
        # JSON-encode so newlines/quotes in the readout survive the trip to JS
        import json
        self._js(f"api.setReadout({json.dumps(text or '')})")

    def set_box_scalars(self, lo: float, hi: float) -> None:
        """Push the trimmed height band [lo, hi] (along the SELECTED box's Z axis) so
        the 'Trim' highlight can show which points the current Trim % is cutting."""
        self._js(f"api.setBoxTrim({_js_num(lo)}, {_js_num(hi)})")

    def _on_box_edited(self, idx, cx, cy, cz, sx, sy, sz, qx, qy, qz, qw, final) -> None:
        box = MeasureBox(cx=cx, cy=cy, cz=cz, sx=sx, sy=sy, sz=sz,
                         qx=qx, qy=qy, qz=qz, qw=qw)
        self.boxEdited.emit(int(idx), box, bool(final))

    # ------------------------------------------------------------ analyze tools
    def set_analyze_tool(self, mode) -> None:
        """Enter/leave a picking tool ('profile'|'distance'|'region'|'point'|None).
        Off also clears the overlay."""
        import json
        self._js(f"api.setAnalyzeMode({json.dumps(mode) if mode else 'null'})")

    def set_analyze_geom(self, markers=None, line=None, marker_r: float = 1.0) -> None:
        """Draw the overlay: sphere markers at `markers` + a polyline through `line`
        (both lists of (x,y,z) in the current unit)."""
        import json
        m = [[_fin(p[0]), _fin(p[1]), _fin(p[2])] for p in (markers or [])]
        ln = [[_fin(p[0]), _fin(p[1]), _fin(p[2])] for p in (line or [])]
        self._js(f"api.setAnalyzeGeom({json.dumps(m)}, {json.dumps(ln)}, {_js_num(marker_r)})")

    def set_analyze_highlight(self, points=None) -> None:
        """Light up the exact points the current tool measured (the region / isolated
        level), so the user can confirm the right zone. `points` is an (N,3) array in
        the display unit; call after set_analyze_geom (which clears the overlay first)."""
        import json
        p = np.asarray(points, np.float64) if points is not None else np.empty((0, 3))
        if p.ndim != 2 or p.shape[0] == 0:
            return
        p = p[np.isfinite(p).all(axis=1)]     # drop non-finite points, keep xyz triplets intact
        if p.shape[0] == 0:
            return
        flat = np.round(p.reshape(-1), 4).tolist()
        self._js(f"api.setAnalyzeHighlight({json.dumps(flat)})")

    def clear_analyze(self) -> None:
        self._js("api.clearAnalyze()")

    # ------------------------------------------------------------ label
    @staticmethod
    def _nice_grid_step(points) -> float:
        p = np.asarray(points)
        span = float((p.max(0) - p.min(0)).max()) if len(p) else 0.0
        if span <= 0:
            return 0.0
        raw = span / 8.0
        mag = 10.0 ** np.floor(np.log10(raw))
        return float(min((1, 2, 5, 10), key=lambda m: abs(m * mag - raw)) * mag)

    def _update_cloud_lbl(self) -> None:
        if not self._n:
            self.cloud_lbl.setText("")
            return
        n = self._n
        if self._model is not None and self._model_names:
            shown = n if not self._hidden else int(np.sum(~np.isin(self._model, list(self._hidden))))
            txt = (f"{shown:,} of {n:,} pts · "
                   f"{len(self._model_names) - len(self._hidden)} of {len(self._model_names)} models")
        elif self._color_mode == "camera" and self._origin is not None:
            nR = int(np.count_nonzero(self._origin == 1))
            txt = f"{n - nR:,} left · {nR:,} right"
        elif self._color_mode == "reliability" and self._reliable is not None:
            pr = 100.0 * float(np.asarray(self._reliable).mean())
            txt = f"{pr:.0f}% reliable · {100 - pr:.0f}% occluded"
        else:
            txt = f"{n:,} pts"
        if (self._model is None and self._color_mode in ("camera", "reliability")
                and not self._has_right):
            txt += "  · single eye"
        if self._grid_step > 0:
            txt += f"  · grid {self._grid_step:g} {self._unit}"
        self.cloud_lbl.setText(txt)
