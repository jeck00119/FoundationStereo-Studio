"""Viewers: the 2D scalar/RGB image view (colormap + pixel probe), built on
pyqtgraph, and the tab stack that arranges every view.

The 3D point cloud is NOT here — it is ``web_cloud.WebCloudView`` (three.js in
a QWebEngineView), which ViewerStack only hosts as a tab. Nothing in this
module touches OpenGL."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QHBoxLayout,
                               QLabel, QPushButton, QTabWidget, QVBoxLayout,
                               QWidget)

from .compare import ComparePanel
from .dtypes import UNIT_DECIMALS
from .repeat import RepeatabilityView
from .web_cloud import WebCloudView
from .widgets import ModelBar, no_wheel

pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)

COLORMAPS = ["turbo", "viridis", "magma", "inferno", "plasma", "jet", "gray"]
_LUT_CACHE: dict[str, np.ndarray] = {}
def build_lut(name: str) -> np.ndarray:
    if name in _LUT_CACHE:
        return _LUT_CACHE[name]
    import matplotlib

    cmap = matplotlib.colormaps[name]
    lut = (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.ubyte)
    _LUT_CACHE[name] = lut
    return lut


class ImageView2D(QWidget):
    """Displays an RGB image or a scalar map (with colormap + level control)
    and reports the pixel under the cursor."""

    hovered = Signal(int, int)      # image x, y
    pixelClicked = Signal(int, int)  # image x, y — a left-click inside the image
    roiChanged = Signal(object)     # (x0, y0, w, h) in LEFT-image pixels, or None
    findShiftRequested = Signal()   # "Find shift" pressed (the window does the match)
    siteMarked = Signal(str, int, int)   # ("pin"|"ref", x, y) in LEFT-image pixels

    def __init__(self, scalar: bool = True, unit: str = "", pair: bool = False,
                 dec: int = 2, parent=None) -> None:
        super().__init__(parent)
        self._scalar = scalar
        self._unit = unit
        self._dec = dec          # decimals for the value readout / range label
        self._pair = pair
        self._left: np.ndarray | None = None
        self._right: np.ndarray | None = None
        self._side = "left"
        self._arr: np.ndarray | None = None
        self._levels = (0.0, 1.0)
        self._guides: list = []   # horizontal row-alignment reference lines (pair mode)

        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground(None)
        self.vb = self.glw.addViewBox()
        self.vb.setAspectLocked(True)
        self.vb.invertY(True)
        self.vb.setMenuEnabled(False)
        self.img = pg.ImageItem()
        self.vb.addItem(self.img)

        pen = pg.mkPen((255, 255, 255, 90), width=1)
        self.vline = pg.InfiniteLine(angle=90, movable=False, pen=pen)
        self.hline = pg.InfiniteLine(angle=0, movable=False, pen=pen)
        for ln in (self.vline, self.hline):
            ln.setZValue(10)
            ln.hide()
            self.vb.addItem(ln, ignoreBounds=True)

        # floating readout
        self.readout = QLabel("", self.glw)
        self.readout.setStyleSheet(
            "background: rgba(10,12,18,0.82); color:#fff; padding:4px 8px;"
            'border-radius:6px; font-family:"Cascadia Mono","Consolas",monospace;'
            "font-size:11px;"
        )
        self.readout.move(10, 10)
        self.readout.hide()

        # bottom control bar
        bar = QHBoxLayout()
        bar.setContentsMargins(2, 0, 2, 0)
        bar.setSpacing(8)
        if scalar:
            self.model_bar = ModelBar()      # hidden until a comparison exists
            self.cmap = QComboBox()
            self.cmap.addItems(COLORMAPS)
            self.cmap.setFixedWidth(110)
            self.cmap.setToolTip("Color scheme for this map. Visual only — it doesn't change the values.")
            self.cmap.currentTextChanged.connect(self._apply_cmap)
            self.range_lbl = QLabel("—")
            self.range_lbl.setProperty("role", "muted")
            self.auto_btn = QPushButton("Auto range")
            self.auto_btn.setToolTip("Rescale the colors to fit the current data range (2–98th percentile).")
            self.auto_btn.clicked.connect(self._auto_levels)
            bar.addWidget(self.model_bar)
            bar.addWidget(QLabel("Colormap"))
            bar.addWidget(self.cmap)
            bar.addWidget(self.auto_btn)
            bar.addStretch(1)
            bar.addWidget(self.range_lbl)
        elif pair:
            self.btn_left = QPushButton("Left")
            self.btn_right = QPushButton("Right")
            grp = QButtonGroup(self)
            grp.setExclusive(True)
            for b in (self.btn_left, self.btn_right):
                b.setObjectName("Seg")
                b.setCheckable(True)
                b.setFixedWidth(60)
                grp.addButton(b)
            self.btn_left.setChecked(True)
            self.btn_left.setToolTip("Show the LEFT image (the reference frame).")
            self.btn_right.setToolTip("Show the RIGHT image. Zoom/pan stays put — flip back "
                                      "and forth to see the stereo shift (blink comparator).")
            self.btn_left.clicked.connect(lambda: self._show_side("left"))
            self.btn_right.clicked.connect(lambda: self._show_side("right"))
            self.fit_btn = QPushButton("Fit")
            self.fit_btn.setToolTip("Reset zoom to fit the image.")
            self.fit_btn.clicked.connect(lambda: self.vb.autoRange(padding=0.02))
            self.guide_chk = QCheckBox("Guides")
            self.guide_chk.setToolTip("Draw horizontal reference lines. A correctly "
                                      "rectified pair keeps a feature on the SAME line in "
                                      "Left and Right — flip between them to check.")
            self.guide_chk.toggled.connect(self._toggle_guides)
            self.side_lbl = QLabel("")
            self.side_lbl.setProperty("role", "muted")
            # --- ROI: the crop the run is restricted to -----------------------
            self.roi_chk = QCheckBox("ROI")
            self.roi_chk.setToolTip(
                "Restrict the run to a region of the LEFT image — drag the box "
                "over the parts you measure.\n\nA macro pair's disparity is huge "
                "(~500 px here) but varies only a few px across a flat board, so "
                "cropping to what you measure and pre-aligning the right side "
                "lets the whole run fit at FULL resolution instead of a shrunken "
                "one — several times finer depth for less memory.")
            self.roi_chk.toggled.connect(self._toggle_roi)
            self.shift_btn = QPushButton("Find shift")
            self.shift_btn.setEnabled(False)
            self.shift_btn.setToolTip(
                "Match the ROI's pixels in the right image to measure how far the "
                "two views are offset. Needs no calibration and no prior run.")
            self.shift_btn.clicked.connect(self.findShiftRequested)
            self.roi_lbl = QLabel("")
            self.roi_lbl.setProperty("role", "muted")
            # --- study sites: the pins to measure and what to measure them
            # AGAINST. Marked on the IMAGE, not in 3D: the measurement tracks
            # them per capture and takes the difference within one frame, which
            # is what makes it immune to the rig's frame drift and to the
            # baseline variation that swamps absolute depth.
            self.mark_combo = no_wheel(QComboBox())
            self.mark_combo.addItems(["Mark: off", "Mark: pin", "Mark: reference"])
            self.mark_combo.setToolTip(
                "Click the image to mark measurement sites.\n\n"
                "PIN — a feature whose height you want.\n"
                "REFERENCE — a nearby textured surface to measure it against; each "
                "pin uses the closest one.\n\nPick references with TEXTURE: bare "
                "solder mask reconstructs poorly and makes a noisy reference.")
            self.sites_lbl = QLabel("no sites")
            self.sites_lbl.setProperty("role", "muted")
            self.sites_clear = QPushButton("Clear sites")
            self.sites_clear.setToolTip("Remove every marked pin and reference.")
            self.sites_clear.clicked.connect(self.clear_sites)
            bar.addWidget(QLabel("View"))
            bar.addWidget(self.btn_left)
            bar.addWidget(self.btn_right)
            bar.addWidget(self.fit_btn)
            bar.addWidget(self.guide_chk)
            bar.addWidget(self.roi_chk)
            bar.addWidget(self.shift_btn)
            bar.addWidget(self.mark_combo)
            bar.addWidget(self.sites_clear)
            bar.addStretch(1)
            bar.addWidget(self.sites_lbl)
            bar.addWidget(self.roi_lbl)
            bar.addWidget(self.side_lbl)
            # The box itself. Snapped to a multiple of 32 (see _sync_roi) because
            # the network pads its input up to that anyway — an unsnapped crop
            # pays for pixels the user did not ask for.
            self.roi = pg.RectROI([0, 0], [256, 256], pen=pg.mkPen("#f4883f", width=2),
                                  handlePen=pg.mkPen("#f4883f", width=2))
            self.roi.addScaleHandle([0, 0], [1, 1])
            self.roi.setZValue(20)
            self.roi.hide()
            self.vb.addItem(self.roi)
            self.roi.sigRegionChangeFinished.connect(self._sync_roi)
            self._sites: list = []            # [{"kind","x","y","name"}]
            self._site_dots = pg.ScatterPlotItem(size=15, pen=pg.mkPen("#0b0d12", width=2))
            self._site_dots.setZValue(25)
            self.vb.addItem(self._site_dots)
            self._site_texts: list = []
        else:
            bar.addStretch(1)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        lay.addWidget(self.glw, 1)
        lay.addLayout(bar)

        self.glw.scene().sigMouseMoved.connect(self._on_move)
        self.glw.scene().sigMouseClicked.connect(self._on_click)
        if scalar:
            self.img.setLookupTable(build_lut("turbo"))

    def _pixel_at(self, scene_pos):
        """(x, y) of the image pixel under a scene position, or None if the
        position isn't over the image. Shared by hover and click so the two can
        never disagree about which pixel you are pointing at.

        floor(), not int(): int() truncates toward zero, so the half-pixel strip
        at (-0.5, 0) would round to 0 and report a hit just outside the image.
        """
        if self._arr is None or not self.vb.sceneBoundingRect().contains(scene_pos):
            return None
        pt = self.vb.mapSceneToView(scene_pos)
        x, y = int(np.floor(pt.x())), int(np.floor(pt.y()))
        h, w = self._arr.shape[:2]
        return (x, y) if (0 <= x < w and 0 <= y < h) else None

    def _on_click(self, ev) -> None:
        """A left-click on a real pixel. pyqtgraph raises sigMouseClicked only for
        a press+release that did NOT drag, so panning the view never fires this."""
        if ev.button() != Qt.LeftButton:
            return
        px = self._pixel_at(ev.scenePos())
        if px is None:
            return
        kind = self._mark_kind()
        if kind:                     # marking mode owns the click
            self.siteMarked.emit(kind, px[0], px[1])
            return
        self.pixelClicked.emit(px[0], px[1])

    # ------------------------------------------------------------- data in
    def _set_rgb(self, arr: np.ndarray) -> None:
        """Show an RGB image at FIXED 0–255 levels for uint8. Bare setImage()
        auto-levels to each array's own min/max, so the two sides of a blink pair
        got different contrast stretches — a global brightness step in the blink
        that is not in the data."""
        if arr.dtype == np.uint8:
            self.img.setImage(arr, levels=(0, 255))
        else:
            self.img.setImage(arr)

    def set_image(self, arr: np.ndarray) -> None:
        self._arr = arr
        if self._scalar:
            self.img.setImage(arr.astype(np.float32), autoLevels=False)
            self._auto_levels()
        else:
            self._set_rgb(arr)  # RGB uint8
        self.vb.autoRange(padding=0.02)

    def set_image_blink(self, arr: np.ndarray) -> None:
        """Swap the image for a comparison blink: keep zoom/pan AND the colour
        levels exactly as they are.

        Holding the levels is the whole point — set_image() auto-ranges to each
        array, so two genuinely different disparity maps would each be scaled to
        their own min/max and look nearly IDENTICAL. Comparing on a shared scale
        is what makes the difference visible (and honest).
        """
        if arr is None:
            return
        rng = self.vb.viewRange()
        lv = self._levels
        self._arr = arr
        if self._scalar:
            self.img.setImage(arr.astype(np.float32), autoLevels=False)
            if lv is not None:
                self._levels = lv
                self.img.setLevels(list(lv))
        else:
            self._set_rgb(arr)
        self.vb.setRange(xRange=rng[0], yRange=rng[1], padding=0)

    def clear(self) -> None:
        self._arr = None
        self.img.clear()
        if hasattr(self, "model_bar"):
            self.model_bar.clear()

    # ------------------------------------------------------------- stereo pair
    def set_pair(self, left: np.ndarray, right: np.ndarray | None) -> None:
        """Load a left/right stereo pair (Input tab). Shows the left image and
        fits it; the Left/Right toggle then swaps between them WITHOUT changing
        zoom/pan, so you can flip back and forth to see the stereo shift."""
        self._left, self._right = left, right
        self._side = "left"
        if hasattr(self, "btn_left"):
            self.btn_left.setChecked(True)
            self.btn_right.setEnabled(right is not None)
        self.set_image(left)          # fits the view (fresh load)
        self._update_side_lbl()
        self._rebuild_guides()        # re-place row guides for this image's height
        if hasattr(self, "roi"):
            # Re-clamp the crop into THIS image and re-announce it. Two cases need
            # it: the box was switched on before any pair was loaded (so no size
            # was known and nothing was ever emitted), and a pair of a DIFFERENT
            # size arriving under a box that now hangs off the edge. Same size and
            # position re-emits an identical rectangle, which the window treats as
            # no change — so a measured Δ survives an ordinary pair reload.
            self.roi.setVisible(self.roi_chk.isChecked())
            self._sync_roi()

    # ---------------------------------------------------------------- sites
    #: marker colours — pins warm, references cool, so a glance says which is which
    SITE_RGB = {"pin": "#f4883f", "ref": "#5ac8fa"}

    def _mark_kind(self) -> str:
        """'pin' | 'ref' | '' — what a click on the image would mark."""
        if not hasattr(self, "mark_combo"):
            return ""
        i = self.mark_combo.currentIndex()
        return {1: "pin", 2: "ref"}.get(i, "")

    def add_site(self, kind: str, x: int, y: int, name: str = "") -> dict:
        """Record a marked site (LEFT-image pixels) and draw it."""
        n = sum(1 for s in self._sites if s["kind"] == kind) + 1
        site = {"kind": kind, "x": int(x), "y": int(y),
                "name": name or (f"Pin {n}" if kind == "pin" else f"Ref {n}")}
        self._sites.append(site)
        self._redraw_sites()
        return site

    def clear_sites(self) -> None:
        self._sites = []
        self._redraw_sites()

    def sites(self) -> list:
        return [dict(s) for s in self._sites]

    def set_sites(self, sites) -> None:
        """Restore saved sites WITHOUT re-emitting — this is not a user edit."""
        self._sites = [{"kind": s.get("kind", "pin"), "x": int(s["x"]),
                        "y": int(s["y"]), "name": s.get("name", "")}
                       for s in (sites or []) if "x" in s and "y" in s]
        self._redraw_sites()

    def _redraw_sites(self) -> None:
        for t in self._site_texts:
            self.vb.removeItem(t)
        self._site_texts = []
        if not self._sites:
            self._site_dots.setData([], [])
            self.sites_lbl.setText("no sites")
            return
        spots = [{"pos": (s["x"], s["y"]), "brush": pg.mkBrush(self.SITE_RGB[s["kind"]]),
                  "symbol": "o" if s["kind"] == "pin" else "s"} for s in self._sites]
        self._site_dots.setData(spots)
        for s in self._sites:
            t = pg.TextItem(s["name"], color=self.SITE_RGB[s["kind"]], anchor=(0, 1.2))
            t.setPos(s["x"], s["y"])
            t.setZValue(26)
            self.vb.addItem(t, ignoreBounds=True)
            self._site_texts.append(t)
        npins = sum(1 for s in self._sites if s["kind"] == "pin")
        nref = len(self._sites) - npins
        plural = "s" if npins != 1 else ""
        self.sites_lbl.setText(f"{npins} pin{plural} · {nref} ref")

    def set_sites_enabled(self, on: bool) -> None:
        """Lock marking while a batch owns the geometry."""
        if hasattr(self, "mark_combo"):
            self.mark_combo.setEnabled(on)
            self.sites_clear.setEnabled(on)
            if not on:
                self.mark_combo.setCurrentIndex(0)

    # ------------------------------------------------------------------- ROI
    def _img_size(self):
        a = self._left if self._left is not None else self._arr
        return (a.shape[1], a.shape[0]) if a is not None else None

    def _toggle_roi(self, on: bool) -> None:
        """Turn the crop on/off. Switching on with no box yet drops a sensible
        one in the middle rather than making the user hunt for a 1-px handle."""
        self.shift_btn.setEnabled(on)
        size = self._img_size()
        if on and size is not None:
            W, H = size
            w = max(32, (min(W, H) // 3 // 32) * 32)
            self.roi.setPos([(W - w) // 2, (H - w) // 2], finish=False)
            self.roi.setSize([w, w], finish=False)
        self.roi.setVisible(on and self._side == "left")
        self._sync_roi()

    def _sync_roi(self) -> None:
        """Snap the box to a whole 32-px grid inside the image and announce it.

        Snapping here rather than at run time keeps what you SEE and what gets
        cropped the same rectangle — rounding it silently later would measure a
        region the box never covered."""
        if not self.roi_chk.isChecked():
            self.roi_lbl.setText("")
            self.roiChanged.emit(None)
            return
        size = self._img_size()
        if size is None:
            self.roi_lbl.setText("load a pair first")
            self.roiChanged.emit(None)
            return
        W, H = size
        p, s = self.roi.pos(), self.roi.size()
        w = int(max(32, min(round(s[0] / 32) * 32, (W // 32) * 32)))
        h = int(max(32, min(round(s[1] / 32) * 32, (H // 32) * 32)))
        x0 = int(max(0, min(round(p[0] / 32) * 32, W - w)))
        y0 = int(max(0, min(round(p[1] / 32) * 32, H - h)))
        self.roi.blockSignals(True)      # writing back must not re-enter this slot
        self.roi.setPos([x0, y0], finish=False)
        self.roi.setSize([w, h], finish=False)
        self.roi.blockSignals(False)
        self.roi_lbl.setText(f"ROI {w}×{h} @ {x0},{y0}")
        self.roiChanged.emit((x0, y0, w, h))

    def set_roi(self, roi) -> None:
        """Restore a saved ROI (from settings) without re-emitting a change."""
        self.roi_chk.blockSignals(True)
        self.roi_chk.setChecked(roi is not None)
        self.roi_chk.blockSignals(False)
        self.shift_btn.setEnabled(roi is not None)
        if roi is not None:
            x0, y0, w, h = (int(v) for v in roi)
            self.roi.blockSignals(True)
            self.roi.setPos([x0, y0], finish=False)
            self.roi.setSize([w, h], finish=False)
            self.roi.blockSignals(False)
            self.roi_lbl.setText(f"ROI {w}×{h} @ {x0},{y0}")
        else:
            self.roi_lbl.setText("")
        self.roi.setVisible(roi is not None and self._side == "left")

    def set_roi_note(self, text: str) -> None:
        self.roi_lbl.setText(text)

    def set_roi_enabled(self, on: bool) -> None:
        """Lock the crop controls (a running batch owns the geometry)."""
        self.roi_chk.setEnabled(on)
        self.shift_btn.setEnabled(on and self.roi_chk.isChecked())
        self.roi.translatable = on
        for hnd in self.roi.getHandles():
            hnd.setVisible(on)

    def _toggle_guides(self, on: bool) -> None:
        self._rebuild_guides()

    def _rebuild_guides(self) -> None:
        """Horizontal reference lines across the image — a correctly rectified pair
        keeps a feature on the same line in Left and Right (blink to check)."""
        for ln in self._guides:
            self.vb.removeItem(ln)
        self._guides = []
        if not (hasattr(self, "guide_chk") and self.guide_chk.isChecked()):
            return
        if self._arr is None:
            return
        h = self._arr.shape[0]
        pen = pg.mkPen((90, 200, 255, 120), width=1, style=Qt.DashLine)
        for i in range(1, 12):
            ln = pg.InfiniteLine(pos=h * i / 12.0, angle=0, movable=False, pen=pen)
            ln.setZValue(9)
            self.vb.addItem(ln, ignoreBounds=True)
            self._guides.append(ln)

    def _show_side(self, side: str) -> None:
        arr = self._left if side == "left" else self._right
        if arr is None:
            return
        rng = self.vb.viewRange()     # capture current zoom/pan
        self._side = side
        self._arr = arr
        self._set_rgb(arr)            # fixed levels — no per-side contrast stretch
        self.vb.setRange(xRange=rng[0], yRange=rng[1], padding=0)  # restore exactly (blink)
        # The ROI is defined on the LEFT image; the right crop sits Δ px away, so
        # drawing the same rectangle over the right view would point at the wrong
        # pixels. Hide it there rather than show a lie.
        if hasattr(self, "roi"):
            self.roi.setVisible(self.roi_chk.isChecked() and side == "left")
        self._update_side_lbl()

    def _update_side_lbl(self) -> None:
        if hasattr(self, "side_lbl") and self._arr is not None:
            h, w = self._arr.shape[:2]
            self.side_lbl.setText(f"{self._side}  ·  {w}×{h}")

    # ------------------------------------------------------------- levels
    def _valid(self) -> np.ndarray | None:
        if self._arr is None:
            return None
        a = self._arr
        if not self._scalar:
            return a
        if a.size > 2_000_000:          # subsample big maps so percentile stays snappy (4K)
            a = a[::4, ::4]
        return a[np.isfinite(a) & (a > 0)]

    def _auto_levels(self) -> None:
        v = self._valid()
        if v is None or v.size == 0:
            return
        lo, hi = np.percentile(v, [2, 98])
        if hi <= lo:
            hi = lo + 1e-3
        self._levels = (float(lo), float(hi))
        self.img.setLevels([lo, hi])
        self.range_lbl.setText(f"{lo:.{self._dec}f} – {hi:.{self._dec}f} {self._unit}".strip())

    def set_unit(self, unit: str, decimals: int | None = None) -> None:
        """Relabel the scalar readout (e.g. a mm/m switch). The underlying array
        is unchanged — the caller rescales it separately — so this only refreshes
        the text of the range label and future hover readouts."""
        self._unit = unit
        if decimals is not None:
            self._dec = int(decimals)
        if self._scalar and self._arr is not None:
            lo, hi = self._levels
            self.range_lbl.setText(f"{lo:.{self._dec}f} – {hi:.{self._dec}f} {self._unit}".strip())

    def _apply_cmap(self, name: str) -> None:
        self.img.setLookupTable(build_lut(name))

    def render_rgb(self) -> np.ndarray | None:
        """The current scalar map as RGB uint8 — same colormap and levels as the
        view. Invalid pixels (non-finite or <=0) are forced to BLACK for a clean
        export; note the live view instead maps them to the colormap's low end
        (or transparent), so exports of invalid regions differ from the screen."""
        if self._arr is None:
            return None
        if not self._scalar:
            return np.ascontiguousarray(self._arr[..., :3]).astype(np.uint8)
        a = self._arr.astype(np.float32)
        lo, hi = self._levels
        lut = build_lut(self.cmap.currentText())
        invalid = ~(np.isfinite(a) & (a > 0))
        # Mask NaN BEFORE the index cast: np.clip passes NaN through unchanged and
        # astype(intp) turns it into a huge negative index → IndexError. A model can
        # emit NaN disparity (e.g. mixed-precision overflow); the live view tolerates
        # it, so export must too — those pixels go black like every invalid one.
        a = np.where(invalid, lo, a)
        # match pyqtgraph makeARGB: idx = clip(floor((v-lo)*256/(hi-lo)), 0, 255)
        idx = np.clip((a - lo) * (256.0 / max(hi - lo, 1e-6)), 0, 255).astype(np.intp)
        rgb = lut[idx]
        rgb[invalid] = 0
        return np.ascontiguousarray(rgb).astype(np.uint8)

    # ------------------------------------------------------------- hover
    def _on_move(self, pos) -> None:
        if self._arr is None or not self.vb.sceneBoundingRect().contains(pos):
            self.readout.hide()
            self.vline.hide()
            self.hline.hide()
            return
        px = self._pixel_at(pos)
        if px is None:
            self.readout.hide()
            return
        x, y = px
        pt = self.vb.mapSceneToView(pos)   # float position — the crosshair is sub-pixel
        self.vline.setPos(pt.x())
        self.hline.setPos(pt.y())
        self.vline.show()
        self.hline.show()
        if self._scalar:
            val = float(self._arr[y, x])
            vtxt = ("—" if (not np.isfinite(val) or val <= 0)
                    else f"{val:.{self._dec}f} {self._unit}".strip())
            self.readout.setText(f"({x}, {y})   {vtxt}")
        else:
            r, g, b = self._arr[y, x][:3]
            self.readout.setText(f"({x}, {y})   rgb {r},{g},{b}")
        self.readout.adjustSize()
        self.readout.show()
        self.hovered.emit(x, y)


class ViewerStack(QTabWidget):
    """Input · Disparity · Depth · 3D Cloud · Repeatability · Compare."""

    hovered = Signal(int, int)
    pixelClicked = Signal(int, int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.input_view = ImageView2D(scalar=False, pair=True)
        self.disp_view = ImageView2D(scalar=True, unit="px")
        self.depth_view = ImageView2D(scalar=True, unit="mm", dec=UNIT_DECIMALS["mm"])
        self.cloud_view = WebCloudView()
        self.repeat_view = RepeatabilityView()
        self.compare_view = ComparePanel()
        self.depth_view.cmap.setCurrentText("viridis")

        self.addTab(self.input_view, "Input")
        self.addTab(self.disp_view, "Disparity")
        self.addTab(self.depth_view, "Depth")
        self.addTab(self.cloud_view, "3D Cloud")
        self.addTab(self.repeat_view, "Repeatability")
        self.addTab(self.compare_view, "Compare")
        self.setTabToolTip(0, "The left image you loaded.")
        self.setTabToolTip(1, "Stereo disparity — how far each pixel shifts between the two "
                              "views. Warmer = closer. Hover to read a value.")
        self._set_depth_tip("mm")
        self.setTabToolTip(3, "Interactive 3D reconstruction (needs calibration). Drag to "
                              "orbit, scroll to zoom.")
        self.setTabToolTip(4, "Log the pin heights across many captures and see the spread "
                              "(mean · σ · range) per pin — your repeatability study.")
        self.setTabToolTip(5, "Run several models on the same pair with settings you choose, "
                              "and compare their results side by side.")

        # only the disparity/depth maps drive the value probe and the click-to-place
        # — the Input tab is full-res RGB (and may be the right image), so its
        # coords don't index the working-scale disparity/depth arrays
        for v in (self.disp_view, self.depth_view):
            v.hovered.connect(self.hovered)
            v.pixelClicked.connect(self.pixelClicked)

    def _set_depth_tip(self, unit: str) -> None:
        self.setTabToolTip(2, f"Real-world depth in {unit} (needs calibration). Hover to read "
                              "the distance at any pixel.")

    def set_units(self, unit: str) -> None:
        """Relabel depth readout + cloud grid for a mm/m switch (data rescaled by
        the caller)."""
        self.depth_view.set_unit(unit, UNIT_DECIMALS.get(unit, 2))
        self.cloud_view.set_units(unit)
        self.compare_view.set_units(unit, UNIT_DECIMALS.get(unit, 2))
        self.repeat_view.set_units(unit)
        self._set_depth_tip(unit)

    def show_input(self, left: np.ndarray, right: np.ndarray | None = None) -> None:
        self.input_view.set_pair(left, right)

    def show_result(self, result, focus: bool = True) -> None:
        # input view keeps the loaded full-res pair (set on load); don't overwrite
        # it with the working-scale left frame from the result
        self.disp_view.set_image(result.disp)
        if result.depth is not None:
            self.depth_view.set_image(result.depth)
        else:
            self.depth_view.clear()
        # focus=False during a comparison: each model landing would otherwise yank
        # the user off the Compare tab they are watching fill in.
        if focus:
            self.setCurrentWidget(self.disp_view)

    def show_cloud(self, cloud, reset_view: bool = False) -> None:
        if cloud is None:
            self.cloud_view.clear()
        else:
            self.cloud_view.set_cloud(
                cloud.points, cloud.colors,
                origin=getattr(cloud, "origin", None),
                reliable=getattr(cloud, "reliable", None),
                reset_view=reset_view,
            )
