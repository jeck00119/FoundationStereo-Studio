"""Small custom widgets that carry the 'professional' feel:
an animated toggle switch, a float slider with live readout, and an
image drop/pick tile. All theme-aware via the active QPalette."""
from __future__ import annotations

import os

from PySide6.QtCore import (Property, QEasingCurve, QEvent, QObject, QPoint,
                            QPointF, QPropertyAnimation, QRect, QRectF, QSize, Qt,
                            Signal)
from PySide6.QtGui import QColor, QPainter, QPalette, QPixmap
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QFrame,
                               QGridLayout, QHBoxLayout, QLabel, QLayout,
                               QPushButton, QScrollArea, QSlider, QToolButton,
                               QVBoxLayout, QWidget)


def _lerp(a: QColor, b: QColor, t: float) -> QColor:
    return QColor(
        int(a.red() + (b.red() - a.red()) * t),
        int(a.green() + (b.green() - a.green()) * t),
        int(a.blue() + (b.blue() - a.blue()) * t),
    )


def set_tip(widget: QWidget, text: str) -> None:
    """Set a tooltip on a widget AND all its child widgets, so composite widgets
    (sliders, toggle rows, spin boxes) show the hint wherever you hover."""
    widget.setToolTip(text)
    for child in widget.findChildren(QWidget):
        child.setToolTip(text)


class _WheelGuard(QObject):
    """Event filter that stops the mouse wheel from ever nudging a control's value:
    the wheel is handed to the enclosing scroll area so the panel scrolls under the
    cursor instead. Change a value by dragging the slider or typing in the spin —
    scrolling past a control must never move it by accident. (An OPEN combo popup is
    a separate widget, so its list still scrolls normally.)"""

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.Type.Wheel:
            area = obj.parent()
            while area is not None and not isinstance(area, QScrollArea):
                area = area.parent()
            if area is not None:
                QApplication.sendEvent(area.viewport(), ev)   # scroll the panel instead
            return True                                       # …but never the control
        return False


_wheel_guard: "_WheelGuard | None" = None


def no_wheel(w: QWidget) -> QWidget:
    """Make a slider / spin box / combo ignore the mouse wheel (it scrolls the panel
    instead), so scrolling past a control never silently changes the setting under
    the cursor. Returns the widget, so it can wrap a construction expression."""
    global _wheel_guard
    if _wheel_guard is None:
        _wheel_guard = _WheelGuard()
    w.installEventFilter(_wheel_guard)
    return w


# --- point-cloud colour palette (shared by the 3D view and its legend) -------
# 0-255 RGB. Kept here, in the leaf widget module, so both the viewer and the
# web-backed cloud view import them from ONE place (and studio/web/cloud.html
# carries a hand-synced copy for the GPU side).
CLOUD_LEFT_RGB = (74, 144, 226)     # left eye  — cool blue
CLOUD_RIGHT_RGB = (245, 145, 60)    # right eye — warm orange
CLOUD_OK_RGB = (70, 200, 130)       # reliable  — green
CLOUD_BAD_RGB = (235, 80, 100)      # occluded  — red
CLOUD_COLOR_MODES = ["Photo", "Camera (L·R)", "Reliability", "Model"]
# one per model, in registry order, deliberately far apart in hue so two models'
# points can be told apart where they land almost on top of each other
CLOUD_MODEL_RGB = [
    (74, 144, 226), (245, 145, 60), (70, 200, 130), (200, 120, 235), (235, 200, 70),
]


class CloudLegend(QWidget):
    """Colour key for the model overlay — and the show/hide for each model.

    One widget for both jobs on purpose: a colour key you can't act on makes you
    hunt for which model owns a colour, and toggles without a key make you guess.
    Hidden entirely unless an overlay is on screen.
    """

    toggled = Signal(int, bool)      # (model index, visible)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._lay = QHBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(10)
        self._boxes: list = []
        self.hide()

    def set_models(self, names: list) -> None:
        for b in self._boxes:
            self._lay.removeWidget(b)
            b.deleteLater()
        self._boxes = []
        for i, name in enumerate(names):
            r, g, b_ = CLOUD_MODEL_RGB[i % len(CLOUD_MODEL_RGB)]
            cb = QCheckBox(name)
            cb.setChecked(True)
            # the label IS the swatch — colouring the text keys the model to its
            # points without spending a second widget on a square
            cb.setStyleSheet(f"color: rgb({r},{g},{b_}); font-weight:600;")
            cb.setToolTip(f"Show or hide {name}'s points in the overlay.")
            cb.toggled.connect(lambda on, idx=i: self.toggled.emit(idx, on))
            self._lay.addWidget(cb)
            self._boxes.append(cb)
        self.setVisible(bool(names))

    def clear(self) -> None:
        self.set_models([])
        self.hide()


class ModelBar(QWidget):
    """Segmented 'which model am I looking at' selector — the blink comparator
    for a model comparison.

    One instance sits in each result view (disparity / depth / 3D). They are kept
    in sync by the window, so flipping model on one tab flips them all. Hidden
    entirely until a comparison has produced more than one result, so the normal
    single-model workflow looks exactly as before.
    """

    picked = Signal(str)   # backend key the user wants to look at

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._grp = QButtonGroup(self)
        self._grp.setExclusive(True)
        self._btns: dict[str, QPushButton] = {}
        self._lay = QHBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(4)
        self._title = QLabel("Model")
        self._title.setProperty("role", "muted")
        self._lay.addWidget(self._title)
        self.hide()

    def set_models(self, models: list, current: str | None = None) -> None:
        """models: [(key, short_label, tooltip)] — rebuilt whenever a comparison
        finishes. Fewer than 2 entries = nothing to compare, so stay hidden."""
        for b in list(self._btns.values()):
            self._grp.removeButton(b)
            self._lay.removeWidget(b)
            b.deleteLater()
        self._btns.clear()
        for key, label, tip in models:
            b = QPushButton(label)
            b.setObjectName("Seg")          # picks up the segmented QSS
            b.setCheckable(True)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, k=key: self.picked.emit(k))
            self._grp.addButton(b)
            self._lay.addWidget(b)
            self._btns[key] = b
        self.setVisible(len(models) > 1)
        if current:
            self.set_current(current)

    def set_current(self, key: str) -> None:
        """Reflect the shown model WITHOUT re-emitting (these bars mirror each
        other, so an echo here would loop)."""
        b = self._btns.get(key)
        if b is not None and not b.isChecked():
            b.blockSignals(True)
            b.setChecked(True)
            b.blockSignals(False)

    def clear(self) -> None:
        self.set_models([])


class ToggleSwitch(QCheckBox):
    """iOS-style animated switch (still a QCheckBox → .isChecked()/.toggled)."""

    def __init__(self, checked: bool = False, parent=None) -> None:
        super().__init__(parent)
        self.setChecked(checked)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(42, 24)
        self._offset = 1.0 if checked else 0.0
        self._anim = None
        self.toggled.connect(self._animate)

    def _animate(self, on: bool) -> None:
        self._anim = QPropertyAnimation(self, b"offset", self)
        self._anim.setDuration(150)
        self._anim.setStartValue(self._offset)
        self._anim.setEndValue(1.0 if on else 0.0)
        self._anim.setEasingCurve(QEasingCurve.InOutCubic)
        self._anim.start()

    def getOffset(self) -> float:
        return self._offset

    def setOffset(self, v: float) -> None:
        self._offset = v
        self.update()

    offset = Property(float, getOffset, setOffset)

    def sizeHint(self) -> QSize:
        return QSize(42, 24)

    def hitButton(self, pos) -> bool:
        return self.contentsRect().contains(pos)

    def paintEvent(self, _e) -> None:
        pal = self.palette()
        accent = pal.highlight().color()
        off_track = _lerp(pal.window().color(), pal.windowText().color(), 0.20)
        track = _lerp(off_track, accent, self._offset)
        if not self.isEnabled():
            track = _lerp(track, pal.window().color(), 0.5)

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        rad = r.height() / 2
        p.setPen(Qt.NoPen)
        p.setBrush(track)
        p.drawRoundedRect(r, rad, rad)

        d = r.height() - 6
        x = r.left() + 3 + self._offset * (r.width() - d - 6)
        p.setBrush(QColor("#FFFFFF"))
        p.drawEllipse(QRectF(x, r.top() + 3, d, d))
        p.end()


class StatSlider(QWidget):
    """Name + live value on top, slider below. Maps the int slider to a float
    range/step and emits floats."""

    valueChanged = Signal(float)

    def __init__(self, name, minv, maxv, value, step=1.0, fmt="{:.0f}",
                 suffix="", tip="", parent=None) -> None:
        super().__init__(parent)
        self._min, self._max, self._step = float(minv), float(maxv), float(step)
        self._fmt, self._suffix = fmt, suffix
        self._steps = max(1, int(round((self._max - self._min) / self._step)))

        self.name_lbl = QLabel(name)
        self.name_lbl.setProperty("role", "muted")
        self.val_lbl = QLabel("")
        self.val_lbl.setProperty("role", "value")
        self.val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, self._steps)
        no_wheel(self.slider)   # scrolling the panel must not drag the value

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(self.name_lbl)
        top.addStretch(1)
        top.addWidget(self.val_lbl)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(5)
        lay.addLayout(top)
        lay.addWidget(self.slider)

        self.slider.valueChanged.connect(self._on_slide)
        self.setValue(value)
        if tip:
            set_tip(self, tip)

    def _i2f(self, i: int) -> float:
        return round(self._min + i * self._step, 6)

    def _f2i(self, v: float) -> int:
        return int(round((v - self._min) / self._step))

    def _on_slide(self, i: int) -> None:
        v = self._i2f(i)
        self.val_lbl.setText(self._fmt.format(v) + self._suffix)
        self.valueChanged.emit(v)

    def value(self) -> float:
        return self._i2f(self.slider.value())

    def setValue(self, v: float) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(self._f2i(float(v)))
        self.slider.blockSignals(False)
        self.val_lbl.setText(self._fmt.format(self.value()) + self._suffix)

    def reconfigure(self, minv, maxv, value, step=None, fmt=None, suffix=None) -> None:
        """Change the range/step/format/suffix in place (e.g. a unit switch) and
        set a new value. Does NOT emit valueChanged — the slider is re-scaled, so
        callers apply the equivalent change themselves. The value is clamped to
        the new range by QSlider."""
        self._min, self._max = float(minv), float(maxv)
        if step is not None:
            self._step = float(step)
        if fmt is not None:
            self._fmt = fmt
        if suffix is not None:
            self._suffix = suffix
        self._steps = max(1, int(round((self._max - self._min) / self._step)))
        self.slider.blockSignals(True)
        self.slider.setRange(0, self._steps)
        self.slider.blockSignals(False)
        self.setValue(value)   # setValue already blocks signals + refreshes the label


IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".ppm", ".webp")


class ImageDrop(QFrame):
    """Clickable + drag-drop tile showing a thumbnail and filename."""

    clicked = Signal()
    fileDropped = Signal(str)

    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("ImageDrop")
        self.setAcceptDrops(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(60)
        self._title = title

        self.thumb = QLabel()
        self.thumb.setFixedSize(48, 40)
        self.thumb.setAlignment(Qt.AlignCenter)
        self.thumb.setObjectName("Thumb")

        self.name_lbl = QLabel(title)
        self.name_lbl.setProperty("role", "mono")
        self.sub_lbl = QLabel("click or drop an image")
        self.sub_lbl.setProperty("role", "muted")
        self.sub_lbl.setStyleSheet("font-size:11px;")

        text = QVBoxLayout()
        text.setContentsMargins(0, 0, 0, 0)
        text.setSpacing(1)
        text.addStretch(1)
        text.addWidget(self.name_lbl)
        text.addWidget(self.sub_lbl)
        text.addStretch(1)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(9, 8, 9, 8)
        lay.setSpacing(10)
        lay.addWidget(self.thumb)
        lay.addLayout(text, 1)
        # NOTE: styled via the global theme QSS (#ImageDrop / #Thumb) — never
        # call setStyleSheet() from changeEvent(PaletteChange): it re-emits
        # PaletteChange and recurses into a stack overflow.

    def set_image(self, path: str, pixmap: QPixmap, w: int, h: int) -> None:
        self.name_lbl.setText(os.path.basename(path))
        self.sub_lbl.setText(f"{w}×{h}")
        self.thumb.setPixmap(
            pixmap.scaled(48, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def reset(self) -> None:
        self.name_lbl.setText(self._title)
        self.sub_lbl.setText("click or drop an image")
        self.thumb.clear()

    def mousePressEvent(self, e) -> None:
        if e.button() == Qt.LeftButton:
            self.clicked.emit()

    def dragEnterEvent(self, e) -> None:
        urls = e.mimeData().urls()
        if urls and urls[0].toLocalFile().lower().endswith(IMG_EXT):
            e.acceptProposedAction()

    def dropEvent(self, e) -> None:
        path = e.mimeData().urls()[0].toLocalFile()
        if path.lower().endswith(IMG_EXT):
            self.fileDropped.emit(path)


class SectionLabel(QWidget):
    """Section header: uppercase title + an optional right-aligned status pill.
    kind='live'  -> the settings below update the current result instantly;
    kind='rerun' -> they need a Run to take effect."""

    def __init__(self, text: str, tag: str = "", kind: str = "", parent=None) -> None:
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        title = QLabel(text.upper())
        title.setProperty("role", "section")
        lay.addWidget(title)
        lay.addStretch(1)
        if tag:
            pill = QLabel(tag.upper())
            pill.setObjectName("Pill")
            pill.setProperty("pill", kind or "live")
            pill.setToolTip(
                "LIVE — these settings update the current result instantly, no re-run needed."
                if (kind or "live") == "live"
                else "NEEDS RUN — changing these takes effect only after you press Run."
            )
            lay.addWidget(pill)


class Collapsible(QWidget):
    """A subtle 'click to fold/unfold' section: a ▸/▾ header over a body you fill.

    Add content through ``body_layout()``. Used to tuck the precise numeric box
    position/size out of the way — it is the occasional exact-entry path, not the
    primary control (the 3D handles are), so it starts collapsed.
    """

    def __init__(self, title: str, expanded: bool = False, parent=None) -> None:
        super().__init__(parent)
        self._title = title
        self.header = QToolButton()
        self.header.setObjectName("Collapse")
        self.header.setCheckable(True)
        self.header.setChecked(expanded)
        self.header.setCursor(Qt.PointingHandCursor)
        self.header.setAutoRaise(True)
        self.header.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.header.setText(self._label(expanded))

        self.body = QWidget()
        self._body_lay = QVBoxLayout(self.body)
        self._body_lay.setContentsMargins(0, 4, 0, 2)
        self._body_lay.setSpacing(6)
        self.body.setVisible(expanded)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        lay.addWidget(self.header)
        lay.addWidget(self.body)
        self.header.toggled.connect(self._on_toggle)

    def _label(self, on: bool) -> str:
        # '&&' so a literal ampersand in the title isn't eaten as a QToolButton
        # mnemonic accelerator ("Position & size" would otherwise show "Position _size").
        return ("▾  " if on else "▸  ") + self._title.replace("&", "&&")

    def _on_toggle(self, on: bool) -> None:
        self.header.setText(self._label(on))
        self.body.setVisible(on)

    def body_layout(self) -> QVBoxLayout:
        return self._body_lay


class _ClickRow(QWidget):
    """A row that emits `clicked` on a left press anywhere on it — the whole header
    bar of a CollapsibleSection, so clicking the title (not just an arrow) folds it."""

    clicked = Signal()

    def mousePressEvent(self, e) -> None:
        if e.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(e)


class CollapsibleSection(QWidget):
    """A foldable panel section: a click-anywhere header (fold arrow + UPPERCASE
    title + optional LIVE / NEEDS-RUN pill) over a body filled via add()/add_layout().

    Drop-in for the old SectionLabel + loose widgets: same 'section' title style and
    Pill, but the whole category folds. ``toggled(bool)`` fires on a user fold so the
    panel can remember the open/closed state across restarts.
    """

    toggled = Signal(bool)

    def __init__(self, title: str, tag: str = "", kind: str = "",
                 expanded: bool = True, body_spacing: int = 13, parent=None) -> None:
        super().__init__(parent)
        self._expanded = expanded

        self.header = _ClickRow()
        self.header.setObjectName("SectionHead")
        self.header.setCursor(Qt.PointingHandCursor)
        hl = QHBoxLayout(self.header)
        hl.setContentsMargins(4, 3, 4, 3)
        hl.setSpacing(7)
        self._arrow = QLabel()
        self._arrow.setObjectName("SectionArrow")
        title_lbl = QLabel(title.upper())
        title_lbl.setProperty("role", "section")
        hl.addWidget(self._arrow)
        hl.addWidget(title_lbl)
        hl.addStretch(1)
        if tag:
            pill = QLabel(tag.upper())
            pill.setObjectName("Pill")
            pill.setProperty("pill", kind or "live")
            pill.setToolTip(
                "LIVE — these settings update the current result instantly, no re-run needed."
                if (kind or "live") == "live"
                else "NEEDS RUN — changing these takes effect only after you press Run.")
            hl.addWidget(pill)

        self.body = QWidget()
        self._body_lay = QVBoxLayout(self.body)
        self._body_lay.setContentsMargins(0, 0, 0, 0)
        self._body_lay.setSpacing(body_spacing)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        lay.addWidget(self.header)
        lay.addWidget(self.body)
        self.header.clicked.connect(self._on_click)
        self._sync()

    def _on_click(self) -> None:
        self.set_expanded(not self._expanded)
        self.toggled.emit(self._expanded)

    def _sync(self) -> None:
        self._arrow.setText("▾" if self._expanded else "▸")
        self.body.setVisible(self._expanded)

    def set_expanded(self, on: bool) -> None:
        self._expanded = bool(on)
        self._sync()

    def is_expanded(self) -> bool:
        return self._expanded

    def add(self, w: QWidget) -> None:
        self._body_lay.addWidget(w)

    def add_layout(self, lay) -> None:
        self._body_lay.addLayout(lay)

    def body_layout(self) -> QVBoxLayout:
        return self._body_lay


class MetricCard(QFrame):
    """A compact result card for a single measurement: a faint eyebrow (the metric
    name), one big headline value + its unit, an optional caption, and a right-aligned
    key/value table under a hairline. Swaps to a single muted line for hints/errors.

    All colour comes from role-based QSS (``role`` = section/muted/value + the
    #InfoCard/#CardHeadline/#InfoRule object names), so it tracks the dark/light theme
    with no hard-coded hex — and the table stays aligned however long the values get.
    """

    def __init__(self, hint: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("InfoCard")
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 11)
        root.setSpacing(6)

        self._eyebrow = QLabel()
        self._eyebrow.setProperty("role", "section")

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(6)
        self._value = QLabel()
        self._value.setObjectName("CardHeadline")
        self._unit = QLabel()
        self._unit.setObjectName("CardUnit")
        self._unit.setAlignment(Qt.AlignLeft | Qt.AlignBottom)
        head.addWidget(self._value)
        head.addWidget(self._unit)
        head.addStretch(1)
        self._head = head

        self._caption = QLabel()
        self._caption.setProperty("role", "muted")
        self._caption.setStyleSheet("font-size:11px;")
        self._caption.setWordWrap(True)

        self._rule = QFrame()
        self._rule.setObjectName("InfoRule")
        self._rule.setFixedHeight(1)

        self._grid = QGridLayout()
        self._grid.setContentsMargins(0, 1, 0, 0)
        self._grid.setHorizontalSpacing(10)
        self._grid.setVerticalSpacing(5)
        self._grid.setColumnStretch(0, 1)     # label column soaks up the slack…
        self._grid.setColumnStretch(1, 0)     # …so values sit flush to the right edge

        self._msg = QLabel()
        self._msg.setProperty("role", "muted")
        self._msg.setWordWrap(True)

        root.addWidget(self._eyebrow)
        root.addLayout(head)
        root.addWidget(self._caption)
        root.addWidget(self._rule)
        root.addLayout(self._grid)
        root.addWidget(self._msg)

        self.show_message(hint or "Pick points on the cloud to measure.")

    def _clear_grid(self) -> None:
        while self._grid.count():
            it = self._grid.takeAt(0)
            w = it.widget()
            if w is not None:
                w.deleteLater()

    def show_message(self, text: str) -> None:
        """Collapse to a single muted line — the empty/hint/error state."""
        self._clear_grid()
        for w in (self._eyebrow, self._value, self._unit, self._caption, self._rule):
            w.hide()
        self._msg.setText(text)
        self._msg.show()

    def show_result(self, eyebrow: str, value: str, unit: str = "",
                    rows=None, caption: str = "") -> None:
        """Render a measurement: EYEBROW, a big ``value`` + ``unit``, an optional
        ``caption``, then ``rows`` = list of (label, value[, kind]) where kind in
        {'', 'best', 'warn'} tints the value."""
        self._clear_grid()
        self._msg.hide()
        self._eyebrow.setText(eyebrow.upper()); self._eyebrow.show()
        self._value.setText(str(value)); self._value.show()
        self._unit.setText(unit); self._unit.setVisible(bool(unit))
        self._caption.setText(caption); self._caption.setVisible(bool(caption))
        rows = rows or []
        self._rule.setVisible(bool(rows))
        for i, row in enumerate(rows):
            label, val = str(row[0]), str(row[1])
            kind = row[2] if len(row) > 2 else ""
            lab = QLabel(label)
            lab.setProperty("role", "muted")
            v = QLabel(val)
            v.setProperty("role", "value")
            if kind:
                v.setProperty("stat", kind)
            v.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            v.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self._grid.addWidget(lab, i, 0, Qt.AlignLeft | Qt.AlignVCenter)
            self._grid.addWidget(v, i, 1)


class FlowLayout(QLayout):
    """Lays widgets left-to-right and wraps to the next row when the width runs
    out (Qt's classic FlowLayout). The 3D-view control strip uses it so it reflows
    onto a second line instead of overflowing when several models overlay at once.
    """

    def __init__(self, parent=None, margin: int = 0, spacing: int = 8) -> None:
        super().__init__(parent)
        self.setContentsMargins(margin, margin, margin, margin)
        self._spacing = spacing
        self._items: list = []

    def addItem(self, item) -> None:
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, i):
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._do(QRect(0, 0, width, 0), test=True)

    def setGeometry(self, rect) -> None:
        super().setGeometry(rect)
        self._do(rect, test=False)

    def sizeHint(self) -> QSize:
        return self.minimumSize()

    def minimumSize(self) -> QSize:
        s = QSize()
        for it in self._items:
            s = s.expandedTo(it.minimumSize())
        m = self.contentsMargins()
        s += QSize(m.left() + m.right(), m.top() + m.bottom())
        return s

    def _do(self, rect, test: bool) -> int:
        m = self.contentsMargins()
        x = rect.x() + m.left()
        y = rect.y() + m.top()
        right = rect.right() - m.right()
        line_h = 0
        for it in self._items:
            if it.isEmpty():          # a hidden widget (e.g. the model bar / legend
                continue              # in single-model mode) — reserve no row space
            w, h = it.sizeHint().width(), it.sizeHint().height()
            nx = x + w + self._spacing
            if nx - self._spacing > right and line_h > 0:   # wrap to next row
                x = rect.x() + m.left()
                y += line_h + self._spacing
                nx = x + w + self._spacing
                line_h = 0
            if not test:
                it.setGeometry(QRect(QPoint(x, y), QSize(w, h)))
            x = nx
            line_h = max(line_h, h)
        return y + line_h - rect.y() + m.bottom()
