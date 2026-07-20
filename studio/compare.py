"""The Compare tab — one column per model: how it did, and a way in to its settings.

The card REPORTS, the panel EDITS. Click a column and the normal right-hand
Inference panel points at that model, which remembers every model's settings
separately — so the editor you already know is the only editor there is, and no
knob exists in two places to disagree with itself. Clicking loads NOTHING; the
engine is untouched until Run comparison, so setting up every model is free.

What is SHARED and what is PER-MODEL is a deliberate split: input scale,
calibration and the point-cloud settings are shared — they define the scene, and
varying them would mean the models weren't solving the same problem. Everything a
model does to that scene internally is its own.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QFrame, QHBoxLayout, QLabel,
                               QPushButton, QScrollArea, QVBoxLayout, QWidget)

from .backends import BACKENDS
from .dtypes import UNIT_DECIMALS
from .widgets import SectionLabel

# (stats key, row label, which way is better: -1 lower, +1 higher, 0 = no winner)
#
# Only time / VRAM / valid% get a winner marked. Median depth SHOULD agree across
# models (it's a sanity check, not a contest), point count is not a virtue, and
# plane σ deliberately has no winner — a model that over-smooths real detail
# scores low on it, so marking a "best" would actively mislead.
_STATS = [
    ("net_s", "time", -1),
    ("peak_gb", "peak VRAM", -1),
    ("valid_pct", "valid", +1),
    ("med_depth", "median depth", 0),
    ("plane_rms", "plane σ", 0),
    ("n_pts", "points", 0),
]

PLANE_TIP = (
    "Spread of the points around a best-fit plane, over the flattest 80% so that "
    "components and edges don't dominate it.\n\n"
    "• Only meaningful when your subject IS flat (a bare-ish PCB).\n"
    "• Compare it BETWEEN models on one scene — it is not an absolute accuracy figure.\n"
    "• Lower is smoother, which is usually less noise — but a model that over-smooths "
    "real detail also scores low, so check the map too. That's why no winner is marked.")

_TIPS = {
    "net_s": "How long the network itself took — excludes loading the model.",
    "peak_gb": "Peak GPU memory this run RESERVED from the driver — memory nothing "
               "else on the card could use while it ran.\n\n"
               "This is what decides whether a run fits, so compare it against your "
               "whole card, not against what's free: other apps are already holding "
               "some of it (see the VRAM readout, bottom right).\n\n"
               "Going over does NOT fail cleanly. The Windows driver backs the "
               "overflow with system RAM over PCIe instead of erroring, so the run "
               "silently takes minutes instead of seconds. If this number is close to "
               "your card's total, lower Input scale.",
    "valid_pct": "Share of pixels the model produced a disparity for.",
    "med_depth": "Median distance over the valid pixels. All models should broadly AGREE "
                 "here — a model that doesn't is the one to distrust.",
    "plane_rms": PLANE_TIP,
    "n_pts": "Points in the 3D cloud after the shared z-range / denoise settings. "
             "More is not automatically better.",
}


def _rule() -> QFrame:
    f = QFrame()
    f.setObjectName("CardRule")
    f.setFixedHeight(1)
    return f


def _fmt(skey: str, val, unit: str, dec: int) -> str:
    if val is None:
        return "—"
    if skey == "net_s":
        return f"{val:.2f} s"
    if skey == "peak_gb":
        return f"{val:.1f} GB" if val else "—"
    if skey == "valid_pct":
        return f"{val:.1f} %"
    if skey == "n_pts":
        return f"{int(val):,}" if val else "—"
    return f"{val:.{dec}f} {unit}"          # med_depth / plane_rms are unit-bearing


class ModelCard(QFrame):
    """One model's column: what it scored, and a click-target for its settings.

    Holds no settings widgets of its own and no longer restates them either — the
    numbers are the reason you're on this tab, and the settings are one click away
    in the panel that owns them."""

    changed = Signal()              # included / excluded
    editRequested = Signal(str)     # "point the settings panel at me"
    showRequested = Signal(str)     # "show me this model's result"

    def __init__(self, spec, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("ModelCard")
        self.spec = spec
        self.key = spec.key
        self.setFixedWidth(252)
        self.setProperty("shown", "false")
        self.setProperty("editing", "false")
        available, reason = spec.availability()
        self._available = available
        self._running = False       # a sweep is in flight — the card is inert
        if available:
            self.setCursor(Qt.PointingHandCursor)
            self.setToolTip(
                f"Show {spec.display_name}'s settings in the Inference panel.\n\n"
                "Nothing loads — this only swaps which model the panel is editing, so "
                "you can set up every model in seconds. Run comparison loads them.")

        name, _, sub = spec.display_name.partition("·")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(13, 12, 13, 12)
        lay.setSpacing(8)

        self.check = QCheckBox(name.strip())
        self.check.setObjectName("CardTitle")
        self.check.setChecked(available)
        self.check.setEnabled(available)
        self.check.setToolTip(
            "Include this model in the comparison." if available
            else f"Unavailable — {reason}")
        self.check.toggled.connect(lambda *_: self.changed.emit())
        lay.addWidget(self.check)

        self.sub_lbl = QLabel(sub.strip() if available else f"⚠  {reason}")
        self.sub_lbl.setProperty("role", "muted")
        self.sub_lbl.setStyleSheet("font-size:11px;")
        self.sub_lbl.setWordWrap(True)
        self.sub_lbl.setToolTip(spec.description)
        lay.addWidget(self.sub_lbl)

        # Which weights ran — the one identifying fact that ISN'T in the panel, and
        # the only thing left here that a model's settings could be confused with.
        self.ckpt_lbl = QLabel("")
        self.ckpt_lbl.setProperty("role", "muted")
        self.ckpt_lbl.setStyleSheet("font-size:10px;")
        self.ckpt_lbl.setWordWrap(True)
        lay.addWidget(self.ckpt_lbl)

        # --- results ---
        lay.addWidget(_rule())
        lay.addWidget(SectionLabel("Result"))
        self.status = QLabel("not run yet")
        self.status.setProperty("role", "muted")
        self.status.setStyleSheet("font-size:11px;")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)

        self._vals: dict = {}
        for skey, label, _better in _STATS:
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            lbl = QLabel(label)
            lbl.setProperty("role", "muted")
            lbl.setStyleSheet("font-size:11px;")
            val = QLabel("—")
            val.setProperty("role", "value")
            val.setStyleSheet("font-size:11px;")
            val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            tip = _TIPS.get(skey, "")
            if tip:
                lbl.setToolTip(tip)
                val.setToolTip(tip)
            row.addWidget(lbl)
            row.addStretch(1)
            row.addWidget(val)
            lay.addLayout(row)
            self._vals[skey] = val

        self.show_btn = QPushButton("Show")
        self.show_btn.setEnabled(False)
        self.show_btn.setToolTip("Show this model's disparity, depth and 3D cloud "
                                 "in the other tabs.")
        self.show_btn.clicked.connect(lambda: self.showRequested.emit(self.key))
        lay.addWidget(self.show_btn)
        lay.addStretch(1)

    # ------------------------------------------------------------- settings
    @property
    def available(self) -> bool:
        return self._available

    @property
    def included(self) -> bool:
        return self._available and self.check.isChecked()

    def mousePressEvent(self, e) -> None:
        """The whole card is the way in to this model's settings.

        The tick box and Show are real controls and consume their own clicks before
        this ever runs, so ticking a model can't also re-point the panel. Inert
        mid-sweep: re-pointing then would only reach the models not yet run."""
        if self._available and not self._running:
            self.editRequested.emit(self.key)
        super().mousePressEvent(e)

    def set_ckpt(self, ckpt_label: str) -> None:
        self.ckpt_lbl.setText(f"weights: {ckpt_label}" if ckpt_label else "")

    def set_running(self, running: bool) -> None:
        self._running = running
        self.setCursor(Qt.ArrowCursor if running or not self._available
                       else Qt.PointingHandCursor)

    def set_editing(self, editing: bool) -> None:
        """Mark the model the settings panel is pointed at. With the Edit button
        gone this highlight is the ONLY thing saying which model you're editing, so
        it has to be unmissable — see the [editing] rule in theme.py."""
        self.setProperty("editing", "true" if editing else "false")
        self.style().unpolish(self)
        self.style().polish(self)

    # -------------------------------------------------------------- results
    def set_status(self, text: str) -> None:
        self.status.setText(text)

    def set_shown(self, shown: bool) -> None:
        self.setProperty("shown", "true" if shown else "false")
        self.style().unpolish(self)
        self.style().polish(self)

    def render_stats(self, stats: dict | None, unit: str, dec: int,
                     best: dict | None = None) -> None:
        """Numbers only — whether Show is clickable is the panel's call, since it
        depends on the comparison being finished as well as on having a result."""
        best = best or {}
        for skey, _label, _better in _STATS:
            val = self._vals[skey]
            val.setText("—" if not stats else _fmt(skey, stats.get(skey), unit, dec))
            val.setProperty("stat", "best" if best.get(skey) else "")
            val.style().unpolish(val)
            val.style().polish(val)

    def clear_results(self) -> None:
        self.render_stats(None, "", 2)
        self.show_btn.setEnabled(False)
        self.set_status("not run yet")
        self.set_shown(False)


class ComparePanel(QWidget):
    """The Compare tab: which models, what each will run with, Run, results."""

    runRequested = Signal()
    editRequested = Signal(str)
    showRequested = Signal(str)
    changed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._unit, self._dec = "mm", UNIT_DECIMALS["mm"]
        self._stats: dict = {}
        self._running = False

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 12, 16, 12)
        lay.setSpacing(10)

        head = QLabel("Run each ticked model on the loaded pair, then flip between "
                      "the results in the Disparity, Depth and 3D Cloud tabs.")
        head.setWordWrap(True)
        lay.addWidget(head)

        # One line, not a paragraph. The rest is a tooltip: the split matters, but
        # it's something you read once, and the columns are what you came for.
        shared = QLabel("Click a card to set that model up — scale, calibration and "
                        "cloud settings are shared by all of them.")
        shared.setProperty("role", "muted")
        shared.setStyleSheet("font-size:11px;")
        shared.setWordWrap(True)
        shared.setToolTip(
            "Input scale, calibration and the point-cloud settings are SHARED — they "
            "define the scene, and varying them between models would mean they "
            "weren't solving the same problem. Set them in the side panels.\n\n"
            "Everything a model does internally is its own: click its card and the "
            "Inference panel edits that model. Nothing loads until Run comparison.")
        lay.addWidget(shared)

        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(12)
        self.cards: dict = {}
        for spec in BACKENDS.values():
            card = ModelCard(spec)
            card.showRequested.connect(self.showRequested)
            card.editRequested.connect(self.editRequested)
            card.changed.connect(self._on_card_changed)
            rl.addWidget(card, 0, Qt.AlignTop)
            self.cards[spec.key] = card
        rl.addStretch(1)

        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setWidget(row)
        lay.addWidget(sa, 1)

        foot = QHBoxLayout()
        foot.setContentsMargins(0, 0, 0, 0)
        self.hint = QLabel("")
        self.hint.setProperty("role", "muted")
        self.hint.setStyleSheet("font-size:11px;")
        self.run_btn = QPushButton("▶  Run comparison")
        self.run_btn.setObjectName("Accent")
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(lambda: self.runRequested.emit())
        foot.addWidget(self.hint, 1)
        foot.addWidget(self.run_btn)
        lay.addLayout(foot)

        self._refresh_run_tip()

    # ------------------------------------------------------------ accessors
    def selected_keys(self) -> list:
        return [k for k, c in self.cards.items() if c.included]

    def _on_card_changed(self) -> None:
        self._refresh_run_tip()
        self.changed.emit()

    def _refresh_run_tip(self) -> None:
        n = len(self.selected_keys())
        self.run_btn.setToolTip(
            f"Run {n} models one after another on this pair, each with the settings "
            "shown in its column, and keep every result so you can flip between "
            "them.\n\nOnly one model is held in GPU memory at a time, so this "
            "loads them one by one — that's most of the wall-clock."
            if n >= 2 else "Tick at least two models to compare.")

    # ---------------------------------------------------------------- state
    def set_ckpt(self, key: str, ckpt_label: str) -> None:
        c = self.cards.get(key)
        if c is not None:
            c.set_ckpt(ckpt_label)

    def set_editing(self, key: str | None) -> None:
        for k, c in self.cards.items():
            c.set_editing(k == key)

    def set_run_state(self, enabled: bool, reason: str = "") -> None:
        """The window decides when a run is possible (pair loaded, engine idle,
        ≥2 models ticked) and says why not when it isn't."""
        self.run_btn.setEnabled(enabled)
        self.hint.setText(reason)

    def set_running(self, running: bool) -> None:
        """Lock the tab while a comparison is in flight — re-pointing the settings
        panel mid-sweep would apply to only the models not yet run."""
        self._running = running
        for c in self.cards.values():
            c.check.setEnabled(not running and c.available)
            c.set_running(running)
        self._refresh_show_buttons()

    def _refresh_show_buttons(self) -> None:
        """Show is clickable only for a model that HAS a result, and only once the
        sweep is over — blinking to a model mid-comparison would be overwritten by
        the next one landing anyway."""
        for key, c in self.cards.items():
            c.show_btn.setEnabled(bool(self._stats.get(key)) and not self._running)

    def set_shown(self, key: str | None) -> None:
        for k, c in self.cards.items():
            c.set_shown(k == key)

    def set_status(self, key: str, text: str) -> None:
        c = self.cards.get(key)
        if c is not None:
            c.set_status(text)

    def set_units(self, unit: str, dec: int) -> None:
        self._unit, self._dec = unit, int(dec)
        self._render()

    def show_stats(self, key: str, stats: dict) -> None:
        # a COPY on purpose: the window owns its own mstats and rescales those on a
        # unit switch — sharing the dict would double-apply the factor
        self._stats[key] = dict(stats)
        self._render()

    def rescale_stats(self, factor: float) -> None:
        """A mm⇄m switch rescales the unit-bearing numbers in place (the window
        does the same to the results themselves — nothing re-runs)."""
        for s in self._stats.values():
            for k in ("med_depth", "plane_rms"):
                if s.get(k) is not None:
                    s[k] = s[k] * factor
        self._render()

    def clear_results(self) -> None:
        self._stats = {}
        for c in self.cards.values():
            c.clear_results()

    def _render(self) -> None:
        # mark the winner only where "better" is unambiguous, and only when there
        # is something to win — one result is not a comparison
        best: dict = {}
        for skey, _label, better in _STATS:
            if not better:
                continue
            # `is not None`, not truthiness: a model that scored 0.0 valid pixels
            # LOST, and dropping it from the ranking entirely is not the same
            # thing. peak_gb is the one exception — 0.0 there means UNMEASURED
            # (CPU run / no CUDA counter), which _fmt already renders as "—", and
            # a dash must not win "peak VRAM" wearing the best-highlight.
            vals = [(k, s.get(skey)) for k, s in self._stats.items()
                    if s.get(skey) is not None
                    and not (skey == "peak_gb" and not s.get(skey))]
            if len(vals) < 2:
                continue
            ranked = sorted(vals, key=lambda kv: kv[1], reverse=better > 0)
            (wkey, wval), (_, rval) = ranked[0], ranked[1]
            # A win nobody can see in the numbers is not a win. Three models that all
            # scored 100.0% valid are TIED, but min()/max() silently hand the green
            # to whichever happens to be registered first — so the column crowned
            # FoundationStereo for drawing. Compare at the precision actually shown:
            # if the winner and the runner-up render identically, mark neither.
            if (_fmt(skey, wval, self._unit, self._dec)
                    == _fmt(skey, rval, self._unit, self._dec)):
                continue
            best[skey] = wkey
        for key, card in self.cards.items():
            s = self._stats.get(key)
            card.render_stats(s, self._unit, self._dec,
                              {sk: (bk == key) for sk, bk in best.items()})
        self._refresh_show_buttons()

    # ------------------------------------------------------------ persistence
    def values(self) -> dict:
        """Only what this tab owns — which models are ticked. The settings
        themselves belong to the Inference panel and are saved with it."""
        return {k: bool(c.check.isChecked()) for k, c in self.cards.items()}

    def restore(self, blob) -> None:
        if not isinstance(blob, dict):
            return
        for k, c in self.cards.items():
            if k in blob and c.available:
                c.check.setChecked(bool(blob[k]))
        self._refresh_run_tip()
