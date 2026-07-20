"""Repeatability log — the measurement-system-analysis view.

The measure boxes give you a pin height per capture; this is where you find out
whether that height REPEATS. Click "Log" on each capture (or feed it a batch) and it
accumulates the per-pin readings, showing mean · σ · range · N per pin — the spread
that IS the repeatability. Heights are held in canonical metres so a mm⇄m switch just
re-labels the table; export drops the raw rows + the per-pin summary to CSV.
"""
from __future__ import annotations

import csv

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QFileDialog, QHBoxLayout, QHeaderView, QLabel,
                               QMessageBox, QPushButton, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from .dtypes import UNIT_DECIMALS
from .engine import UNIT_PER_M


def _num(text: str) -> QTableWidgetItem:
    it = QTableWidgetItem(text)
    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
    return it


class RepeatabilityView(QWidget):
    """Accumulates per-pin height readings and shows their spread. Public surface:
    set_pins() · add_record() · clear() · set_units() · count()."""

    logRequested = Signal()   # the "Log current reading" button (window does the measure)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._records: list = []   # [{"label": str, "vals": {pin: height_m | None}}]
        self._pins: list = []      # column order — pin names, in first-seen order
        self._unit = "mm"

        self.log_btn = QPushButton("＋  Log current reading")
        self.log_btn.setObjectName("Accent")
        self.log_btn.setToolTip("Snapshot the current pin heights (one per measure box) "
                                "into the table below as one capture.")
        self.log_btn.clicked.connect(self.logRequested)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setToolTip("Discard all logged readings and start a fresh study.")
        self.clear_btn.clicked.connect(self.clear)
        self.export_btn = QPushButton("Export CSV…")
        self.export_btn.setToolTip("Write the raw readings + the per-pin summary to a .csv.")
        self.export_btn.clicked.connect(self._export)
        self.count_lbl = QLabel("no readings yet")
        self.count_lbl.setProperty("role", "muted")

        bar = QHBoxLayout()
        bar.setSpacing(8)
        bar.addWidget(self.log_btn)
        bar.addWidget(self.clear_btn)
        bar.addWidget(self.export_btn)
        bar.addStretch(1)
        bar.addWidget(self.count_lbl)

        cap = QLabel("PER-PIN REPEATABILITY")
        cap.setProperty("role", "section")
        self.stat_table = QTableWidget(0, 7)
        self.stat_table.setObjectName("StatTable")
        self.stat_table.verticalHeader().setVisible(False)
        self.stat_table.verticalHeader().setDefaultSectionSize(30)
        self.stat_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.stat_table.setSelectionMode(QTableWidget.NoSelection)
        self.stat_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        cap2 = QLabel("READINGS")
        cap2.setProperty("role", "section")
        self.log_table = QTableWidget(0, 1)
        self.log_table.verticalHeader().setVisible(False)
        self.log_table.verticalHeader().setDefaultSectionSize(28)
        self.log_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.log_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(9)
        lay.addLayout(bar)
        lay.addWidget(cap)
        lay.addWidget(self.stat_table)
        lay.addWidget(cap2)
        lay.addWidget(self.log_table, 1)
        self._refresh()

    # ------------------------------------------------------------------ data
    def count(self) -> int:
        return len(self._records)

    def set_pins(self, names) -> None:
        """Preset the columns from the current boxes, so the table shows the right
        pins before anything is logged. New names are appended, never reordered."""
        changed = False
        for n in names:
            if n not in self._pins:
                self._pins.append(n)
                changed = True
        if changed:
            self._refresh()

    def add_record(self, label: str, vals: dict) -> None:
        """Append one capture. ``vals`` maps pin name -> height in METRES (or None
        where a box caught nothing).

        Appends ONE row + recomputes the small summary instead of rebuilding both
        tables: the full rebuild allocated O(N·pins) QTableWidgetItems per record,
        so a long clouds batch (QTimer-paced, no inference to hide behind) went
        quadratic and visibly stalled the GUI as the study grew. A record that
        introduces a NEW pin still rebuilds fully (the column set changed)."""
        rec = {"label": str(label), "vals": dict(vals)}
        new_pin = any(n not in self._pins for n in vals)
        for n in vals:
            if n not in self._pins:
                self._pins.append(n)
        self._records.append(rec)
        if new_pin:
            self._refresh()
            return
        self._append_log_row(rec)
        self._refresh_stats()
        self._update_count()

    def clear(self) -> None:
        self._records = []
        self._refresh()

    def set_locked(self, locked: bool) -> None:
        """Batch lock. A running batch feeds this table and parks the user on it,
        so its buttons stay visible — but Log would inject a mislabeled extra row,
        Clear would wipe the accumulating study, and Export would race the writer.
        The window locks them for the batch's duration."""
        for b in (self.log_btn, self.clear_btn, self.export_btn):
            b.setEnabled(not locked)

    def set_units(self, unit: str) -> None:
        if unit in UNIT_PER_M and unit != self._unit:
            self._unit = unit
            self._refresh()

    # ---------------------------------------------------------------- render
    def _vals_m(self, pin: str) -> np.ndarray:
        v = [r["vals"].get(pin) for r in self._records]
        return np.array([x for x in v if x is not None], dtype=float)

    def _append_log_row(self, rec: dict) -> None:
        """One new row at the bottom of the raw-readings table (columns unchanged)."""
        u = self._unit
        f = UNIT_PER_M[u]
        dec = UNIT_DECIMALS.get(u, 2)
        r = self.log_table.rowCount()
        self.log_table.setRowCount(r + 1)
        self.log_table.setItem(r, 0, QTableWidgetItem(rec["label"]))
        for c, p in enumerate(self._pins):
            h = rec["vals"].get(p)
            self.log_table.setItem(
                r, c + 1, _num("—" if h is None else f"{h * f:.{dec}f}"))
        self.log_table.scrollToBottom()

    def _refresh_stats(self) -> None:
        """Rebuild the small per-pin summary (rows = pins — cheap, ≤ a few rows)."""
        u = self._unit
        f = UNIT_PER_M[u]
        dec = UNIT_DECIMALS.get(u, 2)
        sdec = dec + 1                      # σ / range are small — one extra digit
        pins = self._pins
        self.stat_table.setHorizontalHeaderLabels(
            ["Pin", "N", f"mean ({u})", f"σ ({u})", f"min ({u})", f"max ({u})", f"range ({u})"])
        self.stat_table.setRowCount(len(pins))
        for r, p in enumerate(pins):
            a = self._vals_m(p) * f
            self.stat_table.setItem(r, 0, QTableWidgetItem(p))
            if len(a):
                n = len(a)
                # σ / range are undefined for a single reading — show "—", not 0.000
                # (0.000 would read as "perfectly repeatable" rather than "N too small")
                sd = f"{a.std(ddof=1):.{sdec}f}" if n > 1 else "—"
                rng = f"{a.max() - a.min():.{sdec}f}" if n > 1 else "—"
                cells = [str(n), f"{a.mean():.{dec}f}", sd,
                         f"{a.min():.{dec}f}", f"{a.max():.{dec}f}", rng]
            else:
                cells = ["0", "—", "—", "—", "—", "—"]
            for c, txt in enumerate(cells):
                self.stat_table.setItem(r, c + 1, _num(txt))
        # size the summary to show every pin (up to 9) without its own scrollbar
        self.stat_table.setFixedHeight(34 + 30 * min(max(len(pins), 1), 9) + 4)

    def _update_count(self) -> None:
        n = len(self._records)
        self.count_lbl.setText("no readings yet" if n == 0
                               else f"{n} reading{'s' if n != 1 else ''} logged")

    def _refresh(self) -> None:
        """Full rebuild — unit switch, Clear, or a record that adds a new pin column."""
        u = self._unit
        f = UNIT_PER_M[u]
        dec = UNIT_DECIMALS.get(u, 2)
        pins = self._pins

        # --- raw readings: rows = captures, cols = Capture + one per pin ---
        self.log_table.setColumnCount(1 + len(pins))
        self.log_table.setHorizontalHeaderLabels(
            ["Capture"] + [f"{p} ({u})" for p in pins])
        self.log_table.setRowCount(len(self._records))
        for r, rec in enumerate(self._records):
            self.log_table.setItem(r, 0, QTableWidgetItem(rec["label"]))
            for c, p in enumerate(pins):
                h = rec["vals"].get(p)
                self.log_table.setItem(
                    r, c + 1, _num("—" if h is None else f"{h * f:.{dec}f}"))
        self.log_table.scrollToBottom()

        self._refresh_stats()
        self._update_count()

    # ---------------------------------------------------------------- export
    def _export(self) -> None:
        if not self._records:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export repeatability CSV", "repeatability.csv", "CSV (*.csv)")
        if not path:
            return
        u = self._unit
        f = UNIT_PER_M[u]
        dec = UNIT_DECIMALS.get(u, 2)
        sdec = dec + 1                      # match the on-screen table's σ/range digits
        try:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                # unit on the pin columns (they hold the heights), not the label column
                w.writerow(["capture"] + [f"{p} ({u})" for p in self._pins])
                for rec in self._records:
                    w.writerow([rec["label"]] + [
                        "" if rec["vals"].get(p) is None else round(rec["vals"][p] * f, dec)
                        for p in self._pins])
                w.writerow([])
                w.writerow(["summary", "N", f"mean ({u})", f"sigma ({u})",
                            f"min ({u})", f"max ({u})", f"range ({u})"])
                for p in self._pins:
                    a = self._vals_m(p) * f
                    if len(a):
                        n = len(a)
                        # same rounding as the table, so the CSV can't disagree with it
                        sd = round(float(a.std(ddof=1)), sdec) if n > 1 else ""
                        rng = round(float(a.max() - a.min()), sdec) if n > 1 else ""
                        w.writerow([p, n, round(float(a.mean()), dec),
                                    sd, round(float(a.min()), dec),
                                    round(float(a.max()), dec), rng])
                    else:
                        w.writerow([p, 0, "", "", "", "", ""])
        except OSError as exc:
            # PermissionError here is Windows daily life: the CSV is open in Excel.
            # Swallowing it told the user the export worked when nothing was written.
            QMessageBox.critical(
                self, "Export failed",
                f"Couldn't write {path}:\n{exc}\n\n"
                "If the file is open in Excel, close it there and export again.")
