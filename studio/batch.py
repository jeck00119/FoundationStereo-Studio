"""Folder / file batch feeders for the Repeatability study.

Two input sources, "the user chooses":
  • Stereo pairs — a folder of left/right images; each pair is run through the
    loaded model and the measure boxes, logging one row per capture.
  • Point clouds — saved .ply/.npy clouds re-measured through the boxes directly,
    NO model, instant (measure.py is pure numpy).

This module owns input DISCOVERY (pairing left/right images; loading clouds) and
the setup/progress DIALOG. The per-item loops live in MainWindow's batch state
machine, beside the compare/overlay ones they mirror.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np
from PySide6.QtWidgets import (QComboBox, QDialog, QFileDialog, QHBoxLayout,
                               QLabel, QLineEdit, QListWidget, QProgressBar,
                               QPushButton, QVBoxLayout, QWidget)

# the ONE loader + extension list, shared with the Input panel (studio.pairs) —
# a batched pair is by construction fed to the engine exactly as a hand-dropped one
from .pairs import IMG_EXTS, load_rgb  # noqa: F401  (load_rgb re-exported for callers)

CLOUD_EXTS = {".ply", ".npy", ".pcd", ".xyz"}


def load_cloud(path: str):
    """Read a saved cloud to (points Nx3 float32, colors Nx3 uint8 | None). Points
    are in whatever unit the file was written in — the dialog's unit selector
    reconciles that to the working unit. Raises ValueError on anything that isn't
    a point set (e.g. a depth-map .npy, which is HxW, not Nx3)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.asarray(np.load(path))
        if arr.ndim != 2 or arr.shape[1] not in (3, 4, 6):
            raise ValueError(f"expected an (N,3) point array, got shape {arr.shape}")
        return arr[:, :3].astype(np.float32).copy(), None
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points, np.float32)
    if pts.size == 0:
        raise ValueError("no points in the file")
    cols = None
    if pcd.has_colors():
        cols = (np.asarray(pcd.colors) * 255.0).clip(0, 255).astype(np.uint8)
    return pts, cols


# ------------------------------------------------------------------ discovery
@dataclass
class PairScan:
    pairs: list                              # [(label, left_path, right_path)]
    method: str                              # human description of how they matched
    unpaired: list = field(default_factory=list)   # leftover image filenames

    def __bool__(self) -> bool:
        return bool(self.pairs)


def _images_in(folder: str) -> list:
    try:
        names = os.listdir(folder)
    except OSError:
        return []
    return sorted(f for f in names
                  if os.path.splitext(f)[1].lower() in IMG_EXTS
                  and os.path.isfile(os.path.join(folder, f)))


# Left/right SUBFOLDER name pairs, most specific first. Each entry is
# (left-name aliases, right-name aliases), all lower-case.
_LR_DIRNAMES = [
    (("left", "cam_left", "camleft", "left_cam", "leftcam"),
     ("right", "cam_right", "camright", "right_cam", "rightcam")),
    (("cam0", "cam_0", "camera0", "cam00", "view0", "im0", "image0"),
     ("cam1", "cam_1", "camera1", "cam01", "view1", "im1", "image1")),
    (("l",), ("r",)),
    (("0",), ("1",)),
]

# Single-folder filename conventions, most specific first. Each regex runs on the
# file STEM (no extension); the 's' group's lower-case value maps to a side.
_SUFFIX_FAMILIES = [
    (re.compile(r"^(?P<key>.+?)[ _.\-]+(?P<s>left|right)$", re.I),
     {"left": "L", "right": "R"}, "name_left / name_right"),
    (re.compile(r"^(?P<key>.+?)[ _.\-]*cam[ _.\-]*(?P<s>0|1)$", re.I),
     {"0": "L", "1": "R"}, "name_cam0 / name_cam1"),
    (re.compile(r"^(?P<key>.+?)[ _.\-]+(?P<s>l|r)$", re.I),
     {"l": "L", "r": "R"}, "name_L / name_R"),
    (re.compile(r"^(?P<key>.+?)[ _.\-]+(?P<s>0|1)$", re.I),
     {"0": "L", "1": "R"}, "name_0 / name_1"),
]
_PREFIX_FAMILIES = [
    (re.compile(r"^(?P<s>left|right)[ _.\-]+(?P<key>.+)$", re.I),
     {"left": "L", "right": "R"}, "left_name / right_name"),
    (re.compile(r"^(?P<s>l|r)[ _.\-]+(?P<key>.+)$", re.I),
     {"l": "L", "r": "R"}, "L_name / R_name"),
]


def _find_lr_subdirs(folder: str):
    """(left_dir, right_dir) if `folder` holds a left/right pair of subfolders."""
    try:
        subs = [d for d in os.listdir(folder)
                if os.path.isdir(os.path.join(folder, d))]
    except OSError:
        return None
    low = {d.lower(): d for d in subs}
    for lefts, rights in _LR_DIRNAMES:
        L = next((low[n] for n in lefts if n in low), None)
        R = next((low[n] for n in rights if n in low), None)
        if L and R:
            return os.path.join(folder, L), os.path.join(folder, R)
    return None


def _sibling_lr(folder: str):
    """The chosen folder may BE the left (or right) folder, with its partner
    sitting next to it. Match by folder name against a sibling."""
    base = os.path.basename(os.path.normpath(folder)).lower()
    parent = os.path.dirname(os.path.normpath(folder))
    if not parent or not os.path.isdir(parent):
        return None
    try:
        sibs = {d.lower(): d for d in os.listdir(parent)
                if os.path.isdir(os.path.join(parent, d))}
    except OSError:
        return None
    for lefts, rights in _LR_DIRNAMES:
        if base in lefts:
            R = next((sibs[n] for n in rights if n in sibs), None)
            if R:
                return folder, os.path.join(parent, R)
        if base in rights:
            L = next((sibs[n] for n in lefts if n in sibs), None)
            if L:
                return os.path.join(parent, L), folder
    return None


def _stem_map(files: list):
    """{stem: filename} plus the files whose stem collided with an earlier one
    (e.g. 1.png and 1.jpg) — those are ambiguous, kept as 'dropped' so they're
    reported rather than silently vanishing.

    Stems are lower-cased for MATCHING (values keep the real filename) — the same
    rule _pair_by_family documents: Windows filesystems are case-insensitive, so
    left/CAP_01.PNG must pair with right/cap_01.png instead of silently falling
    back to positional-order pairing."""
    m, dropped = {}, []
    for f in files:
        stem = os.path.splitext(f)[0].lower()
        if stem in m:
            dropped.append(f)
        else:
            m[stem] = f
    return m, dropped


def _pair_two_dirs(ld: str, rd: str):
    """Pair images across two folders by identical filename stem; fall back to
    positional order only if the counts match but no stems do."""
    limg, ldrop = _stem_map(_images_in(ld))
    rimg, rdrop = _stem_map(_images_in(rd))
    pairs = [(stem, os.path.join(ld, limg[stem]), os.path.join(rd, rimg[stem]))
             for stem in sorted(limg) if stem in rimg]
    if pairs:
        unpaired = ([limg[s] for s in sorted(limg) if s not in rimg]
                    + [rimg[s] for s in sorted(rimg) if s not in limg]
                    + ldrop + rdrop)               # same-stem duplicates, reported not dropped
        return pairs, unpaired, "matched by filename"
    li, ri = _images_in(ld), _images_in(rd)
    if li and len(li) == len(ri):
        pairs = [(os.path.splitext(a)[0], os.path.join(ld, a), os.path.join(rd, b))
                 for a, b in zip(li, ri)]
        return pairs, [], "paired by order — names differ, check the preview"
    return [], li + ri, "no matching filenames"


def _pair_by_family(files: list, regex, sidemap):
    """Group single-folder files into (key, left, right) by one naming family."""
    groups: dict = {}
    for f in files:
        m = regex.match(os.path.splitext(f)[0])
        if not m:
            continue
        side = sidemap.get(m.group("s").lower())
        if side is not None:
            # lower-case the key so IMG_L pairs with img_R (regex is re.I; on
            # Windows the filesystem is case-insensitive so this can't merge two
            # genuinely distinct captures)
            groups.setdefault(m.group("key").lower(), {})[side] = f
    return [(k, g["L"], g["R"]) for k, g in sorted(groups.items())
            if "L" in g and "R" in g]


def find_pairs(folder: str) -> PairScan:
    """Discover stereo pairs in `folder`. Tries, in order: left/right subfolders
    inside it, the folder + a sibling left/right folder, then single-folder
    filename conventions (name_L/name_R, name_left/name_right, cam0/cam1, …)."""
    folder = os.path.normpath(folder)
    if not os.path.isdir(folder):
        return PairScan([], "not a folder")

    lr, where = _find_lr_subdirs(folder), "subfolders"
    if lr is None:
        lr, where = _sibling_lr(folder), "folder + sibling"
    if lr is not None:
        ld, rd = lr
        pairs, unpaired, how = _pair_two_dirs(ld, rd)
        if pairs:
            method = f"{os.path.basename(ld)} / {os.path.basename(rd)} {where}, {how}"
            return PairScan(pairs, method, unpaired)

    files = _images_in(folder)
    for regex, sidemap, desc in _SUFFIX_FAMILIES + _PREFIX_FAMILIES:
        pairs = _pair_by_family(files, regex, sidemap)
        if pairs:
            paths = [(k, os.path.join(folder, l), os.path.join(folder, r))
                     for k, l, r in pairs]
            used = {l for _, l, r in pairs} | {r for _, l, r in pairs}
            unpaired = [f for f in files if f not in used]
            return PairScan(paths, f"filenames: {desc}", unpaired)

    return PairScan([], "no left/right pairs found", files)


# --------------------------------------------------------------------- dialog
class BatchDialog(QDialog):
    """Pick a source (a folder of stereo pairs, or saved cloud files), preview it,
    run — then watch progress. The window drives the run and calls on_progress()/
    on_finished() back; kept non-modal so the Repeatability table fills visibly."""

    def __init__(self, current_unit: str = "mm", parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Batch — feed the Repeatability table")
        self.setModal(False)
        self.setMinimumWidth(580)
        self._mode = "pairs"
        self._scan: PairScan | None = None      # pairs-mode discovery
        self._files: list = []                  # clouds-mode selection
        self._running = False
        # the window sets these — plain callbacks so it owns the whole lifecycle
        self._on_run_cb = None                  # pairs: (pairs_list)
        self._on_run_clouds_cb = None           # clouds: (files, file_unit)
        self._on_cancel_cb = None

        v = QVBoxLayout(self)
        v.setContentsMargins(18, 16, 18, 16)
        v.setSpacing(11)

        self.intro = QLabel("")
        self.intro.setWordWrap(True)
        self.intro.setProperty("role", "muted")
        v.addWidget(self.intro)

        # source switch -----------------------------------------------------
        seg = QHBoxLayout()
        seg.setSpacing(0)
        self.mode_pairs = QPushButton("Stereo pairs")
        self.mode_clouds = QPushButton("Point clouds")
        for b, m in ((self.mode_pairs, "pairs"), (self.mode_clouds, "clouds")):
            b.setObjectName("Seg")
            b.setCheckable(True)
            b.clicked.connect(lambda _=False, mm=m: self._set_mode(mm))
            seg.addWidget(b)
        seg.addStretch(1)
        v.addLayout(seg)

        # source row --------------------------------------------------------
        row = QHBoxLayout()
        row.setSpacing(8)
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        self.browse_btn = QPushButton("Browse folder…")
        self.browse_btn.clicked.connect(self._browse)
        row.addWidget(self.path_edit, 1)
        row.addWidget(self.browse_btn)
        v.addLayout(row)

        # unit row (clouds only) -------------------------------------------
        self.unit_row = QWidget()
        ur = QHBoxLayout(self.unit_row)
        ur.setContentsMargins(0, 0, 0, 0)
        ur.setSpacing(8)
        ulbl = QLabel("These files are saved in:")
        ulbl.setProperty("role", "muted")
        self.unit_combo = QComboBox()
        # every unit the app can EXPORT in — µm was missing, so a µm session's own
        # PLYs had to be declared mm and were scaled ×1000 (boxes caught nothing)
        self.unit_combo.addItems(["mm", "µm", "m"])
        self.unit_combo.setCurrentText(current_unit if current_unit in ("mm", "µm", "m") else "mm")
        self.unit_combo.setFixedWidth(70)
        ur.addWidget(ulbl)
        ur.addWidget(self.unit_combo)
        ur.addStretch(1)
        v.addWidget(self.unit_row)

        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        self.summary.setProperty("role", "section")
        v.addWidget(self.summary)

        self.preview = QListWidget()
        self.preview.setObjectName("BatchPreview")
        self.preview.setMinimumHeight(200)
        v.addWidget(self.preview, 1)

        self.warn = QLabel("")
        self.warn.setWordWrap(True)
        self.warn.setProperty("role", "muted")
        self.warn.hide()
        v.addWidget(self.warn)

        self.note = QLabel("")
        self.note.setWordWrap(True)
        self.note.setProperty("role", "muted")
        v.addWidget(self.note)

        self.bar = QProgressBar()
        self.bar.hide()
        v.addWidget(self.bar)

        btns = QHBoxLayout()
        btns.addStretch(1)
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        self.run_btn = QPushButton("Run batch")
        self.run_btn.setObjectName("Accent")
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self._on_run)
        btns.addWidget(self.close_btn)
        btns.addWidget(self.run_btn)
        v.addLayout(btns)

        self._set_mode("pairs")

    # --------------------------------------------------------------- mode
    def _set_mode(self, mode: str) -> None:
        if self._running:
            return
        self._mode = mode
        self.mode_pairs.setChecked(mode == "pairs")
        self.mode_clouds.setChecked(mode == "clouds")
        self.unit_row.setVisible(mode == "clouds")
        self._scan = None
        self._files = []
        self.preview.clear()
        self.warn.hide()
        self.summary.setText("")
        self.path_edit.clear()
        self.run_btn.setEnabled(False)
        if mode == "pairs":
            self.intro.setText("Run every stereo pair in a folder through the loaded model "
                               "and the measure boxes you placed — one logged row per capture.")
            self.browse_btn.setText("Browse folder…")
            self.path_edit.setPlaceholderText("Choose a folder of stereo pairs…")
            self.note.setText("Uses the current model, calibration and boxes. Readings are "
                              "added to the table — Clear it first for a fresh study.")
        else:
            self.intro.setText("Re-measure saved point-cloud files through your boxes — no "
                               "model needed, so it's instant.")
            self.browse_btn.setText("Add files…")
            self.path_edit.setPlaceholderText("Choose .ply / .npy cloud files…")
            self.note.setText("Measures your boxes on each cloud — no model or calibration "
                              "needed. Readings are added to the table.")

    # ------------------------------------------------------------- setup phase
    def _browse(self) -> None:
        if self._mode == "pairs":
            d = QFileDialog.getExistingDirectory(self, "Choose a folder of stereo pairs")
            if d:
                self.path_edit.setText(d)
                self._scan_folder(d)
        else:
            files, _ = QFileDialog.getOpenFileNames(
                self, "Choose point-cloud files", "",
                "Point clouds (*.ply *.npy *.pcd *.xyz);;All files (*)")
            if files:
                # the button says "Add files…" — append (dedup), don't replace;
                # switching modes is what clears the selection
                self._set_files(self._files
                                + [f for f in files if f not in self._files])

    def _scan_folder(self, folder: str) -> None:
        self._scan = find_pairs(folder)
        n = len(self._scan.pairs)
        self.preview.clear()
        if n:
            self.summary.setText(f"Found {n} pair{'s' if n != 1 else ''}  ·  {self._scan.method}")
            for label, l, r in self._scan.pairs[:40]:
                self.preview.addItem(
                    f"{label}      {os.path.basename(l)}  ↔  {os.path.basename(r)}")
            if n > 40:
                self.preview.addItem(f"…and {n - 40} more")
            self.run_btn.setEnabled(True)
        else:
            self.summary.setText(f"No stereo pairs found — {self._scan.method}.")
            self.run_btn.setEnabled(False)
        up = self._scan.unpaired
        if up:
            shown = ", ".join(up[:6]) + (f"  +{len(up) - 6} more" if len(up) > 6 else "")
            self.warn.setText(f"⚠ {len(up)} image(s) not paired — they'll be skipped: {shown}")
            self.warn.show()
        else:
            self.warn.hide()

    def _set_files(self, files: list) -> None:
        self._files = list(files)
        n = len(self._files)
        self.path_edit.setText(f"{n} file{'s' if n != 1 else ''} selected")
        self.summary.setText(f"{n} cloud file{'s' if n != 1 else ''} selected")
        self.preview.clear()
        for f in self._files[:40]:
            self.preview.addItem(os.path.basename(f))
        if n > 40:
            self.preview.addItem(f"…and {n - 40} more")
        self.warn.hide()
        self.run_btn.setEnabled(n > 0)

    # ---------------------------------------------------------- run / progress
    def _on_run(self) -> None:
        if self._running:
            if self._on_cancel_cb:
                self._on_cancel_cb()
            self.run_btn.setEnabled(False)
            self.run_btn.setText("Cancelling…")
            return
        if self._mode == "pairs":
            if not self._scan or not self._scan.pairs:
                return
            self._enter_running(len(self._scan.pairs))
            if self._on_run_cb:
                self._on_run_cb(list(self._scan.pairs))
        else:
            if not self._files:
                return
            self._enter_running(len(self._files))
            if self._on_run_clouds_cb:
                self._on_run_clouds_cb(list(self._files), self.unit_combo.currentText())

    def _enter_running(self, total: int) -> None:
        self._running = True
        self.bar.setRange(0, total)
        self.bar.setValue(0)
        self.bar.show()
        self.run_btn.setText("Cancel")
        self.close_btn.setEnabled(False)
        self.browse_btn.setEnabled(False)
        self.preview.setEnabled(False)
        self.mode_pairs.setEnabled(False)
        self.mode_clouds.setEnabled(False)
        self.unit_combo.setEnabled(False)

    def on_progress(self, done: int, total: int, label: str,
                    logged: int, empty: int, failed: int) -> None:
        if not self._running:
            return
        self.bar.setMaximum(total)
        self.bar.setValue(done)
        msg = f"Processing {done}/{total} — {label}      ✓ {logged} logged"
        if empty:
            msg += f"  ·  ∅ {empty} empty"
        if failed:
            msg += f"  ·  ✗ {failed} failed"
        self.summary.setText(msg)

    def on_finished(self, summary: str, failed: list | None = None) -> None:
        self._running = False
        # do NOT force the bar to maximum: a batch aborted at 10/900 (engine
        # death) showed a FULL bar under a "Batch stopped" summary. on_progress
        # already left it at the true count.
        self.summary.setText(summary)
        self.run_btn.setText("Run batch")
        # the scan/selection is still valid — allow re-running it (readings are
        # ADDED to the table, as the note says; Clear the table for a fresh study)
        self.run_btn.setEnabled(bool(self._scan and self._scan.pairs)
                                if self._mode == "pairs" else bool(self._files))
        self.close_btn.setEnabled(True)
        self.browse_btn.setEnabled(True)
        self.preview.setEnabled(True)
        self.mode_pairs.setEnabled(True)
        self.mode_clouds.setEnabled(True)
        self.unit_combo.setEnabled(True)
        if failed:
            shown = "\n".join(f"  · {lbl}: {why}" for lbl, why in failed[:12])
            more = f"\n  …and {len(failed) - 12} more" if len(failed) > 12 else ""
            self.warn.setText(f"Failed captures:\n{shown}{more}")
            self.warn.show()

    def closeEvent(self, event) -> None:
        # Closing the monitor during a run = cancel it (takes effect after the
        # current item). Without this, Esc/the window-X would strand the batch: the
        # UI stays locked with no cancel affordance and no way to reopen the monitor.
        if self._running and self._on_cancel_cb:
            self._on_cancel_cb()
        super().closeEvent(event)
