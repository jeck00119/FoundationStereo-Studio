"""InputPanel — the left panel: the image pair, calibration/rectification, and
the model + checkpoint pickers."""
from __future__ import annotations

import math
import os

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QComboBox, QFileDialog, QHBoxLayout, QLabel,
                               QPushButton, QVBoxLayout, QWidget)

from ..backends import BACKENDS, get_spec
from ..dtypes import UNIT_PER_M
from ..pairs import IMG_FILTER, load_rgb
from ..rectify import Rectifier, StereoCalibration
from ..widgets import (CollapsibleSection, ImageDrop, no_wheel, set_tip)
from ._common import _BASELINE_CFG, np_to_qpixmap, make_spin, field_row


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
        self._warn_if_unrectified()         # raw pair in 'already rectified' mode?

    def _warn_if_unrectified(self) -> None:
        """Cheap row-alignment probe for 'already rectified' mode: if matched
        features between the pair are VERTICALLY misaligned by more than a few
        px, the pair is raw (this rig's raw pairs sit at ~19 px) — and the
        stereo model, which assumes row-aligned input, would quietly produce a
        garbage cloud. Saying so here turns a mystery into a mode switch."""
        if self._rect_mode_on or self.left_rgb is None or self.right_rgb is None:
            return
        try:
            import cv2
            gl = cv2.cvtColor(self.left_rgb[::4, ::4], cv2.COLOR_RGB2GRAY)
            gr = cv2.cvtColor(self.right_rgb[::4, ::4], cv2.COLOR_RGB2GRAY)
            orb = cv2.ORB_create(500)
            kl, dl = orb.detectAndCompute(gl, None)
            kr, dr = orb.detectAndCompute(gr, None)
            if dl is None or dr is None or len(kl) < 30 or len(kr) < 30:
                return
            m = sorted(cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(dl, dr),
                       key=lambda x: x.distance)[:120]
            if len(m) < 25:
                return
            dy = float(np.median([kr[x.trainIdx].pt[1] - kl[x.queryIdx].pt[1]
                                  for x in m])) * 4.0
            if abs(dy) > 3.0:
                self.notice.emit(
                    f"⚠ This pair does not look rectified — features are vertically "
                    f"misaligned by ~{abs(dy):.0f} px. If these are raw captures, switch "
                    "Calibration to 'Raw — rectify with calibration' and load your "
                    "calibration file (not k_rectified.txt).")
        except Exception:   # noqa: BLE001 — a probe must never break image loading
            pass

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
        self._warn_if_unrectified()    # entering 'already rectified' with a raw pair?

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

    def process_pair(self, left_raw, right_raw, params=None):
        """Apply the active rectification — and the run's ROI crop — to a RAW pair
        (passthrough when not in raw mode / no rectifier). Used by the folder batch
        so batched pairs are fed exactly like a hand-dropped one. Raises on a size
        mismatch so a folder whose resolution differs from the reference pair fails
        LOUDLY per-image (the batch banks it as failed) instead of silently
        remapping through wrong-sized maps.

        ``params`` carries the ROI. Cropping HERE rather than in the engine child
        is what keeps a whole 73 MB frame off the socket every pair; when we are
        rectifying anyway it also remaps only the ROI's pixels (measured 104 → 6 ms
        per image), because the two crop windows are known before any remapping
        has to happen."""
        from .rectify import crop_pair, roi_rects

        self._ensure_rectifier()
        if self._rect_mode_on and self._rectifier is not None:
            size = (left_raw.shape[1], left_raw.shape[0])
            if self._rectifier.size != size:
                raise ValueError(
                    f"image is {size[0]}×{size[1]} but the calibration/reference pair is "
                    f"{self._rectifier.size[0]}×{self._rectifier.size[1]}; batch images must "
                    "match the reference resolution")
            rects = roi_rects(params, size[0], size[1]) if params is not None else None
            if rects is None:
                return (self._rectifier.rectify(left_raw, "L"),
                        self._rectifier.rectify(right_raw, "R"))
            (lx, ly, lw, lh), (rx, ry, rw, rh) = rects
            return (self._rectifier.rectify_roi(left_raw, "L", lx, ly, lw, lh),
                    self._rectifier.rectify_roi(right_raw, "R", rx, ry, rw, rh))
        if params is not None:
            return crop_pair(left_raw, right_raw, params)   # already-rectified input
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


