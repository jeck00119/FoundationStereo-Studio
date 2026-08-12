"""The ROI crop and its disparity pre-shift.

Owns the two numbers that decide WHICH pixels a run reconstructs: the rectangle
drawn on the Input tab, and Δ — how far left the right crop starts. They live
here rather than in the parameter panel because they are a region of the
picture, not a knob; the window attaches them to every run's params.

See ``StereoParams.roi`` for what the pair buys (full-resolution macro stereo on
a memory budget that could not otherwise reach it).
"""
from __future__ import annotations

import json


class RoiController:
    """Draw-a-box crop + the measured pre-shift. Public surface:
    ``roi`` · ``disp_shift`` · ``dispatch_pair()`` · ``on_roi_changed()`` ·
    ``on_find_shift()`` · ``save()`` · ``restore()``."""

    #: how far under the measured disparity Δ is placed, in full-res px. The
    #: match reports the disparity at ONE depth; parts stand proud of it, and a Δ
    #: past the smallest disparity present would clip the far end of the scene to
    #: zero — which reads as "no match" rather than "too far".
    SHIFT_MARGIN_PX = 24.0

    def __init__(self, win) -> None:
        self.win = win
        self.roi: tuple | None = None    # (x0, y0, w, h), rectified full-res px
        self.disp_shift: float = 0.0     # Δ, measured automatically when roi moves
        self.last_note = ""              # the measured-shift half of the ROI label

    # ------------------------------------------------------------- dispatch
    def dispatch_pair(self, params):
        """The loaded pair, cropped to the run's ROI — exactly what goes over the
        socket to the engine child.

        Every non-batch dispatch site goes through here so none of them can be
        missed: sending full frames alongside params that declare an ROI produces
        a silently wrong reconstruction (shifted K and an un-shifted disparity
        applied to uncropped pixels), not a visible failure. The batch has its own
        path only because it rectifies each file as it goes — and it crops in the
        same step. left_rgb/right_rgb are already rectified (the Input tab shows
        them), which is the frame the ROI was drawn in.
        """
        from ..rectify import crop_pair

        panel = self.win.input_panel
        return crop_pair(panel.left_rgb, panel.right_rgb, params)

    # ---------------------------------------------------------------- edits
    def on_roi_changed(self, roi) -> None:
        """The drawn crop moved. A different crop is a different reconstruction,
        so the shown result is stale — and Δ belonged to the OLD rectangle, so it
        is dropped rather than silently re-used against a region it never
        matched."""
        if self.win._batching:
            return                       # geometry is frozen for the study
        same = (roi == self.roi)
        self.roi = tuple(roi) if roi is not None else None
        if not same:
            # Δ belonged to the OLD rectangle, so it cannot be kept — but making
            # the user re-press a button for it was a trap: the run then saturates
            # and the failure looks like a disparity-range problem. Measuring it
            # is a ~100 ms template match on images already in memory, so just do
            # it. Moving the box is the normal action; it should not break a run.
            self.disp_shift = 0.0
            self.win._mark_stale()
            if self.roi is not None:
                self.measure_shift(quiet=True)
        self.save()
        self.refresh_note()

    def on_find_shift(self) -> None:
        """The manual re-measure. Δ is measured automatically whenever the box
        moves; this is for when the pair changed under a box that did not."""
        self.measure_shift(quiet=False)

    def measure_shift(self, quiet: bool = False) -> bool:
        """Measure Δ by matching the ROI's pixels in the right image."""
        from ..rectify import find_disparity_shift

        win = self.win
        if self.roi is None:
            if not quiet:
                win._set_status("Draw an ROI first.")
            return False
        left, right = win.input_panel.left_rgb, win.input_panel.right_rgb
        if left is None or right is None:
            if not quiet:
                win._set_status("Load a pair first — the shift is measured from the images.")
            return False
        r = find_disparity_shift(left, right, self.roi)
        view = win.viewer.input_view
        if not r["ok"]:
            self.last_note = "no shift — move onto the parts"
            self.refresh_note()
            if quiet:      # auto-measure: a bad spot mid-drag is not an error
                win._set_status(
                    "No confident match for this region — it needs texture. "
                    "Move the box onto the parts you measure.")
                return False
            win._report_error(
                f"Couldn't measure the shift for this ROI (confidence "
                f"{r['score']:.2f}).\n\nThe region probably has too little texture "
                "to match. Draw the box over the parts you measure — pins, "
                "silkscreen, connectors — not a bare area of board.")
            return False
        self.disp_shift = max(0.0, float(r["shift"]) - self.SHIFT_MARGIN_PX)
        win._mark_stale()
        self.save()
        note = f"Δ {self.disp_shift:.0f}  ·  m{r['score']:.2f}"
        self.last_note = note
        self.refresh_note()
        if abs(r["dy"]) > 2:
            win._set_status(
                f"Shift found, but the match sits {r['dy']:+d} px off its own row — "
                "a rectified pair should be row-aligned. Check the calibration.")
        elif not quiet:
            win._set_status(
                f"Shift {self.disp_shift:.0f} px (matched {r['shift']:.0f}, "
                f"confidence {r['score']:.2f}). Run to reconstruct just this region.")
        return True

    # ------------------------------------------------------- engine readiness
    def engine_ready(self):
        """Is a TRT engine already built for this crop? None when the question
        does not apply (another backend, or no ROI yet)."""
        from ..backends.registry import cached_engine_sizes, padded_size

        win = self.win
        if self.roi is None:
            return None
        try:
            if win.input_panel.current_backend_key() != "fast_foundation_stereo_trt":
                return None
            p = win._current_params()
            hp, wp = padded_size(self.roi[2], self.roi[3], p.scale)
            mp = p.model_params or {}
            key = (hp, wp, int(mp.get("valid_iters", 8)), int(mp.get("max_disp", 192)))
            return key in cached_engine_sizes(win.input_panel.current_checkpoint_path())
        except Exception:   # noqa: BLE001 — a label must never break the UI
            return None

    def refresh_note(self) -> None:
        """The ROI label: the measured shift, and — the part that saves an
        afternoon — whether this SIZE already has an engine.

        Only the SIZE decides, never the position, so dragging the box around the
        board is always free; a resize is what can cost an hour. Saying so on the
        label is the difference between that being obvious and being a surprise.
        """
        view = self.win.viewer.input_view
        if self.roi is None:
            view.set_roi_note("")
            return
        bits = [self.last_note]
        ready = self.engine_ready()
        if ready is True:
            bits.append("engine ✓")
        elif ready is False:
            bits.append("⚠ new size — first run builds an engine (~1 h)")
        view.set_roi_note("  ·  ".join(b for b in bits if b))

    # ----------------------------------------------------------- persistence
    def save(self) -> None:
        try:
            self.win.settings.setValue("roi", json.dumps(
                {"roi": list(self.roi) if self.roi else None,
                 "shift": float(self.disp_shift)}))
        except Exception:   # noqa: BLE001 — a settings write must never break the UI
            pass

    def restore(self, blob) -> None:
        """Reload a saved crop. set_roi() deliberately does NOT re-emit, so this
        cannot look like a user edit — which would drop the Δ saved beside it."""
        if not (isinstance(blob, dict) and blob.get("roi")):
            return
        try:
            self.roi = tuple(int(v) for v in blob["roi"])
            self.disp_shift = float(blob.get("shift", 0.0))
        except (TypeError, ValueError):   # a bad blob must not wedge startup
            self.roi, self.disp_shift = None, 0.0
            return
        self.win.viewer.input_view.set_roi(self.roi)
        if self.disp_shift:
            self.last_note = f"Δ {self.disp_shift:.0f}"
        self.refresh_note()


class SitesController:
    """Measurement sites marked on the IMAGE: pins, and the references they are
    measured against.

    Deliberately image-space rather than world-space measure boxes. A world box
    is fixed in the camera frame, and this rig's frame does not hold still: the
    CNC's step (the stereo baseline) repeats to ~0.5 %, which walks ABSOLUTE
    depth by >1 mm, and the image itself drifts ~15 px in x and ~34 px in y over
    a long run. Boxes seeded from one capture were empty on 19 of 20 later ones.

    Marked sites survive both, because the measurement tracks each site per
    capture and takes pin-minus-reference WITHIN one frame: the baseline error is
    common to both terms and cancels (measured: absolute sigma 2200 um vs
    differential ~600 um).

    Pick references with TEXTURE. Bare solder mask measured 3.4 grey-levels of
    local contrast and the metal bar 3.6 — the network has nothing to match there
    and its depth is a guess; the component body (23.8) made a better reference.
    """

    def __init__(self, win) -> None:
        self.win = win

    def on_marked(self, kind: str, x: int, y: int) -> None:
        if self.win._batching:
            return                       # geometry is frozen for the study
        site = self.win.viewer.input_view.add_site(kind, x, y)
        self.save()
        self.win._set_status(
            f"Marked {site['name']} at ({x}, {y}). "
            + ("Mark a reference near it — a textured surface, not bare board."
               if kind == "pin" else "Each pin uses the closest reference."))

    def on_cleared(self) -> None:
        self.save()
        self.win._set_status("Measurement sites cleared.")

    def sites(self) -> list:
        return self.win.viewer.input_view.sites()

    def save(self) -> None:
        try:
            self.win.settings.setValue("study_sites", json.dumps(self.sites()))
        except Exception:   # noqa: BLE001 — a settings write must never break the UI
            pass

    def restore(self, blob) -> None:
        if isinstance(blob, list) and blob:
            self.win.viewer.input_view.set_sites(blob)
