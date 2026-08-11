"""Level-to-plane — the fixed rotation that flattens the board.

Fits the dominant plane once and then applies the SAME rotation to every cloud
the session produces, so a whole fixed-fixture batch reconstructs straight and
pin heights read perpendicular to the board rather than to the camera.

The rotation is stored here; each cloud carries its own un-levelled points as
``CloudResult.raw_points`` so re-levelling is always derived from the original
rather than compounded on an already-rotated cloud.
"""
from __future__ import annotations

import numpy as np

from ..analyze import board_plane
from ..dtypes import ANGLE_DECIMALS, UNIT_PER_M
from ..measure import fit_plane, rotation_to_axis


class LevelController:
    """The board-levelling rotation. Public surface: ``R`` · ``c_m`` ·
    ``ingest()`` · ``apply()`` · ``relevel_current()`` · ``on_toggled()`` ·
    ``board_plane()`` · ``state()`` · ``restore()``."""

    def __init__(self, win) -> None:
        self.win = win
        self.R = None        # 3x3 rotation, or None (off)
        self.c_m = None      # rotation centre, canonical metres

    @property
    def on(self) -> bool:
        return self.R is not None

    # ------------------------------------------------------------- applying
    def ingest(self, cloud):
        """Record the raw points ON the cloud and apply the active rotation, so
        every cloud (single run, compare, or batch) reconstructs in the levelled
        frame. With level off, raw_points is the same array as points (no copy)."""
        if self.win._has_points(cloud):
            cloud.raw_points = cloud.points
            if self.R is not None:
                cloud.points = self.apply(cloud.points)
        return cloud

    def apply(self, points):
        """Rotate points into the levelled frame about the stored centre. The centre
        is canonical metres; scale it to the display unit the points are in."""
        if self.R is None:
            return points
        c = np.asarray(self.c_m, np.float64) * UNIT_PER_M[self.win._units]
        return (np.asarray(points) - c) @ self.R.T + c

    def relevel_current(self) -> None:
        """Re-derive EVERY cached cloud from its own raw points under the current
        level state, then re-show. All clouds, not just the shown one: the overlay
        and the multi-target measure read the compare caches directly, so leaving
        them in the old frame mixed levelled and unlevelled points in one readout."""
        win = self.win
        for c in win._all_clouds():
            raw = getattr(c, "raw_points", None)
            if raw is not None:
                c.points = self.apply(raw)      # level off -> returns raw itself
        if not win._has_points(win.cloud):
            return
        if win._overlay_on:
            win._show_overlay()         # re-stacks the re-levelled caches (+ measures)
        else:
            win.viewer.show_cloud(win.cloud, reset_view=True)
            win._apply_measure()
        win.analyze.reset_overlay()     # picks were in the pre-level frame
        win.analyze.reapply_deviation()  # heatmap must reference the re-levelled plane

    # -------------------------------------------------------------- toggling
    def on_toggled(self, on: bool) -> None:
        """Level button: fit the board plane, rotate the cloud (and the boxes) so the
        board is flat, and remember the rotation for every subsequent cloud."""
        win = self.win
        if on:
            raw = getattr(win.cloud, "raw_points", None) if win.cloud is not None else None
            if raw is None or len(raw) < 500:
                win._set_status("No cloud to level yet — run a pair first.")
                win.param_panel.set_level_checked(False)
                return
            n, c = fit_plane(raw)
            if n[2] > 0:                 # point the board normal toward the camera (−Z)
                n = -n
            R = rotation_to_axis(n, (0.0, 0.0, -1.0))
            tilt = float(np.degrees(np.arccos(np.clip(-n[2], -1.0, 1.0))))
            self.c_m = np.asarray(c, np.float64) / UNIT_PER_M[win._units]
            self.R = R
            win.param_panel.transform_boxes(R, self.c_m, inverse=False)
            self.relevel_current()
            win._set_status(
                f"Levelled to the board plane — removed {tilt:.{ANGLE_DECIMALS}f}° of tilt.")
        else:
            if self.R is not None:
                win.param_panel.transform_boxes(self.R, self.c_m, inverse=True)
            self.R = self.c_m = None
            self.relevel_current()
            win._set_status("Levelling off — showing the raw camera-frame cloud.")

    # --------------------------------------------------------- reference plane
    def board_plane(self):
        """(normal, centroid) reference plane for the analyze tools — the levelling
        plane if levelling is on (board normal = −Z), else a fresh fit."""
        if self.R is not None:
            c = np.asarray(self.c_m, np.float64) * UNIT_PER_M[self.win._units]
            return np.array([0.0, 0.0, -1.0]), c
        return board_plane(self.win.cloud.points)

    # ----------------------------------------------------------- persistence
    def state(self) -> dict:
        if self.R is None:
            return {}
        return {"R": self.R.tolist(), "c": np.asarray(self.c_m).tolist()}

    def restore(self, blob) -> None:
        """A persisted rotation applies to every cloud this session too."""
        if not (isinstance(blob, dict) and blob.get("R") and blob.get("c")):
            return
        try:
            self.R = np.array(blob["R"], np.float64).reshape(3, 3)
            self.c_m = np.array(blob["c"], np.float64).reshape(3)
            self.win.param_panel.set_level_checked(True)
        except Exception:   # noqa: BLE001 — a bad blob must not wedge startup
            self.R = self.c_m = None
