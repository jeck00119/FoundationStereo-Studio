"""The analyze tools — pick points on the cloud and measure surfaces.

Profile, distance, region flatness, point probe, pin analysis, the
deviation-from-plane heatmap and the flat-reference zero. All of it references
the BOARD PLANE, which the level controller owns, so heights read as true
stand-off rather than camera-frame depth.

The maths lives in ``studio.analyze`` (pure numpy, headless-testable); this is
the interaction around it — what is picked, what is shown, what is highlighted.
Its eight pieces of state are owned here rather than on the window, which is the
point of the split: only ``tool`` is read from outside (the measure box gizmo
must not swallow a click meant to pick a point).
"""
from __future__ import annotations

import numpy as np

from ..analyze import (deviation, pin_analysis, point_distance, region_flatness,
                       surface_profile)
from ..dtypes import ANGLE_DECIMALS, UNIT_DECIMALS, UNIT_PER_M
from ..measure import points_in_box


class AnalyzeController:
    """Pick-and-measure tools over the shown cloud. Public surface:
    ``tool`` · ``on_tool()`` · ``on_point_picked()`` · ``refresh()`` ·
    ``reset_overlay()`` · ``reapply_deviation()`` · ``on_flat_ref()`` ·
    ``on_isolate_layer()`` · ``on_deviation()`` · ``on_pin_analyze()``."""

    def __init__(self, win) -> None:
        self.win = win
        self.tool = ""             # '' | profile | distance | region | point
        self.picked: list = []     # points clicked for the current measurement
        self.dev_on = False        # deviation-from-plane heatmap active
        self.isolate = False       # region/profile: keep only the picked Z level
        self.z_offset_m = None     # flat-reference zero offset (canonical metres)
        self.z_ref_pp_m = 0.0      # the reference zone's max−min (flatness uncertainty)
        self.last_region = None    # last region_flatness result (to zero from)
        self.last_shown = None     # what the card shows now: tool name | 'pin' | None

    def on_tool(self, tool: str) -> None:
        self.tool = tool or ""
        self.picked = []
        cv = self.win.viewer.cloud_view
        cv.set_analyze_tool(tool or None)
        cv.clear_analyze()
        self.win.param_panel.set_profile(None, None)
        self.win.param_panel.set_analyze_out(
            "Click two points on the cloud." if tool in ("profile", "distance", "region")
            else "Click a point on the cloud." if tool == "point" else "")
        self.win._apply_measure()   # re-push box editability: freeze the gizmo while armed

    def reset_overlay(self) -> None:
        """Drop the picked points, the 3D overlay, the profile plot and the readout —
        called on ANY change to the cloud's frame/scale (unit, level) or identity
        (new pair, model switch/blink), so a stale pick can't pair with a fresh one in
        a mismatched frame and old markers can't float over a new cloud."""
        self.picked = []
        self.last_region = None        # its points are gone — can't zero from it anymore
        self.win.param_panel.set_flat_ref_available(False)   # (an APPLIED ref stays removable)
        self.last_shown = None       # nothing shown in the card now
        self.win.viewer.cloud_view.clear_analyze()
        self.win.param_panel.set_profile(None, None)
        self.win.param_panel.set_analyze_out("")

    def reapply_deviation(self) -> None:
        """Re-paint the deviation heatmap after a cloud repaint. It's pushed as the
        cloud's colours, so every photo repaint (rebuild, level, unit, blink) wipes
        it; re-applying here keeps it live instead of silently reverting."""
        if self.win._overlay_on:
            return   # recoloring here would tint the whole overlay by one model's plane
        if self.dev_on and self.win._has_points(self.win.cloud):
            n, c = self.win.level.board_plane()
            d, rng = deviation(self.win.cloud.points, n, c)
            # colors-only push: the full set_cloud re-serialized the entire cloud
            # (n×15 bytes + JS geometry rebuild) TWICE per repaint — once for the
            # photo repaint, once more just to change these colors
            self.win.viewer.cloud_view.set_cloud_colors(self._turbo(d, -rng, rng))

    def on_point_picked(self, x: float, y: float, z: float) -> None:
        if not self.tool or not self.win._has_points(self.win.cloud):
            return
        need = 1 if self.tool == "point" else 2
        p = np.array([x, y, z], np.float64)
        self.picked = [p] if len(self.picked) >= need else self.picked + [p]
        self.win.viewer.cloud_view.set_analyze_geom(
            markers=[list(q) for q in self.picked], line=None)
        if len(self.picked) >= need:
            self.compute()

    def compute(self) -> None:
        # no up-front float64 copy of the whole cloud: point/distance never touch
        # it (the copy was ~2× cloud memory per click for nothing), and profile/
        # region convert internally
        pts = self.win.cloud.points
        n, c = self.win.level.board_plane()
        u, dec = self.win._units, UNIT_DECIMALS.get(self.win._units, 2)
        off = self.z_off()                       # flat-reference correction (0 if none)
        zed = "  (zeroed)" if off else ""
        cv, tool = self.win.viewer.cloud_view, self.tool
        self.last_shown = tool                 # remember what's shown (to re-run on offset/rebuild)
        try:
            if tool == "point":
                P = self.picked[0]
                self.win.param_panel.set_analyze_result(
                    "Point · height", f"{float((P - c) @ n) - off:.{dec}f}", u,
                    rows=[("x", f"{P[0]:.{dec}f} {u}"),
                          ("y", f"{P[1]:.{dec}f} {u}"),
                          ("z", f"{P[2]:.{dec}f} {u}")],
                    caption="height above the board plane" + zed)
            elif tool == "distance":
                A, B = self.picked
                d = point_distance(A, B)
                cv.set_analyze_geom(markers=[list(A), list(B)], line=[list(A), list(B)])
                self.win.param_panel.set_analyze_result(
                    "Distance", f"{d['dist']:.{dec}f}", u,
                    rows=[("Δx", f"{d['dx']:+.{dec}f} {u}"),
                          ("Δy", f"{d['dy']:+.{dec}f} {u}"),
                          ("Δz", f"{d['dz']:+.{dec}f} {u}")])
            elif tool == "profile":
                A, B = self.picked
                r = surface_profile(pts, A, B, n, c, isolate=self.isolate)
                if r is None:
                    self.win.param_panel.set_analyze_out("No surface between those points — pick two on the part.")
                    return
                cv.set_analyze_geom(markers=[list(A), list(B)],
                                    line=[list(q) for q in r["poly"]])
                self.highlight_used(r.get("used"))
                self.win.param_panel.set_profile(r["t"], r["h"])
                self.win.param_panel.set_analyze_result(
                    "Surface angle", f"{r['angle']:+.{ANGLE_DECIMALS}f}°", "",
                    rows=[("rise", f"{r['d_height']:+.{dec}f} {u}"),
                          ("distance", f"{r['dist']:.{dec}f} {u}"),
                          ("samples", f"{r['n_pts']:,}")],
                    caption="slope vs the board plane")
            elif tool == "region":
                A, B = self.picked
                r = region_flatness(pts, A, B, n, c, isolate=self.isolate)
                if r is None:
                    self.win.param_panel.set_analyze_out("Empty region — pick two corners over the board.")
                    return
                cv.set_analyze_geom(markers=[list(A), list(B)], line=r["corners"])
                self.highlight_used(r.get("used"))
                self.last_region = r          # the raw result — what a flat-reference zeroes from
                self.win.param_panel.set_flat_ref_available(True)   # now there IS something to zero to
                self.win.param_panel.set_analyze_result(
                    "Region flatness", f"{r['rms']:.{dec}f}", u,
                    rows=[("max − min", f"{r['z_range']:.{dec}f} {u}"),
                          ("avg Z", f"{r['z_mean'] - off:+.{dec}f} {u}"),
                          ("local tilt", f"{r['local_tilt']:.{ANGLE_DECIMALS}f}°"),
                          ("size u", f"{r['size_u']:.{dec}f} {u}"),
                          ("size v", f"{r['size_v']:.{dec}f} {u}"),
                          ("points", f"{r['n_pts']:,}")],
                    caption="RMS vs patch plane · Z above board" + zed)
        except Exception as exc:   # noqa: BLE001 — analysis must never crash the UI
            self.win.param_panel.set_analyze_out(f"couldn't measure: {exc}")

    def z_off(self) -> float:
        """The active flat-reference offset in the DISPLAY unit (0 if no reference)."""
        if self.z_offset_m is None:
            return 0.0
        return float(self.z_offset_m) * UNIT_PER_M[self.win._units]

    def on_flat_ref(self, on: bool) -> None:
        """Zero board-referenced heights to the last flat Region (on), or clear (off).
        The offset is the region's average height (the cloud's systematic error at a zone
        that should read 0); it's stored in canonical metres so it survives a unit switch
        and applies to every cloud of the same fixture."""
        if on:
            r = self.last_region
            if r is None:
                self.win.param_panel.set_flat_ref_checked(False)
                self.win._set_status("Measure a Region on a flat zone first, then zero to it.")
                return
            self.z_offset_m = float(r["z_mean"]) / UNIT_PER_M[self.win._units]
            self.z_ref_pp_m = float(r["z_range"]) / UNIT_PER_M[self.win._units]
            self.win._set_status("Flat reference set — board-referenced heights are now corrected.")
        else:
            self.z_offset_m = None
            # un-applying may leave nothing to re-apply to (the region was reset)
            self.win.param_panel.set_flat_ref_available(self.last_region is not None)
        self.update_ref_label()
        self.refresh()       # re-run whatever's shown (incl. a pin) with/without the correction

    def refresh(self) -> None:
        """Re-run whatever the analyze card currently shows against the current cloud +
        settings (flat-ref offset, isolate) — so the readout never goes stale after the
        offset toggles or the cloud is rebuilt live. No-op if the card shows nothing."""
        if self.last_shown == "pin":
            self.on_pin_analyze()
        elif self.tool and self.picked and \
                len(self.picked) >= (1 if self.tool == "point" else 2):
            self.compute()

    def update_ref_label(self) -> None:
        u, dec = self.win._units, UNIT_DECIMALS.get(self.win._units, 2)
        if self.z_offset_m is None:
            self.win.param_panel.set_flat_ref_text("")
            return
        off = self.z_offset_m * UNIT_PER_M[u]
        pp = self.z_ref_pp_m * UNIT_PER_M[u]
        self.win.param_panel.set_flat_ref_text(
            f"correcting {-off:+.{dec}f} {u}   ·   flatness ±{pp / 2:.{dec}f} {u}")

    def highlight_used(self, used) -> None:
        """Light up the exact points a region/profile measured, so the user can confirm
        the right zone/level is used. Uniformly subsampled — a verification cue, not a
        full re-render — so the runJavaScript payload stays small."""
        if used is None or len(used) == 0:
            self.win.viewer.cloud_view.set_analyze_highlight(None)
            return
        used = np.asarray(used)
        cap = 5000
        if len(used) > cap:
            used = used[np.linspace(0, len(used) - 1, cap).astype(np.int64)]
        self.win.viewer.cloud_view.set_analyze_highlight(used)

    def on_isolate_layer(self, on: bool) -> None:
        """Toggle 'measure only the picked Z level' — re-run the live region/profile."""
        self.isolate = bool(on)
        if self.tool in ("region", "profile") and len(self.picked) >= 2:
            self.compute()

    def on_deviation(self, on: bool) -> None:
        if on and self.win._overlay_on:
            # the heatmap is one model's distance to ONE board plane — over a stack
            # of different models' points it's meaningless, and pushing it would
            # silently collapse the overlay to the single shown cloud
            self.win.param_panel.set_deviation_checked(False)
            self.win._set_status("Deviation heatmap shows a single model — untick Overlay first.")
            return
        self.dev_on = bool(on)
        if not self.win._has_points(self.win.cloud):
            return
        if on:
            n, c = self.win.level.board_plane()
            d, rng = deviation(self.win.cloud.points, n, c)
            # colors-only: toggling the heatmap re-tints the cloud on screen —
            # re-shipping every position for that was the single biggest
            # avoidable transfer in the app (~60 MB at 4M points)
            self.win.viewer.cloud_view.set_cloud_colors(self._turbo(d, -rng, rng))
            dec = UNIT_DECIMALS.get(self.win._units, 4)
            self.win._set_status(f"Deviation heatmap — ±{rng:.{dec}f} {self.win._units} about the board plane.")
        else:
            self.win.viewer.cloud_view.set_cloud_colors(self.win.cloud.colors)

    @staticmethod
    def _turbo(vals, lo, hi):
        import cv2
        t = np.clip((np.asarray(vals) - lo) / (hi - lo + 1e-12), 0.0, 1.0)
        lut = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(-1, 1),
                                cv2.COLORMAP_TURBO).reshape(-1, 3)[:, ::-1]   # BGR→RGB
        return np.ascontiguousarray(lut[(t * 255).astype(np.uint8)], np.uint8)

    def on_pin_analyze(self) -> None:
        if not self.win.param_panel.measure_on or not self.win._has_points(self.win.cloud):
            self.win._set_status("Turn on the Volume box and select a pin box first.")
            return
        specs = self.win.param_panel.box_specs()
        sel = self.win.param_panel.selected_index()
        if not specs or not (0 <= sel < len(specs)):
            self.win._set_status("Select a measure box on a pin.")
            return
        _name, box, _trim = specs[sel]
        mask = points_in_box(self.win.cloud.points, box)
        n, c = self.win.level.board_plane()
        r = pin_analysis(np.asarray(self.win.cloud.points)[mask], n, c)
        u, dec = self.win._units, UNIT_DECIMALS.get(self.win._units, 2)
        if r is None:
            self.win.param_panel.set_analyze_out("Pin box too sparse — place it tighter on the pin.")
            return
        off = self.z_off()            # flat-reference correction (0 if none)
        self.last_shown = "pin"     # remember (so an offset toggle / rebuild re-runs it)
        vert = f"{r['verticality']:.{ANGLE_DECIMALS}f}°" if r["verticality"] is not None else "—"
        self.win.param_panel.set_analyze_result(
            "Pin height", f"{r['height'] - off:.{dec}f}", u,
            rows=[("verticality", vert), ("points", f"{r['n_pts']:,}")],
            caption="above the board plane" + ("  (zeroed)" if off else ""))

