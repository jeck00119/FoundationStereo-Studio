"""Exporting a run: disparity/depth as PNG or raw .npy, the cloud as PLY.

Self-contained — it reads the shown result and writes files, and touches no
other part of the window's state. Failures route through the window's
``_report_error`` rather than its engine error slot: that one advances the
comparison state machine, so a disk error there used to pop the queue and bank
whichever model was mid-run as 'failed'.
"""
from __future__ import annotations

import os

import numpy as np
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QFileDialog, QMenu

#: (menu label, kind) — kind is what run() switches on
MENU = [
    ("Disparity image (PNG)…", "disp_png"),
    ("Depth image (PNG)…", "depth_png"),
    ("Disparity — raw (.npy)…", "disp_npy"),
    ("Depth — raw (.npy)…", "depth_npy"),
    ("Point cloud (.ply)…", "ply"),
    ("Everything → folder…", "all"),
]


class ExportController:
    """Public surface: ``build_menu()`` · ``run(kind)``."""

    def __init__(self, win) -> None:
        self.win = win

    def build_menu(self) -> None:
        win = self.win
        m = QMenu(win)
        for label, kind in MENU:
            a = QAction(label, win)
            a.triggered.connect(lambda _=False, k=kind: self.run(k))
            m.addAction(a)
        win.export_btn.setMenu(m)

    def run(self, kind: str) -> None:
        win = self.win
        if win.result is None:
            return
        import imageio.v2 as imageio

        from ..engine import StereoEngine

        r = win.result
        if kind == "all":
            d = QFileDialog.getExistingDirectory(win, "Export everything to…")
            if not d:
                return
            try:
                base = os.path.join(d, "fs_output")
                imageio.imwrite(base + "_disparity.png", win.viewer.disp_view.render_rgb())
                np.save(base + "_disparity.npy", r.disp)
                if r.depth is not None:
                    imageio.imwrite(base + "_depth.png", win.viewer.depth_view.render_rgb())
                    np.save(base + f"_depth_{win._units}.npy", r.depth)
                if win.cloud is not None:
                    StereoEngine.save_cloud(base + "_cloud.ply", win.cloud)
                win._set_status(f"Exported to {d}")
            except Exception as exc:  # noqa: BLE001
                win._report_error(str(exc))
            return

        specs = {
            "disp_png": ("Save disparity image", "PNG (*.png)", ".png"),
            "depth_png": ("Save depth image", "PNG (*.png)", ".png"),
            "disp_npy": ("Save raw disparity", "NumPy (*.npy)", ".npy"),
            "depth_npy": (f"Save depth ({win._units})", "NumPy (*.npy)", ".npy"),
            "ply": ("Save point cloud", "PLY (*.ply)", ".ply"),
        }
        title, filt, ext = specs[kind]
        path, _ = QFileDialog.getSaveFileName(win, title, "", filt)
        if not path:
            return
        if not path.lower().endswith(ext):
            path += ext
        try:
            if kind == "disp_png":
                imageio.imwrite(path, win.viewer.disp_view.render_rgb())
            elif kind == "depth_png":
                if r.depth is None:
                    raise ValueError("No depth — set calibration first.")
                imageio.imwrite(path, win.viewer.depth_view.render_rgb())
            elif kind == "disp_npy":
                np.save(path, r.disp)
            elif kind == "depth_npy":
                if r.depth is None:
                    raise ValueError("No depth — set calibration first.")
                np.save(path, r.depth)
            elif kind == "ply":
                if win.cloud is None:
                    raise ValueError("No point cloud — set calibration and run.")
                StereoEngine.save_cloud(path, win.cloud)
            win._set_status(f"Saved {os.path.basename(path)}")
        except Exception as exc:  # noqa: BLE001
            win._report_error(str(exc))
