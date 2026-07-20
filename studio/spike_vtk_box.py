"""Standalone spike: does the VTK box widget + eye-dome lighting feel professional
on a REAL cloud, embedded in PySide6? Nothing here touches the app — it only reuses
measure.py so the numbers behind the box are the exact ones the app already reports.

Run:  .venv\\Scripts\\python.exe -m studio.spike_vtk_box  [optional_cloud.ply]

Drag the box's handles (6 faces + centre). The readout is live.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_API", "pyside6")   # tell qtpy which binding, before it loads

import numpy as np

DEFAULT_PLY = r"C:\Users\andre\Desktop\s2ms_test.ply"


def _load_cloud(path):
    """(points Nx3 float32, colors Nx3 uint8) from a PLY, via open3d (reliable
    vertex colours). Same world frame the app measures in."""
    import open3d as o3d

    pc = o3d.io.read_point_cloud(path)
    pts = np.asarray(pc.points, np.float32)
    if pc.has_colors():
        cols = (np.asarray(pc.colors) * 255.0).astype(np.uint8)
    else:
        cols = np.full((len(pts), 3), 200, np.uint8)
    return pts, cols


def main():
    from PySide6.QtWidgets import (QApplication, QLabel, QMainWindow, QVBoxLayout,
                                   QWidget)
    from pyvistaqt import QtInteractor

    from studio.measure import MeasureBox, measure_box

    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PLY
    pts, cols = _load_cloud(path)
    med = np.median(pts, axis=0)

    app = QApplication(sys.argv)
    win = QMainWindow()
    win.setWindowTitle(f"VTK box spike — {os.path.basename(path)}  ·  {len(pts):,} pts")
    win.resize(1180, 820)

    central = QWidget()
    lay = QVBoxLayout(central)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(0)
    plotter = QtInteractor(central)   # this IS a QWidget (QVTKRenderWindowInteractor)
    lay.addWidget(plotter)
    readout = QLabel("Drag the box handles — the readout is live.")
    readout.setStyleSheet(
        "background:#0b0e14; color:#e7ecf4; padding:8px 12px;"
        'font-family:"Cascadia Mono","Consolas",monospace; font-size:12px;')
    lay.addWidget(readout)
    win.setCentralWidget(central)

    # --- the cloud, shaded like a professional viewer would ---
    plotter.set_background("#0a0c12", top="#141924")     # subtle gradient
    plotter.add_points(
        pts, scalars=cols, rgb=True, point_size=2.0,
        render_points_as_spheres=False, name="cloud")
    plotter.enable_eye_dome_lighting()                    # depth-cued edges
    plotter.enable_parallel_projection()                  # measurement-friendly (no perspective foreshortening)

    # --- the measure box widget: orthogonal (no rotation), 6 face + centre handles ---
    def on_box(box, *_):
        b = box.bounds     # (xmin, xmax, ymin, ymax, zmin, zmax) in WORLD coords — no flip
        mb = MeasureBox(cx=(b[0] + b[1]) / 2, cy=(b[2] + b[3]) / 2, cz=(b[4] + b[5]) / 2,
                        sx=b[1] - b[0], sy=b[3] - b[2], sz=b[5] - b[4])
        m = measure_box(pts, mb, trim_pct=2.0, voxel=0.5)
        if m is None:
            readout.setText("▣  box is empty — no points inside it")
            return
        fill = "—" if m["fill_pct"] is None else f"{m['fill_pct']:.1f}%"
        readout.setText(
            f"▣  {m['n']:,} pts   ·   "
            f"z {m['z_min']:.2f}→{m['z_max']:.2f} mm (span {m['z_span']:.2f})   ·   "
            f"trim2% span {m['z_span_t']:.2f} mm   ·   "
            f"extent {m['ext'][0]:.2f}×{m['ext'][1]:.2f}×{m['ext'][2]:.2f}   ·   "
            f"filled {fill}")

    s = 6.0
    b0 = (med[0] - s / 2, med[0] + s / 2, med[1] - s / 2, med[1] + s / 2,
          med[2] - s / 2, med[2] + s / 2)
    plotter.add_box_widget(callback=on_box, bounds=b0, rotation_enabled=False,
                           color="#f5d95c", use_planes=False)

    plotter.add_text("VTK box widget · drag a face or the centre", font_size=9,
                     position="upper_left", color="#9fb0c8", name="hdr")
    plotter.reset_camera()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
