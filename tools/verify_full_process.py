"""Full-process live verification: app start → images → calibration → model
load → inference → point cloud → display → measure → export.

Drives the REAL app on the bundled demo pair (assets/left.png + right.png with
its auto-loaded K.txt), loading the actual model on the actual GPU. Takes a
model-load (~4–20 s) plus one inference. Run after pipeline-level changes:

    .venv/Scripts/python.exe tools/verify_full_process.py
"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from PySide6.QtCore import QCoreApplication, QElapsedTimer, QEventLoop, Qt, QTimer

QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
from PySide6.QtWidgets import QApplication

app = QApplication([])
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(("PASS  " if ok else "FAIL  ") + name + (f"   [{detail}]" if detail and not ok else ""))


def sleep_ms(ms):
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def wait_for(pred, timeout_ms, step=200):
    t = QElapsedTimer()
    t.start()
    while t.elapsed() < timeout_ms:
        if pred():
            return True
        sleep_ms(step)
    return False


import json


def js(page, code):
    loop = QEventLoop()
    out = {}

    def done(r):
        out["v"] = r
        loop.quit()

    page.runJavaScript(code, 0, done)
    loop.exec()
    return out.get("v")


from studio.main_window import MainWindow
from studio.theme import apply_theme

apply_theme(app, "dark")
win = MainWindow("dark")
win.show()
errors = []
win.worker.error.connect(lambda m: errors.append(str(m)))

# ---- 1. initialization ------------------------------------------------------
check("1.1 engine child spawned and connected", wait_for(lambda: win.worker.alive, 15000))
check("1.2 nothing loaded at startup (Run reads Load & Run)", win._needs_load())
check("1.3 Measure/Analyze gated (no cloud yet)",
      win.param_panel.sec_measure.is_gated() and win.param_panel.sec_analyze.is_gated())

# ---- 2. load the demo pair (auto-K.txt calibration) -------------------------
win.input_panel.rect_mode.setCurrentIndex(0)          # already-rectified mode
left = os.path.join(REPO, "assets", "left.png")
right = os.path.join(REPO, "assets", "right.png")
win.input_panel.load_image(left, "left")
win.input_panel.load_image(right, "right")
check("2.1 pair loaded", win.input_panel.ready)
check("2.2 K.txt auto-loaded and plausible for this pair",
      win.input_panel.has_calibration
      and abs(win.input_panel.fx.value() - 754.668) < 0.01)
unit_f = {"mm": 1000.0, "µm": 1e6, "m": 1.0}[win._units]
check("2.3 baseline converted to the display unit",
      abs(win.input_panel.baseline.value() - 0.063 * unit_f) < 1e-6 * unit_f,
      f"unit={win._units} value={win.input_panel.baseline.value()}")
check("2.4 cloud-settings section ungated by calibration",
      not win.param_panel.sec_cloud.is_gated())

# ---- 3. run: load the real model, infer, build the cloud --------------------
# the demo scene sits at ~0.5–1.5 m; the panel's PCB-tuned default z-far
# (250 mm) would clip ALL of it (the app now says so in the status bar)
win.param_panel.z_far.setValue(2.0 * unit_f)          # 2 m in the display unit
model = win.input_panel.current_backend_key()
print(f"      (loading model '{model}' on the real GPU…)")
win._run()
check("3.1 model loads", wait_for(lambda: win._model_ready or errors, 180000, 500)
      and not errors, "; ".join(errors)[:200])
check("3.2 inference lands", wait_for(lambda: win.result is not None or errors, 240000, 500)
      and not errors, "; ".join(errors)[:200])
check("3.3 cloud lands", wait_for(lambda: win.cloud is not None or errors, 120000, 500)
      and not errors, "; ".join(errors)[:200])

r, c = win.result, win.cloud
if r is not None:
    scale = win.param_panel.scale.value()
    check("3.4 disparity at the working scale",
          r.disp.shape == (int(540 * scale), int(960 * scale)), str(r.disp.shape))
    valid = r.disp > 0
    check("3.5 model produced dense disparity", valid.mean() > 0.9, f"{valid.mean():.2%}")
    med = float(np.median(r.depth[r.depth > 0]))
    check("3.6 median depth plausible for the demo scene (0.3–3 m)",
          0.3 * unit_f <= med <= 3.0 * unit_f, f"med={med:.4g} {win._units}")
    check("3.7 depth = fx·B/disp spot check",
          abs(float(r.K[0, 0]) * r.baseline / float(r.disp[r.disp > 0].flat[0])
              - float(r.depth[r.disp > 0].flat[0])) < 1e-3)
if c is not None:
    check("3.8 cloud has substance", c.n > 50000, f"n={c.n}")
    check("3.9 cloud z respects z-far",
          c.n > 0 and float(c.points[:, 2].max()) <= win.param_panel.z_far.value() + 1e-6)
    page = win.viewer.cloud_view.view.page()
    got = wait_for(lambda: (lambda d: d and d.get("N") == c.n)(
        (lambda v: json.loads(v) if v else None)(js(page, "window._dbg?window._dbg():null"))), 20000)
    check("3.10 cloud displayed in the 3D view (N matches)", got)
    check("3.11 Measure/Analyze ungated by the cloud",
          not win.param_panel.sec_measure.is_gated())

# ---- 4. measure on the real cloud ------------------------------------------
# The user's SAVED pin boxes restore at startup — snapshot them, work on a
# clean slate, and hand the real layout back before close (closeEvent persists
# whatever the panel holds; clobbering a hand-built pin layout is not ok).
saved_boxes = win.param_panel.boxes_blob()
if c is not None:
    while win.param_panel.has_boxes():
        win.param_panel.remove_box()
    win.param_panel.measure_sw.setChecked(True)        # auto-adds a snapped, centred box
    sleep_ms(400)
    specs = win.param_panel.box_specs()
    check("4.1 a fresh box was auto-added and selected", len(specs) == 1)
    from studio.measure import measure_box
    m = measure_box(c.points, specs[0][1], trim_pct=specs[0][2])
    check("4.2 the fresh box lands ON the cloud (snap-to-surface)",
          m is not None and m["n"] >= 1, str(None if m is None else m["n"]))
    # grow it to a scene-appropriate size through the spin path
    for w in (win.param_panel.box_sx, win.param_panel.box_sy, win.param_panel.box_sz):
        w.setValue(0.15 * unit_f)                      # a 15 cm cube
    sleep_ms(300)
    specs = win.param_panel.box_specs()
    m = measure_box(c.points, specs[0][1], trim_pct=specs[0][2])
    check("4.3 the grown box catches a real patch", m is not None and m["n"] > 500,
          str(None if m is None else m["n"]))

# ---- 5. export --------------------------------------------------------------
if r is not None:
    out = os.path.join(os.environ.get("TEMP", "."), "fs_verify_export")
    os.makedirs(out, exist_ok=True)
    import imageio.v2 as imageio
    from studio.engine import StereoEngine

    np.save(os.path.join(out, "disp.npy"), r.disp)
    imageio.imwrite(os.path.join(out, "disp.png"), win.viewer.disp_view.render_rgb())
    if c is not None:
        StereoEngine.save_cloud(os.path.join(out, "cloud.ply"), c)
    ok = (os.path.getsize(os.path.join(out, "disp.npy")) > 1000
          and os.path.getsize(os.path.join(out, "disp.png")) > 1000
          and (c is None or os.path.getsize(os.path.join(out, "cloud.ply")) > 10000))
    check("5.1 exports written (npy/png/ply)", ok, out)

win.param_panel.restore_boxes(saved_boxes)   # the user's real pin layout goes back
win.close()                                  # …and closeEvent persists THAT
sleep_ms(800)
fails = [x for x in RESULTS if not x[1]]
print(f"\n{len(RESULTS) - len(fails)}/{len(RESULTS)} checks passed")
sys.exit(1 if fails else 0)
