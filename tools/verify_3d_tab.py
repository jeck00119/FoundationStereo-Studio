"""Live end-to-end verification of every 3D Cloud tab setting.

Launches the REAL WebCloudView and a real MainWindow, pushes synthetic clouds,
drives every control, and asserts INSIDE the WebGL page (via window._dbg) what
each one actually did. Not part of the pytest suite (it opens windows and takes
~30 s) — run it after touching web_cloud.py / cloud.html:
    .venv/Scripts/python.exe tools/verify_3d_tab.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PySide6.QtCore import QCoreApplication, QElapsedTimer, QEventLoop, Qt, QTimer

QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
from PySide6.QtWidgets import QApplication

app = QApplication([])
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(("PASS  " if ok else "FAIL  ") + name + (f"   [{detail}]" if detail and not ok else ""))


def js(page, code):
    loop = QEventLoop()
    out = {}

    def done(r):
        out["v"] = r
        loop.quit()

    page.runJavaScript(code, 0, done)
    loop.exec()
    return out.get("v")


def dbg(page):
    v = js(page, "window._dbg ? window._dbg() : null")
    return json.loads(v) if v else None


def sleep_ms(ms):
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def wait_dbg(page, pred, timeout=10000):
    t = QElapsedTimer()
    t.start()
    while t.elapsed() < timeout:
        d = dbg(page)
        if d is not None and pred(d):
            return d
        sleep_ms(100)
    return None


def combo_items(cb):
    return [cb.itemText(i) for i in range(cb.count())]


# ======================================================= Stage A: WebCloudView
from studio.web_cloud import WebCloudView
from studio.measure import MeasureBox

cv = WebCloudView()
cv.resize(900, 600)
cv.show()
page = cv.view.page()
check("A0 page + bridge ready", wait_dbg(page, lambda d: True) is not None and
      (lambda: [sleep_ms(100) for _ in range(50) if not cv._ready] and None or cv._ready)() or cv._ready)

rng = np.random.default_rng(5)
NL = NR = 600
left = np.column_stack([rng.uniform(-10, -1, NL), rng.uniform(-5, 5, NL), rng.uniform(40, 60, NL)])
right = np.column_stack([rng.uniform(1, 10, NR), rng.uniform(-5, 5, NR), rng.uniform(40, 60, NR)])
pts = np.vstack([left, right]).astype(np.float32)
cols = np.full((NL + NR, 3), 120, np.uint8)
origin = np.r_[np.zeros(NL, np.uint8), np.ones(NR, np.uint8)]
reliable = (np.arange(NL + NR) % 2 == 0)

cv.set_cloud(pts, cols, origin=origin, reliable=reliable, reset_view=True)
d = wait_dbg(page, lambda d: d["N"] == NL + NR and d["hasOri"] and d["hasRel"])
check("A1 cloud arrives in JS (N, origin, reliable)", d is not None, str(dbg(page)))
check("A2 color modes follow the data",
      combo_items(cv.color_combo) == ["Photo", "Camera (L·R)", "Reliability"],
      str(combo_items(cv.color_combo)))
check("A3 color group visible", cv._color_group.isVisibleTo(cv))
d = dbg(page)
check("A4 label grid step == drawn grid step", abs(d["gridStep"] - cv._grid_step) < 1e-9,
      f"js={d['gridStep']} py={cv._grid_step}")

cv.color_combo.setCurrentText("Camera (L·R)")
check("A5 camera mode reaches JS", wait_dbg(page, lambda d: d["colorMode"] == "camera") is not None)
lc = json.loads(js(page, "window._dbgCol(0)"))
rc = json.loads(js(page, f"window._dbgCol({NL})"))
check("A6 left point painted LEFT blue", np.allclose(lc, np.array([74, 144, 226]) / 255, atol=1e-3), str(lc))
check("A7 right point painted RIGHT orange", np.allclose(rc, np.array([245, 145, 60]) / 255, atol=1e-3), str(rc))

cv.color_combo.setCurrentText("Reliability")
wait_dbg(page, lambda d: d["colorMode"] == "reliability")
okc = json.loads(js(page, "window._dbgCol(0)"))      # even index = reliable
badc = json.loads(js(page, "window._dbgCol(1)"))     # odd index = occluded
check("A8 reliable point green", np.allclose(okc, np.array([70, 200, 130]) / 255, atol=1e-3), str(okc))
check("A9 occluded point red", np.allclose(badc, np.array([235, 80, 100]) / 255, atol=1e-3), str(badc))

cv.set_point_size(4.0)
check("A10 point size uniform = 4×dpr",
      wait_dbg(page, lambda d: abs(d["uSize"] - 4 * js(page, "devicePixelRatio")) < 1e-6) is not None)

box = MeasureBox(cx=5.5, cy=0, cz=50, sx=9.0, sy=10.0, sz=20.0)
cv.set_boxes([box], 0, True)
check("A11 gizmo active on editable box",
      wait_dbg(page, lambda d: d["nBoxes"] == 1 and d["tcEnabled"]) is not None)
check("A12 in-box group visible", cv._inbox_group.isVisibleTo(cv))
check("A13 hint switched to gizmo keys",
      wait_dbg(page, lambda d: "W move" in d["hint"]) is not None)
cv.hi_combo.setCurrentText("Height map")
check("A14 height-map mode + ramp fits box contents",
      wait_dbg(page, lambda d: d["uMode"] == 2 and -10.5 < d["uHRange"][0] < d["uHRange"][1] < 10.5) is not None,
      str(dbg(page)["uHRange"]))
cv.hi_combo.setCurrentText("Trim")
cv.set_box_scalars(-1.25, 1.25)
check("A15 trim band in uniforms",
      wait_dbg(page, lambda d: d["uMode"] == 3 and d["uTrim"] == [-1.25, 1.25]) is not None)
cv.isolate_chk.setChecked(True)
check("A16 isolate + box uniforms live",
      wait_dbg(page, lambda d: d["uIsolate"] == 1 and d["uBoxOn"] == 1 and abs(d["uBoxCx"] - 5.5) < 1e-6) is not None)
cv.isolate_chk.setChecked(False)
cv.hi_combo.setCurrentText("Off")

js(page, "window._dbgSetCam(500,500,-2000,400,400,0)")
js(page, "api.resetView(); '';")
d = dbg(page)
diag = float(np.linalg.norm(pts.max(0) - pts.min(0)))
check("A17 Fit reframes the cloud", abs(d["camDist"] - 0.9 * diag) < 0.05 * diag,
      f"dist={d['camDist']:.1f} expect={0.9*diag:.1f}")
check("A18 Fit refits orbit limits", abs(d["maxDist"] - 15 * diag) < 0.05 * diag)

cv.color_combo.setCurrentText("Camera (L·R)")
wait_dbg(page, lambda d: d["colorMode"] == "camera")
js(page, "window._dbgStamp(7)")
heat = np.zeros((NL + NR, 3), np.uint8)
heat[:, 0] = 255
cv.set_cloud_colors(heat)
check("A19 recolor forces Photo mode (deviation-visibility fix)",
      cv.color_combo.currentText() == "Photo"
      and wait_dbg(page, lambda d: d["colorMode"] == "photo") is not None)
check("A20 heat colors actually shown",
      wait_dbg(page, lambda d: (lambda c: abs(c[0] - 1.0) < 1e-3 and c[1] < 1e-3)(
          json.loads(js(page, "window._dbgCol(0)")))) is not None)
check("A21 geometry NOT rebuilt (colors-only path)", dbg(page)["stamp"] == 7)

cv.set_cloud(pts, cols, reset_view=False)      # plain cloud: no origin/reliable
d = wait_dbg(page, lambda d: d["colorMode"] == "photo" and not d["hasOri"] and d["stamp"] is None)
check("A22 modes fall back with the data", d is not None and combo_items(cv.color_combo) == ["Photo"],
      str(combo_items(cv.color_combo)))
check("A23 color group hides when photo-only", not cv._color_group.isVisibleTo(cv))

a = np.random.default_rng(1).uniform(-5, 5, (500, 3)).astype(np.float32)
b = np.random.default_rng(2).uniform(-5, 5, (800, 3)).astype(np.float32)
cv.set_cloud(a, np.full((500, 3), 90, np.uint8), reset_view=False)
cv.set_cloud(b, np.full((800, 3), 90, np.uint8), reset_view=False)
check("A24 last-issued cloud wins", wait_dbg(page, lambda d: d["N"] == 800) is not None)
sleep_ms(700)
check("A25 …and stays won (no late-fetch flip)", dbg(page)["N"] == 800)

cv.set_units("µm")
check("A26 unit/decimals reach the drag readout",
      wait_dbg(page, lambda d: d["liveUnit"] == "µm" and d["liveDec"] == 1) is not None)
cv.clear()
check("A27 clear empties the page",
      wait_dbg(page, lambda d: d["N"] == 0 and not d["hasPoints"]) is not None)
check("A28 clear hides contextual groups + hint reverts",
      not cv._inbox_group.isVisibleTo(cv) and not cv._color_group.isVisibleTo(cv)
      and cv.cloud_lbl.text() == "" and "hover a point" in dbg(page)["hint"])

cv.close()
del cv

# ==================================================== Stage B: window wiring
from studio.main_window import MainWindow
from studio.theme import apply_theme
from studio.dtypes import CloudResult
from studio.measure import fit_plane

apply_theme(app, "dark")
win = MainWindow("dark")
win.show()
page2 = win.viewer.cloud_view.view.page()
check("B0 window page ready", wait_dbg(page2, lambda d: True) is not None)

ang = np.deg2rad(10.0)
xy = rng.uniform(-15, 15, (30000, 2))
z = 55.0 + np.tan(ang) * xy[:, 1] + rng.normal(0, 0.05, len(xy))
bpts = np.column_stack([xy[:, 0], xy[:, 1], z]).astype(np.float32)
bcols = np.full((len(bpts), 3), 140, np.uint8)
cloud = win._ingest_level(CloudResult(points=bpts, colors=bcols, n=len(bpts)))
win.cloud = cloud
win.viewer.show_cloud(cloud, reset_view=True)
win._apply_measure()
check("B1 fabricated cloud reaches JS",
      wait_dbg(page2, lambda d: d["N"] == len(bpts)) is not None)
check("B2 Measure/Analyze ungated by the cloud",
      not win.param_panel.sec_measure.is_gated() and not win.param_panel.sec_analyze.is_gated())

builds0 = dbg(page2)["builds"]
n0, _ = fit_plane(win.cloud.points)
tilt0 = np.degrees(np.arccos(min(1.0, abs(n0[2]))))
win._on_level_toggled(True)
n1, _ = fit_plane(win.cloud.points)
check("B3 Level flattens the board", abs(n1[2]) > 0.9999 and tilt0 > 9.0,
      f"tilt before={tilt0:.2f} after={np.degrees(np.arccos(min(1.0, abs(n1[2])))):.4f}")
check("B4 raw points keep the tilt (per-cloud raw)", abs(fit_plane(win.cloud.raw_points)[0][2]) < 0.999)
win._on_level_toggled(False)
check("B5 Level off restores the raw frame",
      np.allclose(win.cloud.points, win.cloud.raw_points, atol=1e-9))

# the level pushes must LAND before stamping the geometry. Two rapid pushes
# produce ONE build by design (the superseded fetch is dropped), so expect
# >= +1 and then confirm stability.
d = wait_dbg(page2, lambda d: d["builds"] >= builds0 + 1 and d["N"] == len(bpts))
sleep_ms(500)
check("B5b level pushes settled (rapid pushes coalesce to one build)",
      d is not None and dbg(page2)["builds"] == d["builds"])
js(page2, "window._dbgStamp(9)")
win._on_deviation(True)
check("B6 deviation heatmap shows (colors change)",
      wait_dbg(page2, lambda d: (lambda c: abs(c[0] - 140 / 255) > 0.05)(
          json.loads(js(page2, "window._dbgCol(0)")))) is not None)
check("B7 deviation is colors-only (no rebuild)", dbg(page2)["stamp"] == 9)
win._on_deviation(False)
check("B8 deviation off restores photo colors",
      wait_dbg(page2, lambda d: (lambda c: abs(c[0] - 140 / 255) < 1e-3)(
          json.loads(js(page2, "window._dbgCol(0)")))) is not None)

p0 = float(win.cloud.points[0, 0])
zf0 = win.param_panel.z_far.value()
win._set_units("µm")
check("B9 cloud points rescaled ×1000", abs(float(win.cloud.points[0, 0]) - p0 * 1000) < 1e-3)
check("B10 z-far slider rescaled ×1000", abs(win.param_panel.z_far.value() - zf0 * 1000) < 1e-6)
check("B11 page reloaded in µm",
      wait_dbg(page2, lambda d: d["liveUnit"] == "µm" and d["pos0"] is not None
               and abs(d["pos0"] - p0 * 1000) < 1) is not None)
check("B12 profile axes carry the unit",
      "µm" in win.param_panel.analyze_plot.getAxis("left").labelText)
win._set_units("mm")
check("B13 back to mm losslessly", abs(float(win.cloud.points[0, 0]) - p0) < 1e-6)

win.close()
sleep_ms(500)
fails = [r for r in RESULTS if not r[1]]
print(f"\n{len(RESULTS) - len(fails)}/{len(RESULTS)} checks passed")
sys.exit(1 if fails else 0)
