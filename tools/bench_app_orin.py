"""Whole-app validation of the Orin default config: drives the REAL MainWindow
(GUI + QtWebEngine 3D view + engine child) through one rig-sized pair at the
recommended Input scale, and reports wall time and the system-memory floor.

    DISPLAY=:0 .venv/bin/python tools/bench_app_orin.py <scale> [left.png right.png]

Follows verify_full_process.py's QSettings discipline: the user's saved measure
boxes are snapshotted and handed back before close.
"""
import os
import sys
import threading
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

SCALE = float(sys.argv[1]) if len(sys.argv) > 1 else 0.30


def meminfo_available_mb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024.0


class Floor(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.min_mb = meminfo_available_mb()
        self.stop = threading.Event()

    def run(self):
        while not self.stop.is_set():
            self.min_mb = min(self.min_mb, meminfo_available_mb())
            time.sleep(0.2)


# rig-sized pair on disk (upscaled demo unless a real pair is given)
if len(sys.argv) > 3:
    left_path, right_path = sys.argv[2], sys.argv[3]
else:
    import cv2
    import imageio.v2 as iio
    d = os.path.join(os.environ.get("TMPDIR", "/tmp"), "fs_bench_pair")
    os.makedirs(d, exist_ok=True)
    left_path = os.path.join(d, "left.png")
    right_path = os.path.join(d, "right.png")
    if not (os.path.exists(left_path) and os.path.exists(right_path)):
        for src, dst in (("left.png", left_path), ("right.png", right_path)):
            img = iio.imread(os.path.join(REPO, "assets", src))[..., :3]
            iio.imwrite(dst, cv2.resize(img, (2664, 2304),
                                        interpolation=cv2.INTER_CUBIC))

from PySide6.QtCore import QCoreApplication, QElapsedTimer, QEventLoop, Qt, QTimer

QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
from PySide6.QtWidgets import QApplication

app = QApplication([])


def sleep_ms(ms):
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def wait_for(pred, timeout_ms, step=250):
    t = QElapsedTimer()
    t.start()
    while t.elapsed() < timeout_ms:
        if pred():
            return True
        sleep_ms(step)
    return False


from studio.main_window import MainWindow
from studio.theme import apply_theme

apply_theme(app, "dark")
win = MainWindow("dark")
win.show()
errors = []
win.worker.error.connect(lambda m: errors.append(str(m)))
assert wait_for(lambda: win.worker.alive, 20000), "engine child did not connect"

floor = Floor()
start_mb = floor.min_mb
floor.start()

win.input_panel.rect_mode.setCurrentIndex(0)          # already-rectified mode
win.input_panel.load_image(left_path, "left")
win.input_panel.load_image(right_path, "right")
# plausible close-range rig intrinsics — geometry only has to be self-consistent
win.input_panel.fx.setValue(2000.0)
win.input_panel.fy.setValue(2000.0)
win.input_panel.cx.setValue(1332.0)
win.input_panel.cy.setValue(1152.0)
win.input_panel.baseline.setValue(5.0)                # mm display unit
win.param_panel.scale.setValue(SCALE)
win.param_panel.z_far.setValue(10000.0)

saved_boxes = win.param_panel.boxes_blob()
t0 = time.time()
win._run()
ok_model = wait_for(lambda: win._model_ready or errors, 300000, 500) and not errors
t_model = time.time() - t0
ok_res = wait_for(lambda: win.result is not None or errors, 300000, 500) and not errors
t_res = time.time() - t0
ok_cloud = wait_for(lambda: win.cloud is not None or errors, 120000, 500) and not errors
t_cloud = time.time() - t0

sleep_ms(1500)                                        # let the 3D view ingest
floor.stop.set()
floor.join(timeout=2)

r, c = win.result, win.cloud
print(f"scale={SCALE}  model_ok={ok_model}({t_model:.1f}s)  "
      f"infer_ok={ok_res}({t_res:.1f}s)  cloud_ok={ok_cloud}({t_cloud:.1f}s)")
if errors:
    print("ERRORS:", "; ".join(errors)[:300])
if r is not None:
    d = r.disp
    print(f"disp {d.shape[1]}x{d.shape[0]}  valid {(d > 0).mean():.1%}")
if c is not None:
    print(f"cloud n={c.n}")
print(f"mem: start {start_mb:.0f} MB avail → floor {floor.min_mb:.0f} MB avail "
      f"(app+engine dip {start_mb - floor.min_mb:.0f} MB)")

win.param_panel.restore_boxes(saved_boxes)
win.close()
sleep_ms(800)
sys.exit(0 if (ok_model and ok_res and ok_cloud) else 1)
