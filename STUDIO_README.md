# FoundationStereo Studio

A Windows desktop app for close-range stereo metrology (PCB / pin-height
measurement) built on top of this FoundationStereo checkout. Upstream's own
readme is `readme.md`; this file covers only the local additions.

## Run

```
run_studio.bat            (console-less; logs land in %TEMP%\fs_studio_*.log)
```
or, with a console: `.venv\Scripts\python.exe -m studio.app`

The engine (CUDA + the model) runs in a separate child process, so the GUI
never freezes; each model switch gets a fresh child with clean VRAM.

## Layout

| Path | What it is |
|---|---|
| `studio/` | the app: `main_window.py` (workflows), `panels.py` (inputs/params), `viewers.py` + `web_cloud.py` + `web/` (2D views + three.js 3D view), `worker.py`/`engine_process.py` (GUI↔engine), `backends/` (FoundationStereo · Fast-FoundationStereo · S²M²), `infer.py`/`cloud.py` (shared pipeline), `pairs.py` (Qt-free loading + pair discovery), `measure.py`/`analyze.py`/`repeat.py`/`compare.py`/`batch.py` (metrology tools) |
| `tools/calibrate.py` | single-camera CNC-rig checkerboard calibration → `calib.json` (+ optional `k_rectified.txt`) |
| `data/` | your captures & calibration outputs (git-ignored — keep user data OUT of `assets/`, which holds upstream's demo pair and its `K.txt`) |
| `tests/` | unit tests for the pure-math/pipeline modules: `.venv\Scripts\python.exe -m pytest tests` |
| `requirements.txt` | pinned environment (see its header for the torch/cu128 install step) |

Fast-FoundationStereo and S²M² are expected as sibling clones
(`..\Fast-FoundationStereo`, `..\s2m2`) — see `studio/backends/registry.py`.

## Calibrate the rig

```
.venv\Scripts\python.exe tools\calibrate.py <checkerboard_folder> --cols 9 --rows 6 --square 20 --out data\calib\calib.json
```
Load the resulting `calib.json` in the app's "Raw — rectify with calibration"
mode (baseline unit = the `--unit` you calibrated in). The tool refuses to
write a solve whose reprojection RMS fails sanity gates unless `--force`.

## Conventions worth knowing

- `K.txt` files are metres (upstream convention); the app converts on load and
  refuses to auto-apply a `K.txt` whose principal point can't belong to the
  loaded image (guards against picking up the demo calibration by accident).
- Depth/cloud units follow the display unit (mm/µm/m); everything rescales
  losslessly on a unit switch — no re-run.
- The input pair must be rectified (or use raw mode with a calibration); left
  really means the left-side camera.
