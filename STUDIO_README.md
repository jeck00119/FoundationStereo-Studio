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
never freezes; each model switch gets a fresh child with clean VRAM. With
Windows Triton installed (it is, in this venv), Fast-FoundationStereo runs its
compiled cost-volume path — the first run after changing Input scale or Max
disparity pauses once while kernels compile.

## Layout

| Path | What it is |
|---|---|
| `studio/` | the app: `main_window.py` (workflows), `panels.py` (inputs/params), `viewers.py` + `web_cloud.py` + `web/` (2D views + three.js 3D view), `worker.py`/`engine_process.py` (GUI↔engine), `backends/` (FoundationStereo · Fast-FoundationStereo · S²M²), `infer.py`/`cloud.py` (shared pipeline), `pairs.py` (Qt-free loading + pair discovery), `measure.py`/`analyze.py`/`repeat.py`/`compare.py`/`batch.py` (metrology tools) |
| `tools/calibrate.py` | stereo calibration → `calib.json` (+ optional `k_rectified.txt`). Checkerboard or **ChArUco** (`--charuco CXxRY --marker … --dict …`, partial views OK); `--simple-lens` pins the principal point and zeroes distortion — use it for machine-vision lenses whose image circle dwarfs the sensor |
| `tools/pair_captures.py` | verifies + names a CNC capture session into `poseNN_left/right.jpg` by the pair's optical signature (pure horizontal shift, rows preserved) |
| `tools/verify_3d_tab.py` | live check of every 3D-tab control against the real WebGL page (~30 s); run after touching `web_cloud.py`/`cloud.html` |
| `tools/verify_full_process.py` | full-chain live check on the real GPU: images → calibration → model → cloud → measure → export |
| `data/` | your captures & calibration outputs (git-ignored): `calib/calib_provisional.json`, exported clouds, … |
| `tests/` | pytest suite (ground-truth math for the pipeline, calibration, measurement, GUI behavior): `.venv\Scripts\python.exe -m pytest tests` |
| `requirements.txt` | pinned environment (see its header for the torch/cu128 install step) |

Fast-FoundationStereo and S²M² are expected as sibling clones
(`..\Fast-FoundationStereo`, `..\s2m2`) — see `studio/backends/registry.py`.

## Calibrate the rig (ChArUco workflow)

Per board pose: shoot at CNC position A → jog the measurement step (+X) →
shoot at position B → then reposition/tilt the board. 10–15 poses, mixed
heights and tilts, corners reaching the frame edges. Then:

```
.venv\Scripts\python.exe tools\pair_captures.py <capture_folder>
.venv\Scripts\python.exe tools\calibrate.py <capture_folder>\paired --charuco 11x8 --square <MEASURED> --marker <MEASURED*2/3> --simple-lens --out data\calib\calib.json --krect
```

`--square` is the caliper-MEASURED size (printers rescale by ~1 %). The tool
refuses solves whose reprojection RMS fails sanity gates unless `--force`.
Load the resulting `calib.json` in the app's **"Raw — rectify with
calibration"** mode (baseline unit = your `--unit`). `k_rectified.txt` is only
for images already rectified offline — never for raw captures.

## Current rig quick-start (Basler acA4024 + 35 mm lens, ~200 mm distance)

Raw—rectify + `data\calib\calib_provisional.json` (mm) · Input scale **0.50** ·
Max disparity **416** · z-near **195** / z-far **208 mm** · Denoise on.
Rule of thumb: this scene needs ≈ scale × 546 px of disparity — keep Max
disparity above that (the app warns on saturation). The z-clip removes
out-of-focus tall structures the optics cannot measure (DOF ≈ 1–3 mm).

## Conventions worth knowing

- `K.txt` files are metres (upstream convention); the app converts on load and
  refuses to auto-apply a `K.txt` whose principal point can't belong to the
  loaded image.
- Depth/cloud units follow the display unit (mm/µm/m); everything rescales
  losslessly on a unit switch — no re-run.
- The input pair must be rectified, or use raw mode with a calibration; the
  app warns when a pair loaded as "already rectified" has misaligned rows.
- Left really means the left-side camera position (the shot whose features sit
  further right in the image).
