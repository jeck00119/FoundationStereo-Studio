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

## Map of the repository

Every top-level entry, labeled — this clone hosts two things at once: NVIDIA's
FoundationStereo research code (leave as-is, it stays upstream-mergeable) and
the local Studio app built on top of it.

| Entry | Whose | What it is |
|---|---|---|
| `studio/` | **local** | the app. Flat, descriptively-named modules: `main_window.py` (workflows) · `panels.py` (input/param panels) · `viewers.py`, `web_cloud.py`, `web/` (2D views + three.js 3D view) · `worker.py`/`engine_process.py` (GUI↔engine child) · `backends/` (FoundationStereo · Fast-FS · S²M²) · `infer.py`/`cloud.py` (shared pipeline) · `pairs.py` (Qt-free loading/pairing) · `measure.py`/`analyze.py`/`repeat.py`/`compare.py`/`batch.py` (metrology tools) |
| `tools/` | **local** | CLIs + diagnostics: `calibrate.py` (checkerboard or ChArUco → calib.json; `--simple-lens` for near-distortion-free lenses) · `pair_captures.py` (verifies/names CNC A-B sessions) · `verify_3d_tab.py` (~30 s live 3D-view check) · `verify_full_process.py` (full-chain live check on the GPU) |
| `tests/` | **local** | pytest suite (pipeline/calibration/measurement ground truth + GUI behavior): `.venv\Scripts\python.exe -m pytest tests` |
| `data/` | **local, git-ignored** | your data: `calib/` (calibration files only) · `captures/` (capture sessions) · `exports/` (generated clouds/renders) |
| `run_studio.bat`, `requirements.txt`, `STUDIO_README.md` | **local** | launcher · pinned env (torch/cu128 note in its header) · this file |
| `core/`, `dinov2/`, `depth_anything/`, `scripts/`, `Utils.py`, `teaser/`, `docker/` | upstream | the FoundationStereo model + demos (scripts carry two small local fixes) — don't refactor |
| `readme.md`, `readme_jetson.md`, `environment.yml`, `LICENSE` | upstream | upstream docs; `environment.yml` is upstream's conda env — **not** this app's environment (use `requirements.txt`) |
| `assets/` | upstream | the bundled demo pair + its `K.txt` — keep your own data out of here |
| `pretrained_models/`, `.venv/`, `.cache/` | generated, git-ignored | FoundationStereo weights · the Python env · HF model cache |

Fast-FoundationStereo and S²M² are expected as sibling clones
(`..\Fast-FoundationStereo`, `..\s2m2`) — see `studio/backends/registry.py`.

## Calibrate the rig (ChArUco workflow)

Per board pose: shoot at CNC position A → jog the measurement step (+X) →
shoot at position B → then reposition/tilt the board. 10–15 poses, mixed
heights and tilts, corners reaching the frame edges. Put the session under
`data\captures\<session>` and:

```
.venv\Scripts\python.exe tools\pair_captures.py data\captures\<session>
.venv\Scripts\python.exe tools\calibrate.py data\captures\<session>\paired --charuco 11x8 --square <MEASURED> --marker <MEASURED*2/3> --simple-lens --out data\calib\calib.json --krect
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

## Working across machines (git)

- `origin` = **github.com/jeck00119/FoundationStereo-Studio** (public) — your
  repo, the one both machines sync through.
- `upstream` = github.com/NVlabs/FoundationStereo — NVIDIA's repo, pull-only
  (`git fetch upstream` to bring in their updates).
- **`master` is the single source of truth for BOTH platforms** — the code is
  platform-neutral, so fixes land here once. **`orin`** is the Jetson
  bring-up branch: commit device-side experiments there, merge into `master`
  whatever generalizes, and once bring-up is done the Orin runs `master` too.
  (Two permanent per-platform branches would force every fix to be committed
  twice — deliberately avoided.)

On the Orin:

```
git clone https://github.com/jeck00119/FoundationStereo-Studio.git FoundationStereo
cd FoundationStereo && git checkout orin
./setup_jetson.sh && ./run_studio.sh
```

## Linux / Jetson Orin Nano

The app code is platform-neutral; the launcher and environment differ:

```
git clone <this repo> && cd FoundationStereo
./setup_jetson.sh          # one-shot: system libs, venv, Jetson torch wheels,
                           # requirements, offscreen test-suite validation
./run_studio.sh            # the run_studio.bat equivalent
```

Notes for the Orin Nano 8 GB (unified CPU+GPU memory):
- **Fast-FoundationStereo is the practical model** — FoundationStereo ViT-L
  does not fit. Keep Input scale modest and watch peak VRAM.
- torch comes from the Jetson wheel index (standard PyPI torch is x86-only);
  `setup_jetson.sh` handles it, override via `JETSON_TORCH_INDEX=...`.
- No flash-attn on Jetson (FoundationStereo falls back automatically); open3d
  is optional — without it the denoise step skips itself and PLY export
  reports the missing dependency.
- Your `data/calib/*.json` calibration is device-independent — copy it over.
- If the 3D tab stays black, see the commented `QTWEBENGINE` lines in
  `run_studio.sh`.

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
