# FoundationStereo Studio

A desktop app for close-range stereo metrology (PCB / pin-height measurement)
built on top of this FoundationStereo checkout. Runs on the Windows rig
machine and on the Jetson Orin Nano (Linux port validated on-device
2026-07-22). Upstream's own readme is `readme.md`; this file covers only the
local additions.

## Deploy on a new machine

**A clone is not enough on its own.** Four things git cannot bring: the model
repos (they are SIBLING directories, not submodules), their weights, the venv,
and your captures (`data/` is ignored apart from `data/calib/*.json`).

`tools/check_setup.py` checks every one of those and prints the fix for each —
run it first, fix what it lists, run it again until it says `Ready`.

**The short way — one script does all of it:**

```
git clone https://github.com/jeck00119/FoundationStereo-Studio.git FoundationStereo
cd FoundationStereo && python install.py
```

`install.py` clones the sibling model repo, fetches the weights, builds the
environment for THIS platform, verifies with `check_setup.py` + the test suite,
and drops a desktop launcher (`.desktop` on Linux, `.lnk` on Windows). It is
idempotent and asks before the two slow steps, so re-run it after fixing
whatever it stops on. The manual sequence below is the same thing by hand.

**1. Both repos, side by side**

```
git clone https://github.com/jeck00119/FoundationStereo-Studio.git FoundationStereo
git clone https://github.com/NVlabs/Fast-FoundationStereo.git
```

They must be siblings — `studio/backends/registry.py` resolves `../Fast-FoundationStereo`.

**2. Weights** → `Fast-FoundationStereo/weights/hf-c-release/`

`huggingface.co/nvidia/c-fast-foundationstereo` → `cfg.yaml` +
`model_best_bp2_serialize.pth`. (The readme's Drive folder is routinely
quota-blocked; the HF drop is the v1.0 unpruned flagship anyway.)

**3. Environment**

| | |
|---|---|
| **Jetson** (JetPack 6) | `./setup_jetson.sh` — system libs, venv, Jetson torch wheels, requirements, offscreen test run. Idempotent; self-repairs the known JetPack 6 traps (cudss, ptxas, QtWebEngine's webp/minizip sonames). |
| **Windows** (Python 3.12) | `py -3.12 -m venv .venv`<br>`.venv\Scripts\pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128`<br>`.venv\Scripts\pip install -r requirements.txt` |

torch must come from the platform index BEFORE `requirements.txt` — plain pip
fetches CPU wheels on Windows and x86 wheels on Jetson.

**4. Check, then run**

```
python tools/check_setup.py      # exit 1 = nothing can run yet
./run_studio.sh                  # or run_studio.bat
```

**Which model on which machine.** TensorRT is a Jetson-only speed option
(1.13 s vs 1.86 s at the same config) and now reads as unavailable elsewhere.
Everything else — the ROI crop, the pre-shift, the marked sites, the
measurement — is backend-agnostic, so **Fast-FoundationStereo (PyTorch) gives
the same answer on any machine with no engine build at all**. A bigger GPU
lifts the Jetson's 563k-pixel engine ceiling; re-derive it there rather than
assuming these numbers.

## Run

```
run_studio.bat            (Windows, console-less; logs land in %TEMP%\fs_studio_*.log)
./run_studio.sh           (Linux / Jetson)
```
or, with a console: `.venv\Scripts\python.exe -m studio.app`

The engine (CUDA + the model) runs in a separate child process, so the GUI
never freezes; each model switch gets a fresh child with clean VRAM. Triton is pinned in
`requirements.txt`, which is what lets Fast-FoundationStereo run its COMPILED
cost-volume path — without it the eager fallback is correct but takes 4.2x the
peak memory. The first run after changing Input scale or Max disparity pauses
once while kernels compile.

## Map of the repository

Every top-level entry, labeled — this clone hosts two things at once: NVIDIA's
FoundationStereo research code (leave as-is, it stays upstream-mergeable) and
the local Studio app built on top of it.

| Entry | Whose | What it is |
|---|---|---|
| `studio/` | **local** | the app. `main_window.py` (workflows) · `window/` (parts lifted out of it: roi · level · export · analyze) · `panels/` (input · params · shared widgets) · `viewers.py`, `web_cloud.py`, `web/` (2D views + three.js 3D view) · `worker.py`/`engine_process.py` (GUI↔engine child) · `backends/` (FoundationStereo · Fast-FS · Fast-FS TensorRT · S²M²) · `infer.py`/`cloud.py`/`rectify.py` (shared pipeline; `rectify` also owns the ROI crop + pre-shift) · `pairs.py` (Qt-free loading/pairing) · `measure.py`/`analyze.py`/`sites_measure.py`/`repeat.py`/`compare.py`/`batch.py` (metrology) |
| `tools/` | **local** | CLIs + diagnostics: `check_setup.py` (**run this first on a new machine** — the model repos are siblings, not submodules) · `show_sites.py` (what the GUI saved) · `study_pin_heights.py` (headless run of the app's own measurement over a capture run) · `build_roi_engine.py` · `rehearse_study.py` · `calibrate.py` (checkerboard or ChArUco → calib.json; `--simple-lens` for near-distortion-free lenses) · `pair_captures.py` (verifies/names CNC A-B sessions) · `verify_3d_tab.py` (~30 s live 3D-view check) · `verify_full_process.py` (full-chain live check on the GPU) · `bench_orin.py` / `bench_app_orin.py` (Jetson perf/memory harnesses behind the measured defaults) |
| `tests/` | **local** | pytest suite (pipeline/calibration/measurement ground truth + GUI behavior): `.venv\Scripts\python.exe -m pytest tests` |
| `data/` | **local, git-ignored** | your data: `calib/` (calibration files only) · `captures/` (capture sessions) · `exports/` (generated clouds/renders) |
| `run_studio.bat`, `requirements.txt`, `STUDIO_README.md` | **local** | Windows launcher · its pinned env (torch/cu128 note in its header) · this file |
| `run_studio.sh`, `setup_jetson.sh`, `requirements-jetson.txt` | **local** | Jetson/Linux launcher · one-shot device setup (self-repairs the known JetPack 6 traps) · its env |
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

## Current rig quick-start (Basler acA4024 + 35 mm lens)

Raw—rectify + `data/calib/calib_provisional.json` (mm). Measured on this rig,
not inherited: rectified fx **21103.6**, baseline **5.1785 mm**, working
distance **211–223 mm**.

| setting | value | why |
|---|---|---|
| rectify mode | Raw — rectify with calibration | the ROI is drawn on the RECTIFIED image |
| Input scale | **1.00** | the ROI is what makes full resolution fit |
| Max disparity | **64** | the pre-shift leaves only the few px that vary |
| ROI | 512×1024, over the parts | ≤563k px, the only size proven to build an engine here |
| z-near / z-far | **180 / 260 mm** | must bracket 211–223; the old 195/208 returns an EMPTY cloud |
| Denoise | **off** | 3.5 s/pair on this device and buys ~0.4 µm — the 2 % trim already removes flyers |

**Why not "scale 0.30 · max_disp 192"?** That was the setting before the ROI
existed, when the whole frame had to be shrunk to fit. It still works, and it is
what a full-frame run needs — but it resolves 208 µm per 0.1 px against the
ROI's 42 µm, for more memory. The old rule of thumb (needed disparity ≈
scale × 546 px) applies only WITHOUT a pre-shift; with one, `max_disp` covers
the scene's disparity SPAN (~30 px here), not its absolute value (~500 px).

The z-clip also removes out-of-focus tall structures the optics cannot measure
(DOF ≈ 1–3 mm).

## Measuring pin heights over a capture run

The metrology workflow, in the order the app expects:

1. Load a pair; set **Raw — rectify with calibration** and load your `calib.json`.
2. Pick a model. **TensorRT is a Jetson speed option only** — everything else is
   backend-agnostic, so PyTorch gives the same answer anywhere with no engine.
3. Tick **ROI** on the Input tab and drag it over the parts you measure. Δ (the
   right-crop pre-shift) is re-measured automatically whenever the box moves;
   the label shows it plus `engine ✓` or a warning that this SIZE would need a
   build. Position is always free — only a resize can cost one.
4. **Run** once. This is the reference capture site tracking cuts templates from.
5. `Mark: pin` → click each pin. `Mark: reference` → click a **textured** surface
   near each (bare solder mask reconstructs poorly and makes a noisy reference).
   `tools/show_sites.py` prints what was saved and the pin→reference pairing.
6. **Batch…** → the capture folder. One row per capture in Repeatability; export CSV.

Why sites rather than the 3D measure boxes: a box is fixed in the camera frame
and this rig's frame does not hold still. Boxes seeded from one capture came
back **empty on 19 of 20** later ones. Sites are tracked per capture and
differenced inside the frame, which also cancels the ~0.5 % variation in the CNC
step — i.e. in the stereo baseline (σ 2200 µm absolute vs ~350–500 µm
referenced). Measure boxes remain for saved-cloud batches, which have no image.

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

On the Orin (bring-up is done and merged — the Orin runs `master` like the
Windows machine; `orin` is only re-branched for risky device experiments):

```
git clone https://github.com/jeck00119/FoundationStereo-Studio.git FoundationStereo
cd FoundationStereo
./setup_jetson.sh && ./run_studio.sh
```

## Linux / Jetson Orin Nano

Validated on-device (JetPack 6 / L4T R36.4.7): test suite 143/143, 3D-view
checks 44/44, full pipeline 22/22. The app code is platform-neutral;
the launcher and environment differ:

See **Deploy on a new machine** above for the full sequence; on a Jetson it is
`./setup_jetson.sh` then `./run_studio.sh`.

Notes for the Orin Nano 8 GB (unified CPU+GPU memory):
- **Fast-FoundationStereo is the practical model** — FoundationStereo ViT-L
  does not fit. With an ROI use **scale 1.00 · Max disparity 64** (see the
  quick-start above); for a whole-frame run, **scale 0.30 · Max disparity 192**
  is the device default and the Windows 0.50 · 416 profile gets OOM-killed.
- **The TensorRT backend is the Orin default**: same accuracy as the
  PyTorch backend (median 0.034 px difference), **1.30 s vs 1.93 s** per
  run at the device config, ~5 s cold start with no warm-up. First run at a
  NEW size/disparity builds an engine once (status bar narrates; a
  `.build.log` sits beside the engine; up to ~2 h) — the standard configs
  are already cached. Scales ≥0.35 exceed this export flavor's memory on
  8 GB — details and the plugin-path plan are in CLAUDE.md.
- Weights: the readme's Google Drive folder is routinely quota-blocked —
  NVIDIA's official Hugging Face drop (`nvidia/c-fast-foundationstereo`) is
  the reliable source; it appears in the model picker as `hf-c-release` and
  is the **v1.0 unpruned flagship**, i.e. the family's best-accuracy tier
  (evidence chain in `studio/backends/registry.py`). The Drive runs add
  only the faster/pruned variants.
- The very first inference on a fresh device compiles for **>10 minutes**
  with no visible progress (one-time; caches persist across reboots after
  that). Each app session's first Run then takes ~30 s, later Runs ~2 s.
- Benchmarking: switch to MAXN_SUPER + `jetson_clocks` first (see CLAUDE.md);
  `tools/bench_orin.py` / `tools/bench_app_orin.py` reproduce the numbers.
- torch comes from the Jetson wheel index (standard PyPI torch is x86-only);
  `setup_jetson.sh` handles it, override via `JETSON_TORCH_INDEX=...`.
- No flash-attn on Jetson (FoundationStereo falls back automatically); open3d
  is optional — without it the denoise step skips itself and PLY export
  reports the missing dependency.
- Your `data/calib/*.json` calibration is device-independent (and ships in
  git already — `data/calib/calib_provisional.json`).
- If the 3D tab stays black, see the commented `QTWEBENGINE` lines in
  `run_studio.sh` (not needed on the dev kit's own display — verified).

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
