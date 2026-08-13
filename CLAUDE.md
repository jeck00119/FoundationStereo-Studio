# CLAUDE.md — FoundationStereo Studio

Two things live in this repo: NVIDIA's FoundationStereo research code
(upstream) and **FoundationStereo Studio** — a PySide6 desktop app for
close-range PCB stereo metrology (pin heights, flatness, repeatability) using
a single camera on a CNC that steps ~5 mm to form stereo pairs. The full
human-facing map, workflows and platform notes are in **STUDIO_README.md** —
read it first.

## Commands

- Run: `run_studio.bat` (Windows) / `./run_studio.sh` (Linux/Jetson)
- Tests (fast, no GPU): `.venv/bin/python -m pytest tests` (Windows: `.venv\Scripts\python.exe`)
- Live verification after touching the 3D view: `tools/verify_3d_tab.py`;
  after pipeline changes: `tools/verify_full_process.py` (loads the real model)
- Calibration: `tools/pair_captures.py <session>` then
  `tools/calibrate.py <session>/paired --charuco 11x8 --square <MEASURED> --marker <MEASURED*2/3> --simple-lens --out data/calib/calib.json --krect`

## Hard rules

- **Never refactor the upstream dirs** (`core/`, `dinov2/`, `depth_anything/`,
  `scripts/`, `Utils.py`) — they stay mergeable with NVlabs (`upstream` remote).
- The studio code is dense with deliberate, comment-documented workarounds —
  **read the surrounding comments before "fixing" anything odd**.
- Verify before claiming: run the pytest tier, and the relevant `tools/verify_*`
  harness for UI/pipeline work. State numbers, not adjectives.
- Any script that drives the real app must **snapshot and restore the user's
  QSettings state** (saved measure boxes, rectify mode) — see how
  `tools/verify_full_process.py` does it.
- Git: `master` = single platform-neutral truth for BOTH machines; `orin` =
  Jetson bring-up scratch, merged back when things generalize. `origin` =
  jeck00119/FoundationStereo-Studio (public); `upstream` = NVlabs, pull-only.
- User data lives in `data/` (git-ignored): `calib/`, `captures/`, `exports/`.
  Keep it out of `assets/` (upstream demo data — a stray `K.txt` there once
  silently hijacked calibration).

## Current rig + calibration state (2026-07-21)

- Camera: Basler acA4024-29uc (1.85 µm, 4024×3036, rolling shutter) + Fujinon
  CF35ZA-1S 35 mm at ~200–230 mm working distance (~96 px/mm, FOV ≈ 42×32 mm,
  DOF only 1–3 mm — tall structures can't be measured; z-clip them).
- Working calibration: `data/calib/calib_provisional.json` (ChArUco **11x8**,
  `--simple-lens` required for this lens). Validated: board flat to 0.37 mm,
  square pitch 6.03 mm, PCB surface RMS 0.16 mm.
- Settled PCB run (Windows / 12 GB card): Raw—rectify + that calib (mm) ·
  scale 0.50 · Max disparity 416 (rule: needed disparity ≈ scale × 546 px) ·
  z-near 195 / z-far 208 mm. The 0.50 · 416 pair OOM-kills the 8 GB Orin —
  its defaults are in the Jetson section below.
- Pending for metrology grade: glue the ChArUco print flat (2.5 px RMS floor =
  paper waviness) and caliper the square size (~3 % print scale error), then
  recapture ~12 A/B pose-pairs and re-run the calibration commands above.

## On a Jetson Orin Nano (8 GB)

Port **done** (2026-07-22, JetPack 6 / L4T R36.4.7). Setup, model and
defaults are settled — this section is the device's operating manual.

**Setup**: `./setup_jetson.sh` (idempotent, self-repairs its known traps:
libcudss for torch≥2.10 — `--no-deps` is load-bearing, SBSA deps poison
Tegra; ptxas for the triton wheel; focal libwebp6 + libminizip1 for
QtWebEngine; system TensorRT bindings symlinked in). JetPack 6 is a hard
prerequisite — it fails fast on JetPack 5 with the reason. open3d is
optional (denoise self-skips).

**Weights**: the readme's Drive folder is routinely quota-blocked ~24 h.
Use NVIDIA's HF drop —
`huggingface.co/nvidia/c-fast-foundationstereo/resolve/main/{cfg.yaml,
model_best_bp2_serialize.pth}` → `weights/hf-c-release/`. It is the **v1.0
unpruned flagship** (23-36-37 accuracy class; evidence chain in
`backends/registry.py`), so the best run needs no Drive. Its pickle predates
some upstream args — the adapter backfills them (`_NEWER_ARG_DEFAULTS`;
extend it if a new "Missing key …" ever appears at forward time).

**Defaults** — backend **Fast-FS · TensorRT**, scale **0.30** · max_disp
**192**. Measured (MAXN_SUPER + `jetson_clocks`, rig-sized 2664×2304 pair,
iters 8, warm):

| backend | warm | cold start | note |
|---|---|---|---|
| TensorRT (default) | **1.30 s** | ~5 s, no warm-up ever | Δ vs torch: median 0.034 px, p95 0.156, valid 100 % |
| PyTorch/triton | 1.93 s | 10.7 s + ~30 s first Run per session | fallback, one click away |

Full app press-to-cloud ≈ 3 s. Scale rule: needed disparity ≈ scale × 546 px
≤ max_disp. Orin ≈ 7.7× slower than the PC's 3060 at identical config.

**The engine ceiling is PIXELS, not cost volume (measured 2026-08-11)**.
An earlier version of this file said "scale 0.30" and blamed the cost
volume (H·W·D). That is wrong and a build disproved it: 1024×1024 · d64 is
only **67M** H·W·D — well under the 108M that builds — and still failed, at
`/AveragePool_1`. Pooling and convolutions scale with **pixels alone**;
`max_disp` never enters them. Every attempt over **563k px** has failed:

| padded | H·W·D | result |
|---|---|---|
| 512×1024 · d64 (524k px) | 34M | ✅ built, 76 min |
| 704×800 · d192 (563k px) | 108M | ✅ built |
| 832×960 · d224 (799k px) | 179M | ❌ `PWN(/Mul_1)`, 4.29 GB, Error 10 |
| 928×1088 · d256 (1010k px) | 258M | ❌ fused `PWN(…)`, 4.65 GB, Error 4 |
| 1024×1024 · d64 (1049k px) | 67M | ❌ `/AveragePool_1` |

So **keep any crop ≤563k px and max_disp is nearly free**. Failure logs are
kept beside the engines. PyTorch reaches 0.40 engine-only but is OOM-adjacent
with the GUI up; 0.50·416 is kernel-OOM-killed.

**Full resolution fits anyway — use an ROI + pre-shift.** A macro pair's
disparity is huge in absolute terms (~500 px here) but varies only a few px
across a flat board, so ~95 % of any cost volume searches disparities that
never occur. Crop to what you measure, start the RIGHT crop Δ px further
left, and `max_disp` collapses to its 64 floor: **512×1024 at scale 1.00,
0.64 s/run, 42 µm per 0.1 px** — better resolution than full-frame 0.20 for
less memory. Δ is measured from the images (no calibration, no prior run)
and placed **0.6·max_disp** below the match so the observed band lands mid-
range; a fixed 24 px margin put it at the edge and cost 5× in σ (2284 vs
435 µm). Details in `StereoParams.roi` and `studio/window/roi.py`.

**TRT engine cache**: per (padded size · iters · max_disp) under
`weights/<run>/trt/`, survives reboots. Building is **opt-in** — the
"Build engine if missing" toggle, default OFF; otherwise the backend refuses
a new size and lists the engines it does have. TensorRT is ONLY a speed
optimisation (1.13 s vs 1.86 s at the ROI config, measured); the crop,
pre-shift, sites and measurement are backend-agnostic, so the PyTorch
backend gives the same answer with no engine at all. The ROI label says
`engine ✓` or warns before you pay. A new size builds once — ~1¾ h at
default opt level (`FS_TRT_OPT_LEVEL=2` for far faster probe builds),
narrated by minute heartbeats with a kept `.build.log`. ONNX export runs
CPU-side in a throwaway subprocess **on purpose**: GPU-side tracing pins
>5 GB of unswappable memory and once starved the box until the Tegra
watchdog reset it.

**Device quirks**: the mis-flashed DTB makes `nvpower.sh` relink the
non-super nvpmodel conf every boot — the enabled `fsstudio-maxn-super`
oneshot re-applies MAXN_SUPER after it; run `jetson_clocks` manually before
timing work. TorchInductor's parallel compile pool dies here, so the torch
adapter pins serial compile + a persistent cache dir (first-ever compile on
a fresh device is >10 min with no output — let it finish once via
`tools/bench_orin.py warmup`).

## Measuring pin heights on this rig (measured over 1000 real captures)

Mark **sites** on the Input tab (`Mark: pin` / `Mark: reference`), not
world-space measure boxes — boxes seeded from one capture were **empty on 19
of 20** later ones. Each site is tracked per capture and a height is only ever
**pin − reference within one frame**. What the run taught, all of it measured:

- **The CNC repeats its step to 28 µm (0.53 %)** — not the limiter. But that
  step IS the stereo baseline, so it walks ABSOLUTE depth by >1 mm: σ 2200 µm
  absolute vs ~350–500 µm differential. Always reference inside the frame.
- **The reference must have TEXTURE.** Local contrast: bare solder mask 3.4
  grey levels, metal bar 3.6 — the network guesses there. Component body 23.8,
  leads 57–69. This mattered more than any other single choice.
- **Don't seed geometry from the first capture.** loop001 sat 13 px (5.4 mm)
  off every later one.
- Current best: **σ 335–495 µm** on specular connector leads. That is stereo
  matching noise, not the machine. 42 µm/0.1 px is the SENSOR's resolution,
  not the rig's repeatability.

Tools: `tools/show_sites.py` (what the GUI saved) · `tools/study_pin_heights.py`
(headless run of the app's own measurement) · `tools/build_roi_engine.py` ·
`tools/rehearse_study.py`.

**Validate**: `pytest tests` (offscreen: `QT_QPA_PLATFORM=offscreen`) →
`tools/verify_full_process.py` → `tools/verify_trt_backend.py all [--rig]`
(TRT-vs-torch gate). Benchmarks: `tools/bench_orin.py`,
`tools/bench_app_orin.py`. Work lands on `master`; re-branch `orin` only for
risky device experiments.
