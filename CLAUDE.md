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

**Ceilings (measured, do not re-litigate)**: TRT engines cannot exceed
scale 0.30 on 8 GB — at ≥0.35 every tactic for one fused cost-volume op
wants 4.1–4.7 GB against the ~2.6 GB actually free, and trtexec gives up on
that node (this is also NVlabs issue #43's unexplained report). The node and
error code vary with size, so recognise the pattern, not the string: 832×960
· d224 died at `PWN(/Mul_1)`, 4.29 GB, Error 10 (insufficient device memory);
928×1088 · d256 at a longer fused `PWN(...)` chain, 4.65 GB, Error 4
(insufficient workspace). Both logs are kept beside the engines.
PyTorch reaches 0.40 engine-only but is OOM-adjacent with the GUI up;
0.50·416 is kernel-OOM-killed. **Next project if >0.30 is ever needed here:
upstream's TRT plugin path (PR #55, fused GWC kernel).** Until then 0.35+
belongs to the Windows machine.

**TRT engine cache**: per (padded size · iters · max_disp) under
`weights/<run>/trt/`, survives reboots. A new size builds once — ~1¾ h at
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

**Validate**: `pytest tests` (offscreen: `QT_QPA_PLATFORM=offscreen`) →
`tools/verify_full_process.py` → `tools/verify_trt_backend.py all [--rig]`
(TRT-vs-torch gate). Benchmarks: `tools/bench_orin.py`,
`tools/bench_app_orin.py`. Work lands on `master`; re-branch `orin` only for
risky device experiments.
