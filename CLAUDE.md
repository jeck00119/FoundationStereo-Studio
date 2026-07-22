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

Status: **port DONE on-device 2026-07-22** (JetPack 6, L4T R36.4.7). On that
day: offscreen suite 69/69, `verify_3d_tab.py` 44/44 on the real display,
`verify_full_process.py` 22/22, whole-app rig-sized run clean. What a fresh
agent needs to know:

0. **JetPack 6.x (L4T R36+) is a hard prerequisite** — a JetPack 5 flash
   (R35: Ubuntu 20.04, Python 3.8, CUDA 11.4, no Triton wheels) cannot run
   this stack; `setup_jetson.sh` fails fast there with the reason. Reflash
   with SDK Manager first. Verified 2026-07: the live torch index is
   `pypi.jetson-ai-lab.io/jp6/cu126/+simple` (the old `.dev` host is dead)
   and the PySide6 6.8.0.2 aarch64 Addons wheel DOES contain QtWebEngine.
1. `./setup_jetson.sh` is proven end-to-end and self-repairs its known traps
   (libcudss for torch≥2.10 — --no-deps matters, SBSA deps poison Tegra;
   ptxas for the triton wheel; focal libwebp6 + libminizip1 for QtWebEngine).
   open3d is optional (denoise self-skips without it).
2. Weights: the readme's Drive folder
   (`drive.google.com/drive/folders/1HuTt7UIp7gQsMiDvJwVuWmKpvFzIIMap`,
   `gdown --folder`) is routinely quota-blocked for ~24 h. The reliable
   source is NVIDIA's official HF drop — files
   `huggingface.co/nvidia/c-fast-foundationstereo/resolve/main/{cfg.yaml,
   model_best_bp2_serialize.pth}` into `weights/hf-c-release/`. Identified
   2026-07-22 (registry comment has the full evidence chain): it is the
   **v1.0 unpruned flagship** — 23-36-37 accuracy class, the author's
   announced "commercial version" (issue #53) — so the BEST run is available
   without Drive. Bit-identity with 23-36-37 remains unverified. That pickle
   predates some upstream args — the adapter backfills them (`normalize`,
   cf. upstream PR #41 fixing the same miss elsewhere); extend
   `_NEWER_ARG_DEFAULTS` if a future "Missing key …" appears at forward
   time.
3. Compile behavior (the adapter pins inductor serial + a persistent cache
   dir `~/.cache/fsstudio-inductor`; the parallel worker pool dies on this
   device): the FIRST-ever compile on a fresh device runs >10 min with no
   output — let it finish once (`python tools/bench_orin.py warmup`). After
   that: new-shape compile 20–40 s, engine restart seconds, first Run in
   each app session ~30 s, later Runs ~2 s.
4. Measured defaults (MAXN_SUPER + jetson_clocks, rig-sized 2664×2304 pair,
   valid_iters 8, warm engine, whole numbers = `tools/bench_orin.py sweep`):
   **Input scale 0.30 · Max disparity 192 is the device default** — 1.8 s,
   1.08 GB GPU peak, and 2.6 GB system floor with the FULL app resident
   (`tools/bench_app_orin.py 0.30`). 0.35 · 224 fits when needed (≈2.7 s,
   ~1.7 GB peak, floor ≈2 GB); 0.40 · 256 leaves ~1.1 GB engine-only — GUI
   on top makes it OOM-adjacent; the Windows profile 0.50 · 416 is
   **OOM-killed by the kernel** (6.6 GB RSS, dmesg receipt). Disparity rule
   still applies: needed ≈ scale × 546 px ≤ Max disparity.
   Orin ≈ 7.7× slower than the 3060 at identical config (2.68 s vs 0.35 s at
   932×806·192) with byte-identical 1.49 GB GPU peak — the compiled cost
   volume behaves the same everywhere.
5. Power: stock nvpmodel.conf lacks the Super modes — repoint the symlink to
   `/etc/nvpmodel/nvpmodel_p3767_0003_super.conf`, then `nvpmodel -m 2`
   (MAXN_SUPER) + `jetson_clocks` before benchmarking.
6. Validate with `pytest tests` (offscreen: `QT_QPA_PLATFORM=offscreen`), then
   `tools/verify_full_process.py`. Commit fixes to `orin`; merge to `master`
   what applies to both platforms. A TensorRT backend (fixed input size,
   FP16 engine via upstream `scripts/make_onnx.py` + trtexec) remains the
   planned next step for speed — community calibration for it (NVlabs issue
   #43): pruned run 20-30-48 does 4–5 FPS at 448×640 PyTorch on an Orin
   Nano Super, and some TRT versions fail the engine build
   ("PWN(/Mul_1)" on TRT 10.7), so pin/verify the JetPack TRT version
   before building that adapter.
