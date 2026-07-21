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
  private jeck00119/FoundationStereo-Studio; `upstream` = NVlabs, pull-only.
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
- Settled PCB run: Raw—rectify + that calib (mm) · scale 0.50 · Max disparity
  416 (rule: needed disparity ≈ scale × 546 px) · z-near 195 / z-far 208 mm.
- Pending for metrology grade: glue the ChArUco print flat (2.5 px RMS floor =
  paper waviness) and caliper the square size (~3 % print scale error), then
  recapture ~12 A/B pose-pairs and re-run the calibration commands above.

## On a Jetson Orin Nano (8 GB)

You are likely here to finish the Linux port. Status: the app code is
platform-neutral and `setup_jetson.sh` / `run_studio.sh` exist but are
**untested on-device**. Do this:

1. `git checkout orin`, run `./setup_jetson.sh`, and FIX WHAT BREAKS — the two
   expected weak points are the Jetson torch wheel index (override with
   `JETSON_TORCH_INDEX=...`) and PySide6/QtWebEngine aarch64 wheels (the 3D
   view). open3d is optional (denoise self-skips without it).
2. 8 GB is unified CPU+GPU memory: **Fast-FoundationStereo is the only
   practical model** (sibling clone `../Fast-FoundationStereo` + weights);
   FoundationStereo ViT-L does not fit. Start at Input scale ≤ 0.30; if
   Triton is unavailable, the adapter falls back to eager at ~4× memory —
   lower the scale further. A TensorRT backend (new adapter in
   `studio/backends/`, fixed input size, FP16 engine via upstream
   `scripts/make_onnx.py` + trtexec) is the planned proper solution.
3. Validate with `pytest tests` (offscreen: `QT_QPA_PLATFORM=offscreen`), then
   `tools/verify_full_process.py`. Commit fixes to `orin`; merge to `master`
   what applies to both platforms.
