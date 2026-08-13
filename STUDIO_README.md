# FoundationStereo Studio

Desktop app for close-range stereo metrology — measuring pin heights on a PCB
with one camera on a CNC that steps ~5 mm between two shots. Runs on Windows
and on a Jetson Orin.

Upstream's own readme is `readme.md`. Deep detail and measured device numbers
are in `CLAUDE.md`.

## Install

```
git clone https://github.com/jeck00119/FoundationStereo-Studio.git FoundationStereo
cd FoundationStereo
python install.py
```

That clones the model repo beside this one, fetches the weights, builds the
environment, verifies it, and puts an icon on your desktop. Re-run it after
fixing anything it stops on — it skips what is already done.

Then launch from the desktop icon, or `run_studio.bat` / `./run_studio.sh`.

Something missing later? `python tools/check_setup.py` says what and how to fix it.

## Measure

1. **Load** a left/right pair.
2. **Calibration** → *Raw — rectify with calibration* → load your `calib.json`.
3. **Model** → Fast-FoundationStereo. TensorRT is a Jetson-only speed option;
   PyTorch gives the same answer everywhere.
4. **ROI** → tick it, drag the box over your parts. The shift is measured for
   you; the label shows `engine ✓` when the size is ready to run.
5. **Run**. Check the depth looks sane.
6. **Mark** → `Mark: pin`, click each pin. `Mark: reference`, click a *textured*
   surface near them.
7. **Batch…** → your capture folder. One row per capture in **Repeatability**.
   Export CSV.

Steps 1–6 are setup and stick between sessions. Step 7 runs unattended
(~20 min for 1000 pairs on the Orin).

## Settings that matter

| | value |
|---|---|
| Input scale | 1.00 |
| Max disparity | 64 |
| ROI size | 512×1024 |
| z-near / z-far | 180 / 260 mm |
| Denoise | off |

Measured on this rig: working distance 211–223 mm, rectified fx 21103.6,
baseline 5.1785 mm.

## Things that will bite you

- **The reference must have texture.** Bare solder mask reconstructs as a guess.
  Aim at a component body or silkscreen, not empty board.
- **Measure pin against reference, never absolute depth.** The CNC step is the
  stereo baseline and varies ~0.5 %, which moves absolute depth by >1 mm.
- **Don't set up on the first capture of a run** — it is often an outlier.
- **Resizing the ROI** can trigger a ~1 h engine build on the Orin. Moving it
  never does. The label warns you first.
- `data/` is not in git — your captures and marked sites do not travel with a
  clone.

Expect ~0.3–0.5 mm repeatability on shiny metal leads. That is the stereo
matching, not the machine.

## Where things are

| | |
|---|---|
| `studio/` | the app — `main_window.py`, `window/` (roi, level, export, analyze), `panels/`, `backends/`, `sites_measure.py` |
| `tools/` | `check_setup.py`, `show_sites.py`, `calibrate.py`, `study_pin_heights.py` |
| `tests/` | `python -m pytest tests` |
| `data/` | yours, git-ignored: `calib/`, `captures/`, `exports/` |
| `core/`, `dinov2/`, `scripts/`, `Utils.py` | upstream NVIDIA code — don't refactor |

Fast-FoundationStereo is a **sibling** directory, not a submodule. `install.py`
handles that.

## Calibrating a new rig

Shoot the ChArUco board at CNC position A, jog the step, shoot B, then move the
board. 10–15 poses.

```
python tools/pair_captures.py data/captures/<session>
python tools/calibrate.py data/captures/<session>/paired --charuco 11x8 \
    --square <MEASURED> --marker <MEASURED*2/3> --simple-lens \
    --out data/calib/calib.json
```

`--square` is caliper-measured, not nominal — printers rescale by ~1 %.

## Git

`master` is the truth for both machines; `orin` is a scratch branch for risky
device experiments. `upstream` is NVIDIA's repo, pull-only.
