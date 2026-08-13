"""Check that this machine can actually run the app, and say how to fix it.

A fresh clone is not enough on its own: the model repos are SIBLINGS rather
than submodules, the weights are a separate download, torch has to come from a
platform-specific index, and Triton decides whether Fast-FoundationStereo runs
in 1.5 GB or 6.3 GB. None of that is visible until something fails deep in a
dropdown or an engine child. This says it up front.

    python tools/check_setup.py [--strict]

Exit 0 if the app can run at all, 1 if something essential is missing.
--strict also fails on the merely-degrading problems (missing Triton, no CUDA).
Read-only: imports things and looks at the filesystem, writes nothing.
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

OK, WARN, BAD = "ok", "warn", "bad"
_MARK = {OK: "  OK  ", WARN: "  --  ", BAD: "  !!  "}
_rows = []


def say(state, what, detail="", fix=""):
    _rows.append((state, what, detail, fix))
    print(f"{_MARK[state]}{what}" + (f"  —  {detail}" if detail else ""))
    if fix and state != OK:
        for line in fix.splitlines():
            print(f"        {line}")


def check_python():
    v = sys.version_info
    say(OK if v >= (3, 10) else BAD, f"Python {v.major}.{v.minor}.{v.micro}",
        fix="Python 3.10+ is required (3.12 on Windows, 3.10 on JetPack 6).")


def check_torch():
    try:
        import torch
    except ImportError:
        say(BAD, "torch", "not installed", fix=(
            "Windows:  pip install torch==2.7.0 torchvision==0.22.0 "
            "--index-url https://download.pytorch.org/whl/cu128\n"
            "Jetson:   ./setup_jetson.sh  (PyPI torch is x86-only)"))
        return None
    cuda = torch.cuda.is_available()
    name = torch.cuda.get_device_name(0) if cuda else "no CUDA device"
    say(OK if cuda else WARN, f"torch {torch.__version__}", name, fix=(
        "" if cuda else
        "Installed without CUDA, or the driver is not visible. The app will\n"
        "run on CPU, which is far too slow to be useful."))
    return torch


def check_triton(torch):
    """Not optional: it is the memory budget, not a speed knob."""
    if torch is None:
        return
    try:
        from torch.utils._triton import has_triton
        good = bool(has_triton())
    except Exception:
        good = False
    say(OK if good else WARN, "triton", "usable" if good else "missing or unusable",
        fix=("" if good else
             "Fast-FoundationStereo falls back to an EAGER cost volume: correct,\n"
             "but measured at 4.2x the peak memory and slower (6.29 GB vs 1.49 GB\n"
             "on a 3060). Windows:  pip install triton-windows\n"
             "Jetson:  it comes from the Jetson wheel index — re-run ./setup_jetson.sh"))


def check_qt():
    try:
        import PySide6
        from PySide6 import QtWebEngineWidgets  # noqa: F401 — the 3D view needs it
        say(OK, f"PySide6 {PySide6.__version__}", "incl. QtWebEngine")
    except ImportError as e:
        say(BAD, "PySide6", str(e).split("\n")[0],
            fix="pip install -r requirements.txt   (QtWebEngine drives the 3D tab)")


def check_open3d():
    try:
        import open3d
        say(OK, f"open3d {open3d.__version__}")
    except ImportError:
        say(WARN, "open3d", "not installed", fix=(
            "Optional: cloud denoise self-skips and PLY export reports it.\n"
            "Denoise is off for metrology anyway (3.5 s/pair on Jetson)."))


def check_backends():
    """The model repos are SIBLINGS of this one — a clone has none of them."""
    from studio.backends import BACKENDS

    usable = 0
    for key, spec in BACKENDS.items():
        ok, why = spec.availability()
        usable += bool(ok)
        fix = ""
        if not ok and "clone it next to" in why:
            name = os.path.basename(spec.repo_dir.rstrip("/src"))
            fix = (f"git clone <{name}> {os.path.dirname(REPO)}/{name}\n"
                   "It must sit BESIDE this repo, not inside it.")
        elif not ok and "no weights" in why:
            # Each family has its OWN weights and its own source; pointing at
            # the wrong one is worse than saying nothing.
            if key.startswith("fast_foundation_stereo"):
                fix = ("huggingface.co/nvidia/c-fast-foundationstereo -> "
                       "<sibling>/weights/hf-c-release/\n"
                       "  {cfg.yaml, model_best_bp2_serialize.pth}   "
                       "(the readme's Drive folder is often quota-blocked)")
            elif key == "foundation_stereo":
                fix = ("NVIDIA's FoundationStereo checkpoints (readme.md) -> "
                       "pretrained_models/23-51-11/\n"
                       "  ViT-L needs a large GPU; it does not fit an 8 GB Jetson.")
            else:
                fix = f"place the checkpoint at the path above"
        say(OK if ok else WARN, f"backend {key}", why, fix=fix)
    say(OK if usable else BAD, f"{usable} of {len(BACKENDS)} backends usable",
        fix="" if usable else "No model can run. Fix at least one backend above.")


def check_calibration():
    p = os.path.join(REPO, "data", "calib")
    found = [f for f in os.listdir(p)] if os.path.isdir(p) else []
    jsons = [f for f in found if f.endswith(".json")]
    say(OK if jsons else WARN, "calibration", ", ".join(jsons) or "none in data/calib",
        fix="" if jsons else (
            "Metric depth needs one. calib_provisional.json ships in git; if it\n"
            "is missing, re-solve with tools/calibrate.py."))


def check_captures():
    p = os.path.join(REPO, "data", "captures")
    if not os.path.isdir(p):
        say(WARN, "captures", "data/captures/ absent", fix=(
            "Expected — data/ is git-ignored, so captures do not travel with a\n"
            "clone. Copy them across, or pair a session with tools/pair_captures.py."))
        return
    sets = [d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))]
    say(OK if sets else WARN, "captures", f"{len(sets)} session(s): {', '.join(sets[:4])}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true",
                    help="also fail on degrading problems (no Triton, no CUDA)")
    args = ap.parse_args()
    print(f"FoundationStereo Studio — setup check\n{REPO}\n")
    check_python()
    torch = check_torch()
    check_triton(torch)
    check_qt()
    check_open3d()
    print()
    check_backends()
    print()
    check_calibration()
    check_captures()

    bad = [r for r in _rows if r[0] == BAD]
    warn = [r for r in _rows if r[0] == WARN]
    print()
    if bad:
        print(f"{len(bad)} blocking problem(s) — the app cannot run:")
        for _, what, _, _ in bad:
            print(f"  - {what}")
        return 1
    if warn and args.strict:
        print(f"{len(warn)} warning(s), --strict:")
        for _, what, detail, _ in warn:
            print(f"  - {what}: {detail}")
        return 1
    print("Ready." + (f"  ({len(warn)} warning(s) above — read them if something"
                      " is missing from the model picker.)" if warn else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
