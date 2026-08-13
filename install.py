"""One-shot install: dependencies, weights, verification, desktop shortcut.

    python install.py [--no-icon] [--no-tests] [--yes]

Everything a fresh machine needs, in the order it needs it, on either platform.
A clone alone does not run: the model repo is a SIBLING directory rather than a
submodule, the weights are a separate download, and torch has to come from a
platform-specific index BEFORE the rest of the requirements (plain pip fetches
CPU wheels on Windows and x86 wheels on Jetson). Each of those used to be found
one failure at a time, deep inside a dropdown or an engine child.

Idempotent — safe to re-run after fixing whatever it stopped on. It never
deletes anything, and it asks before the two steps that cost real time (a 68 MB
weights download, and the Jetson's sudo system-package install).
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import urllib.request

REPO = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(REPO)
FAST_REPO = os.path.join(PARENT, "Fast-FoundationStereo")
FAST_GIT = "https://github.com/NVlabs/Fast-FoundationStereo.git"
WEIGHTS_DIR = os.path.join(FAST_REPO, "weights", "hf-c-release")
HF = "https://huggingface.co/nvidia/c-fast-foundationstereo/resolve/main/"
WEIGHT_FILES = ["cfg.yaml", "model_best_bp2_serialize.pth"]

IS_WIN = os.name == "nt"
IS_JETSON = (not IS_WIN) and os.path.exists("/etc/nv_tegra_release")
VENV = os.path.join(REPO, ".venv")
VPY = os.path.join(VENV, "Scripts" if IS_WIN else "bin",
                   "python.exe" if IS_WIN else "python")

_step = 0


def head(msg: str) -> None:
    global _step
    _step += 1
    print(f"\n\033[1m[{_step}] {msg}\033[0m" if not IS_WIN else f"\n[{_step}] {msg}")


def run(cmd, **kw) -> int:
    print("    $ " + " ".join(str(c) for c in cmd))
    return subprocess.call(cmd, cwd=kw.pop("cwd", REPO), **kw)


def die(msg: str, fix: str = "") -> None:
    print(f"\n  STOPPED: {msg}")
    if fix:
        for line in fix.splitlines():
            print(f"           {line}")
    sys.exit(1)


def ask(question: str, auto: bool) -> bool:
    if auto:
        return True
    try:
        return input(f"    {question} [Y/n] ").strip().lower() in ("", "y", "yes")
    except EOFError:      # non-interactive: assume yes rather than hanging
        return True


# ----------------------------------------------------------------- steps
def check_python() -> None:
    head("Python")
    v = sys.version_info
    print(f"    {sys.executable}  ({v.major}.{v.minor}.{v.micro})")
    if v < (3, 10):
        die(f"Python {v.major}.{v.minor} is too old",
            "Needs 3.10+ (3.12 on Windows, 3.10 on JetPack 6).")


def sibling_repo(auto: bool) -> None:
    head("Fast-FoundationStereo (sibling repo)")
    if os.path.isdir(os.path.join(FAST_REPO, ".git")):
        print(f"    present: {FAST_REPO}")
        return
    if not shutil.which("git"):
        die("git not found", f"Clone it manually to {FAST_REPO}")
    print(f"    missing: {FAST_REPO}")
    print("    It must sit BESIDE this repo — registry.py resolves "
          "../Fast-FoundationStereo.")
    if not ask("Clone it now?", auto):
        die("cannot continue without the model repo")
    if run(["git", "clone", FAST_GIT, FAST_REPO], cwd=PARENT) != 0:
        die("clone failed", f"Clone {FAST_GIT} to {FAST_REPO} by hand.")


def weights(auto: bool) -> None:
    head("Weights (NVIDIA HF drop, ~68 MB)")
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    missing = [f for f in WEIGHT_FILES
               if not os.path.isfile(os.path.join(WEIGHTS_DIR, f))]
    if not missing:
        print(f"    present: {WEIGHTS_DIR}")
        return
    print(f"    missing: {', '.join(missing)}")
    if not ask("Download from huggingface.co/nvidia/c-fast-foundationstereo?", auto):
        print("    skipped — the model picker will show 'no weights' until you add them")
        return
    for f in missing:
        dest = os.path.join(WEIGHTS_DIR, f)
        print(f"    downloading {f} …")
        try:
            urllib.request.urlretrieve(HF + f, dest + ".part")
            os.replace(dest + ".part", dest)     # never leave a half file behind
            print(f"      -> {os.path.getsize(dest)/1e6:.1f} MB")
        except Exception as exc:                 # noqa: BLE001
            if os.path.exists(dest + ".part"):
                os.remove(dest + ".part")
            die(f"download failed: {exc}",
                f"Fetch {HF}{f}\nby hand into {WEIGHTS_DIR}")


def environment(auto: bool) -> None:
    head("Python environment")
    if IS_JETSON:
        # setup_jetson.sh owns the device-specific traps (system libs, the
        # Jetson torch wheel index, ptxas, QtWebEngine's sonames). Duplicating
        # them here would mean two things to keep in step.
        script = os.path.join(REPO, "setup_jetson.sh")
        if os.path.isfile(VPY):
            print("    .venv present — re-running setup_jetson.sh (idempotent)")
        if not ask("Run ./setup_jetson.sh? (installs system packages with sudo)", auto):
            die("cannot continue without an environment")
        os.chmod(script, 0o755)
        if run(["bash", script]) != 0:
            die("setup_jetson.sh failed", "Read its output above; it is re-runnable.")
        return
    if not os.path.isfile(VPY):
        print(f"    creating {VENV}")
        if run([sys.executable, "-m", "venv", VENV]) != 0:
            die("venv creation failed")
    pip = [VPY, "-m", "pip"]
    run(pip + ["install", "--upgrade", "pip", "--quiet"])
    print("    torch first, from the CUDA index — plain pip gets CPU wheels")
    if run(pip + ["install", "torch==2.7.0", "torchvision==0.22.0",
                  "--index-url", "https://download.pytorch.org/whl/cu128"]) != 0:
        die("torch install failed",
            "Check the CUDA index is reachable, then re-run this script.")
    if run(pip + ["install", "-r", os.path.join(REPO, "requirements.txt")]) != 0:
        die("requirements install failed")


def verify(run_tests: bool) -> bool:
    head("Verify")
    rc = run([VPY, os.path.join(REPO, "tools", "check_setup.py")])
    if rc != 0:
        print("\n    check_setup.py reports blocking problems (above).")
        print("    Fix them and re-run this script — it will skip what is done.")
        return False
    if run_tests:
        env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
        if run([VPY, "-m", "pytest", "tests", "-q"], env=env) != 0:
            print("    tests failed — the install is usable but something is wrong")
            return False
    return True


def desktop_icon() -> None:
    """A launcher matching the ones already on this rig's desktop."""
    head("Desktop shortcut")
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    if not os.path.isdir(desktop):
        print(f"    no Desktop directory at {desktop} — skipped")
        return
    if IS_WIN:
        target = os.path.join(REPO, "run_studio.bat")
        lnk = os.path.join(desktop, "FoundationStereo Studio.lnk")
        ps = (f"$s=(New-Object -COM WScript.Shell).CreateShortcut('{lnk}');"
              f"$s.TargetPath='{target}';$s.WorkingDirectory='{REPO}';"
              f"$s.Description='Stereo PCB metrology';$s.Save()")
        if run(["powershell", "-NoProfile", "-Command", ps]) == 0:
            print(f"    created {lnk}")
        else:
            print(f"    could not create a shortcut — run {target} directly")
        return
    launcher = os.path.join(REPO, "run_studio.sh")
    os.chmod(launcher, 0o755)
    icon = "applications-graphics"
    for cand in (os.path.expanduser("~/.local/share/icons/fsstudio.png"),
                 os.path.join(REPO, "teaser", "fsd_sample.png")):
        if os.path.isfile(cand):
            icon = cand
            break
    path = os.path.join(desktop, "FoundationStereo Studio.desktop")
    with open(path, "w") as f:
        f.write("[Desktop Entry]\nType=Application\n"
                "Name=FoundationStereo Studio\n"
                "Comment=Stereo PCB metrology\n"
                f"Exec={launcher}\nPath={REPO}\nIcon={icon}\n"
                "Terminal=false\nCategories=Development;Graphics;\n"
                "StartupNotify=true\n")
    os.chmod(path, 0o755)
    # GNOME/Cinnamon refuse to launch a .desktop that is not marked trusted
    subprocess.call(["gio", "set", path, "metadata::trusted", "true"],
                    stderr=subprocess.DEVNULL)
    print(f"    created {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Full setup for FoundationStereo Studio")
    ap.add_argument("--no-icon", action="store_true")
    ap.add_argument("--no-tests", action="store_true")
    ap.add_argument("--yes", "-y", action="store_true", help="assume yes to prompts")
    args = ap.parse_args()

    kind = "Jetson" if IS_JETSON else ("Windows" if IS_WIN else "Linux")
    print(f"FoundationStereo Studio — setup\n  {kind} · {platform.machine()}\n  {REPO}")

    check_python()
    sibling_repo(args.yes)
    weights(args.yes)
    environment(args.yes)
    good = verify(not args.no_tests)
    if not args.no_icon:
        desktop_icon()

    print("\n" + "=" * 62)
    if good:
        print("Ready.  Launch from the desktop icon, or:")
        print(f"  {'run_studio.bat' if IS_WIN else './run_studio.sh'}")
        print("\nNext: load a pair, load your calibration, draw the ROI, Run,")
        print("mark pins and references, then Batch.  See STUDIO_README.md.")
        return 0
    print("Setup incomplete — see the problems above, then re-run this script.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
