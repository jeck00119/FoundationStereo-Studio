#!/usr/bin/env bash
# One-shot FoundationStereo Studio setup for a Jetson Orin Nano (8 GB, JetPack 6.x).
#
#   git clone <this repo> && cd FoundationStereo && ./setup_jetson.sh
#
# What it does, in order: sanity-checks the platform, installs the system
# libraries Qt/WebEngine need, creates .venv, installs torch/torchvision from
# the Jetson wheel index (standard PyPI torch is x86-only), installs the rest
# from requirements-jetson.txt, then validates with the offscreen test suite.
# Idempotent — safe to re-run after a failure.
set -e
cd "$(dirname "$(readlink -f "$0")")"

say()  { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
fail() { printf '\n\033[1;31mXX %s\033[0m\n' "$*"; exit 1; }

# ---- 1. platform sanity ------------------------------------------------------
say "Checking platform"
[ "$(uname -m)" = "aarch64" ] || fail "This script is for Jetson (aarch64); got $(uname -m)."
if [ -f /etc/nv_tegra_release ]; then
    cat /etc/nv_tegra_release
else
    echo "warning: /etc/nv_tegra_release not found — is this a Jetson with JetPack flashed?"
fi
PY=python3
$PY --version || fail "python3 not found"

# ---- 2. system packages ------------------------------------------------------
say "Installing system packages (sudo)"
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3-venv python3-pip python3-dev \
    libgl1 libegl1 libopengl0 libglib2.0-0 libfontconfig1 libdbus-1-3 \
    libxkbcommon-x11-0 libxcb-cursor0 libxcb-icccm4 libxcb-keysyms1 \
    libxcb-shape0 libxcb-xinerama0 libnss3 libasound2 libxcomposite1 \
    libxdamage1 libxrandr2 libxtst6

# ---- 3. venv -----------------------------------------------------------------
say "Creating .venv"
[ -d .venv ] || $PY -m venv .venv
PIP=".venv/bin/pip"
$PIP install -q --upgrade pip wheel

# ---- 4. torch for Jetson -----------------------------------------------------
# Standard PyPI torch wheels are x86-only. The Jetson AI Lab index publishes
# aarch64 CUDA wheels per JetPack release; cu126 matches JetPack 6.x. If this
# index moves or your JetPack differs, see:
#   https://pypi.jetson-ai-lab.dev   and   https://forums.developer.nvidia.com
# for the current torch-for-JetPack instructions, install torch+torchvision
# manually into .venv, then RE-RUN this script — it will skip ahead.
say "Installing torch/torchvision (Jetson wheels)"
if ! .venv/bin/python -c "import torch" 2>/dev/null; then
    JETSON_INDEX="${JETSON_TORCH_INDEX:-https://pypi.jetson-ai-lab.dev/jp6/cu126}"
    echo "using index: $JETSON_INDEX   (override with JETSON_TORCH_INDEX=...)"
    $PIP install torch torchvision --index-url "$JETSON_INDEX" \
        || fail "torch install failed — install Jetson torch manually (see note above), then re-run."
fi
.venv/bin/python - <<'EOF'
import torch
print(f"torch {torch.__version__}  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device: {torch.cuda.get_device_name(0)}")
else:
    print("warning: CUDA not available — the JetPack/torch combination is wrong.")
EOF

# ---- 5. everything else ------------------------------------------------------
say "Installing app requirements"
$PIP install -q -r requirements-jetson.txt \
    || echo "warning: some optional packages failed (open3d has no wheel for every
python/aarch64 combo — the app runs without it; denoise + PLY export degrade)."

# ---- 6. validate -------------------------------------------------------------
say "Validating (offscreen test suite)"
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests -q \
    || fail "Test suite failed — paste the output into the support session."

say "Done"
cat <<'EOF'
Setup complete. Remaining manual items:
  1. Model weights: put the FoundationStereo checkpoint under
     pretrained_models/23-51-11/  and/or clone Fast-FoundationStereo as a
     SIBLING of this repo with its weights/ (see studio/backends/registry.py).
     On 8 GB unified memory, Fast-FoundationStereo is the practical model.
  2. Your calibration transfers as-is: copy data/calib/*.json from the
     Windows machine (device-independent).
  3. Start the app:   ./run_studio.sh
     First run of a model compiles/loads slowly; if the 3D tab stays black,
     see the commented QTWEBENGINE lines in run_studio.sh.
EOF
