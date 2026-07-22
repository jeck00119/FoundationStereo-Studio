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

# JetPack generation gate. L4T R36.x = JetPack 6. Anything older (R35 =
# JetPack 5: Ubuntu 20.04 / Python 3.8 / CUDA 11.4) cannot run this stack:
# requirements' scipy>=1.11 and matplotlib>=3.8 need Python >=3.9, no Triton
# wheels exist for JetPack 5 (Fast-FoundationStereo would fall back to eager
# at ~4x memory on the very device where memory is the constraint), and the
# Jetson torch index ships cp310 wheels only. Fail here, in seconds and with
# the real reason, instead of minutes later inside pip with a misleading one.
L4T_MAJOR=$(sed -n 's/^# R\([0-9][0-9]*\).*/\1/p' /etc/nv_tegra_release 2>/dev/null | head -1)
if [ -n "$L4T_MAJOR" ] && [ "$L4T_MAJOR" -lt 36 ] && [ -z "$JETSON_SKIP_L4T_CHECK" ]; then
    fail "L4T R$L4T_MAJOR = JetPack 5 or older; this port needs JetPack 6.x (L4T R36+).
   Reflash the device with NVIDIA SDK Manager, picking JetPack 6.x, then re-run this
   script. (To attempt an unsupported setup anyway: JETSON_SKIP_L4T_CHECK=1 ./setup_jetson.sh)"
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
    libxdamage1 libxrandr2 libxtst6 libminizip1

# ---- 3. venv -----------------------------------------------------------------
say "Creating .venv"
[ -d .venv ] || $PY -m venv .venv
PIP=".venv/bin/pip"
$PIP install -q --upgrade pip wheel

# ---- 4. torch for Jetson -----------------------------------------------------
# Standard PyPI torch wheels don't serve the Jetson iGPU (x86, or SBSA-server
# aarch64). The Jetson AI Lab index publishes per-JetPack aarch64 CUDA wheels;
# jp6/cu126 matches JetPack 6.x. The original host (pypi.jetson-ai-lab.dev)
# went dark in 2026 — the .io devpi mirror below is live (torch 2.8–2.11,
# cp310) and proxies PyPI, so torch's own deps (sympy, filelock, …) resolve
# through it too. If it moves again, check forums.developer.nvidia.com for the
# current torch-for-JetPack instructions, install torch+torchvision+triton
# manually into .venv, then RE-RUN this script — it will skip ahead.
#
# triton rides along deliberately: it is what lets Fast-FoundationStereo run
# its compiled cost-volume path (~4x less peak memory than the eager fallback
# — decisive on a shared-memory 8 GB Orin). PyPI's triton has no Jetson build;
# this index's does.
say "Installing torch/torchvision/triton (Jetson wheels)"
if ! .venv/bin/python -c "import torch, triton" 2>/dev/null; then
    JETSON_INDEX="${JETSON_TORCH_INDEX:-https://pypi.jetson-ai-lab.io/jp6/cu126/+simple}"
    echo "using index: $JETSON_INDEX   (override with JETSON_TORCH_INDEX=...)"
    $PIP install torch torchvision triton --index-url "$JETSON_INDEX" \
        || fail "torch install failed — install Jetson torch manually (see note above), then re-run."
    # torch >=2.10 jp6 wheels link libcudss.so.0, which JetPack does not ship.
    # PyPI's aarch64 nvidia-cudss wheel provides it — but --no-deps is load-
    # bearing: the wheel's declared deps are SBSA (server-ARM) cublas/nvrtc
    # 12.9 builds without Orin (sm_87) kernels, and once loaded under the same
    # sonames they shadow JetPack's Tegra libs — cublasCreate then fails with
    # CUBLAS_STATUS_ALLOC_FAILED on the first matmul. Without deps, cudss
    # resolves cublas from the system (JetPack) via ldconfig, which is correct.
    # 0.7.x is the CUDA-12.6-era line (0.8+ pairs with newer toolkits).
    # The symlink matters too: torch's _preload_cuda_deps list has no cudss
    # entry and libtorch_cuda's RUNPATH is bare $ORIGIN, so the pip nvidia/
    # tree is invisible to the loader until the lib sits in torch/lib itself.
    $PIP install -q --no-deps "nvidia-cudss-cu12==0.7.*" \
        || fail "nvidia-cudss-cu12 install failed — torch >=2.10 cannot import without it."
    CUDSS_LIB=$(find .venv/lib -path "*/nvidia/*" -name "libcudss.so.0" | head -1)
    TORCH_LIB_DIR=$(dirname "$(find .venv/lib -name "libtorch_cuda.so" | head -1)")
    [ -n "$CUDSS_LIB" ] && [ -n "$TORCH_LIB_DIR" ] \
        || fail "could not locate libcudss.so.0 / torch lib dir for the RUNPATH link."
    ln -sfr "$CUDSS_LIB" "$TORCH_LIB_DIR/libcudss.so.0"
fi
# The jp6 triton wheel bundles only a Blackwell (sm_100+) ptxas variant, so
# the first kernel compile dies with "Cannot find ptxas". JetPack's own
# cuda-nvcc package carries the real 12.6 one — link it where triton looks.
# Outside the install-gate above on purpose: torch/triton IMPORT fine without
# it, so a re-run must be able to repair this even when the gate skips.
TRITON_BIN_DIR=$(ls -d .venv/lib/python*/site-packages/triton/backends/nvidia/bin 2>/dev/null | head -1)
if [ -n "$TRITON_BIN_DIR" ] && [ ! -e "$TRITON_BIN_DIR/ptxas" ]; then
    [ -x /usr/local/cuda/bin/ptxas ] \
        || fail "no ptxas at /usr/local/cuda/bin (is cuda-nvcc-12-6 installed?) — triton cannot compile without it."
    ln -sf /usr/local/cuda/bin/ptxas "$TRITON_BIN_DIR/ptxas"
fi

.venv/bin/python - <<'EOF'
import torch
print(f"torch {torch.__version__}  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device: {torch.cuda.get_device_name(0)}")
else:
    print("warning: CUDA not available — the JetPack/torch combination is wrong.")
try:
    import triton
    print(f"triton {triton.__version__}")
except Exception:
    print("warning: triton missing — Fast-FoundationStereo will run eager at ~4x memory.")
EOF

# ---- 5. everything else ------------------------------------------------------
say "Installing app requirements"
$PIP install -q -r requirements-jetson.txt \
    || echo "warning: some optional packages failed (open3d has no wheel for every
python/aarch64 combo — the app runs without it; denoise + PLY export degrade)."

# ---- 5b. QtWebEngine's pre-22.04 sonames -------------------------------------
# The PySide6 6.8 aarch64 QtWebEngine binaries were built against an Ubuntu
# 20.04-era sysroot: besides libminizip.so.1 (apt: libminizip1, in the list
# above) they need libwebp.so.6 — which Ubuntu 22.04 does NOT ship (it moved to
# libwebp7, a bumped soname). Symlinking .so.7 under the .so.6 name invites
# missing-symbol crashes deep inside Chromium; instead drop the REAL focal
# 0.6.1 lib into PySide6's private Qt/lib — that dir is every Qt lib's $ORIGIN
# RUNPATH, so nothing outside this venv ever sees the old library.
QT_LIB_DIR=$(ls -d .venv/lib/python*/site-packages/PySide6/Qt/lib 2>/dev/null | head -1)
if [ -n "$QT_LIB_DIR" ] && [ ! -e "$QT_LIB_DIR/libwebp.so.6" ]; then
    say "Fetching libwebp.so.6 (Ubuntu focal) for QtWebEngine"
    WEBP_POOL="http://ports.ubuntu.com/pool/main/libw/libwebp/"
    WEBP_DEB=$(curl -s "$WEBP_POOL" | grep -oE 'libwebp6_[^"]*arm64\.deb' | sort -uV | tail -1)
    [ -n "$WEBP_DEB" ] || fail "no focal libwebp6 arm64 deb found at $WEBP_POOL"
    WEBP_TMP=$(mktemp -d)
    curl -s -o "$WEBP_TMP/$WEBP_DEB" "$WEBP_POOL$WEBP_DEB" \
        && dpkg-deb -x "$WEBP_TMP/$WEBP_DEB" "$WEBP_TMP/x" \
        && cp -a "$WEBP_TMP/x"/usr/lib/aarch64-linux-gnu/libwebp.so.6* "$QT_LIB_DIR/" \
        || fail "libwebp6 fetch/extract failed — the 3D (QtWebEngine) view cannot load without it."
    rm -rf "$WEBP_TMP"
fi

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
