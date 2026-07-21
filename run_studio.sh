#!/usr/bin/env bash
# FoundationStereo Studio launcher (Linux / Jetson).
# Mirror of run_studio.bat: run from anywhere, uses the repo's own venv.
set -e
cd "$(dirname "$(readlink -f "$0")")"

if [ ! -x .venv/bin/python ]; then
    echo "No .venv found — run ./setup_jetson.sh first (or create .venv yourself)."
    exit 1
fi

# QtWebEngine (the 3D view) on some Jetson/driver combinations needs one of
# these — uncomment only if the 3D tab stays black:
# export QTWEBENGINE_DISABLE_SANDBOX=1
# export QTWEBENGINE_CHROMIUM_FLAGS="--ignore-gpu-blocklist"

exec .venv/bin/python -m studio.app "$@"
