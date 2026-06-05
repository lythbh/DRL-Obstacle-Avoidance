#!/bin/bash
# Local Linux runner — runs validation.py sequentially across all static and moving worlds.

set -e

# Re-launch inside a detached screen session so validation survives logout.
# Reattach later with: screen -r drl_validation
if [ -z "${STY}" ] && [ -z "${TMUX}" ]; then
    exec screen -dmS drl_validation bash "$0" "$@"
fi

# ── CONFIG ────────────────────────────────────────────────────────────────────
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
WEBOTS_HOME="${WEBOTS_HOME:-/uio/hume/student-u79/esbrovol/Downloads/webots-R2025a-x86-64/webots}"
VENV="$REPO_DIR/venv"
TMPDIR_BASE="${TMPDIR:-/tmp}/webots_local_$$"
PORT=1234
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p logs

nvidia-smi 2>/dev/null || echo "nvidia-smi not found — GPU may not be available"

# Webots paths
export PATH=$WEBOTS_HOME:$PATH
export LD_LIBRARY_PATH=$WEBOTS_HOME/lib/webots:$HOME/sndio_install/lib:${LD_LIBRARY_PATH:-}

# Per-run Webots tmp dirs live under TMPDIR_BASE
mkdir -p $TMPDIR_BASE
mkdir -p /tmp/webots/$USER
mkdir -p /tmp/.X11-unix

# EGL: use NVIDIA GPU via renderD128 (world-accessible on cupid)
export EGL_PLATFORM=device
export DRM_RENDER_NODE=/dev/dri/renderD128
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Start a single shared virtual display
DISPLAY_NUM=$((90 + $$ % 900))
Xvfb :$DISPLAY_NUM -screen 0 1024x768x24 -nolisten tcp &
XVFB_PID=$!
export DISPLAY=:$DISPLAY_NUM
sleep 3

if ! kill -0 $XVFB_PID 2>/dev/null; then
    echo "ERROR: Xvfb failed to start"
    exit 1
fi
echo "Xvfb started (PID $XVFB_PID, DISPLAY=$DISPLAY)"

# Python env
source "$VENV/bin/activate"
export PYTHONPATH=$REPO_DIR:$WEBOTS_HOME/lib/controller/python:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1
export PPO_FORCE_CPU=${PPO_FORCE_CPU:-0}

python -c "import torch; print('CUDA available:', torch.cuda.is_available())" || echo "torch not found"

cleanup() {
    kill $XVFB_PID 2>/dev/null || true
    rm -rf $TMPDIR_BASE
}
trap cleanup EXIT

LOG="$REPO_DIR/logs/validation.log"
echo "Starting validation → $LOG"

WEBOTS_TMPDIR="$TMPDIR_BASE/validation" python validation.py \
    --webots-cmd "$WEBOTS_HOME/webots" \
    --port $PORT \
    "$@" \
    > "$LOG" 2>&1

echo "Validation finished. Log: $LOG"
