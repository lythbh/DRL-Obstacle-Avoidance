#!/bin/bash
# Local Linux runner — equivalent to slurm.sh but without SLURM or module system.
# Edit the variables in the CONFIG section below to match your machine.

set -e

# ── CONFIG ────────────────────────────────────────────────────────────────────
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
WEBOTS_HOME="${WEBOTS_HOME:-/uio/hume/student-u79/esbrovol/Downloads/webots-R2025a-x86-64/webots}"  # override via env if needed
VENV="$REPO_DIR/venv"                                    # path to your Python venv
TMPDIR_BASE="${TMPDIR:-/tmp}/webots_local_$$"
# Default curriculum command — override by setting PPO_RUN_COMMAND in the environment
DEFAULT_RUN_COMMAND="python controllers/run.py worker --arch gru --seed 0"
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p logs

nvidia-smi 2>/dev/null || echo "nvidia-smi not found — GPU may not be available"

# Webots paths
export PATH=$WEBOTS_HOME:$PATH
export LD_LIBRARY_PATH=$WEBOTS_HOME/lib/webots:$HOME/sndio_install/lib:${LD_LIBRARY_PATH:-}

# Webots temp dir
export WEBOTS_TMPDIR=$TMPDIR_BASE
mkdir -p $WEBOTS_TMPDIR
mkdir -p /tmp/webots/$USER

# X11 socket dir — Xvfb won't create it as non-root
mkdir -p /tmp/.X11-unix

# EGL: use X11 backend (same fix as on HPC)
export EGL_PLATFORM=x11
export LIBGL_ALWAYS_SOFTWARE=1

# Start virtual display
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
    rm -rf $WEBOTS_TMPDIR
}
trap cleanup EXIT

RUN_COMMAND="${PPO_RUN_COMMAND:-$DEFAULT_RUN_COMMAND}"
echo "Running: $RUN_COMMAND"
eval "$RUN_COMMAND"
echo "Training finished."
