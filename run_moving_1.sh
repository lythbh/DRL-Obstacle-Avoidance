#!/bin/bash
# Local Linux runner — launches 9 parallel moving-world curriculum runs (3 archs × 3 seeds),
# continuing training from the existing PPO checkpoints via run_moving_worlds.py.

set -e

# Re-launch inside a detached screen session so training survives logout.
# Reattach later with: screen -r drl_moving
if [ -z "${STY}" ] && [ -z "${TMUX}" ]; then
    exec screen -dmS drl_moving bash "$0" "$@"
fi

# ── CONFIG ────────────────────────────────────────────────────────────────────
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
WEBOTS_HOME="${WEBOTS_HOME:-/uio/hume/student-u79/esbrovol/Downloads/webots-R2025a-x86-64/webots}"
VENV="$REPO_DIR/venv"
TMPDIR_BASE="${TMPDIR:-/tmp}/webots_local_$$"
ARCHS=(gru lstm none)
SEEDS=(0 1 2)
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
    kill "${PIDS[@]}" 2>/dev/null || true
    rm -rf $TMPDIR_BASE
}
trap cleanup EXIT

# Launch 9 runs in parallel, each with its own log file and tmp dir
PIDS=()
RUN_INDEX=0
for ARCH in "${ARCHS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        LOG="$REPO_DIR/logs/moving_${ARCH}_seed${SEED}.log"
        WEBOTS_TMPDIR="$TMPDIR_BASE/${ARCH}_seed${SEED}"
        mkdir -p "$WEBOTS_TMPDIR"
        PORT=$((2000 + RUN_INDEX * 12))
        echo "Starting arch=$ARCH seed=$SEED port=$PORT → $LOG"
        WEBOTS_TMPDIR="$WEBOTS_TMPDIR" WEBOTS_PORT="$PORT" \
            python run_moving_worlds.py --arch "$ARCH" --seed "$SEED" \
            > "$LOG" 2>&1 &
        PIDS+=($!)
        RUN_INDEX=$((RUN_INDEX + 1))
    done
done

echo "Launched ${#PIDS[@]} moving-world runs. Logs in logs/moving_<arch>_seed<N>.log"

# Wait for all runs and collect exit codes
FAILED=0
for PID in "${PIDS[@]}"; do
    wait "$PID" || FAILED=$((FAILED + 1))
done

if [ "$FAILED" -gt 0 ]; then
    echo "$FAILED run(s) failed — check logs for details."
    exit 1
fi
echo "All runs finished successfully."

