#!/bin/bash
# Local Linux runner — launches 6 parallel moving-world curriculum runs (gru × 6 seeds),
# resuming from each seed's own train_10_full checkpoint.

set -e

if [ -z "${STY}" ] && [ -z "${TMUX}" ]; then
    exec screen -dmS drl_moving_gru bash "$0" "$@"
fi

# ── CONFIG ────────────────────────────────────────────────────────────────────
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
WEBOTS_HOME="${WEBOTS_HOME:-/uio/hume/student-u79/esbrovol/Downloads/webots-R2025a-x86-64/webots}"
VENV="$REPO_DIR/venv"
TMPDIR_BASE="${TMPDIR:-/tmp}/webots_local_$$"
CKPT_ROOT="/uio/hume/student-u79/esbrovol/fys5429/DRL-Obstacle-Avoidance/controllers/PPO/checkpoints"

declare -A SEED_CHECKPOINTS
SEED_CHECKPOINTS[0]="$CKPT_ROOT/20260525_141956_gru_seed00_stage10_train_10_full/final_20260525_141956_gru_seed00_stage10_train_10_full.pth"
SEED_CHECKPOINTS[1]="$CKPT_ROOT/20260525_141956_gru_seed01_stage10_train_10_full/final_20260525_141956_gru_seed01_stage10_train_10_full.pth"
SEED_CHECKPOINTS[2]="$CKPT_ROOT/20260525_141956_gru_seed02_stage10_train_10_full/final_20260525_141956_gru_seed02_stage10_train_10_full.pth"
SEED_CHECKPOINTS[3]="$CKPT_ROOT/20260527_131325_gru_seed03_stage10_train_10_full/final_20260527_131325_gru_seed03_stage10_train_10_full.pth"
SEED_CHECKPOINTS[4]="$CKPT_ROOT/20260527_131325_gru_seed04_stage10_train_10_full/final_20260527_131325_gru_seed04_stage10_train_10_full.pth"
SEED_CHECKPOINTS[5]="$CKPT_ROOT/20260527_131325_gru_seed05_stage10_train_10_full/final_20260527_131325_gru_seed05_stage10_train_10_full.pth"
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p logs

nvidia-smi 2>/dev/null || echo "nvidia-smi not found — GPU may not be available"

export PATH=$WEBOTS_HOME:$PATH
export LD_LIBRARY_PATH=$WEBOTS_HOME/lib/webots:$HOME/sndio_install/lib:${LD_LIBRARY_PATH:-}

mkdir -p $TMPDIR_BASE
mkdir -p /tmp/webots/$USER
mkdir -p /tmp/.X11-unix

export EGL_PLATFORM=device
export DRM_RENDER_NODE=/dev/dri/renderD128
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json

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

PIDS=()
RUN_INDEX=0
for SEED in 0 1 2 3 4 5; do
    LOG="$REPO_DIR/logs/moving_gru_seed${SEED}.log"
    WEBOTS_TMPDIR="$TMPDIR_BASE/gru_seed${SEED}"
    mkdir -p "$WEBOTS_TMPDIR"
    PORT=$((2000 + RUN_INDEX * 12))
    RESUME="${SEED_CHECKPOINTS[$SEED]}"
    echo "Starting arch=gru seed=$SEED port=$PORT resume=$RESUME → $LOG"
    WEBOTS_TMPDIR="$WEBOTS_TMPDIR" WEBOTS_PORT="$PORT" \
        python run_moving_worlds.py --arch gru --seed "$SEED" \
        --resume-from "$RESUME" \
        > "$LOG" 2>&1 &
    PIDS+=($!)
    RUN_INDEX=$((RUN_INDEX + 1))
done

echo "Launched ${#PIDS[@]} GRU moving-world runs. Logs in logs/moving_gru_seed<N>.log"

FAILED=0
for PID in "${PIDS[@]}"; do
    wait "$PID" || FAILED=$((FAILED + 1))
done

if [ "$FAILED" -gt 0 ]; then
    echo "$FAILED run(s) failed — check logs for details."
    exit 1
fi
echo "All GRU runs finished successfully."
