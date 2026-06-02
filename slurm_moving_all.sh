#!/bin/bash
#SBATCH --job-name=drl_moving_gru_none
#SBATCH --account=ec12
#SBATCH --time=20:00:00
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=110G
#SBATCH --output=/fp/homes01/u01/ec-esbrovol/fys5429/DRL-Obstacle-Avoidance/logs/moving_gru_none_%j.out
#SBATCH --error=/fp/homes01/u01/ec-esbrovol/fys5429/DRL-Obstacle-Avoidance/logs/moving_gru_none_%j.err

set -e
mkdir -p logs

module load Python/3.10.8-GCCcore-12.2.0
module load GCCcore/12.2.0
module load CUDA/12.1.1

nvidia-smi

REPO_DIR="/fp/homes01/u01/ec-esbrovol/fys5429/DRL-Obstacle-Avoidance"
WEBOTS_HOME="/fp/homes01/u01/ec-esbrovol/fys5429/webots"
# NOTE: checkpoints must exist at this path on Fox before submitting.
# If they were trained on the local Linux machine, sync first:
#   rsync -av /uio/hume/student-u79/esbrovol/fys5429/DRL-Obstacle-Avoidance/controllers/PPO/checkpoints/ \
#       fox:/fp/homes01/u01/ec-esbrovol/fys5429/DRL-Obstacle-Avoidance/controllers/PPO/checkpoints/
CKPT_ROOT="$REPO_DIR/controllers/PPO/checkpoints"

export PATH=$WEBOTS_HOME:$PATH
export LD_LIBRARY_PATH=$WEBOTS_HOME/lib/webots:/cluster/software/EL9/easybuild/software/X11/20221110-GCCcore-12.2.0/lib:${LD_LIBRARY_PATH:-}

WEBOTS_TMPDIR_BASE="/fp/homes01/u01/ec-esbrovol/fys5429/tmp/webots_${SLURM_JOB_ID}"
mkdir -p "$WEBOTS_TMPDIR_BASE"
mkdir -p /tmp/.X11-unix

export EGL_PLATFORM=x11
export LIBGL_ALWAYS_SOFTWARE=1

DISPLAY_NUM=$((90 + SLURM_JOB_ID % 900))
Xvfb :$DISPLAY_NUM -screen 0 1024x768x24 -nolisten tcp &
XVFB_PID=$!
export DISPLAY=:$DISPLAY_NUM
sleep 3

if ! kill -0 $XVFB_PID 2>/dev/null; then
    echo "ERROR: Xvfb failed to start"
    exit 1
fi
echo "Xvfb started (PID $XVFB_PID, DISPLAY=$DISPLAY)"

source "$REPO_DIR/.venv/bin/activate"
export PYTHONPATH=/fp/homes01/u01/ec-esbrovol/.local/lib/python3.10/site-packages:$REPO_DIR:$WEBOTS_HOME/lib/controller/python:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1
export PPO_FORCE_CPU=${PPO_FORCE_CPU:-0}

python -c "import torch; print('CUDA available:', torch.cuda.is_available())" || echo "torch not found"

cleanup() {
    kill $XVFB_PID 2>/dev/null || true
    kill "${PIDS[@]}" 2>/dev/null || true
    rm -rf "$WEBOTS_TMPDIR_BASE"
}
trap cleanup EXIT

# ── Per-seed checkpoints ───────────────────────────────────────────────────────
declare -A GRU_CKPTS
GRU_CKPTS[0]="$CKPT_ROOT/20260525_141956_gru_seed00_stage10_train_10_full/final_20260525_141956_gru_seed00_stage10_train_10_full.pth"
GRU_CKPTS[1]="$CKPT_ROOT/20260525_141956_gru_seed01_stage10_train_10_full/final_20260525_141956_gru_seed01_stage10_train_10_full.pth"
GRU_CKPTS[2]="$CKPT_ROOT/20260525_141956_gru_seed02_stage10_train_10_full/final_20260525_141956_gru_seed02_stage10_train_10_full.pth"
GRU_CKPTS[3]="$CKPT_ROOT/20260527_131325_gru_seed03_stage10_train_10_full/final_20260527_131325_gru_seed03_stage10_train_10_full.pth"
GRU_CKPTS[4]="$CKPT_ROOT/20260527_131325_gru_seed04_stage10_train_10_full/final_20260527_131325_gru_seed04_stage10_train_10_full.pth"
GRU_CKPTS[5]="$CKPT_ROOT/20260527_131325_gru_seed05_stage10_train_10_full/final_20260527_131325_gru_seed05_stage10_train_10_full.pth"

declare -A NONE_CKPTS
NONE_CKPTS[0]="$CKPT_ROOT/20260525_141956_none_seed00_stage10_train_10_full/final_20260525_141956_none_seed00_stage10_train_10_full.pth"
NONE_CKPTS[1]="$CKPT_ROOT/20260525_141956_none_seed01_stage10_train_10_full/final_20260525_141956_none_seed01_stage10_train_10_full.pth"
NONE_CKPTS[2]="$CKPT_ROOT/20260525_141956_none_seed02_stage10_train_10_full/final_20260525_141956_none_seed02_stage10_train_10_full.pth"
NONE_CKPTS[3]="$CKPT_ROOT/20260527_131325_none_seed03_stage10_train_10_full/final_20260527_131325_none_seed03_stage10_train_10_full.pth"
NONE_CKPTS[4]="$CKPT_ROOT/20260527_131325_none_seed04_stage10_train_10_full/final_20260527_131325_none_seed04_stage10_train_10_full.pth"
NONE_CKPTS[5]="$CKPT_ROOT/20260527_131325_none_seed05_stage10_train_10_full/final_20260527_131325_none_seed05_stage10_train_10_full.pth"
# ─────────────────────────────────────────────────────────────────────────────

PIDS=()
RUN_INDEX=0

for ARCH in gru none; do
    for SEED in 0 1 2 3 4 5; do
        case $ARCH in
            gru)  RESUME="${GRU_CKPTS[$SEED]}" ;;
            none) RESUME="${NONE_CKPTS[$SEED]}" ;;
        esac

        LOG="$REPO_DIR/logs/fox_moving_${ARCH}_seed${SEED}_${SLURM_JOB_ID}.log"
        WEBOTS_TMPDIR="$WEBOTS_TMPDIR_BASE/${ARCH}_seed${SEED}"
        mkdir -p "$WEBOTS_TMPDIR"
        PORT=$((2000 + RUN_INDEX * 12))

        echo "Starting arch=$ARCH seed=$SEED port=$PORT resume=$RESUME → $LOG"
        WEBOTS_TMPDIR="$WEBOTS_TMPDIR" WEBOTS_PORT="$PORT" \
            python "$REPO_DIR/run_moving_worlds.py" --arch "$ARCH" --seed "$SEED" \
            --resume-from "$RESUME" \
            > "$LOG" 2>&1 &
        PIDS+=($!)
        RUN_INDEX=$((RUN_INDEX + 1))
    done
done

echo "Launched ${#PIDS[@]} moving-world runs (gru + none × 6 seeds). Per-seed logs in logs/fox_moving_<arch>_seed<N>_<jobid>.log"

FAILED=0
for PID in "${PIDS[@]}"; do
    wait "$PID" || FAILED=$((FAILED + 1))
done

if [ "$FAILED" -gt 0 ]; then
    echo "$FAILED run(s) failed — check per-seed logs for details."
    exit 1
fi
echo "All 12 moving-world runs finished successfully."
