#!/bin/bash
#SBATCH --job-name=drl_ppo
#SBATCH --account=ec12
#SBATCH --time=00:30:00
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/ppo_%j.out
#SBATCH --error=logs/ppo_%j.err

set -e
mkdir -p logs

module load Python/3.10.8-GCCcore-12.2.0
module load GCCcore/12.2.0
module load CUDA/12.1.1

nvidia-smi

# Confirm CUDA is reachable from Python (will print True if GPU training will work)
python -c "import sys; sys.path.insert(0,'/fp/homes01/u01/ec-esbrovol/.local/lib/python3.10/site-packages'); import torch; print('CUDA available:', torch.cuda.is_available())" || echo "torch not found yet"

# Webots paths
export WEBOTS_HOME=/fp/homes01/u01/ec-esbrovol/fys5429/webots
export PATH=$WEBOTS_HOME:$PATH
export LD_LIBRARY_PATH=$WEBOTS_HOME/lib/webots:/cluster/software/EL9/easybuild/software/X11/20221110-GCCcore-12.2.0/lib:$LD_LIBRARY_PATH

# Webots temp dir (keep off shared /tmp)
export WEBOTS_TMPDIR=/fp/homes01/u01/ec-esbrovol/fys5429/tmp/webots_$SLURM_JOB_ID
mkdir -p $WEBOTS_TMPDIR
mkdir -p /tmp/webots/ec-esbrovol

# Fix 1: create X11 socket dir — Xvfb won't create it as non-root
mkdir -p /tmp/.X11-unix

# Fix 2: tell EGL to use the X11 backend instead of scanning /dev/dri devices
# (LIBGL_ALWAYS_SOFTWARE doesn't affect Webots' bundled libEGL, but EGL_PLATFORM=x11 does)
export EGL_PLATFORM=x11
export LIBGL_ALWAYS_SOFTWARE=1

# Start virtual display (Qt xcb plugin requires X11 — offscreen is not in Webots' Qt build)
Xvfb :99 -screen 0 1024x768x24 -nolisten tcp &
XVFB_PID=$!
export DISPLAY=:99
sleep 3

if ! kill -0 $XVFB_PID 2>/dev/null; then
    echo "ERROR: Xvfb failed to start"
    exit 1
fi
echo "Xvfb started (PID $XVFB_PID)"

# Python env
source /fp/homes01/u01/ec-esbrovol/fys5429/DRL-Obstacle-Avoidance/.venv/bin/activate
# torch installed to ~/.local (venv site-packages was read-only), so include it explicitly
export PYTHONPATH=/fp/homes01/u01/ec-esbrovol/.local/lib/python3.10/site-packages:/fp/homes01/u01/ec-esbrovol/fys5429/DRL-Obstacle-Avoidance:$WEBOTS_HOME/lib/controller/python:$PYTHONPATH
export PYTHONUNBUFFERED=1

# Launch Webots headlessly in background
# --stdout --stderr: pipe the controller's stdout/stderr into this job's log
webots --no-rendering --batch --minimize --stdout --stderr --mode=fast \
    /fp/homes01/u01/ec-esbrovol/fys5429/DRL-Obstacle-Avoidance/worlds/ObstacleCourse.wbt &
WEBOTS_PID=$!
echo "Webots launched (PID $WEBOTS_PID), waiting for it to load..."

# Wait for Webots to load — check every 2s that it hasn't crashed
for i in $(seq 1 10); do
    sleep 2
    if ! kill -0 $WEBOTS_PID 2>/dev/null; then
        echo "ERROR: Webots exited unexpectedly after $((i*2))s"
        rm -rf $WEBOTS_TMPDIR
        exit 1
    fi
done
echo "Webots has been running for 20s..."

# Webots launches controllers/PPO/PPO.py internally (controller "PPO" in world file).
# Do NOT run python PPO.py here — that would be a second unconnected process.
# Just wait for Webots (and its controller) to finish.
wait $WEBOTS_PID
echo "Webots finished."

# Cleanup
kill $XVFB_PID 2>/dev/null || true
rm -rf $WEBOTS_TMPDIR