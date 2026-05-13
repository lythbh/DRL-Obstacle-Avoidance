# DRL Obstacle Avoidance (Webots ALTINO)

This project trains recurrent PPO and recurrent SAC controllers for ALTINO obstacle avoidance in Webots, with shared environment, sensor, reward, and SLAM processing.

## Environment Setup
1. Create and activate a Python 3.10.19 environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. In Webots, set Preferences -> Python command to your project environment Python.
   Example:
   ```text
   C:\Users\ErikH\miniconda3\envs\FYS5429\python.exe
   ```

## Repository Layout
- `controllers/PPO/PPO.py`: recurrent PPO training controller.
- `controllers/SAC/SAC.py`: recurrent SAC training controller.
- `controllers/RNN/`: recurrent policy/value modules used by PPO.
- `controllers/Webots/webots_env.py`: Webots environment, reward logic, success criteria, and SLAM hooks.
- `controllers/SLAM/`: IMU/IEKF/map processing modules.
- `run_model.py`: deterministic inference runner for PPO or SAC checkpoints.
- `worlds/ObstacleCourse.wbt`: PPO world.
- `worlds/Simple.wbt`: SAC world.

## Current Pipeline
1. Webots starts the selected world/controller (PPO or SAC).
2. Controller initializes a global Supervisor and creates `WebotsEnv`.
3. `WebotsEnv` builds sensors, motor control, SLAM processor, reward computer, and observation vector.
4. Default observation size is 33 features:
   - 16 LiDAR sector minima
   - 7 pose/goal features
   - 10 IMU features
5. Action is continuous `[steering, speed]` with environment clipping.
6. Success is strict: goal is counted only when the robot reaches the goal region and is below the stop-speed threshold.
7. Both controllers can run recurrent policies with `gru` or `lstm` encoders.
8. SAC now uses a small fixed-size replay buffer of recurrent sequence windows, so it is off-policy again without the overhead of a large transition replay system.

## Training Behavior
Both algorithms now report aligned episode metrics:
- `r`, `avg10`, `steps`, `succ10`, `touch10`, `col10`, `to10`, `min_d`, `end`, `t`

Where:
- `succ10`: rolling 10-episode success rate (goal reached and stopped)
- `touch10`: rolling 10-episode goal-touch rate
- `col10`: rolling 10-episode collision rate
- `to10`: rolling 10-episode timeout (`max_steps`) rate

## PPO vs SAC (Current Code)
### PPO
- Recurrent options: `gru` or `lstm`
- Update style: on-policy rollout trajectories
- Update schedule: every `update_every` episodes
- Loss style: clipped PPO objective + value loss + entropy regularization

### SAC
- Recurrent options: `gru` or `lstm`
- Update style: off-policy sequence replay sampled from a compact ring buffer of fixed-length episode windows
- Update schedule: after `update_after_steps`, sample `replay_batch_size` windows and run `updates_per_step` optimizer steps
- Loss style: twin critics, target critics (soft update with `tau`), entropy temperature `alpha` (auto-tuning optional), burn-in masking for recurrent sequences

## What Changed in SAC
The SAC pipeline was updated to keep recurrent training fast while restoring off-policy learning:
- Added a compact ring replay buffer that stores fixed-length recurrent sequence windows.
- Replay stores preallocated NumPy arrays and samples batches directly to tensors to avoid the old Python-heavy slowdown.
- Added `sequence_stride`, `replay_capacity`, `replay_batch_size`, and `min_replay_sequences` to control buffer size and sampling.
- SAC now samples random replay windows after episodes, so updates are off-policy again.
- Removed the old feedforward (`ff`) recurrent-cell path; SAC now uses only `gru` and `lstm`.

Important note:
- This is not a large classic transition replay buffer. It is a compact sequence replay buffer designed for recurrent SAC.

## Checkpoints
Both controllers save best checkpoints into algorithm-specific dated folders:
- PPO: `controllers/PPO/checkpoints/<timestamp>/best_model.pth`
- SAC: `controllers/SAC/checkpoints/<timestamp>/best_model.pth`

Both save final model to controller root:
- PPO: `controllers/PPO/final_model.pth`
- SAC: `controllers/SAC/final_model.pth`

## Inference
Run deterministic inference with:
```bash
python run_model.py --algorithm ppo --episodes 10
python run_model.py --algorithm sac --episodes 10
```

Optional args:
- `--model-path <path>` to force a specific checkpoint
- `--quiet` to reduce per-episode output

If `--model-path` is omitted, `run_model.py` loads the newest dated `best_model.pth` for the selected algorithm.

