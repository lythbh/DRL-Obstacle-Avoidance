# DRL Obstacle Avoidance (Webots ALTINO)

This project trains recurrent PPO and SAC controllers for ALTINO obstacle avoidance in Webots, with shared environment, sensor, reward, and mapping processing across algorithms. Curriculum training is orchestrated by `controllers/run.py` through 10 progressively harder worlds.

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
- `controllers/run.py`: HPC/local launcher for curriculum training (worker + submit subcommands).
- `controllers/PPO/PPO.py`: recurrent PPO training controller + `FeedForwardActorCritic`.
- `controllers/PPO/PPO_defaults.py`: PPO config constants, `PPODefaults`, `RecurrentDefaults`.
- `controllers/PPO/PPO_rewards.py`: PPO-specific reward computation.
- `controllers/SAC/SAC.py`: SAC training controller with replay buffer.
- `controllers/SAC/SAC_defaults.py`: SAC config constants, `SACDefaults`.
- `controllers/SAC/SAC_rewards.py`: SAC-specific reward computation.
- `controllers/DDPG/DDPG.py`: DDPG controller (legacy/reference).
- `controllers/RNN/`: `base.py` (shared base), `gru.py`, `lstm.py`, `__init__.py`.
- `controllers/Webots/webots_env.py`: Webots environment, mapping hooks, Altino driver.
- `controllers/state_estimation/`: IMU/IEKF/map processing modules.
- `controllers/common/`: checkpoint utilities, metrics logger, seed, training_utils.
- `run_model.py`: deterministic inference runner for PPO and SAC checkpoints.
- `slurm.sh`: SLURM job script for HPC training.
- `worlds/generate_training_worlds.py`: generates 10 curriculum `.wbt` worlds.
- `worlds/generate_validation_worlds.py`: generates 5 validation `.wbt` worlds.
- `worlds/training/`: curriculum world files (`train_1_empty.wbt` – `train_10_full.wbt`).
- `worlds/validation/`: validation world files (`val_1_empty_center.wbt` – `val_5_dense.wbt`).
- `worlds/testing/`: manual test worlds (`ObstacleCourse.wbt`, `Simple.wbt`, `SimplePPO.wbt`).
- `tests/test_algorithms.py`: PPO/SAC unit tests (no Webots required).

## Complete Pipeline

### 1. Entry Points

| Component | File | Function |
|---|---|---|
| Curriculum Launcher | `controllers/run.py` | `worker` (single arch), `submit` (SLURM batch) |
| PPO Training | `controllers/PPO/PPO.py` | `train()` |
| SAC Training | `controllers/SAC/SAC.py` | `train()` |
| Inference | `run_model.py` | `run_inference()` |

PPO and SAC are started by Webots as robot controllers. `controllers/run.py` orchestrates sequential world runs, passing final checkpoints between worlds for curriculum transfer. `run_model.py` loads a saved checkpoint and runs deterministic episodes independently.

#### `controllers/run.py` usage:
```bash
# Single-architecture worker through all curriculum worlds
python controllers/run.py worker --arch gru --seed 0
# Resume from a previous checkpoint
python controllers/run.py worker --arch gru --seed 0 --resume-from path/to/checkpoint.pth
# Override training duration
python controllers/run.py worker --arch gru --seed 0 --episodes 2500 --max-steps 6000
# Submit SLURM jobs for all architectures × N seeds
python controllers/run.py submit --sessions 10 --episodes 2500
# Submit for specific worlds only
python controllers/run.py worker --arch none --worlds worlds/training/train_1_empty.wbt worlds/training/train_3_three_obs.wbt
```

Architecture aliases: `none`, `mlp`, `feedforward`, `ff` all map to feed-forward (no recurrent core).

### 2. Configuration System

| File | Provides |
|---|---|
| `controllers/PPO/PPO_defaults.py` | `PPODefaults`, `RecurrentDefaults`, environment constants, mapping flags, reward coefficients |
| `controllers/SAC/SAC_defaults.py` | `SACDefaults`, environment constants, reward coefficients |

**PPO defaults**: `episodes=500`, `update_every=4`, `epochs=4`, `batch_size=128`, `lr=1e-4`, `entropy_coef=0.005`, `gamma=0.99`, `gae_lambda=0.98`, `epsilon=0.1`, `max_steps=2500`, `save_every=100`, `enable_mapping=False`, `save_mapping_plots=False`, `REW_SCALE=0.01`.

**Recurrent defaults**: `sequence_length=32`, `burn_in=8`, `sequence_stride=16`.

**Environment variables** override Config fields per-run (`PPO_ARCH`, `PPO_EPISODES`, `PPO_MAX_STEPS`, `PPO_SAVE_EVERY`, `PPO_FORCE_CPU`, `PPO_SEED`, `PPO_LOAD_MODEL`, `PPO_RUN_ID`).

### 3. Environment & Sensor Pipeline

#### Initialization Order
```
_init_supervisor()              # Creates Webots Supervisor, sets FAST mode
  └─ WebotsEnv(config, reward_computer)  # Builds entire simulation stack
       ├─ AltinoDriver(config)           # Initializes all Webots devices
       │    ├─ Motors: left_steer, right_steer, 4 wheels
       │    ├─ LiDAR: enabled, range image, FOV
       │    ├─ GPS: 3D position
       │    ├─ Accelerometer: 3-axis
       │    ├─ Gyro: 3-axis yaw rate
       │    └─ MappingProcessor(config)     # IMU filtering + IEKF + occupancy map
       ├─ PPORewardComputer (or SACRewardComputer)
       └─ _sync_endpoint_from_world()   # Reads GOAL_MARKER position from world file
```

#### Sensor Read Pipeline (`AltinoDriver.read_sensors()`)
```
lidar.getRangeImage() ──────┐
gps.getValues() ────────────┤
accelerometer.getValues() ──┤──> MappingProcessor.process()
gyro.getValues() ───────────┘    │
                                  ├─ sector_lidar()    → 16 normalized sector minima
                                  ├─ imu_proc.step()   → EKF quaternion + bias
                                  ├─ iekf.propagate()  → dead-reckoned position/heading
                                  └─ mapping_map.update() → occupancy grid via Bresenham
```

#### Observation Vector (31 features by default, `_build_observation()`)

| Segment | Size | Source |
|---|---|---|
| LiDAR sectors | 16 | `MappingProcessor.sector_lidar()` — min range per angular bin, normalized [0,1] |
| Direction features | 5 | `sin(heading), cos(heading), sin(goal_error), cos(goal_error), goal_distance / reference_distance` |
| IMU features | 10 | 3 accel_body, 3 gyro_body, 4 quaternion — all normalized |
| Occupancy grid | variable | Downsampled mapping occupancy probabilities (disabled by default, `ENV_OCCUPANCY_GRID_SHAPE=None`) |

Inference uses the observation layout from the training checkpoint; the reported obs_size equals `lidar_sector_dim + pose_goal_dim + imu_feature_dim + occupancy_grid_size`.

#### Action Space (`WebotsEnv.step()`)
- **Raw**: `[steering (-0.9, 0.9), speed (0.0, 6.0)]`
- **Adaptive speed cap**: `max_speed * obstacle_factor * steering_factor` where `obstacle_factor` is derived from min lidar norm and `steering_factor` penalizes sharp turns at high speed
- **Output**: `robot.set_steering(angle)`, `robot.set_speed(speed)` — steering via position control, wheels via velocity control

#### Episode Termination
- **Goal success**: robot distance < `goal_threshold` (0.3m) — sets `terminated=True`
- **Collision**: raw lidar minimum < `collision_threshold` (0.1m) — sets `terminated=True`
- **Low score**: cumulative reward <= `low_score_threshold` (-500.0) — terminates early
- **Timeout**: `current_step >= max_steps` — sets `truncated=True`, applies timeout penalty
- **Overshoot**: robot leaves goal region without stopping — applies overshoot penalty (-12.0)

#### Episode Reset (`WebotsEnv.reset()`)
Each episode begins with:
1. Reset mapping map
2. Stop robot
3. Randomize goal Y position (if `randomize_goal=True`, within `goal_y_range` ±1.5m)
4. Move obstacles into travel corridor with 75% probability (`AltinoDriver.randomize_obstacles()`)
5. Reset robot position with noise (`start_position_noise=0.08m`, `start_yaw_noise=0.8rad`)
6. Settle for `reset_settle_steps=10` timesteps
7. Reset mapping state

Goal endpoint is synced from the `GOAL_MARKER` node's position in the world file during `WebotsEnv.__init__()`, ensuring the agent always knows the actual goal location.

### 4. Mapping Modules (`controllers/state_estimation/`)

| Module | File | Purpose |
|---|---|---|
| `IMUProcessor` | `imu_filter.py` | EKF with 7 states (quaternion w,x,y,z + gyro bias bx,by,bz). Predict from gyro, correct from accelerometer. Outputs `IMUState` with quaternion, body-frame accel/gyro, world-frame accel. |
| `IEKFBackend` | `iekf_backend.py` | 8-state dead-reckoning IEKF [px, py, theta, vx, vy, b_omega_z, b_ax, b_ay]. Propagates from wheel speed + gyro z-rate. Used for heading estimate when mapping is enabled. |
| `MappingMap` | `mapping.py` | Log-odds occupancy grid with Bresenham ray-casting + keyframe trajectory tracker. `KEYFRAME_DIST=0.3m`, `KEYFRAME_ANGLE=0.15rad`. Saves PNG plots via matplotlib. |
| `OccupancyMap` | `mapping.py` (same) | Core grid: `FREE_LOG_ODDS=-0.5`, `OCC_LOG_ODDS=1.5`, clip range [-5,5]. Resolution 0.05m, 40x40m grid. |
| `NavigatingController` | `navigation_controller.py` | Standalone deterministic controller using potential fields: goal attraction + LiDAR repulsion + map repulsion. Not used by RL training. |

### 5. Reward Computation

#### PPO (`PPORewardComputer`)

`PPORewardComputer.compute(collision, pos, step, prev_distance, goal_error, min_lidar_norm, speed_norm, reached_new_best, accel)` returns `(reward, new_distance)`.

Components:
- **Collision penalty**: -50
- **Progress reward**: forward progress * 2.0 * proximity_factor
- **Distance penalty**: -distance_ratio * 0.05
- **Heading reward**: cos(goal_error) * 0.5
- **Safety penalty**: -(1 - min_lidar) * 0.2
- **Motion reward**: speed_norm * 0.05
- **Slow speed penalty**: -0.02 when speed_norm < 0.25
- **High speed bonus**: +0.05 when speed_norm > 0.6
- **New best distance bonus**: +0.05 when closest approach improves
- **Step penalty**: -0.015
- **Goal success**: +200 (when distance < goal_threshold) + 50 hold reward
- **Goal overshoot penalty**: -12.0 (robot leaves goal region without stopping)
- **Goal speed penalty**: -10.0 (configured, not applied per-step)

Rewards are clipped to `[-100, 100]` in PPO.episode collection before storage. Before GAE computation, rewards are scaled by `REW_SCALE=0.01`.

#### SAC (`SACRewardComputer`)

Same component structure as PPO with separate defaults in `controllers/SAC/SAC_defaults.py`. Can be configured independently.

### 6. Neural Network Architecture (`controllers/RNN/` + `controllers/PPO/PPO.py`)

Recurrent networks derive from `RecurrentActorCriticBase` (`base.py`). `GRUActorCritic` (`gru.py`) and `LSTMActorCritic` (`lstm.py`) are shared across algorithms. `FeedForwardActorCritic` is defined in `controllers/PPO/PPO.py` for PPO only.

```
Observation (31)
  ├─ ObstacleEncoder (16→64)  ──┐
  ├─ PoseGoalEncoder (5→64)    ──┤→ concat → FusionMLP → Latent (128)
  ├─ IMUEncoder (10→64)        ──┘
  └─ GridEncoder (optional)    ──┘
                                      ↓
                              Recurrent Core
                        ┌──────────────┬──────────────┐
                   ┌────┴────┐   ┌────┴────┐   ┌─────┴─────┐
                   │   GRU   │   │  LSTM   │   │ FeedForward│
                   │ 128, 1  │   │ 128, 1  │   │  (PPO only)│
                   └─────────┘   └─────────┘   └───────────┘
                         │              │
                    ┌────┴────┐   ┌─────┴─────┐
                    │Policy   │   │ Value     │
                    │Head → 2 │   │Head → 1   │
                    └─────────┘   └───────────┘
```

Key architecture details:
- **Encoder branches**: each is `Linear(hidden=128, ReLU, Linear(latent//2 or 128), ReLU)`
- **Fusion MLP**: `Linear(3*64 + grid, 128, ReLU) + Linear(128, 128, ReLU)`
- **GRU**: `nn.GRU(128, 128, layers=1, batch_first=True)` — hidden state shape `(1, B, 128)`
- **LSTM**: `nn.LSTM(128, 128, layers=1, batch_first=True)` — hidden + cell state shape `(1, B, 128)` each
- **FeedForward**: same encoder branches + fusion, then direct policy/value heads (no recurrent core). `get_initial_state()` returns `None`.
- **Recurrent forward**: per-timestep loop with done-mask state reset
- **Recurrent options**: `gru` (default), `lstm`, or `none` (feedforward)
- **Grid encoder**: CNN (Conv2d) when `occupancy_grid_shape` is 2D/3D, MLP when grid features are flat, absent when grid is disabled
- **SAC architecture**: same recurrent encoder + GRU/LSTM core, with separate actor and twin Q-networks sharing the encoder. Uses target networks with soft updates.

#### Action Distribution
`Normal(mean, std)` with `actor_log_std` as a learnable parameter (initialized -0.5). Tanh squashing + rescale to action range. Log-probability includes tanh correction and scale correction. Analytical entropy: `0.5 * D * (1 + log(2π)) + sum(log_std)`.

#### Architecture-Aware Entropy Schedule
Entropy coefficient starts at `base * arch_scale` where `arch_scale` is `1.35` for `lstm` and `1.0` for `gru`/`none`. Decays linearly to 70% of initial value over training (30% reduction).

### 7. PPO Training Loop (`controllers/PPO/PPO.py:train()`)

```
for episode in range(episodes):
    ┌───────────────────────────────────────────────┐
    │ 1. Reset env → initial observation            │
    │ 2. Agent.state = zeros (GRU/LSTM)             │
    │ 3. Collect episode:                            │
    │    while not done:                             │
    │      a = agent.select_action(obs, state, done) │
    │      obs, r, done, info = env.step(a)          │
    │      clip reward to [-100, 100]                │
    │      store obs, a, log_prob, reward            │
    │                                              │
    │ 4. Evaluate episode: values via model forward  │
    │ 5. Bootstrap value if truncated (max_steps)    │
    │ 6. Scale rewards by REW_SCALE (0.01)           │
    │ 7. GAE(lambda): advantages + returns           │
    │ 8. Append trajectory to rollout buffer         │
    │                                              │
    │ 9. LR warmup (linear ramp first 25 episodes)   │
    │10. Every update_every episodes:                │
    │    a. Sanitize trajectories (NaN/Inf guard)    │
    │    b. Normalize advantages across episodes     │
    │    c. Split into sequences (seq_len, stride)   │
    │    d. Shuffle, mini-batch (batch_size)         │
    │    e. For each epoch:                          │
    │       - evaluate_sequences → log_prob, value   │
    │       - clipped surrogate objective (ε=0.1)    │
    │       - smooth L1 value loss                   │
    │       - analytical entropy bonus               │
    │       - learn_mask excludes burn-in steps      │
    │       - gradient clipping per component:       │
    │         actor=0.5, critic=5.0, rnn=1.0, enc=0.5│
    │       - early stop if approx_kl > 0.05         │
    │    f. Clear rollout buffer                     │
    │                                              │
    │11. Anneal entropy_coef (arch-aware, 30% decay) │
    │12. Log episode/update metrics                  │
    │13. Save best_goal checkpoint if new goal best  │
    │14. Save best checkpoint if no goal episode yet │
    │15. Save checkpoint every save_every episodes   │
    └───────────────────────────────────────────────┘
```

**Advantage (GAE)** formulation:
```
delta_t = r_t + γ*V(s_{t+1}) - V(s_t)
A_t = δ_t + (γλ)*δ_{t+1} + (γλ)²*δ_{t+2} + ...
```
Bootstrap value for truncated episodes: `V(s_T)` from model forward on final state.

**PPO loss** for a single timestep:
```
L = -min(ratio*A, clip(ratio, 1-ε, 1+ε)*A) + 0.5*SmoothL1(V, R) - β*H[π]
  where ratio = exp(log π_new - log π_old)
```
All terms masked by `learn_mask` (excludes burn-in and padding), averaged over `valid_count`.

**Gradient clipping per component**:
| Component | Max Norm | Parameters |
|---|---|---|
| Actor | 0.5 | policy_head weights/bias + actor_log_std |
| Critic | 5.0 | value_head weights/bias |
| RNN | 1.0 | gru/lstm parameters (if recurrent) |
| Encoder | 0.5 | all remaining parameters |

### 8. Burn-In Masking

`sequence_loss_mask(valid_mask, burn_in)` in `controllers/common/training_utils.py`:
```
start_index = min(burn_in, valid_length - 1)
learn_mask = valid_mask * (timestep >= start_index)
```
The first `burn_in` (default 8) timesteps of each sequence are excluded from gradient computation, allowing the recurrent state to warm up on real data before contributing to the loss.

### 9. Checkpoint System (`controllers/common/checkpoints.py`)

| Function | Purpose |
|---|---|
| `run_checkpoint_dir()` | Create folder: `checkpoints/<run_id>/` |
| `run_checkpoint_path()` | Build path: `checkpoints/<run_id>/<prefix>_<run_id>.pth` |
| `load_checkpoint()` | `torch.load` with `weights_only=False` fallback |
| `save_checkpoint_file()` | Save under prefix, return path |
| `make_checkpoint_header()` | Construct `{episode, reward, goal_episode, algorithm, config}` dict |

**PPO checkpoint contents**: `{algorithm, episode, reward, goal_episode, config, obs_size, action_dim, recurrent_cell, model, actor_log_std}`

**SAC checkpoint contents**: `{algorithm, episode, reward, goal_episode, config, obs_size, action_dim, architecture, actor_enc, actor_mean, actor_log_std_head, q1_enc, q1_head, q2_enc, q2_head, target_q1_enc, target_q1_head, target_q2_enc, target_q2_head, log_alpha}`

**Save triggers** (PPO):
- `best_goal`: every episode ending in "goal" with new best reward → `best_<run_id>.pth`
- `best`: first non-goal best reward (only if no goal episode yet) → `best_<run_id>.pth`
- `checkpoint`: every `save_every` episodes → `checkpoint_<run_id>.pth`
- `final`: end of training → `final_<run_id>.pth`

### 10. Inference Pipeline (`run_model.py`)

```
run_inference(config):
  _init_supervisor()
  checkpoint = torch.load(model_path)
  Build: env = WebotsEnv(train_config, reward_computer)
         agent = PPOAgent or SACAgent (from checkpoint)
  agent.load_model(model_path)  # PPO: load_model(), SAC: load()
  agent.model.eval()

  for episode in range(episodes):
    obs = env.reset()
    while not done:
      a = agent.select_action(obs, state, deterministic=True)
      obs, r, done, info = env.step(a)
      accumulate reward, steps
    Log: reward, steps, min_distance, end_reason
  Summary: avg_reward, std_reward, success_rate
```

- `--algorithm ppo` (default) or `sac`
- `--model-path`: optional override of checkpoint path
- `--quiet`: suppresses per-episode output
- Default model path: most recently modified `.pth` in `controllers/<algo>/checkpoints/`

### 11. Metrics Logging (`controllers/common/metrics_logger.py`)

Three CSV files written per run (e.g. `ppo_episodes.csv`, etc.):

| File | Contents |
|---|---|
| `{algo}_hyperparams.csv` | All config fields + recurrent_cell, obs_size, action_dim |
| `{algo}_episodes.csv` | Per-episode: reward, avg10, length, success, goal_touched, collision, timeout, min_dist, avg_speed, end_reason, elapsed_s, act0/act1 stats, obs stats, aggregated update metrics, replay_buffer_size, recurrent_cell |
| `{algo}_updates.csv` | Per-update: actor_loss, critic_loss, policy_entropy, entropy_coef, value_residual, grad_norm_actor, grad_norm_critic, grad_norm_rnn, lr_actor, lr_critic, alpha, alpha_loss, target_update_magnitude, recurrent_cell |

The logger auto-detects which fields are relevant per algorithm (e.g., `alpha` and `target_update_magnitude` for SAC).

Static helpers: `compute_action_stats()`, `compute_obs_stats()`, `compute_grad_norm()`, `compute_value_residual()`, `aggregate_update_metrics()`.

### 12. Curriculum Training (`controllers/run.py`)

Training proceeds through 10 progressively harder worlds, each building on the checkpoint from the previous world:

```
train_1_empty      (0 obstacles,  goal fixed at y=0.0)     ──→ final.pth
train_2_one_obs    (1 obstacle,   goal fixed at y=0.0)     ──→ final.pth
train_3_three_obs  (3 obstacles,  goal fixed at y=0.0)     ──→ final.pth
train_4_five_obs   (5 obstacles,  goal fixed at y=0.0)     ──→ final.pth
train_5_goal_shift_pos  (7  obstacles, goal at y=+0.20)    ──→ final.pth
train_6_goal_shift_neg  (9  obstacles, goal at y=-0.30)    ──→ final.pth
train_7_goal_offset_pos (11 obstacles, goal at y=+0.50)    ──→ final.pth
train_8_goal_offset_neg (13 obstacles, goal at y=-0.60)    ──→ final.pth
train_9_dense      (15 obstacles, goal fixed at y=0.0)     ──→ final.pth
train_10_full      (18 obstacles, goal fixed at y=0.0)     ──→ final.pth
```

The `worker` command handles the serial transfer automatically. The `submit` command generates one SLURM job per (architecture × seed), each running the full curriculum pipeline.

### 13. Training Worlds (`worlds/training/`)

Generated by `worlds/generate_training_worlds.py` from `worlds/testing/ObstacleCourse.wbt` template:

| World | Obstacles | Goal Y | Barrier gap centered on goal |
|---|---|---|---|
| `train_1_empty` | 0 | 0.0 | y=0.0 |
| `train_2_one_obs` | 1 | 0.0 | y=0.0 |
| `train_3_three_obs` | 3 | 0.0 | y=0.0 |
| `train_4_five_obs` | 5 | 0.0 | y=0.0 |
| `train_5_goal_shift_pos` | 7 | +0.20 | y=+0.20 |
| `train_6_goal_shift_neg` | 9 | -0.30 | y=-0.30 |
| `train_7_goal_offset_pos` | 11 | +0.50 | y=+0.50 |
| `train_8_goal_offset_neg` | 13 | -0.60 | y=-0.60 |
| `train_9_dense` | 15 | 0.0 | y=0.0 |
| `train_10_full` | 18 | 0.0 | y=0.0 |

All obstacles are drawn from a pre-verified pool of 18 positions with minimum centre-to-centre spacing of 0.70m. Barrier walls at x=1.5 create a gap ±1.55m around the goal Y. Obstacles include cylinders (r=0.15, h=0.3) and boxes (0.3×0.3×0.3). The `GOAL_MARKER` node's translation matches the world's goal Y.

### 14. Validation Worlds (`worlds/validation/`)

Generated by `worlds/generate_validation_worlds.py`:

| World | Obstacles | Goal Y | Description |
|---|---|---|---|
| `val_1_empty_center` | 0 | 0.0 | Empty arena, centred goal |
| `val_2_empty_offset` | 0 | 0.5 | Empty arena, shifted goal |
| `val_3_sparse_a` | 5 | 0.0 | Open corridor layout |
| `val_4_sparse_b` | 5 | 0.0 | Partial corridor block layout |
| `val_5_dense` | 10 | 0.0 | Dense obstacle layout |

### 15. HPC / SLURM (`slurm.sh`)

The SLURM script handles headless Webots execution on GPU nodes:
- Loads Python 3.10.8, GCC, CUDA 12.1 modules
- Starts Xvfb virtual display (Qt xcb plugin requires X11)
- Sets `EGL_PLATFORM=x11`, `WEBOTS_TMPDIR`, Webots paths
- Activates project `.venv`, exports `PYTHONPATH` for torch + Webots controller
- Either runs `PPO_RUN_COMMAND` (from `controllers/run.py submit`) or launches Webots directly

Usage:
```bash
# Generate and print sbatch commands (no submission)
python controllers/run.py submit --no-submit --episodes 2500
# Submit 10 seeds × 3 architectures = 30 jobs
python controllers/run.py submit --sessions 10 --episodes 2500 --account ec12 --time 00:30:00
```

### 16. SAC Algorithm (`controllers/SAC/SAC.py`)

Soft Actor-Critic with GRU/LSTM recurrent encoder for off-policy training:
- **Policy**: tanh-squashed Gaussian with learnable `actor_log_std`
- **Q-networks**: twin Q-networks sharing recurrent encoder, with target networks (soft update via `tau=0.005`)
- **Entropy tuning**: auto-α via dual gradient descent (optional), target entropy = `-action_dim * target_entropy_scale`
- **Replay buffer**: sequence-aware replay with `replay_capacity`, `min_replay_sequences`, `replay_batch_size`
- **Training**: gradient steps per episode after warmup (`update_after_steps`), using clipped double-Q, delayed policy updates
- **Sequence handling**: uses same `sequence_loss_mask` and recurrent state management as PPO

### 17. Tests (`tests/test_algorithms.py`)

Self-contained tests that mock the Webots `controller` C-extension, exercising:
- PPO and SAC Config initialization and validation
- PPO `FeedForwardActorCritic` forward pass
- `GRUActorCritic` and `LSTMActorCritic` forward pass + recurrent state
- PPO `select_action()` with deterministic and stochastic modes
- PPO `update()` pipeline (sanitize → normalize → split → batch)
- SAC network construction and forward pass
- GAE computation with and without bootstrap
- `sequence_loss_mask` correctness
- Reward computer component shapes

Run with:
```bash
python -m pytest tests/ -v
```

### 18. Data Flow Summary

```
Controllers/run.py (curriculum orchestrator)
  └── Webots (headless) → World File (.wbt)
       └── ALTINO Robot Controller (PPO.py or SAC.py)
            │
            ├── Config (dataclass + env overrides)
            │
            ├── WebotsEnv
            │    ├── AltinoDriver
            │    │    ├── Sensors → raw lidar, GPS, IMU
            │    │    ├── MappingProcessor
            │    │    │    ├── IMUProcessor (EKF quaternion)
            │    │    │    ├── IEKFBackend (dead-reckoning)
            │    │    │    └── MappingMap (occupancy grid)
            │    │    └── randomize_goal / randomize_obstacles
            │    ├── _sync_endpoint_from_world() → reads GOAL_MARKER
            │    ├── _build_observation() → 31-feature vector
            │    ├── step() → adaptive speed, reward, termination
            │    └── RewardComputer (PPO or SAC)
            │
            ├── PPOAgent (or SACAgent)
            │    ├── RNN Actor-Critic (GRU/LSTM/FeedForward)
            │    │    ├── Encoder branches (obstacle, pose, IMU, grid)
            │    │    ├── Recurrent core (gru.py / lstm.py)
            │    │    └── Policy + Value heads
            │    │    (SAC: + Twin Q-networks + Target networks)
            │    ├── select_action() → [steering, speed]
            │    └── update() → gradient steps
            │
            ├── Rollout Buffer (PPO) / Replay Buffer (SAC)
            │
            ├── Checkpoint Manager (controllers/common/checkpoints.py)
            │
            └── Metrics Logger (3 CSV files per run)
```

## Training Behavior

PPO reports the following episode metrics:
- `r`, `avg10`, `steps`, `succ10`, `touch10`, `col10`, `to10`, `min_d`, `end`, `t`

Where:
- `succ10`: rolling 10-episode success rate (goal reached and stopped)
- `touch10`: rolling 10-episode goal-touch rate
- `col10`: rolling 10-episode collision rate
- `to10`: rolling 10-episode timeout (`max_steps`) rate

SAC reports similar metrics with replay buffer statistics.

## PPO Summary
- Recurrent options: `gru`, `lstm`, or `none` (feedforward)
- Update style: on-policy rollout trajectories
- Update schedule: every `update_every` episodes
- Loss style: clipped PPO objective + value loss + analytical entropy regularization
- GAE lambda: 0.98, value bootstrap for truncated episodes
- Reward clipping: [-100, 100] per step
- Reward scaling: REW_SCALE=0.01 applied before GAE
- Advantage normalization: per-update batch, clipped to [-5, 5]
- LR warmup: linear ramp first 25 episodes (0.25× → 1.0×)
- Entropy schedule: architecture-aware, 30% decay over training
- Gradient clipping: per-component (actor 0.5, critic 5.0, rnn 1.0, encoder 0.5)
- Early stopping: per-epoch, breaks if approx_kl > 0.05

## SAC Summary
- Recurrent options: `gru` or `lstm`
- Update style: off-policy with replay buffer
- Update schedule: `gradient_steps_per_episode` after `update_after_steps` warmup
- Loss style: clipped double-Q + policy gradient + auto-α entropy tuning
- Target networks: soft update with `tau=0.005`
- Sequence replay: preserves temporal structure via sequence-aware sampling

## Checkpoints
- PPO: `controllers/PPO/checkpoints/<run_id>/best_<run_id>.pth`, `final_<run_id>.pth`
- SAC: `controllers/SAC/checkpoints/<run_id>/best_<run_id>.pth`, `final_<run_id>.pth`
- Run ID format: `<timestamp>_<arch>_seed<NN>_stage<NN>_<world_name>` (curriculum)
- Checkpoints are saved under algorithm-specific checkpoint directories

## Inference
Run deterministic inference with:
```bash
python run_model.py --algorithm ppo --episodes 10
python run_model.py --algorithm sac --episodes 10 --model-path path/to/checkpoint.pth
```

Optional args:
- `--algorithm ppo` or `sac` (default: `sac`)
- `--model-path <path>` to force a specific checkpoint
- `--quiet` to reduce per-episode output

If `--model-path` is omitted, the most recently modified `.pth` in the algorithm's checkpoint directory is used.
