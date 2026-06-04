# DRL Obstacle Avoidance (Webots ALTINO)

Recurrent PPO controller for ALTINO obstacle avoidance in Webots, with shared environment, sensor, reward, and state estimation processing. Training is orchestrated by `run_model.py` through a curriculum of progressively harder worlds, including static and moving-obstacle stages.

## Environment Setup

1. Create and activate a Python 3.10.19 environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. In Webots, set Preferences → Python command to your project environment Python:
   ```text
   C:\Users\ErikH\miniconda3\envs\FYS5429\python.exe
   ```

## Repository Layout

```
run_model.py                          # CLI: inference, curriculum worker, moving-curriculum, SLURM submit
controllers/
  PPO/
    PPO.py                            # Training controller: train(), evaluate()
    PPO_agent.py                      # PPOAgent, sequence utilities, checkpoint save/load
    PPO_config.py                     # Config dataclass + env-var overrides
    PPO_defaults.py                   # Shared constants: environment, reward, training defaults
    PPO_feedforward.py                # FeedForwardActorCritic (no recurrent core)
    PPO_rewards.py                    # PPORewardComputer
  RNN/
    base.py                           # RecurrentActorCriticBase (shared encoder + recurrent interface)
    gru.py                            # GRUActorCritic
    lstm.py                           # LSTMActorCritic
    __init__.py                       # Exports: GRUActorCritic, LSTMActorCritic, RecurrentState
  state_estimation/
    imu_filter.py                     # IMUEKF (7-state) + IMUProcessor
    iekf_backend.py                   # IEKFBackend (8-state dead-reckoning)
    mapping.py                        # OccupancyMap (log-odds grid) + MappingMap (keyframe tracker)
  Webots/
    webots_env.py                     # WebotsEnv, AltinoDriver, MappingProcessor
    __init__.py
  common/
    checkpoints.py                    # Checkpoint path/load/save utilities
    metrics_logger.py                 # MetricsLogger: 3 CSV files per run
    seed.py                           # set_all_seeds()
worlds/
  training/                           # 13 curriculum .wbt files (static + moving)
  validation/                         # 9 validation .wbt files
plots_paper/
  make_plots.py                       # Paper figure generation from training CSVs
```

## Entry Points

| Command | Description |
|---|---|
| `python run_model.py infer` | Run deterministic inference with a trained checkpoint |
| `python run_model.py worker` | Run one architecture through curriculum worlds serially |
| `python run_model.py moving-curriculum` | Run moving-obstacle curriculum stages |
| `python run_model.py submit` | Generate/submit SLURM jobs for HPC training |

PPO is started by Webots as a robot controller. `run_model.py worker` orchestrates sequential world runs, passing final checkpoints between worlds for curriculum transfer.

### `run_model.py` Usage

```bash
# Inference with latest checkpoint
python run_model.py infer --episodes 10
# Inference with specific checkpoint
python run_model.py infer --model-path controllers/PPO/checkpoints/run_id/final_run_id.pth

# Single-architecture worker through all curriculum worlds
python run_model.py worker --arch gru --seed 0
# Resume from a previous checkpoint
python run_model.py worker --arch gru --seed 0 --resume-from path/to/checkpoint.pth
# Override training duration
python run_model.py worker --arch gru --seed 0 --episodes 2500 --max-steps 6000
# Specific worlds only
python run_model.py worker --arch none --worlds worlds/training/train_1_empty.wbt worlds/training/train_3_three_obs.wbt

# Moving-obstacle curriculum (5 stages, requires a static-curriculum checkpoint)
python run_model.py moving-curriculum --arch gru --seed 0
python run_model.py moving-curriculum --arch gru --seed 0 --resume-from path/to/checkpoint.pth

# SLURM job submission
python run_model.py submit --sessions 10 --episodes 2500
python run_model.py submit --no-submit --episodes 2500   # print commands only
```

Architecture aliases: `none`, `mlp`, `feedforward`, `ff` all map to feed-forward (no recurrent core).

## Configuration

| File | Provides |
|---|---|
| `controllers/PPO/PPO_defaults.py` | Environment constants, reward coefficients, moving-obstacle defaults, `PPODefaults`, `RecurrentDefaults` |
| `controllers/PPO/PPO_config.py` | `Config` dataclass, `_apply_env_overrides()` |

**PPO defaults**: `episodes=500`, `update_every=4`, `epochs=4`, `batch_size=128`, `lr=1e-4`, `entropy_coef=0.005`, `gamma=0.99`, `gae_lambda=0.98`, `epsilon=0.1`, `max_steps=2500`, `save_every=100`, `REW_SCALE=0.01`.

**Recurrent defaults**: `sequence_length=32`, `burn_in=8`, `sequence_stride=16`.

**Environment variables** override Config fields per-run:

| Variable | Effect |
|---|---|
| `PPO_ARCH` / `PPO_RECURRENT_CELL` | Architecture: `none`, `gru`, `lstm` |
| `PPO_EPISODES` | Episode count |
| `PPO_MAX_STEPS` | Max steps per episode |
| `PPO_SAVE_EVERY` | Checkpoint interval |
| `PPO_FORCE_CPU` | Force CPU (`1`/`true`/`yes`) |
| `PPO_SEED` | Random seed |
| `PPO_LOAD_MODEL` | Resume from checkpoint |
| `PPO_RUN_ID` | Override run identifier |
| `PPO_MOVING_OBSTACLE_INDICES` | Comma-separated indices or `all` |
| `PPO_MOVING_OBSTACLE_SPEED` | Oscillation speed (default 0.3) |
| `PPO_MOVING_OBSTACLE_AMPLITUDE` | Oscillation amplitude (default 0.4) |
| `PPO_MOVING_GOAL` | Enable moving goal (`1`/`true`) |
| `PPO_MOVING_GOAL_SPEED` | Goal oscillation speed (default 0.2) |
| `PPO_MOVING_GOAL_AMPLITUDE` | Goal oscillation amplitude (default 0.5) |

## Environment & Sensor Pipeline

### Initialization Order

```
_init_supervisor()                    # Creates Webots Supervisor, sets FAST mode
  └─ WebotsEnv(config, reward_computer)
       ├─ AltinoDriver(config)        # Initializes all Webots devices
       │    ├─ Motors: left_steer, right_steer, 4 wheels
       │    ├─ LiDAR: enabled, range image, FOV
       │    ├─ GPS: 3D position
       │    ├─ Accelerometer: 3-axis
       │    ├─ Gyro: 3-axis yaw rate
       │    └─ MappingProcessor       # IMU filtering + IEKF + optional occupancy map
       ├─ PPORewardComputer
       └─ _sync_endpoint_from_world() # Reads GOAL_MARKER position from world file
```

### Sensor Read Pipeline (`AltinoDriver.read_sensors()`)

```
lidar.getRangeImage() ──────┐
gps.getValues() ────────────┤
accelerometer.getValues() ──┤──> MappingProcessor.process()
gyro.getValues() ───────────┘    │
                                 ├─ sector_lidar()    → 16 normalized sector minima
                                 ├─ imu_proc.step()   → EKF quaternion + bias
                                 ├─ iekf.propagate()  → dead-reckoned position/heading
                                 └─ mapping_map       → occupancy grid via Bresenham (optional)
```

### Observation Vector (31 features, `_build_observation()`)

| Segment | Size | Source |
|---|---|---|
| LiDAR sectors | 16 | `MappingProcessor.sector_lidar()` — min range per angular bin, normalized [0,1] |
| Direction features | 5 | `sin(heading), cos(heading), sin(goal_error), cos(goal_error), goal_distance / reference_distance` |
| IMU features | 10 | 3 accel_body, 3 gyro_body, 4 quaternion — all normalized |

### Action Space (`WebotsEnv.step()`)

- **Raw**: `[steering (-0.9, 0.9), speed (0.0, 6.0)]`
- **Adaptive speed cap**: `max_speed * obstacle_factor * steering_factor` where `obstacle_factor` is derived from min lidar norm and `steering_factor` penalizes sharp turns at high speed
- **Output**: `robot.set_steering(angle)`, `robot.set_speed(speed)`

### Episode Termination

| Condition | Trigger | Effect |
|---|---|---|
| Goal success | distance < `goal_threshold` (0.3m) | `terminated=True`, reward +250 |
| Collision | raw lidar min < `collision_threshold` (0.1m) | `terminated=True`, reward -50 |
| Low score | cumulative reward ≤ -500.0 | `terminated=True` |
| Timeout | `current_step >= max_steps` | `truncated=True`, penalty `-10 - 20 * distance_ratio` |
| Overshoot | robot leaves goal region without stopping | penalty -12.0 |

### Episode Reset (`WebotsEnv.reset()`)

1. Reset mapping state
2. Stop robot
3. Randomize goal Y position (if `randomize_goal=True`, within ±1.5m, barrier walls follow)
4. Move obstacles into travel corridor with 75% probability (`AltinoDriver.randomize_obstacles()`)
5. Reset robot position with noise (`start_position_noise=0.08m`, `start_yaw_noise=0.8rad`)
6. Settle for `reset_settle_steps=10` timesteps
7. Reset state estimation

## State Estimation (`controllers/state_estimation/`)

| Module | File | Purpose |
|---|---|---|
| `IMUProcessor` / `IMUEKF` | `imu_filter.py` | 7-state EKF (quaternion w,x,y,z + gyro bias bx,by,bz). Predict from gyro, correct from accelerometer. Outputs `IMUState` with quaternion, body-frame accel/gyro, world-frame accel. |
| `IEKFBackend` | `iekf_backend.py` | 8-state dead-reckoning IEKF [px, py, theta, vx, vy, b_omega_z, b_ax, b_ay]. Propagates from wheel speed + gyro z-rate. |
| `OccupancyMap` | `mapping.py` | Log-odds occupancy grid with Bresenham ray-casting. `FREE_LOG_ODDS=-0.5`, `OCC_LOG_ODDS=1.5`, clip [-5,5]. Resolution 0.05m, 40×40m grid. |
| `MappingMap` | `mapping.py` | Keyframe trajectory tracker on top of `OccupancyMap`. `KEYFRAME_DIST=0.3m`, `KEYFRAME_ANGLE=0.15rad`. Saves PNG plots. |

Mapping is disabled by default (`enable_mapping=False`). When enabled, the `MappingProcessor` in `webots_env.py` runs the full IMU→IEKF→occupancy pipeline and can save per-episode map visualizations.

## Reward Computation (`PPORewardComputer`)

`PPORewardComputer.compute(collision, current_pos, prev_distance, goal_error, min_lidar_norm, speed_norm, reached_new_best_distance)` returns `(reward, new_distance)`.

| Component | Value | Condition |
|---|---|---|
| Collision penalty | -50.0 | Always on collision |
| Progress reward | `delta * 2.0 * proximity_factor` | Forward progress toward goal |
| Distance penalty | `-distance_ratio * 0.05` | Proportional to normalized distance |
| Heading reward | `cos(goal_error) * 0.5` | Alignment with goal direction |
| Safety penalty | `-(1 - min_lidar) * 0.2` | Proximity to obstacles |
| Motion reward | `speed_norm * 0.05` | Encourages movement |
| Slow speed penalty | -0.02 | When `speed_norm < 0.25` |
| High speed bonus | +0.05 | When `speed_norm > 0.6` |
| New best distance bonus | +0.05 | When closest approach improves |
| Step penalty | -0.015 | Every step |
| Goal success | +250.0 | `distance < goal_threshold` (200 success + 50 hold) |

Rewards are clipped to `[-100, 100]` per step before storage. Before GAE computation, rewards are scaled by `REW_SCALE=0.01`.

## Neural Network Architecture

Recurrent networks derive from `RecurrentActorCriticBase` (`controllers/RNN/base.py`). `GRUActorCritic` and `LSTMActorCritic` are in `controllers/RNN/`. `FeedForwardActorCritic` is in `controllers/PPO/PPO_feedforward.py`.

```
Observation (31)
  ├─ ObstacleEncoder (16→128→64)  ──┐
  ├─ PoseGoalEncoder (5→128→64)   ──┤→ concat (192) → FusionMLP → Latent (128)
  └─ IMUEncoder (10→128→64)       ──┘
                                      ↓
                              Recurrent Core
                        ┌──────────────┬──────────────┐
                   ┌────┴────┐   ┌────┴────┐   ┌─────┴─────┐
                   │   GRU   │   │  LSTM   │   │ FeedForward│
                   │ 128, 1  │   │ 128, 1  │   │  (no core) │
                   └─────────┘   └─────────┘   └───────────┘
                         │              │
                    ┌────┴────┐   ┌─────┴─────┐
                    │Policy   │   │ Value     │
                    │Head → 2 │   │Head → 1   │
                    └─────────┘   └───────────┘
```

- **Encoder branches**: `Linear(in, 128) → ReLU → Linear(128, 64) → ReLU` (each branch outputs 64-dim)
- **Fusion MLP**: `Linear(192, 128) → ReLU → Linear(128, 128) → ReLU` (192 = 3×64, no grid)
- **GRU**: `nn.GRU(128, 128, layers=1, batch_first=True)` — per-timestep loop with done-mask state reset
- **LSTM**: `nn.LSTM(128, 128, layers=1, batch_first=True)` — same pattern
- **FeedForward**: same encoders + fusion, direct policy/value heads. `get_initial_state()` returns `None`.

### Action Distribution

`Normal(mean, std)` with learnable `actor_log_std` (initialized -0.5). Tanh squashing + rescale to action range. Log-probability includes tanh correction and scale correction.

### Entropy Schedule

Architecture-aware: starts at `base * arch_scale` (`arch_scale`: 1.35 for LSTM, 1.0 for GRU/none). Decays linearly to 70% of initial value over training (30% reduction).

## PPO Training Loop

```
for episode in range(episodes):
  1. Reset env → initial observation
  2. Collect episode: select_action → env.step → clip reward [-100, 100]
  3. Evaluate episode values via model forward
  4. Bootstrap value if truncated (max_steps)
  5. Scale rewards by REW_SCALE (0.01)
  6. GAE(lambda): advantages + returns
  7. Append trajectory to rollout buffer
  8. LR warmup (linear ramp first 25 episodes)
  9. Every update_every episodes:
     a. Sanitize trajectories (NaN/Inf guard)
     b. Normalize advantages (clip to [-5, 5])
     c. Split into sequences (seq_len=32, stride=16)
     d. Shuffle, mini-batch (batch_size=128)
     e. For each epoch (4):
        - evaluate_sequences → log_prob, value, entropy
        - clipped surrogate objective (ε=0.1)
        - smooth L1 value loss
        - analytical entropy bonus
        - learn_mask excludes burn-in (8) steps
        - gradient clipping per component:
          actor=0.5, critic=5.0, rnn=1.0, encoder=0.5
        - early stop if approx_kl > 0.05
     f. Clear rollout buffer
  10. Anneal entropy_coef (arch-aware, 30% decay)
  11. Log metrics, save checkpoints
```

**PPO loss** per timestep:
```
L = -min(ratio*A, clip(ratio, 1-ε, 1+ε)*A) + 0.5*SmoothL1(V, R) - β*H[π]
```

## Burn-In Masking

`_sequence_loss_mask(valid_mask, burn_in)` in `controllers/PPO/PPO_agent.py`:
- First `burn_in` (default 8) timesteps of each sequence are excluded from gradient computation
- Allows recurrent state to warm up on real data before contributing to loss

## Checkpoint System (`controllers/common/checkpoints.py`)

**PPO checkpoint contents**: `{algorithm, episode, reward, goal_episode, config, obs_size, action_dim, recurrent_cell, model, actor_log_std}`

**Save triggers**:
- `best_goal`: episode ending in goal with new best reward → `best_<run_id>.pth`
- `best`: first non-goal best reward (only if no goal episode yet) → `best_<run_id>.pth`
- `checkpoint`: every `save_every` episodes → `checkpoint_<run_id>.pth`
- `final`: end of training → `final_<run_id>.pth`

Run ID format: `<timestamp>_<arch>_seed<NN>_stage<NN>_<world_name>`

Checkpoints are saved under `controllers/PPO/checkpoints/<run_id>/`.

## Curriculum Training

### Static Curriculum (10 worlds)

Training proceeds through 10 progressively harder worlds, each building on the checkpoint from the previous world:

| World | Obstacles | Goal Y |
|---|---|---|
| `train_1_empty` | 0 | 0.0 |
| `train_2_one_obs` | 1 | 0.0 |
| `train_3_three_obs` | 3 | 0.0 |
| `train_4_five_obs` | 5 | 0.0 |
| `train_5_goal_shift_pos` | 7 | +0.20 |
| `train_6_goal_shift_neg` | 9 | -0.30 |
| `train_7_goal_offset_pos` | 11 | +0.50 |
| `train_8_goal_offset_neg` | 13 | -0.60 |
| `train_9_dense` | 15 | 0.0 |
| `train_10_full` | 18 | 0.0 |

Obstacles include cylinders (r=0.15, h=0.3) and boxes (0.3×0.3×0.3). Barrier walls at x=1.5 create a gap ±1.55m around the goal Y. The `GOAL_MARKER` node's translation matches the world's goal Y.

### Moving-Obstacle Curriculum (5 stages)

After static curriculum, `moving-curriculum` runs 5 additional stages using worlds 11–13:

| Stage | World | Moving Obstacles | Moving Goal |
|---|---|---|---|
| partial_moving_1 | `train_11_partial_moving` | 1 (index 0) | No |
| partial_moving_3 | `train_11_partial_moving` | 3 (indices 0–2) | No |
| partial_moving_5 | `train_11_partial_moving` | 5 (indices 0–4) | No |
| all_moving | `train_12_all_moving` | All 18 | No |
| moving_goal | `train_13_moving_goal` | All 18 | Yes |

Moving obstacles oscillate sinusoidally in Y: `base_y + amplitude * sin(speed * t + phase)`. Moving goal uses the same pattern with separate speed/amplitude, and barrier walls follow.

### Validation Worlds (9 worlds)

| World | Obstacles | Goal Y | Description |
|---|---|---|---|
| `val_1_empty_center` | 0 | 0.0 | Empty arena, centred goal |
| `val_2_empty_offset` | 0 | 0.5 | Empty arena, shifted goal |
| `val_3_sparse_a` | 5 | 0.0 | Open corridor layout |
| `val_4_sparse_b` | 5 | 0.0 | Partial corridor block layout |
| `val_5_dense` | 10 | 0.0 | Dense obstacle layout |
| `val_6_one_moving` | 5 + 1 moving | 0.0 | Sparse with one moving obstacle |
| `val_7_all_moving` | 10 moving | 0.0 | All obstacles moving |
| `val_8_moving_goal` | 10 moving | moving | Moving obstacles + moving goal |
| `val_9_offset_goal` | 10 | offset | Dense with offset goal |

## Metrics Logging (`controllers/common/metrics_logger.py`)

Three CSV files written per run:

| File | Contents |
|---|---|
| `{algo}_hyperparams.csv` | All config fields + recurrent_cell, obs_size, action_dim |
| `{algo}_episodes.csv` | Per-episode: reward, avg10, length, success, goal_touched, collision, timeout, min_dist, avg_speed, end_reason, elapsed, action/obs stats, aggregated update metrics |
| `{algo}_updates.csv` | Per-update: actor_loss, critic_loss, policy_entropy, entropy_coef, approx_kl, value_residual, grad norms, lr |

Static helpers: `compute_action_stats()`, `compute_obs_stats()`, `compute_grad_norm()`, `aggregate_update_metrics()`.

## Data Flow Summary

```
run_model.py (curriculum orchestrator + inference)
  └── Webots (headless) → World File (.wbt)
       └── ALTINO Robot Controller (PPO.py)
            ├── Config (dataclass + env overrides)
            │
            ├── WebotsEnv
            │    ├── AltinoDriver
            │    │    ├── Sensors → raw lidar, GPS, IMU
            │    │    ├── MappingProcessor
            │    │    │    ├── IMUProcessor (EKF quaternion)
            │    │    │    ├── IEKFBackend (dead-reckoning)
            │    │    │    └── MappingMap (occupancy grid, optional)
            │    │    └── randomize_goal / randomize_obstacles / moving updates
            │    ├── _sync_endpoint_from_world() → reads GOAL_MARKER
            │    ├── _build_observation() → 31-feature vector
            │    ├── step() → adaptive speed, reward, termination
            │    └── PPORewardComputer
            │
            ├── PPOAgent
            │    ├── Actor-Critic (GRU / LSTM / FeedForward)
            │    │    ├── Encoder branches (obstacle, pose-goal, IMU)
            │    │    ├── Recurrent core (gru.py / lstm.py)
            │    │    └── Policy + Value heads
            │    ├── select_action() → [steering, speed]
            │    └── update() → gradient steps
            │
            ├── Rollout Buffer (on-policy trajectories)
            │
            ├── Checkpoint Manager (controllers/common/checkpoints.py)
            │
            └── Metrics Logger (3 CSV files per run)
```

## PPO Summary

| Property | Value |
|---|---|
| Recurrent options | `gru`, `lstm`, or `none` (feedforward) |
| Update style | On-policy rollout trajectories |
| Update schedule | Every `update_every` episodes |
| Loss | Clipped PPO + smooth L1 value + analytical entropy |
| GAE lambda | 0.98, value bootstrap for truncated episodes |
| Reward clipping | [-100, 100] per step |
| Reward scaling | REW_SCALE=0.01 before GAE |
| Advantage normalization | Per-update batch, clipped to [-5, 5] |
| LR warmup | Linear ramp first 25 episodes (0.25× → 1×) |
| Entropy schedule | Architecture-aware, 30% decay over training |
| Gradient clipping | Actor 0.5, critic 5.0, RNN 1.0, encoder 0.5 |
| Early stopping | Per-epoch, breaks if approx_kl > 0.05 |

## Inference

```bash
python run_model.py infer --episodes 10
python run_model.py infer --model-path path/to/checkpoint.pth --episodes 10 --quiet
```

If `--model-path` is omitted, the most recently modified `.pth` in `controllers/PPO/checkpoints/` is used. The `--quiet` flag suppresses per-episode output.

PPO also exposes `evaluate()` in `controllers/PPO/PPO.py` for programmatic evaluation, returning a summary dict with `mean_reward`, `success_rate`, `goal_touch_rate`, `collision_rate`, `timeout_rate`.

---

<!-- Add your own notes, citations, or additional information below this line -->

LLM level: 4 - LLM generated this file, gone through and made sure it is correct.
