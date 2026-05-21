# DRL Obstacle Avoidance (Webots ALTINO)

This project trains a recurrent PPO controller for ALTINO obstacle avoidance in Webots, with shared environment, sensor, reward, and SLAM processing.

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
- `controllers/RNN/`: recurrent policy/value modules used by PPO.
- `controllers/Webots/webots_env.py`: Webots environment, reward logic, success criteria, and SLAM hooks.
- `controllers/SLAM/`: IMU/IEKF/map processing modules.
- `controllers/PPO/PPO_rewards.py`: PPO-specific reward computation.
- `controllers/PPO/PPO_metrics_logger.py`: PPO CSV metrics logger.
- `controllers/common/`: shared defaults, checkpoint utilities, metrics logger base, seed, and curriculum manager.
- `run_model.py`: deterministic inference runner for PPO checkpoints.
- `worlds/ObstacleCourse.wbt`: PPO world.

## Complete Pipeline

### 1. Entry Points

| Component | File | Function |
|---|---|---|
| PPO Training | `controllers/PPO/PPO.py` | `train()` |
| Inference | `run_model.py` | `run_inference()` |

PPO is started by Webots as a robot controller. `run_model.py` loads a saved checkpoint and runs deterministic episodes independently.

### 2. Configuration System

| File | Provides |
|---|---|
| `controllers/common/PPO_defaults.py` | `PPODefaults`, `RecurrentDefaults`, environment constants, SLAM flags, reward coefficients |

PPO defaults: `episodes=500`, `update_every=4`, `epochs=4`, `batch_size=128`, `lr=1e-4`, `entropy_coef=0.005`, `gamma=0.99`, `gae_lambda=0.98`, `epsilon=0.1`, `max_steps=6000`, `SLAM_ENABLE=False`, `REW_SCALE=0.01`.

Environment variables override Config fields per-run (`PPO_ARCH`, `PPO_EPISODES`, `PPO_MAX_STEPS`, `PPO_SAVE_EVERY`, `PPO_FORCE_CPU`, `PPO_SEED`, `PPO_LOAD_MODEL`, `PPO_RUN_ID`).

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
       │    └─ SLAMProcessor(config)     # IMU filtering + IEKF + occupancy map
       └─ PPORewardComputer
```

#### Sensor Read Pipeline (`AltinoDriver.read_sensors()`)
```
lidar.getRangeImage() ──────┐
gps.getValues() ────────────┤
accelerometer.getValues() ──┤──> SLAMProcessor.process()
gyro.getValues() ───────────┘    │
                                  ├─ sector_lidar()    → 16 normalized sector minima
                                  ├─ imu_proc.step()   → EKF quaternion + bias
                                  ├─ iekf.propagate()  → dead-reckoned position/heading
                                  └─ slam_map.update() → occupancy grid via Bresenham
```

#### Observation Vector (33 features by default, `_build_observation()`)

| Segment | Size | Source |
|---|---|---|
| LiDAR sectors | 16 | `SLAMProcessor.sector_lidar()` — min range per angular bin, normalized [0,1] |
| Direction features | 5 | `sin(heading), cos(heading), sin(goal_error), cos(goal_error), normalized_goal_distance` |
| IMU features | 10 | 3 accel_body, 3 gyro_body, 4 quaternion — all normalized |
| Occupancy grid | variable | Downsampled SLAM occupancy probabilities (disabled by default) |

#### Action Space (`WebotsEnv.step()`)
- **Raw**: `[steering (-0.9, 0.9), speed (0.0, 6.0)]`
- **Adaptive speed cap**: `max_speed * obstacle_factor * steering_factor` where obstacle_factor is derived from min lidar norm and steering_factor penalizes sharp turns at high speed
- **Output**: `robot.set_steering(angle)`, `robot.set_speed(speed)` — steering via position control, wheels via velocity control

#### Episode Termination
- **Goal success**: robot distance < `goal_threshold` (0.3m) — sets `terminated=True`
- **Collision**: raw lidar minimum < `collision_threshold` (0.1m) — sets `terminated=True`
- **Low score**: cumulative reward <= `low_score_threshold` (-2000) — terminates early
- **Timeout**: `current_step >= max_steps` — sets `truncated=True`, applies timeout penalty
- **Overshoot**: robot leaves goal region without stopping — applies overshoot penalty

### 4. SLAM Modules (`controllers/SLAM/`)

| Module | File | Purpose |
|---|---|---|
| `IMUProcessor` | `imu_filter.py` | EKF with 7 states (quaternion w,x,y,z + gyro bias bx,by,bz). Predict from gyro, correct from accelerometer. Outputs `IMUState` with quaternion, body-frame accel/gyro, world-frame accel. |
| `IEKFBackend` | `iekf_backend.py` | 8-state dead-reckoning IEKF [px, py, theta, vx, vy, b_omega_z, b_ax, b_ay]. Propagates from wheel speed + gyro z-rate. Used for heading estimate when SLAM is enabled. |
| `SLAMMap` | `slam_map.py` | Log-odds occupancy grid with Bresenham ray-casting + keyframe trajectory tracker. `KEYFRAME_DIST=0.3m`, `KEYFRAME_ANGLE=0.15rad`. Saves PNG plots via matplotlib. |
| `OccupancyMap` | `slam_map.py` (same) | Core grid: `FREE_LOG_ODDS=-0.5`, `OCC_LOG_ODDS=1.5`, clip range [-5,5]. Resolution 0.05m, 40x40m grid. |
| `NavigatingController` | `navigation_controller.py` | Standalone deterministic controller using potential fields: goal attraction + LiDAR repulsion + map repulsion. Not used by PPO training. |

### 5. Reward Computation (`PPORewardComputer`)

`PPORewardComputer.compute(collision, pos, prev_distance, goal_error, min_lidar_norm, speed_norm, reached_new_best)` returns `(reward, new_distance)`.

#### Components:
- **Collision penalty**: -50
- **Progress reward**: forward progress * scale * proximity_factor
- **Distance penalty**: -distance_ratio * 0.05
- **Heading reward**: cos(goal_error) * 0.5
- **Safety penalty**: -(1 - min_lidar) * 0.2
- **Motion reward**: speed_norm * 0.05
- **Speed bonuses**: slow penalty below 0.25 (-0.02), high-speed bonus above 0.6 (+0.05)
- **New best distance bonus**: +0.05 when closest approach improves
- **Step penalty**: -0.015
- **Goal success**: +200 (when distance < goal_threshold) + 50 hold reward

### 6. Neural Network Architecture (`controllers/RNN/`)

Networks derive from `RecurrentActorCriticBase` (`base.py`) and `FeedForwardActorCritic` (`PPO.py`):

```
Observation (33)
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
- **GRU**: `nn.GRU(128, 128, layers=1, batch_first=True)`
- **LSTM**: `nn.LSTM(128, 128, layers=1, batch_first=True)`
- **FeedForward**: same encoder branches + fusion, then direct policy/value heads (no recurrent core)
- **Recurrent forward**: per-timestep loop with done-mask state reset
- **Recurrent options**: `gru` (default), `lstm`, or `none` (feedforward)

#### Action Distribution
`Normal(mean, std)` with `actor_log_std` as a learnable parameter (initialized -0.5). Tanh squashing + rescale to action range. Log-probability includes tanh correction and scale correction.

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
    │      store obs, a, log_prob, reward            │
    │                                              │
    │ 4. Evaluate episode: values via model forward  │
    │ 5. Bootstrap value if truncated (max_steps)    │
    │ 6. GAE(lambda): advantages + returns           │
    │ 7. Append trajectory to rollout buffer         │
    │                                              │
    │ 8. Every update_every episodes:                │
    │    a. Sanitize trajectories (NaN/Inf guard)    │
    │    b. Normalize advantages across episodes     │
    │    c. Split into sequences (seq_len, stride)   │
    │    d. Shuffle, mini-batch (batch_size)         │
    │    e. For each epoch:                          │
    │       - evaluate_sequences → log_prob, value   │
    │       - clipped surrogate objective (ε=0.1)    │
    │       - smooth L1 value loss                   │
    │       - entropy bonus                          │
    │       - learn_mask excludes burn-in steps      │
    │       - gradient clipping per component:       │
    │         actor=0.5, critic=5.0, encoder=0.5     │
    │       - early stop if approx_kl > 0.05         │
    │    f. Clear rollout buffer                     │
    │                                              │
    │ 9. Anneal entropy_coef (→ 40% over episodes)  │
    │10. Warmup LR (linear ramp first 25 episodes)  │
    │11. Log episode/update metrics                  │
    │12. Save best checkpoint if new best            │
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

### 8. Burn-In Masking

PPO uses `_sequence_loss_mask(valid_mask, burn_in)`:
```
start_index = min(burn_in, valid_length - 1)
learn_mask = valid_mask * (timestep >= start_index)
```
The first `burn_in` (default 8) timesteps of each sequence are excluded from gradient computation, allowing the recurrent state to warm up on real data before contributing to the loss.

### 9. Checkpoint System (`controllers/common/checkpoints.py`)

| Function | Purpose |
|---|---|
| `run_checkpoint_dir()` | Create dated folder: `checkpoints/<run_id>/` |
| `run_checkpoint_path()` | Build path: `checkpoints/<run_id>/<prefix>_<run_id>.pth` |
| `load_checkpoint()` | `torch.load` with `weights_only=False` fallback |
| `save_checkpoint_file()` | Save under prefix, return path |

**PPO checkpoint contents**: `{algorithm, episode, reward, goal_episode, config, obs_size, action_dim, recurrent_cell, model_state_dict, actor_log_std}`

**Save triggers**:
- `best_goal`: every episode ending in "goal" with new best reward
- `best`: first non-goal best reward (only if no goal episode yet)
- `checkpoint`: every `save_every` episodes
- `final`: end of training

### 10. Inference Pipeline (`run_model.py`)

```
run_inference(config):
  _init_supervisor()
  checkpoint = torch.load(model_path)
  Build: env = WebotsEnv(train_config)
         agent = PPOAgent(obs_size, action_dim, config)
  agent.load_model(model_path)
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

- `--algorithm ppo` (default)
- `--model-path`: optional override of checkpoint path
- `--quiet`: suppresses per-episode output
- Default model path: most recently modified `.pth` in `controllers/PPO/checkpoints/`

### 11. Metrics Logging (`controllers/common/metrics_logger.py`)

Three CSV files written per training run:

| File | Contents |
|---|---|
| `ppo_hyperparams.csv` | All config fields + obs_size, action_dim |
| `ppo_episodes.csv` | Per-episode: reward, avg10, length, success, goal_touched, collision, timeout, min_dist, avg_speed, end_reason, action/obs stats, aggregated update metrics |
| `ppo_updates.csv` | Per-update: actor_loss, critic_loss, policy_entropy, entropy_coef, value_residual, grad_norms, lr |

### 12. Curriculum (Optional, `controllers/common/curriculum.py`)

Three-stage progressive difficulty:
- **Stage 1**: 0 obstacles, fixed goal, low noise — learn basic goal-seeking
- **Stage 2**: 5 obstacles, goal_y ±0.7m, moderate noise
- **Stage 3**: 15 obstacles, goal_y ±1.5m, full noise — target distribution

Advancement: 10 consecutive goals or 2500 max episodes per stage.

### 13. Data Flow Summary

```
Webots Supervisor
  └── World File (ObstacleCourse.wbt)
       └── ALTINO Robot Controller (PPO.py)
            │
            ├── Config (dataclass + env overrides)
            │
            ├── WebotsEnv
            │    ├── AltinoDriver
            │    │    ├── Sensors → raw lidar, GPS, IMU
            │    │    └── SLAMProcessor
            │    │         ├── IMUProcessor (EKF quaternion)
            │    │         ├── IEKFBackend (dead-reckoning)
            │    │         └── SLAMMap (occupancy grid)
            │    ├── _build_observation() → 33-feature vector
            │    ├── step() → adaptive speed, reward, termination
            │    └── PPORewardComputer
            │
            ├── PPOAgent
            │    ├── RNN Actor-Critic (GRU/LSTM/FeedForward)
            │    │    ├── Encoder branches
            │    │    ├── Recurrent core
            │    │    └── Policy + Value heads
            │    ├── select_action() → [steering, speed]
            │    └── update() → PPO gradient step
            │
            ├── Rollout Buffer
            │
            ├── Checkpoint Manager
            │
            └── Metrics Logger (3 CSV files)
```

## Training Behavior

PPO reports the following episode metrics:
- `r`, `avg10`, `steps`, `succ10`, `touch10`, `col10`, `to10`, `min_d`, `end`, `t`

Where:
- `succ10`: rolling 10-episode success rate (goal reached and stopped)
- `touch10`: rolling 10-episode goal-touch rate
- `col10`: rolling 10-episode collision rate
- `to10`: rolling 10-episode timeout (`max_steps`) rate

## PPO Summary
- Recurrent options: `gru` or `lstm` or `none` (feedforward)
- Update style: on-policy rollout trajectories
- Update schedule: every `update_every` episodes
- Loss style: clipped PPO objective + value loss + entropy regularization
- GAE lambda: 0.98, value bootstrap for truncated episodes

## Checkpoints
Best checkpoints saved to `controllers/PPO/checkpoints/<timestamp>/best_model.pth`
Final model saved to `controllers/PPO/final_model.pth`

## Inference
Run deterministic inference with:
```bash
python run_model.py --algorithm ppo --episodes 10
```

Optional args:
- `--model-path <path>` to force a specific checkpoint
- `--quiet` to reduce per-episode output

If `--model-path` is omitted, `run_model.py` loads the newest dated `best_model.pth` for PPO.
