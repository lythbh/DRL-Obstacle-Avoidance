"""Consolidated defaults for environment, SLAM, rewards, and training."""

# --- Environment observation / physics ---
ENV_LIDAR_SECTOR_DIM = 16
ENV_POSE_GOAL_DIM = 5
ENV_IMU_FEATURE_DIM = 10
ENV_OCCUPANCY_GRID_SHAPE = None
ENV_MAX_STEPS = 5000
ENV_COLLISION_THRESHOLD = 0.1
ENV_LOW_SCORE_THRESHOLD = -3000.0   # CHANGED: -2000 -> -3000 (more exploration room)
ENV_ENDPOINT = (2.0, 0.0)
ENV_GOAL_THRESHOLD = 0.3
ENV_GOAL_STOP_SPEED_THRESHOLD = 0.15
ENV_MAX_STEERING_ANGLE = 0.9
ENV_MAX_SPEED = 6.0
ENV_MIN_SPEED = 0.0
ENV_START_POSITION = (-2.0, 0.0, 0.02)
ENV_START_ROTATION = (0.0, 0.0, 1.0, 0.0)
ENV_START_POSITION_NOISE = 0.08
ENV_START_YAW_NOISE = 0.8
ENV_RESET_SETTLE_STEPS = 10

# --- SLAM ---
SLAM_ENABLE = True
SLAM_PROFILE = False
SLAM_PROFILE_INTERVAL = 500
SLAM_SAVE_PLOTS = False
SLAM_FORCE_CPU = True

# --- Reward ---
REW_COLLISION_PENALTY = -200.0      # CHANGED: -400 -> -200 (still 2x original; collision must hurt more than proximity)
REW_PROGRESS_SCALE = 3.0
REW_DISTANCE_SCALE = 0.05
REW_HEADING_SCALE = 0.35            # CHANGED: 0.25 -> 0.35 (restore goal-seeking signal; 0.5 was too strong, 0.25 too weak)
REW_SAFETY_SCALE = 0.8              # CHANGED: 2.0 -> 0.8 (moderate increase over 0.5; 2.0 was lethal)
REW_MOTION_SCALE = 0.2              # CHANGED: 0.1 -> 0.2 (compromise; keep some forward incentive)
REW_SLOW_SPEED_THRESHOLD = 0.25
REW_SLOW_SPEED_PENALTY = -0.02
REW_HIGH_SPEED_THRESHOLD = 0.6
REW_HIGH_SPEED_BONUS = 0.05
REW_NEW_BEST_DISTANCE_BONUS = 0.3   # CHANGED: 0.5 -> 0.3 (exploration nudge without inflation)
REW_STEP_PENALTY = -0.01            # CHANGED: -0.03 -> -0.01 (revert; time pressure was compounding the death spiral)
REW_GOAL_SUCCESS = 100.0
REW_GOAL_STOP_BONUS = 200.0
REW_GOAL_SPEED_PENALTY = -10.0
REW_GOAL_OVERSHOOT_PENALTY = -12.0
REW_SCALE = 0.5
REW_PROXIMITY_SCALE = 0.8             # CHANGED: 0.6 -> 0.8
REW_PROXIMITY_RADIUS = 2.0            # CHANGED: 1.5 -> 2.0 (earlier final-approach bonus)

# --- Training ---
class RecurrentDefaults:
    sequence_length = 32
    burn_in = 8
    sequence_stride = 16

class SACDefaults:
    episodes = 2500
    seed = 42
    update_after_steps = 2000
    gradient_steps_per_episode = 16       # CHANGED: 32 -> 16 (less overfitting)
    save_every = 100
    gamma = 0.99
    tau = 0.005                           # CHANGED: 0.001 -> 0.005 (smoother target updates)
    target_update_interval = 2            # CHANGED: 1 -> 2 (update targets less frequently)
    actor_lr = 2e-4                       # CHANGED: 3e-4 -> 2e-4 (slightly more conservative)
    critic_lr = 2e-4                      # CHANGED: 3e-4 -> 2e-4
    alpha_lr = 0.0005                     # CHANGED: 0.0003 -> 0.0005 (slightly faster decay for precision)
    initial_alpha = 0.5                   # CHANGED: 1 -> 0.5 (less initial noise, balanced by slower decay)
    auto_entropy_tuning = True
    target_entropy_scale = 0.5            # CHANGED: 0.4 -> 0.5 (maintain slightly more exploration)
    hidden_size = 128
    latent_size = 128
    recurrent_cell = "gru"
    recurrent_hidden_size = None
    recurrent_layers = 1
    lstm_hidden_size = 128
    lstm_layers = 1
    log_std_min = -3.0
    log_std_max = 2.0
    replay_capacity = 50000               # CHANGED: 16384 -> 50000 (prevents buffer saturation)
    replay_batch_size = 128
    min_replay_sequences = 128
