"""Consolidated defaults for environment, SLAM, rewards, and training."""

# --- Environment observation / physics ---
ENV_LIDAR_SECTOR_DIM = 16
ENV_POSE_GOAL_DIM = 5
ENV_IMU_FEATURE_DIM = 10
ENV_OCCUPANCY_GRID_SHAPE = None
ENV_MAX_STEPS = 5000
ENV_COLLISION_THRESHOLD = 0.1
ENV_LOW_SCORE_THRESHOLD = -3000.0
ENV_ENDPOINT = (2.0, 0.0)
ENV_GOAL_THRESHOLD = 0.3
ENV_GOAL_STOP_SPEED_THRESHOLD = 0.15
ENV_MAX_STEERING_ANGLE = 0.9
ENV_MAX_SPEED = 6.0
ENV_MIN_SPEED = 0.0
ENV_START_POSITION = (-2.0, 0.0, 0.02)
ENV_START_ROTATION = (0.0, 0.0, 1.0, 0.0)
ENV_START_POSITION_NOISE = 0.08
ENV_START_YAW_NOISE = 0.4
ENV_RESET_SETTLE_STEPS = 10

# --- SLAM ---
SLAM_ENABLE = True
SLAM_PROFILE = False
SLAM_PROFILE_INTERVAL = 500
SLAM_SAVE_PLOTS = False
SLAM_FORCE_CPU = True

# --- Reward ---
REW_COLLISION_PENALTY = -100.0  
REW_PROGRESS_SCALE = 1.0
REW_DISTANCE_SCALE = 0.0
REW_HEADING_SCALE = 1.0
REW_SAFETY_SCALE = 1.0
REW_MOTION_SCALE = 0.0
REW_SLOW_SPEED_THRESHOLD = 0.25
REW_SLOW_SPEED_PENALTY = 0.0
REW_HIGH_SPEED_THRESHOLD = 0.6
REW_HIGH_SPEED_BONUS = 0.0
REW_NEW_BEST_DISTANCE_BONUS = 0.0
REW_STEP_PENALTY = 0.0
REW_GOAL_SUCCESS = 100.0
REW_GOAL_STOP_BONUS = 0.0
REW_GOAL_SPEED_PENALTY = 0.0
REW_GOAL_OVERSHOOT_PENALTY = 0.0
REW_SCALE = 1.0
REW_PROXIMITY_SCALE = 0.0
REW_PROXIMITY_RADIUS = 0.0

# --- Training ---
class RecurrentDefaults:
    sequence_length = 32
    burn_in = 8
    sequence_stride = 16

class SACDefaults:
    episodes = 500
    #seed = 42                        #Optional
    update_after_steps = 2000
    gradient_steps_per_episode = 16
    save_every = 100
    gamma = 0.995
    tau = 0.003
    target_update_interval = 2
    actor_lr = 1e-4
    critic_lr = 1e-4
    alpha_lr = 0.0001
    initial_alpha = 0.7
    auto_entropy_tuning = True
    target_entropy_scale = 0.6
    hidden_size = 128
    latent_size = 128
    recurrent_cell = "gru"
    recurrent_hidden_size = None
    recurrent_layers = 1
    lstm_hidden_size = 128
    lstm_layers = 1
    log_std_min = -3.0
    log_std_max = 2.0
    replay_capacity = 50000
    replay_batch_size = 128
    min_replay_sequences = 128
