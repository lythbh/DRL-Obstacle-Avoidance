"""Consolidated defaults for environment, SLAM, rewards, and training."""

# --- Environment observation / physics ---
ENV_LIDAR_SECTOR_DIM = 16
ENV_POSE_GOAL_DIM = 5
ENV_IMU_FEATURE_DIM = 10
ENV_OCCUPANCY_GRID_SHAPE = None
ENV_MAX_STEPS = 4000
ENV_COLLISION_THRESHOLD = 0.1
ENV_LOW_SCORE_THRESHOLD = -2000.0
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
REW_COLLISION_PENALTY = -100.0
REW_PROGRESS_SCALE = 3.0
REW_DISTANCE_SCALE = 0.1
REW_HEADING_SCALE = 0.05
REW_SAFETY_SCALE = 0.15
REW_MOTION_SCALE = 0.02
REW_SLOW_SPEED_THRESHOLD = 0.25
REW_SLOW_SPEED_PENALTY = -0.02
REW_HIGH_SPEED_THRESHOLD = 0.6
REW_HIGH_SPEED_BONUS = 0.05
REW_NEW_BEST_DISTANCE_BONUS = 0.05
REW_STEP_PENALTY = -0.005
REW_GOAL_SUCCESS = 500.0
REW_GOAL_STOP_BONUS = 200.0
REW_GOAL_HOLD = 0.0
REW_GOAL_SPEED_PENALTY = -10.0
REW_GOAL_OVERSHOOT_PENALTY = -12.0
REW_SCALE = 1.0
REW_PROXIMITY_SCALE = 0.6
REW_PROXIMITY_RADIUS = 1.5

# --- Training ---
class RecurrentDefaults:
    sequence_length = 32
    burn_in = 8
    sequence_stride = 16

class PPODefaults:
    episodes = 2500
    update_every = 4
    epochs = 2
    batch_size = 64
    save_every = 100
    learning_rate = 5e-4
    entropy_coef = 0.01
    gae_lambda = 0.95
    hidden_size = 128
    latent_size = 128
    lstm_hidden_size = 128
    lstm_layers = 1
    recurrent_cell = "gru"
    clip_value_loss = False
    max_grad_norm = 0.5
