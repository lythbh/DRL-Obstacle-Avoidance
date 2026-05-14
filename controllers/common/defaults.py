"""Consolidated defaults for environment, SLAM, rewards, and training."""

# --- Environment observation / physics ---
ENV_LIDAR_SECTOR_DIM = 16
ENV_POSE_GOAL_DIM = 5
ENV_IMU_FEATURE_DIM = 10
ENV_OCCUPANCY_GRID_SHAPE = None
ENV_MAX_STEPS = 4000
ENV_COLLISION_THRESHOLD = 0.1
ENV_LOW_SCORE_THRESHOLD = -2000.0      # relaxed from -500 — let episodes run full duration to discover goal
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
REW_COLLISION_PENALTY = -200.0
REW_PROGRESS_SCALE = 5.0              # increased from 2.0 — stronger incentive for forward progress
REW_DISTANCE_SCALE = 0.15              # increased from 0.15 — stronger penalty for distance from goal
REW_HEADING_SCALE = 0.05               # increased from 0.05 — stronger heading alignment
REW_SAFETY_SCALE = 0.15
REW_MOTION_SCALE = 0.02
REW_SLOW_SPEED_THRESHOLD = 0.25
REW_SLOW_SPEED_PENALTY = -0.02
REW_HIGH_SPEED_THRESHOLD = 0.6
REW_HIGH_SPEED_BONUS = 0.05
REW_NEW_BEST_DISTANCE_BONUS = 0.05
REW_STEP_PENALTY = -0.01              # was 0.0 — time pressure to reach goal faster
REW_GOAL_SUCCESS = 200.0
REW_GOAL_STOP_BONUS = 100.0
REW_GOAL_HOLD = 0.0
REW_GOAL_SPEED_PENALTY = -10.0
REW_GOAL_OVERSHOOT_PENALTY = -12.0
REW_SCALE = 1.0                       # full reward signal � per-step progress rewards need magnitude to guide critic
REW_PROXIMITY_SCALE = 0.6              # doubled from 0.3 — stronger incentive in final approach zone
REW_PROXIMITY_RADIUS = 1.5

# --- Training ---
class RecurrentDefaults:
    sequence_length = 16
    burn_in = 8                         # increased from 4 � more steps for recurrent state to warm up from zero-init
    sequence_stride = 8

class PPODefaults:
    episodes = 2500
    update_every = 4                    # was 1 � collect multiple episodes for more diverse gradient estimates
    epochs = 2
    batch_size = 64
    save_every = 100
    learning_rate = 5e-4
    entropy_coef = 0.01                 # was 0.005 � more exploration; critical before critic converges
    gae_lambda = 0.95
    hidden_size = 128
    latent_size = 128
    lstm_hidden_size = 128
    lstm_layers = 1
    recurrent_cell = "gru"
    clip_value_loss = False            # was True — value clipping blocks critic gradient when returns are large
    max_grad_norm = 0.5

class SACDefaults:
    episodes = 2500
    update_after_steps = 2000
    updates_per_step = 2
    gradient_steps_per_episode = 2
    save_every = 100
    gamma = 0.99
    tau = 0.01                          # reduced from 0.01 � slower target tracking for more stable Q-learning
    actor_lr = 5e-4                      # was 3e-4 � faster actor learning
    critic_lr = 1e-4                     # was 3e-4 � faster critic learning
    alpha_lr = 0.001                     # reduced from 0.003 � slower alpha decay to maintain exploration longer
    initial_alpha = 0.5                # increased from 0.2 — start with more exploration that decays
    auto_entropy_tuning = True
    target_entropy_scale = 0.8           # increased from 0.5 � maintain more entropy/stochasticity in policy
    hidden_size = 128
    latent_size = 128
    recurrent_cell = "gru"
    recurrent_hidden_size = None
    recurrent_layers = 1
    lstm_hidden_size = 128
    lstm_layers = 1
    log_std_min = -5.0
    log_std_max = 2.0
    replay_capacity = 65536
    replay_batch_size = 64
    min_replay_sequences = 256