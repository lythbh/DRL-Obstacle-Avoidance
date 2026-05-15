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
ENV_GOAL_THRESHOLD = 0.5            # CHANGED: 0.3 -> 0.5 (speed-penalty zone starts earlier; forces deceleration)
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
REW_COLLISION_PENALTY = -600.0      # CHANGED: -250 -> -600 (collision must be the dominant negative event)
REW_PROGRESS_SCALE = 3.0
REW_DISTANCE_SCALE = 0.02           # CHANGED: 0.05 -> 0.02 (progress already covers distance; weaken redundant signal)
REW_HEADING_SCALE = 0.2             # CHANGED: 0.3 -> 0.2 (reduce blind "face the goal" pressure)
REW_SAFETY_SCALE = 0.3              # CHANGED: 1.0 -> 0.3 (per-step obstacle tax is the #1 cause of variance)
REW_MOTION_SCALE = 0.1              # CHANGED: 0.15 -> 0.1 (drive fast only when path is clear)
REW_SLOW_SPEED_THRESHOLD = 0.25
REW_SLOW_SPEED_PENALTY = -0.02
REW_HIGH_SPEED_THRESHOLD = 0.6
REW_HIGH_SPEED_BONUS = 0.05
REW_NEW_BEST_DISTANCE_BONUS = 0.3   # CHANGED: 0.5 -> 0.3 (exploration nudge without inflation)
REW_STEP_PENALTY = -0.005           # CHANGED: -0.01 -> -0.005 (very mild time pressure)
REW_GOAL_SUCCESS = 100.0
REW_GOAL_STOP_BONUS = 200.0
REW_GOAL_SPEED_PENALTY = -20.0      # CHANGED: -10 -> -20 (stronger incentive to brake in goal zone)
REW_GOAL_OVERSHOOT_PENALTY = -12.0
REW_SCALE = 0.5
REW_PROXIMITY_SCALE = 0.6           # CHANGED: 1.2 -> 0.6 (halve the goal attractor; fewer high-speed crashes near goal)
REW_PROXIMITY_RADIUS = 2.0            # CHANGED: 1.5 -> 2.0 (earlier final-approach bonus)

# --- Training ---
class RecurrentDefaults:
    sequence_length = 32
    burn_in = 8
    sequence_stride = 16

class SACDefaults:
    episodes = 1000
    #seed = 42                        #Optional
    update_after_steps = 2000
    gradient_steps_per_episode = 12   # CHANGED: 16 -> 12 (reduce replay overfitting)
    save_every = 100
    gamma = 0.99
    tau = 0.003                       # CHANGED: 0.005 -> 0.003 (softer target networks)
    target_update_interval = 4        # CHANGED: 2 -> 4 (halve target-update frequency)
    actor_lr = 2e-4                       # CHANGED: 3e-4 -> 2e-4 (slightly more conservative)
    critic_lr = 2e-4                      # CHANGED: 3e-4 -> 2e-4
    alpha_lr = 0.0002                 # CHANGED: 0.0005 -> 0.0002 (2.5x slower decay)
    initial_alpha = 0.7               # CHANGED: 0.5 -> 0.7 (more initial policy noise)
    auto_entropy_tuning = True
    target_entropy_scale = 0.6        # CHANGED: 0.5 -> 0.6 (higher entropy floor)
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
