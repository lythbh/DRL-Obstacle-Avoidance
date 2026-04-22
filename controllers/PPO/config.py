"""Configuration definitions for PPO training and environment behavior."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Config:
    """Training and environment hyperparameters."""

    # Training
    episodes: int = 500
    update_every: int = 4  # PPO update frequency (episodes)
    epochs: int = 2  # Optimization epochs per update
    batch_size: int = 512
    diagnostics_window: int = 25  # Episode window size for diagnostics/performance logs
    checkpoint_dir: str = "checkpoints"
    checkpoint_name: str = "ppo_best_last_run.pt"

    # PPO Agent
    gamma: float = 0.99  # Discount factor
    epsilon: float = 0.2  # PPO clip parameter
    learning_rate: float = 2e-4
    entropy_coef: float = 0.006  # Entropy regularization (initial)
    entropy_final_coef: float = 0.0005  # Entropy regularization (final)
    max_grad_norm: float = 0.5  # Gradient clipping for stabler PPO updates
    hidden_size: int = 64  # Network hidden layer size
    train_device: str = "auto"  # "auto", "cpu", or "cuda"
    enable_policy_rollback: bool = True  # Reload best model if rolling goal-rate collapses
    rollback_goal_rate_threshold: float = 0.02  # Treat windows below this goal-rate as collapse
    rollback_patience_windows: int = 3  # Consecutive collapsed windows before rollback
    rollback_cooldown_windows: int = 2  # Windows to wait before another rollback
    rollback_min_episodes: int = 100  # Earliest episode to allow rollback checks

    # Environment
    max_steps: int = 3000  # Max steps per episode
    collision_threshold: float = 0.1  # LiDAR distance threshold for collision
    low_score_threshold: float = -300.0  # Episode reset threshold
    collision_penalty: float = -80.0  # Penalty when collision happens
    goal_reward: float = 200.0  # Reward when reaching goal
    progress_reward_scale: float = 6.5  # Scale for distance-progress reward
    progress_away_multiplier: float = 2.0  # Extra penalty factor when moving away from the goal
    distance_penalty_scale: float = 0.16  # Per-step penalty proportional to normalized goal distance
    orbit_distance_threshold: float = 1.2  # Detect circling when within this distance to goal
    orbit_progress_tolerance: float = 0.01  # Treat tiny distance changes as no progress
    orbit_penalty: float = 0.35  # Per-step penalty for near-goal circling
    no_progress_penalty: float = 0.05  # Per-step penalty when progress is below tolerance
    clearance_target_norm: float = 0.18  # Target minimum normalized LiDAR clearance
    clearance_penalty_scale: float = 0.65  # Penalty scale when driving too close to obstacles
    near_collision_norm: float = 0.10  # Early termination threshold before hard collision
    near_collision_penalty: float = -25.0  # Penalty for near-collision early termination
    heading_reward_scale: float = 0.12  # Bonus for facing toward the goal
    step_penalty: float = -0.02  # Small per-step penalty to encourage efficiency
    heading_frame_offset: float = float(np.pi)  # ALTINO forward axis is flipped vs world yaw
    endpoint: Tuple[float, float] = (2, 0)  # Goal location (adjust to match world)
    goal_radius: float = 0.3  # Distance threshold for successful goal reach
    stagnation_patience_steps: int = 70  # End episode if best distance is not improved for too long
    stagnation_progress_delta: float = 0.01  # Minimum distance improvement to reset stagnation counter
    stagnation_penalty: float = -110.0  # Penalty applied when stagnation ends an episode
    visualize_goal: bool = False  # Spawn/move a visible marker at the goal
    goal_marker_height: float = 0.05  # Marker height above floor
    reference_distance: Optional[float] = None  # Start-to-goal distance, filled in at init
    force_fast_mode: bool = True  # Put Webots in FAST mode from controller at startup
    observation_mode: str = "slam_v1"  # Observation schema mode: "baseline" or "slam_v1"

    # SLAM Integration
    enable_slam_runtime: bool = True  # Keep baseline behavior unless explicitly enabled
    slam_cnn_checkpoint_path: Optional[str] = None  # Optional path to CNN landmark checkpoint
    slam_cnn_update_every: int = 5  # Run CNN landmark inference every N environment steps
    slam_pose_graph_optimize_every: int = 50  # Run map optimization every N keyframes
    slam_telemetry_interval: int = 100  # Emit SLAM telemetry every N environment steps

    # Robot Control
    actions: Optional[List[Tuple[float, float]]] = None  # (steering, speed) pairs
    start_position: Optional[List[float]] = field(default_factory=lambda: [-2, 0, 0.02])  # [x, y, z]
    start_rotation: Optional[List[float]] = None  # [x, y, z, w]
    start_position_noise: float = 0.0  # Random position jitter at reset
    start_yaw_noise: float = 0.1  # Random yaw jitter at reset

    # Motor/Sensor Config
    max_speed: float = 8.0
    lidar_bins: int = 128  # Number of pooled lidar bins used in the observation vector
    action_repeat: int = 2  # Simulator steps to repeat per chosen action (>=1)
    reset_settle_steps: int = 10  # Steps to wait for physics to settle after reset

    def __post_init__(self) -> None:
        """Initialize defaults for mutable fields."""
        if self.actions is None:
            self.actions = [
                (-0.85, 0.35 * self.max_speed),  # hard left, slow
                (-0.45, 0.45 * self.max_speed),  # left
                (0.0, 0.65 * self.max_speed),  # straight
                (0.45, 0.45 * self.max_speed),  # right
                (0.85, 0.35 * self.max_speed),  # hard right, slow
            ]
        if self.start_position is None:
            self.start_position = [0.05, -0.1, 0.02]
        if self.start_rotation is None:
            self.start_rotation = [0.0, 0.0, 1.0, 0.0]
        if self.reference_distance is None:
            start_xy = np.array(self.start_position[:2], dtype=np.float32)
            endpoint_xy = np.array(self.endpoint, dtype=np.float32)
            self.reference_distance = float(np.linalg.norm(start_xy - endpoint_xy))
