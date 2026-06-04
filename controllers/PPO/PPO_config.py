"""
PPO configuration and environment-variable overrides.

LLM level: 4 - LLM wrote the majority of the starting code, but we have since iterated on it a lot.
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import controllers.PPO.PPO_defaults as d
from controllers.common.seed import set_all_seeds


@dataclass
class Config:
    episodes: int = d.PPODefaults.episodes
    update_every: int = d.PPODefaults.update_every
    epochs: int = d.PPODefaults.epochs
    batch_size: int = d.PPODefaults.batch_size
    save_every: int = d.PPODefaults.save_every
    gamma: float = d.PPODefaults.gamma
    gae_lambda: float = d.PPODefaults.gae_lambda
    epsilon: float = d.PPODefaults.epsilon 
    learning_rate: float = d.PPODefaults.learning_rate
    entropy_coef: float = d.PPODefaults.entropy_coef
    hidden_size: int = d.PPODefaults.hidden_size
    latent_size: int = d.PPODefaults.latent_size
    recurrent_cell: str = d.PPODefaults.recurrent_cell
    sequence_length: int = d.RecurrentDefaults.sequence_length
    burn_in: int = d.RecurrentDefaults.burn_in
    sequence_stride: int = d.RecurrentDefaults.sequence_stride
    
    collision_penalty: float = d.REW_COLLISION_PENALTY
    progress_reward_scale: float = d.REW_PROGRESS_SCALE
    distance_reward_scale: float = d.REW_DISTANCE_SCALE
    heading_reward_scale: float = d.REW_HEADING_SCALE
    safety_reward_scale: float = d.REW_SAFETY_SCALE
    motion_reward_scale: float = d.REW_MOTION_SCALE
    new_best_distance_bonus: float = d.REW_NEW_BEST_DISTANCE_BONUS
    step_penalty: float = d.REW_STEP_PENALTY
    goal_success_reward: float = d.REW_GOAL_SUCCESS
    goal_hold_reward: float = d.REW_GOAL_HOLD
    lidar_sector_dim: int = d.ENV_LIDAR_SECTOR_DIM
    pose_goal_dim: int = d.ENV_POSE_GOAL_DIM
    imu_feature_dim: int = d.ENV_IMU_FEATURE_DIM
    force_cpu: bool = d.force_cpu
    enable_mapping: bool = d.enable_mapping
    save_mapping_plots: bool = d.save_mapping_plots

    max_steps: int = d.ENV_MAX_STEPS
    endpoint: Tuple[float, float] = d.ENV_ENDPOINT
    goal_threshold: float = d.ENV_GOAL_THRESHOLD
    reference_distance: Optional[float] = None
    start_position: Optional[List[float]] = None
    start_rotation: Optional[List[float]] = None
    max_steering_angle: float = d.ENV_MAX_STEERING_ANGLE
    max_speed: float = d.ENV_MAX_SPEED
    min_speed: float = d.ENV_MIN_SPEED

    moving_obstacle_indices: Optional[List[int]] = None
    moving_obstacle_speed: float = d.MOVING_OBSTACLE_SPEED
    moving_obstacle_amplitude: float = d.MOVING_OBSTACLE_AMPLITUDE
    moving_goal: bool = d.MOVING_GOAL
    moving_goal_speed: float = d.MOVING_GOAL_SPEED
    moving_goal_amplitude: float = d.MOVING_GOAL_AMPLITUDE


    def __post_init__(self) -> None:
        """
        Apply some post-initialization logic for the recurrent cell.
        """
        self.recurrent_cell = self.recurrent_cell.lower().strip()
        aliases = {"mlp": "none", "feedforward": "none", "ff": "none"}
        self.recurrent_cell = aliases.get(self.recurrent_cell, self.recurrent_cell)
        assert self.recurrent_cell in {"none", "lstm", "gru"}, f"Unsupported recurrent_cell: {self.recurrent_cell}"
        
        if self.recurrent_cell == "none":
            self.burn_in = 0    
            self.sequence_length = max(1, self.sequence_length)
        
        if self.start_position is None:
            self.start_position = list(d.ENV_START_POSITION)
        
        if self.start_rotation is None:
            self.start_rotation = list(d.ENV_START_ROTATION)
        
        if self.reference_distance is None:
            start_xy = np.array(self.start_position[:2], dtype=np.float32)
            endpoint_xy = np.array(self.endpoint, dtype=np.float32)
            self.reference_distance = float(np.linalg.norm(start_xy - endpoint_xy))
        
        if self.moving_obstacle_indices is None:
            self.moving_obstacle_indices = []


def _env_bool(name: str, default: bool) -> bool:
    """
    Reads environment variable and returns a boolean value.

    Parameters
    ----------
    name : str
        Name of the environment variable.
    default : bool
        Default value to return if the environment variable is not set.

    Returns
    -------
    bool
        Boolean value of the environment variable.
    """
    value = os.getenv(name)
    if value is None:
        return default
    
    return value.lower().strip() in {"1", "true", "yes", "on"}


def _apply_env_overrides(config: Config) -> tuple[Config, Optional[str], Optional[str]]:
    """
    Applies environment variables to the configuration.
    
    Parameters
    ----------
    config : Config
        Configuration to apply environment variables to.
    
    Returns
    -------
    tuple[Config, Optional[str], Optional[str]]
        Tuple of the configuration, the path to the model file, and the path to the optimizer file.
    """
    arch = os.getenv("PPO_ARCH") or os.getenv("PPO_RECURRENT_CELL")
    if arch:
        config.recurrent_cell = arch
    if os.getenv("PPO_EPISODES"):
        config.episodes = int(os.environ["PPO_EPISODES"])
    if os.getenv("PPO_MAX_STEPS"):
        config.max_steps = int(os.environ["PPO_MAX_STEPS"])
    if os.getenv("PPO_SAVE_EVERY"):
        config.save_every = int(os.environ["PPO_SAVE_EVERY"])
    if os.getenv("PPO_FORCE_CPU"):
        config.force_cpu = _env_bool("PPO_FORCE_CPU", config.force_cpu)
    if os.getenv("PPO_SEED"):
        set_all_seeds(int(os.environ["PPO_SEED"]))
    if os.getenv("PPO_MOVING_OBSTACLE_INDICES"):
        raw = os.environ["PPO_MOVING_OBSTACLE_INDICES"].strip()
        if raw.lower() == "all":
            config.moving_obstacle_indices = list(range(18))
        elif raw:
            config.moving_obstacle_indices = [int(x) for x in raw.split(",")]
    if os.getenv("PPO_MOVING_OBSTACLE_SPEED"):
        config.moving_obstacle_speed = float(os.environ["PPO_MOVING_OBSTACLE_SPEED"])
    if os.getenv("PPO_MOVING_OBSTACLE_AMPLITUDE"):
        config.moving_obstacle_amplitude = float(os.environ["PPO_MOVING_OBSTACLE_AMPLITUDE"])
    if os.getenv("PPO_MOVING_GOAL"):
        config.moving_goal = _env_bool("PPO_MOVING_GOAL", False)
    if os.getenv("PPO_MOVING_GOAL_SPEED"):
        config.moving_goal_speed = float(os.environ["PPO_MOVING_GOAL_SPEED"])
    if os.getenv("PPO_MOVING_GOAL_AMPLITUDE"):
        config.moving_goal_amplitude = float(os.environ["PPO_MOVING_GOAL_AMPLITUDE"])
    config.__post_init__()
    
    return config, os.getenv("PPO_LOAD_MODEL"), os.getenv("PPO_RUN_ID")
