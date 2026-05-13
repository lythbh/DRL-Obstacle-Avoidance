"""Soft Actor-Critic controller for the ALTINO Webots task."""

from __future__ import annotations

import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from controllers.Webots import WebotsEnv, _init_supervisor
from controllers.common.reward_defaults import (
    COLLISION_THRESHOLD,
    COLLISION_PENALTY,
    DISTANCE_REWARD_SCALE,
    ENABLE_SLAM,
    ENDPOINT,
    FORCE_CPU,
    GOAL_HOLD_REWARD,
    GOAL_STOP_SPEED_THRESHOLD,
    GOAL_OVERSHOOT_PENALTY,
    GOAL_SPEED_PENALTY,
    GOAL_STOP_BONUS,
    GOAL_THRESHOLD,
    GOAL_SUCCESS_REWARD,
    HEADING_REWARD_SCALE,
    IMU_FEATURE_DIM,
    LIDAR_SECTOR_DIM,
    LOW_SCORE_THRESHOLD,
    MAX_SPEED,
    MAX_STEERING_ANGLE,
    MAX_STEPS,
    MIN_SPEED,
    MOTION_REWARD_SCALE,
    NEW_BEST_DISTANCE_BONUS,
    OCCUPANCY_GRID_SHAPE,
    POSE_GOAL_DIM,
    PROGRESS_REWARD_SCALE,
    PROFILE_SLAM,
    RESET_SETTLE_STEPS,
    REWARD_SCALE,
    SAFETY_REWARD_SCALE,
    SAVE_SLAM_PLOTS,
    SLAM_PROFILE_INTERVAL,
    START_POSITION,
    START_POSITION_NOISE,
    START_ROTATION,
    START_YAW_NOISE,
    STEP_PENALTY,
)
from controllers.common.training_defaults import RecurrentDefaults, SACDefaults

_CONTROLLER_DIR = Path(__file__).resolve().parent
_CHECKPOINT_DIR = _CONTROLLER_DIR / "checkpoints"

# Use shared checkpoint helpers to avoid duplication across controllers.
from controllers.common.checkpoints import (
    checkpoint_path as _shared_checkpoint_path,
    run_checkpoint_dir as _shared_run_checkpoint_dir,
    run_checkpoint_path as _shared_run_checkpoint_path,
    load_checkpoint as _shared_load_checkpoint,
    save_checkpoint_file as _save_checkpoint_file,
)
from controllers.common.metrics_logger import MetricsLogger


def _checkpoint_path(filename: str) -> str:
    """Controller-local wrapper around shared `checkpoint_path` helper."""
    return _shared_checkpoint_path(_CONTROLLER_DIR, filename)


def _run_checkpoint_dir(run_id: str) -> Path:
    """Controller-local wrapper around shared `run_checkpoint_dir` helper."""
    return _shared_run_checkpoint_dir(_CHECKPOINT_DIR, run_id)


def _run_checkpoint_path(run_id: str, prefix: str, extension: str = "pth") -> str:
    """Controller-local wrapper around shared `run_checkpoint_path` helper."""
    return _shared_run_checkpoint_path(_CHECKPOINT_DIR, run_id, prefix, extension)


def _load_checkpoint(path: str, map_location: torch.device) -> Dict[str, Any]:
    """Controller-local wrapper around shared `load_checkpoint` helper."""
    return _shared_load_checkpoint(path, map_location)


# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass
class Config:
    """
    Training and environment hyperparameters for SAC.
    
    This dataclass consolidates all tunable parameters for the algorithm
    (learning rates, entropy tuning, network architecture) and the environment
    (reward shaping, action bounds, episode configuration). Validation in
    __post_init__ ensures consistency across interdependent parameters.
    """
    episodes: int = SACDefaults.episodes
    update_after_steps: int = SACDefaults.update_after_steps
    updates_per_step: int = SACDefaults.updates_per_step
    save_every: int = SACDefaults.save_every
    gamma: float = SACDefaults.gamma
    tau: float = SACDefaults.tau
    actor_lr: float = SACDefaults.actor_lr
    critic_lr: float = SACDefaults.critic_lr
    alpha_lr: float = SACDefaults.alpha_lr
    initial_alpha: float = SACDefaults.initial_alpha
    auto_entropy_tuning: bool = SACDefaults.auto_entropy_tuning
    target_entropy_scale: float = SACDefaults.target_entropy_scale
    hidden_size: int = SACDefaults.hidden_size
    recurrent_cell: str = SACDefaults.recurrent_cell
    recurrent_hidden_size: Optional[int] = SACDefaults.recurrent_hidden_size
    recurrent_layers: int = SACDefaults.recurrent_layers
    log_std_min: float = SACDefaults.log_std_min
    log_std_max: float = SACDefaults.log_std_max
    sequence_length: int = RecurrentDefaults.sequence_length
    burn_in: int = RecurrentDefaults.burn_in
    sequence_stride: int = RecurrentDefaults.sequence_stride
    replay_capacity: int = SACDefaults.replay_capacity
    replay_batch_size: int = SACDefaults.replay_batch_size
    min_replay_sequences: int = SACDefaults.min_replay_sequences

    lidar_sector_dim: int = LIDAR_SECTOR_DIM
    pose_goal_dim: int = POSE_GOAL_DIM
    imu_feature_dim: int = IMU_FEATURE_DIM
    occupancy_grid_shape: Optional[Tuple[int, ...]] = OCCUPANCY_GRID_SHAPE

    max_steps: int = MAX_STEPS
    collision_threshold: float = COLLISION_THRESHOLD
    low_score_threshold: float = LOW_SCORE_THRESHOLD
    collision_penalty: float = COLLISION_PENALTY
    progress_reward_scale: float = PROGRESS_REWARD_SCALE
    distance_reward_scale: float = DISTANCE_REWARD_SCALE
    heading_reward_scale: float = HEADING_REWARD_SCALE
    safety_reward_scale: float = SAFETY_REWARD_SCALE
    motion_reward_scale: float = MOTION_REWARD_SCALE
    new_best_distance_bonus: float = NEW_BEST_DISTANCE_BONUS
    step_penalty: float = STEP_PENALTY
    endpoint: Tuple[float, float] = ENDPOINT
    goal_threshold: float = GOAL_THRESHOLD
    goal_stop_speed_threshold: float = GOAL_STOP_SPEED_THRESHOLD
    goal_success_reward: float = GOAL_SUCCESS_REWARD
    goal_stop_bonus: float = GOAL_STOP_BONUS
    goal_hold_reward: float = GOAL_HOLD_REWARD
    goal_speed_penalty: float = GOAL_SPEED_PENALTY
    goal_overshoot_penalty: float = GOAL_OVERSHOOT_PENALTY
    reference_distance: Optional[float] = None

    enable_slam: bool = ENABLE_SLAM
    profile_slam: bool = PROFILE_SLAM
    slam_profile_interval: int = SLAM_PROFILE_INTERVAL
    save_slam_plots: bool = SAVE_SLAM_PLOTS
    force_cpu: bool = FORCE_CPU

    max_steering_angle: float = MAX_STEERING_ANGLE
    min_speed: float = MIN_SPEED
    start_position: Optional[List[float]] = None
    start_rotation: Optional[List[float]] = None
    start_position_noise: float = START_POSITION_NOISE
    start_yaw_noise: float = START_YAW_NOISE
    reset_settle_steps: int = RESET_SETTLE_STEPS
    max_speed: float = MAX_SPEED

    def __post_init__(self) -> None:
        self.recurrent_cell = self.recurrent_cell.lower().strip()
        if self.recurrent_cell not in {"gru", "lstm"}:
            raise ValueError(f"Unsupported recurrent_cell: {self.recurrent_cell}")
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be > 0")
        if self.burn_in < 0:
            raise ValueError("burn_in must be >= 0")
        if self.burn_in >= self.sequence_length:
            self.burn_in = max(self.sequence_length - 1, 0)
        if self.sequence_stride <= 0:
            raise ValueError("sequence_stride must be > 0")
        if self.replay_capacity <= 0:
            raise ValueError("replay_capacity must be > 0")
        if self.replay_batch_size <= 0:
            raise ValueError("replay_batch_size must be > 0")
        if self.min_replay_sequences <= 0:
            raise ValueError("min_replay_sequences must be > 0")
        if self.recurrent_hidden_size is None:
            self.recurrent_hidden_size = self.hidden_size
        if self.target_entropy_scale <= 0.0:
            raise ValueError("target_entropy_scale must be greater than 0")
        if self.goal_stop_speed_threshold <= 0.0:
            raise ValueError("goal_stop_speed_threshold must be greater than 0")
        if self.slam_profile_interval <= 0:
            raise ValueError("slam_profile_interval must be greater than 0")
        if self.start_position is None:
            self.start_position = list(START_POSITION)
        if self.start_rotation is None:
            self.start_rotation = list(START_ROTATION)
        if self.reference_distance is None:
            start_xy = np.array(self.start_position[:2], dtype=np.float32)
            endpoint_xy = np.array(self.endpoint, dtype=np.float32)
            self.reference_distance = float(np.linalg.norm(start_xy - endpoint_xy))


class SequenceReplayBuffer:
    """
    Fixed-length recurrent replay buffer for efficient SAC off-policy updates.
    
    Stores variable-length episodes as overlapping fixed-size windows using a ring buffer.
    This approach enables efficient recurrent network processing by pre-allocating storage
    for sequences of observation-action pairs. Each episode is split into windows of
    length sequence_length with stride sequence_stride, allowing the recurrent network
    to maintain state continuity during learning.
    """

    def __init__(self, obs_size: int, action_dim: int, config: Config) -> None:
        self.capacity = config.replay_capacity
        self.seq_len = config.sequence_length
        self.stride = config.sequence_stride
        self.obs = np.zeros((self.capacity, self.seq_len, obs_size), dtype=np.float32)
        self.actions = np.zeros((self.capacity, self.seq_len, action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, self.seq_len, 1), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, self.seq_len, obs_size), dtype=np.float32)
        self.dones = np.zeros((self.capacity, self.seq_len, 1), dtype=np.float32)
        self.valid_mask = np.zeros((self.capacity, self.seq_len), dtype=np.float32)
        self.size = 0
        self.pos = 0

    def __len__(self) -> int:
        return self.size

    def _store_window(
        self,
        obs_window: np.ndarray,
        action_window: np.ndarray,
        reward_window: np.ndarray,
        next_obs_window: np.ndarray,
        done_window: np.ndarray,
    ) -> None:
        length = obs_window.shape[0]
        idx = self.pos
        self.obs[idx].fill(0.0)
        self.actions[idx].fill(0.0)
        self.rewards[idx].fill(0.0)
        self.next_obs[idx].fill(0.0)
        self.dones[idx].fill(0.0)
        self.valid_mask[idx].fill(0.0)

        self.obs[idx, :length] = obs_window
        self.actions[idx, :length] = action_window
        self.rewards[idx, :length, 0] = reward_window
        self.next_obs[idx, :length] = next_obs_window
        self.dones[idx, :length, 0] = done_window.astype(np.float32)
        self.valid_mask[idx, :length] = 1.0

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_episode(
        self,
        episode_obs: List[np.ndarray],
        episode_actions: List[np.ndarray],
        episode_rewards: List[float],
        episode_next_obs: List[np.ndarray],
        episode_dones: List[bool],
    ) -> None:
        if not episode_obs:
            return

        obs = np.asarray(episode_obs, dtype=np.float32)
        actions = np.asarray(episode_actions, dtype=np.float32)
        rewards = np.asarray(episode_rewards, dtype=np.float32)
        next_obs = np.asarray(episode_next_obs, dtype=np.float32)
        dones = np.asarray(episode_dones, dtype=np.bool_)

        total_len = obs.shape[0]
        start = 0
        while start < total_len:
            end = min(start + self.seq_len, total_len)
            self._store_window(
                obs[start:end],
                actions[start:end],
                rewards[start:end],
                next_obs[start:end],
                dones[start:end],
            )
            if end >= total_len:
                break
            start += self.stride

    def can_sample(self, batch_size: int, min_sequences: int) -> bool:
        return self.size >= max(batch_size, min_sequences)

    def sample(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs[indices], dtype=torch.float32, device=device),
            "actions": torch.as_tensor(self.actions[indices], dtype=torch.float32, device=device),
            "rewards": torch.as_tensor(self.rewards[indices], dtype=torch.float32, device=device),
            "next_obs": torch.as_tensor(self.next_obs[indices], dtype=torch.float32, device=device),
            "dones": torch.as_tensor(self.dones[indices], dtype=torch.float32, device=device),
            "valid_mask": torch.as_tensor(self.valid_mask[indices], dtype=torch.float32, device=device),
        }

# ============================================================================
# RECURRENT ENCODER AND POLICY/VALUE NETWORKS
# ============================================================================


class RecurrentEncoder(nn.Module):
    """
    Shared recurrent feature encoder for actor and critic networks.
    
    Processes observations through a trunk of dense layers followed by either GRU or LSTM.
    The recurrent core maintains hidden state across time, enabling the network to learn
    temporal patterns in the observation sequence. Episode boundaries (indicated by
    done_mask) reset the hidden state to prevent information leakage between episodes.
    """
    def __init__(self, obs_size: int, config: Config) -> None:
        super().__init__()
        self.cell = config.recurrent_cell
        self.hidden_size = config.recurrent_hidden_size or config.hidden_size
        self.recurrent_layers = config.recurrent_layers
        self.trunk = nn.Sequential(
            nn.Linear(obs_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, self.hidden_size),
            nn.ReLU(),
        )
        if self.cell == "gru":
            self.core: Optional[nn.Module] = nn.GRU(
                input_size=self.hidden_size,
                hidden_size=self.hidden_size,
                num_layers=self.recurrent_layers,
                batch_first=True,
            )
        elif self.cell == "lstm":
            self.core = nn.LSTM(
                input_size=self.hidden_size,
                hidden_size=self.hidden_size,
                num_layers=self.recurrent_layers,
                batch_first=True,
            )
        else:
            raise ValueError(f"Unsupported recurrent encoder cell: {self.cell}")

    def get_initial_state(self, batch_size: int, device: Optional[torch.device] = None) -> Optional[Any]:
        if self.core is None:
            return None
        if device is None:
            device = next(self.parameters()).device
        shape = (self.recurrent_layers, batch_size, self.hidden_size)
        if self.cell == "lstm":
            return torch.zeros(shape, device=device), torch.zeros(shape, device=device)
        return torch.zeros(shape, device=device)

    def _prepare_done_mask(
        self,
        done_mask: Optional[torch.Tensor],
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """
        Convert done_mask to a standard [B, T] tensor for state reset across batches and sequences.
        
        Handles three input formats: scalar (broadcast), [T] (apply to all batches), or [B, T].
        When a done flag is true, the recurrent state is reset at that timestep.
        """
        if done_mask is None:
            return None
        mask = torch.as_tensor(done_mask, dtype=torch.float32, device=device)
        if mask.ndim == 0:
            mask = mask.view(1, 1).expand(batch_size, seq_len)
        elif mask.ndim == 1:
            mask = mask.view(batch_size, seq_len)
        elif mask.ndim != 2:
            raise ValueError(f"done_mask must be scalar, [T], or [B, T], got shape {tuple(mask.shape)}")
        return mask

    def forward(
        self,
        obs: torch.Tensor,
        recurrent_state: Optional[Any] = None,
        done_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=next(self.parameters()).device)
        obs_tensor = torch.nan_to_num(obs_tensor, nan=0.0, posinf=1.0, neginf=-1.0)
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.view(1, 1, -1)
        elif obs_tensor.ndim == 2:
            obs_tensor = obs_tensor.unsqueeze(1)
        batch_size, seq_len = obs_tensor.shape[:2]
        latent = self.trunk(obs_tensor.reshape(batch_size * seq_len, -1)).reshape(batch_size, seq_len, -1)
        if self.core is None:
            return latent.squeeze(1) if seq_len == 1 else latent, None
        state = recurrent_state if recurrent_state is not None else self.get_initial_state(batch_size)
        mask = self._prepare_done_mask(done_mask, batch_size, seq_len, latent.device)

        outputs: List[torch.Tensor] = []
        if self.cell == "lstm":
            if not isinstance(state, tuple) or len(state) != 2:
                state = self.get_initial_state(batch_size, device=latent.device)
            state = cast(Tuple[torch.Tensor, torch.Tensor], state)
            h_t, c_t = state
            for t in range(seq_len):
                if mask is not None:
                    keep = (1.0 - mask[:, t]).view(1, batch_size, 1)
                    h_t = h_t * keep
                    c_t = c_t * keep
                step_out, (h_t, c_t) = self.core(latent[:, t : t + 1], (h_t, c_t))
                outputs.append(step_out)
            next_state: Any = (h_t, c_t)
        else:
            h_t = state
            for t in range(seq_len):
                if mask is not None:
                    keep = (1.0 - mask[:, t]).view(1, batch_size, 1)
                    h_t = h_t * keep
                step_out, h_t = self.core(latent[:, t : t + 1], h_t)
                outputs.append(step_out)
            next_state = h_t

        core_out = torch.cat(outputs, dim=1)
        return core_out.squeeze(1) if seq_len == 1 else core_out, next_state


class GaussianActor(nn.Module):
    """
    Stochastic policy network parameterized as a Gaussian distribution.
    
    Outputs mean and log-standard-deviation of an action distribution.
    The actual action is sampled via reparameterization and then squashed to
    environment action bounds using tanh. This enables gradient-based optimization
    of the policy through differentiable sampling.
    """
    def __init__(self, obs_size: int, action_dim: int, config: Config) -> None:
        super().__init__()
        self.encoder = RecurrentEncoder(obs_size, config)
        self.mean = nn.Linear(self.encoder.hidden_size, action_dim)
        self.log_std = nn.Linear(self.encoder.hidden_size, action_dim)

    def get_initial_state(self, batch_size: int, device: Optional[torch.device] = None) -> Optional[Any]:
        return self.encoder.get_initial_state(batch_size, device)

    def forward(
        self,
        obs: torch.Tensor,
        recurrent_state: Optional[Any] = None,
        done_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Any]]:
        features, next_state = self.encoder(obs, recurrent_state=recurrent_state, done_mask=done_mask)
        return self.mean(features), self.log_std(features), next_state


class QNetwork(nn.Module):
    """
    Critic network that estimates state-action value (Q-function).
    
    Concatenates encoded recurrent features with the action vector and passes
    through dense layers to produce a scalar Q-value estimate. SAC maintains
    two independent Q-networks to stabilize learning via double Q-learning.
    """
    def __init__(self, obs_size: int, action_dim: int, config: Config) -> None:
        super().__init__()
        self.encoder = RecurrentEncoder(obs_size, config)
        self.head = nn.Sequential(
            nn.Linear(self.encoder.hidden_size + action_dim, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 1),
        )

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        recurrent_state: Optional[Any] = None,
        done_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        features, next_state = self.encoder(obs, recurrent_state=recurrent_state, done_mask=done_mask)
        if features.ndim != action.ndim:
            if features.ndim == 2 and action.ndim == 3 and action.shape[1] == 1:
                action = action.squeeze(1)  # [B,1,A] -> [B,A] to match single-step features.
            elif features.ndim == 3 and action.ndim == 2:
                action = action.unsqueeze(1)  # [B,A] -> [B,1,A] to match sequence features.
        return self.head(torch.cat([features, action], dim=-1)), next_state


# ============================================================================
# SOFT ACTOR-CRITIC AGENT
# ============================================================================


class SACAgent:
    """
    Soft Actor-Critic implementation for continuous control in recurrent settings.
    
    Maintains separate networks for the stochastic policy (actor) and value estimation
    (critics Q1, Q2, and targets). The algorithm optimizes the policy to maximize
    expected return while maintaining entropy regularization for exploration. Entropy
    coefficient alpha is either fixed or automatically tuned via automatic entropy
    temperature adjustment.
    
    Sections:
    - Device and network initialization
    - State management
    - Checkpoint metadata and validation
    - Action sampling and squashing
    - Policy updates (actor and critics)
    - Soft target updates
    """
    def __init__(self, obs_size: int, action_dim: int, config: Config) -> None:
        self.config = config
        self.device = self._get_device()
        self.obs_size = obs_size
        self.action_dim = action_dim

        self.actor = GaussianActor(obs_size, action_dim, config).to(self.device)
        with torch.no_grad():
            if self.actor.mean.bias is not None and action_dim > 1:
                # Mild low-speed bias: keep early rollouts controllable without freezing motion.
                self.actor.mean.bias[1] = -0.8
        self.q1 = QNetwork(obs_size, action_dim, config).to(self.device)
        self.q2 = QNetwork(obs_size, action_dim, config).to(self.device)
        self.target_q1 = QNetwork(obs_size, action_dim, config).to(self.device)
        self.target_q2 = QNetwork(obs_size, action_dim, config).to(self.device)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=config.critic_lr)
        self.log_alpha = torch.tensor(np.log(config.initial_alpha), dtype=torch.float32, device=self.device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.alpha_lr)
        self.target_entropy = -float(action_dim) * float(config.target_entropy_scale)

        self.action_low = torch.tensor([-config.max_steering_angle, config.min_speed], dtype=torch.float32, device=self.device)
        self.action_high = torch.tensor([config.max_steering_angle, config.max_speed], dtype=torch.float32, device=self.device)
        self.action_center = (self.action_high + self.action_low) / 2.0
        self.action_scale = (self.action_high - self.action_low) / 2.0

    def _get_device(self) -> torch.device:
        if self.config.force_cpu or not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device("cuda")

    def get_initial_state(self, batch_size: int = 1) -> Optional[Any]:
        return self.actor.get_initial_state(batch_size, self.device)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def _checkpoint_metadata(self) -> Dict[str, Any]:
        return {
            "obs_size": self.obs_size,
            "action_dim": self.action_dim,
            "architecture": {
                "hidden_size": self.config.hidden_size,
                "recurrent_cell": self.config.recurrent_cell,
                "recurrent_hidden_size": self.config.recurrent_hidden_size,
                "recurrent_layers": self.config.recurrent_layers,
                "log_std_min": self.config.log_std_min,
                "log_std_max": self.config.log_std_max,
                "max_steering_angle": self.config.max_steering_angle,
                "min_speed": self.config.min_speed,
                "max_speed": self.config.max_speed,
            },
            "entropy": {
                "auto_entropy_tuning": self.config.auto_entropy_tuning,
                "initial_alpha": self.config.initial_alpha,
                "alpha_lr": self.config.alpha_lr,
                "target_entropy_scale": self.config.target_entropy_scale,
            },
            "optimization": {
                "reward_scale": REWARD_SCALE,
            },
        }

    def _validate_checkpoint_metadata(self, checkpoint: Dict[str, Any]) -> None:
        if int(checkpoint.get("obs_size", self.obs_size)) != self.obs_size:
            raise ValueError(
                f"Checkpoint observation size {checkpoint.get('obs_size')} does not match current observation size {self.obs_size}."
            )
        if int(checkpoint.get("action_dim", self.action_dim)) != self.action_dim:
            raise ValueError(
                f"Checkpoint action dim {checkpoint.get('action_dim')} does not match current action dim {self.action_dim}."
            )

        saved_architecture = checkpoint.get("architecture")
        if isinstance(saved_architecture, dict):
            current_architecture = self._checkpoint_metadata()["architecture"]
            mismatches: List[str] = []
            for key, expected_value in current_architecture.items():
                if key not in saved_architecture:
                    continue
                if saved_architecture[key] != expected_value:
                    mismatches.append(f"{key}: checkpoint={saved_architecture[key]}, current={expected_value}")
            if mismatches:
                raise ValueError(
                    "Checkpoint architecture is incompatible with the current configuration: " + "; ".join(mismatches)
                )

        saved_entropy = checkpoint.get("entropy")
        if isinstance(saved_entropy, dict):
            current_entropy = self._checkpoint_metadata()["entropy"]
            mismatches = []
            for key, expected_value in current_entropy.items():
                if key not in saved_entropy:
                    continue
                if saved_entropy[key] != expected_value:
                    mismatches.append(f"{key}: checkpoint={saved_entropy[key]}, current={expected_value}")
            if mismatches:
                raise ValueError(
                    "Checkpoint entropy settings are incompatible with the current configuration: " + "; ".join(mismatches)
                )

        saved_optimization = checkpoint.get("optimization")
        if isinstance(saved_optimization, dict):
            current_optimization = self._checkpoint_metadata()["optimization"]
            mismatches = []
            for key, expected_value in current_optimization.items():
                if key not in saved_optimization:
                    continue
                if saved_optimization[key] != expected_value:
                    mismatches.append(f"{key}: checkpoint={saved_optimization[key]}, current={expected_value}")
            if mismatches:
                raise ValueError(
                    "Checkpoint optimization settings are incompatible with the current configuration: "
                    + "; ".join(mismatches)
                )

    def _tensor_obs(self, obs: np.ndarray) -> torch.Tensor:
        """Convert environment observation to batched tensor format for policy network."""
        return torch.as_tensor(obs, dtype=torch.float32, device=self.device).view(1, -1)

    def _squash_action(self, action: torch.Tensor) -> torch.Tensor:
        """Rescale action from [-1, 1] to environment bounds."""
        return action * self.action_scale + self.action_center

    def _sample_policy(
        self,
        obs: torch.Tensor,
        recurrent_state: Optional[Any] = None,
        done_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Any]]:
        """
        Sample action from the policy and compute its log probability.
        
        When deterministic=False, samples via reparameterization. Log probability is adjusted
        for the tanh squashing using the Jacobian correction to account for the bounded action space.
        """
        mean, log_std, next_state = self.actor(obs, recurrent_state=recurrent_state, done_mask=done_mask)
        log_std = log_std.clamp(self.config.log_std_min, self.config.log_std_max)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        pre_tanh = mean if deterministic else dist.rsample()
        tanh_action = torch.tanh(pre_tanh)
        log_prob = dist.log_prob(pre_tanh)
        log_prob -= torch.log(self.action_scale + 1e-6)
        log_prob -= torch.log(1.0 - tanh_action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return self._squash_action(tanh_action), log_prob, next_state

    def select_action(
        self,
        obs: np.ndarray,
        recurrent_state: Optional[Any] = None,
        done: bool = False,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Any]]:
        """
        Select a single action for environment interaction.
        
        Wraps numpy observation as a batch tensor, queries the policy, and returns
        action as numpy along with the updated recurrent state.
        """
        done_mask = torch.tensor([float(done)], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action, _, next_state = self._sample_policy(
                self._tensor_obs(obs),
                recurrent_state=recurrent_state,
                done_mask=done_mask,
                deterministic=deterministic,
            )
        return action.squeeze(0).cpu().numpy(), next_state

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        """Update target network via exponential moving average with the source network."""
        tau = self.config.tau
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)

    @staticmethod
    def _sequence_loss_mask(valid_mask: torch.Tensor, burn_in: int) -> torch.Tensor:
        """
        Create a mask for loss computation, zeroing out burn-in steps in each sequence.
        
        Burn-in allows the recurrent network to warm up its hidden state before learning
        begins. The mask respects sequence boundaries (where valid_mask indicates valid timesteps).
        """
        valid_lengths = valid_mask.sum(dim=1).to(dtype=torch.long)
        burn_tensor = torch.full_like(valid_lengths, burn_in)
        start_index = torch.minimum(burn_tensor, torch.clamp(valid_lengths - 1, min=0))
        time_index = torch.arange(valid_mask.shape[1], device=valid_mask.device).unsqueeze(0)
        return valid_mask * (time_index >= start_index.unsqueeze(1)).to(dtype=valid_mask.dtype)

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute mean of values over valid (masked) elements."""
        weight = mask.unsqueeze(-1).to(dtype=values.dtype)
        return (values * weight).sum() / weight.sum().clamp_min(1.0)

    def _build_episode_batch(
        self,
        episode_obs: List[np.ndarray],
        episode_actions: List[np.ndarray],
        episode_rewards: List[float],
        episode_next_obs: List[np.ndarray],
        episode_dones: List[bool],
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Convert a complete episode to a tensor batch suitable for network forward pass.
        
        Adds a batch dimension and converts all numpy arrays to torch tensors.
        Returns None if the episode is empty.
        """
        if not episode_obs:
            return None

        obs = np.asarray(episode_obs, dtype=np.float32)
        actions = np.asarray(episode_actions, dtype=np.float32)
        rewards = np.asarray(episode_rewards, dtype=np.float32).reshape(-1, 1)
        next_obs = np.asarray(episode_next_obs, dtype=np.float32)
        dones = np.asarray(episode_dones, dtype=np.float32).reshape(-1, 1)
        valid_mask = np.ones((obs.shape[0],), dtype=np.float32)

        return {
            "obs": torch.as_tensor(obs[None, ...], dtype=torch.float32, device=self.device),
            "actions": torch.as_tensor(actions[None, ...], dtype=torch.float32, device=self.device),
            "rewards": torch.as_tensor(rewards[None, ...], dtype=torch.float32, device=self.device),
            "next_obs": torch.as_tensor(next_obs[None, ...], dtype=torch.float32, device=self.device),
            "dones": torch.as_tensor(dones[None, ...], dtype=torch.float32, device=self.device),
            "valid_mask": torch.as_tensor(valid_mask[None, ...], dtype=torch.float32, device=self.device),
        }

    def _build_update_masks(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Construct masks for loss computation and recurrent state resets.
        
        Returns:
        - learn_mask: Mask for computing loss (zeros out burn-in steps)
        - done_mask_obs: Reset mask for current observations
        - done_mask_next: Reset mask for next observations
        """
        valid_mask = batch["valid_mask"]
        burn_in = min(self.config.burn_in, batch["obs"].shape[1] - 1)
        learn_mask = self._sequence_loss_mask(valid_mask, burn_in)
        done_flags = batch["dones"].squeeze(-1)
        done_mask_obs = torch.zeros_like(done_flags)
        done_mask_obs[:, 0] = 1.0
        if done_flags.shape[1] > 1:
            done_mask_obs[:, 1:] = done_flags[:, :-1]
        done_mask_next = torch.cat([torch.ones_like(done_flags[:, :1]), done_flags], dim=1)
        return learn_mask, done_mask_obs, done_mask_next

    def _update_critics(
        self,
        batch: Dict[str, torch.Tensor],
        learn_mask: torch.Tensor,
        done_mask_obs: torch.Tensor,
        done_mask_next: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute critic loss and update Q-networks via temporal difference (TD) error.
        
        Computes the target Q-value using the target networks and next action samples,
        then minimizes MSE between current Q-estimates and the target. Entropy is included
        in the target to encourage exploration.
        """
        scaled_rewards = batch["rewards"] * REWARD_SCALE

        def _ensure_time_dim(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.unsqueeze(1) if tensor.ndim == 2 else tensor

        with torch.no_grad():
            target_obs = torch.cat([batch["obs"][:, :1], batch["next_obs"]], dim=1)
            next_action, next_log_prob, _ = self._sample_policy(target_obs, done_mask=done_mask_next, deterministic=False)
            next_action = _ensure_time_dim(next_action)[:, 1:]
            next_log_prob = _ensure_time_dim(next_log_prob)[:, 1:]
            target_actions = torch.cat([torch.zeros_like(batch["actions"][:, :1]), next_action], dim=1)
            target_q1, _ = self.target_q1(target_obs, target_actions, done_mask=done_mask_next)
            target_q2, _ = self.target_q2(target_obs, target_actions, done_mask=done_mask_next)
            target_q = torch.min(_ensure_time_dim(target_q1), _ensure_time_dim(target_q2))[:, 1:]
            target_q = target_q - self.alpha.detach() * next_log_prob
            target_q = scaled_rewards + (1.0 - batch["dones"]) * self.config.gamma * target_q

        current_q1, _ = self.q1(batch["obs"], batch["actions"], done_mask=done_mask_obs)
        current_q2, _ = self.q2(batch["obs"], batch["actions"], done_mask=done_mask_obs)
        target_q = _ensure_time_dim(target_q)
        current_q1 = _ensure_time_dim(current_q1)
        current_q2 = _ensure_time_dim(current_q2)

        critic_loss = self._masked_mean((current_q1 - target_q).pow(2), learn_mask)
        return critic_loss + self._masked_mean((current_q2 - target_q).pow(2), learn_mask)

    def _update_actor_and_alpha(
        self,
        batch: Dict[str, torch.Tensor],
        learn_mask: torch.Tensor,
        done_mask_obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update policy (actor) and entropy temperature (alpha).
        
        Actor loss encourages the policy to take actions that maximize Q-values
        while maintaining controlled entropy. Alpha is tuned to keep actual entropy
        near the target entropy via automatic entropy temperature adjustment.
        """
        new_action, log_prob, _ = self._sample_policy(batch["obs"], done_mask=done_mask_obs, deterministic=False)
        q1_pi, _ = self.q1(batch["obs"], new_action, done_mask=done_mask_obs)
        q2_pi, _ = self.q2(batch["obs"], new_action, done_mask=done_mask_obs)

        def _ensure_time_dim(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.unsqueeze(1) if tensor.ndim == 2 else tensor

        log_prob = _ensure_time_dim(log_prob)
        q1_pi = _ensure_time_dim(q1_pi)
        q2_pi = _ensure_time_dim(q2_pi)
        actor_loss = self._masked_mean(self.alpha.detach() * log_prob - torch.min(q1_pi, q2_pi), learn_mask)

        alpha_loss = torch.tensor(0.0, device=self.device)
        if self.config.auto_entropy_tuning:
            alpha_loss = -self._masked_mean(self.log_alpha * (log_prob + self.target_entropy).detach(), learn_mask)

        return actor_loss, alpha_loss

    def update(self, batch: Dict[str, torch.Tensor]) -> Optional[Dict[str, float]]:
        """
        Perform a complete SAC update step: update both critics and the actor.
        
        Applies gradient clipping and soft target network updates. Returns a dictionary
        of logged metrics (losses and entropy coefficient) or None if the batch is empty.
        """
        if batch["obs"].shape[1] <= 0:
            return None

        learn_mask, done_mask_obs, done_mask_next = self._build_update_masks(batch)
        critic_loss = self._update_critics(batch, learn_mask, done_mask_obs, done_mask_next)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), max_norm=1.0)
        self.critic_optimizer.step()

        actor_loss, alpha_loss = self._update_actor_and_alpha(batch, learn_mask, done_mask_obs)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        if self.config.auto_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self._soft_update(self.q1, self.target_q1)
        self._soft_update(self.q2, self.target_q2)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.item()),
        }

    def checkpoint(self, episode: Any, reward: float) -> Dict[str, Any]:
        """Create a checkpoint dictionary containing all model and optimizer state."""
        checkpoint = {    
            "episode": episode,
            "reward": reward,
            "algorithm": "sac",
            "actor": self.actor.state_dict(),
            "critic1": self.q1.state_dict(),
            "critic2": self.q2.state_dict(),
            "target_critic1": self.target_q1.state_dict(),
            "target_critic2": self.target_q2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
        }
        checkpoint.update(self._checkpoint_metadata())
        checkpoint["config"] = asdict(self.config)
        return checkpoint

    def save(self, path: str, episode: Any, reward: float) -> None:
        """Save checkpoint to disk."""
        torch.save(self.checkpoint(episode, reward), path)

    def load(self, path: str) -> Dict[str, Any]:
        """Load checkpoint from disk and restore all model and optimizer state."""
        checkpoint = _load_checkpoint(path, map_location=self.device)
        checkpoint_algorithm = str(checkpoint.get("algorithm", "sac")).lower().strip()
        if checkpoint_algorithm != "sac":
            raise ValueError(f"Checkpoint algorithm '{checkpoint_algorithm}' does not match SAC.")
        self._validate_checkpoint_metadata(checkpoint)
        self.actor.load_state_dict(checkpoint["actor"])
        self.q1.load_state_dict(checkpoint["critic1"])
        self.q2.load_state_dict(checkpoint["critic2"])
        self.target_q1.load_state_dict(checkpoint.get("target_critic1", checkpoint["critic1"]))
        self.target_q2.load_state_dict(checkpoint.get("target_critic2", checkpoint["critic2"]))
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        if self.config.auto_entropy_tuning and "alpha_optimizer" in checkpoint:
            self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
        if "log_alpha" in checkpoint:
            self.log_alpha.data.copy_(checkpoint["log_alpha"].to(self.device))
        return checkpoint


# ============================================================================
# TRAINING LOOP
# ============================================================================


def train(config: Optional[Config] = None) -> None:
    """
    Main training loop integrating environment interaction, replay buffer, and optimization.
    
    Collects experience by rolling out the current policy in the Webots simulator,
    stores transitions in the replay buffer, and performs off-policy SAC updates.
    Maintains metrics for progress tracking and saves checkpoints for model evaluation.
    """
    if config is None:
        config = Config()

    _init_supervisor()
    env = WebotsEnv(config)
    env.reset()
    run_id = Path(env.run_folder).name
    agent = SACAgent(env.observation_size, env.action_dim, config)
    replay = SequenceReplayBuffer(env.observation_size, env.action_dim, config)
    checkpoint_dir = _run_checkpoint_dir(run_id)
    final_model_path = _run_checkpoint_path(run_id, "final")
    print(
        f"[TRAIN][SAC] rnn={config.recurrent_cell.upper()} "
        f"weights_dir={checkpoint_dir} final={final_model_path}",
        flush=True,
    )
    print(
        f"[TRAIN][SAC] replay=on cap={config.replay_capacity} seq={config.sequence_length} "
        f"stride={config.sequence_stride} batch={config.replay_batch_size}",
        flush=True,
    )

    total_steps = 0
    best_reward = float("-inf")
    best_goal_reward = float("-inf")
    best_goal_episode: Optional[int] = None
    reward_window: List[float] = []
    success_window: List[float] = []
    goal_touch_window: List[float] = []
    collision_window: List[float] = []
    timeout_window: List[float] = []
    start_time = time.perf_counter()
    metrics_logger = MetricsLogger(env.run_folder, algorithm="sac")

    for episode in range(config.episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        episode_end_reason = "max_steps"
        episode_obs: List[np.ndarray] = []
        episode_actions: List[np.ndarray] = []
        episode_rewards: List[float] = []
        episode_next_obs: List[np.ndarray] = []
        episode_dones: List[bool] = []
        episode_goal_reached = False
        episode_success = False
        actor_state = agent.get_initial_state(batch_size=1)
        prev_done = True

        while not done:
            action, actor_state = agent.select_action(
                obs,
                recurrent_state=actor_state,
                done=prev_done,
                deterministic=False,
            )

            next_obs, reward, terminated, truncated, info = env.step(action)
            # Only mark done on true termination (collision, goal).  Timeout
            # (truncated) is NOT a terminal state — the Bellman target must still
            # bootstrap.  Using (terminated or truncated) here was cutting the
            # Q-value estimate to zero at every episode timeout, preventing the
            # critic from propagating credit across episode boundaries.
            transition_done = bool(terminated)
            episode_obs.append(np.asarray(obs, dtype=np.float32))
            episode_actions.append(np.asarray(action, dtype=np.float32))
            episode_rewards.append(float(reward))
            episode_next_obs.append(np.asarray(next_obs, dtype=np.float32))
            episode_dones.append(transition_done)

            obs = next_obs
            episode_reward += reward
            episode_goal_reached = episode_goal_reached or bool(info.get("goal_reached", False))
            episode_success = bool(info.get("success", False))
            done = transition_done
            prev_done = done
            total_steps += 1

            if done:
                if info.get("reset_reason") == "low_score":
                    episode_end_reason = "low_score"
                elif info.get("reset_reason") == "collision":
                    episode_end_reason = "collision"
                elif info.get("reset_reason") == "goal":
                    episode_end_reason = "goal"
                elif truncated:
                    episode_end_reason = "max_steps"

        replay.add_episode(
            episode_obs,
            episode_actions,
            episode_rewards,
            episode_next_obs,
            episode_dones,
        )

        if total_steps >= config.update_after_steps and replay.can_sample(
            config.replay_batch_size,
            config.min_replay_sequences,
        ):
            # One gradient step per environment step collected this episode
            # (scaled by updates_per_step).  The old formula divided the full
            # replay capacity by the batch size, which grew to 512+ steps/episode
            # as the buffer filled — far too many updates, causing overfitting.
            num_updates = max(1, len(episode_obs)) * config.updates_per_step
            for _ in range(num_updates):
                batch = replay.sample(config.replay_batch_size, agent.device)
                agent.update(batch)

        reward_window.append(episode_reward)
        success_window.append(1.0 if episode_success else 0.0)
        goal_touch_window.append(1.0 if episode_goal_reached else 0.0)
        collision_window.append(1.0 if episode_end_reason == "collision" else 0.0)
        timeout_window.append(1.0 if episode_end_reason == "max_steps" else 0.0)
        checkpoint_flags: List[str] = []

        if episode_end_reason == "goal":
            if episode_reward > best_goal_reward:
                best_goal_reward = episode_reward
                best_goal_episode = episode + 1
                env.robot.slam.save_episode(env.run_folder, episode + 1, episode_reward)
                checkpoint = agent.checkpoint(best_goal_episode, best_goal_reward)
                checkpoint["goal_episode"] = True
                _save_checkpoint_file(_CHECKPOINT_DIR, run_id, "best", checkpoint)
                checkpoint_flags.append("best_goal")
        elif best_goal_episode is None and episode_reward > best_reward:
            best_reward = episode_reward
            env.robot.slam.save_episode(env.run_folder, episode + 1, episode_reward)
            checkpoint = agent.checkpoint(episode + 1, best_reward)
            checkpoint["goal_episode"] = False
            _save_checkpoint_file(_CHECKPOINT_DIR, run_id, "best", checkpoint)
            checkpoint_flags.append("best")

        if config.save_every > 0 and (episode + 1) % config.save_every == 0:
            latest_checkpoint = agent.checkpoint(episode + 1, episode_reward)
            latest_checkpoint["goal_episode"] = episode_end_reason == "goal"
            _save_checkpoint_file(_CHECKPOINT_DIR, run_id, "checkpoint", latest_checkpoint)
            checkpoint_flags.append("latest")

        rolling_reward = float(np.mean(reward_window[-10:]))
        rolling_success = float(np.mean(success_window[-10:]))
        rolling_goal_touch = float(np.mean(goal_touch_window[-10:]))
        rolling_collision = float(np.mean(collision_window[-10:]))
        rolling_timeout = float(np.mean(timeout_window[-10:]))
        elapsed = time.perf_counter() - start_time
        checkpoint_note = f" ckpt={'+'.join(checkpoint_flags)}" if checkpoint_flags else ""
        print(
            f"[TRAIN][SAC] ep={episode + 1:03d}/{config.episodes} "
            f"r={episode_reward:8.2f} avg10={rolling_reward:8.2f} steps={env.current_step:4d} "
            f"succ10={rolling_success:4.2f} touch10={rolling_goal_touch:4.2f} "
            f"col10={rolling_collision:4.2f} to10={rolling_timeout:4.2f} "
            f"min_d={env.min_episode_distance:5.2f} end={episode_end_reason} replay={len(replay):4d} "
            f"t={elapsed:7.1f}s{checkpoint_note}",
            flush=True,
        )
        metrics_logger.log(
            episode=episode + 1,
            reward=round(episode_reward, 4),
            avg10=round(rolling_reward, 4),
            steps=env.current_step,
            success=int(episode_success),
            goal_touched=int(episode_goal_reached),
            collision=int(episode_end_reason == "collision"),
            timeout=int(episode_end_reason == "max_steps"),
            min_dist=round(env.min_episode_distance, 4),
            end_reason=episode_end_reason,
            replay_size=len(replay),
            elapsed_s=round(elapsed, 1),
        )

    metrics_logger.close()
    print(f"[TRAIN][SAC] metrics saved to {metrics_logger.path}", flush=True)

    final_reward = best_goal_reward if best_goal_episode is not None else best_reward
    agent.save(final_model_path, "final", final_reward)
    elapsed = time.perf_counter() - start_time
    print(f"[TRAIN][SAC] final reward={final_reward:.2f} t={elapsed:7.1f}s", flush=True)

    env.robot.motors.stop()
    print("[TRAIN][SAC] done", flush=True)


if __name__ == "__main__":
    train()
