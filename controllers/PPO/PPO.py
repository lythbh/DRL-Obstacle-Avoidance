"""PPO training controller for ALTINO robot in Webots obstacle avoidance task."""
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any, Union
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn.utils.rnn import pad_sequence

RecurrentState = Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from controllers.RNN import GRUActorCritic, LSTMActorCritic
from controllers.Webots.webots_env import WebotsEnv, _init_supervisor
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
from controllers.common.training_defaults import PPODefaults, RecurrentDefaults

_CONTROLLER_DIR = Path(__file__).resolve().parent
_CHECKPOINT_DIR = _CONTROLLER_DIR / "checkpoints"

# Use shared checkpoint helpers to avoid duplication across controllers.
from controllers.common.checkpoints import (
    checkpoint_path as _shared_checkpoint_path,
    run_checkpoint_dir as _shared_run_checkpoint_dir,
    run_checkpoint_path as _shared_run_checkpoint_path,
    load_checkpoint as _shared_load_checkpoint,
    make_checkpoint_header as _make_checkpoint_header,
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


def _load_checkpoint(path: str, map_location: Union[str, torch.device]) -> Dict[str, Any]:
    """Controller-local wrapper around shared `load_checkpoint` helper."""
    return _shared_load_checkpoint(path, map_location)



# ============================================================================
# CONFIGURATION
# ============================================================================


"""
Sections:
 - Config: Training and environment hyperparameters
 - PPOAgent: Recurrent actor-critic model with policy and value updates
 - train(): Main training loop integrating environment, rollout, and optimization
"""

@dataclass
class Config:
    """
    Training and environment hyperparameters for PPO.
    
    Consolidates all hyperparameters for the actor-critic optimization algorithm
    (learning rates, batch size, epochs, entropy coefficient) and environment
    configuration (reward shaping, action bounds, episode setup). Validation in
    __post_init__ ensures internal consistency.
    """
    
    episodes: int = PPODefaults.episodes
    update_every: int = PPODefaults.update_every
    epochs: int = PPODefaults.epochs
    batch_size: int = PPODefaults.batch_size
    save_every: int = PPODefaults.save_every
    
    gamma: float = 0.99
    gae_lambda: float = PPODefaults.gae_lambda
    epsilon: float = 0.2
    learning_rate: float = PPODefaults.learning_rate
    entropy_coef: float = PPODefaults.entropy_coef
    hidden_size: int = PPODefaults.hidden_size
    latent_size: int = PPODefaults.latent_size
    lstm_hidden_size: int = PPODefaults.lstm_hidden_size
    lstm_layers: int = PPODefaults.lstm_layers
    recurrent_cell: str = PPODefaults.recurrent_cell
    sequence_length: int = RecurrentDefaults.sequence_length
    burn_in: int = RecurrentDefaults.burn_in
    sequence_stride: int = RecurrentDefaults.sequence_stride
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
    
    max_speed: float = MAX_SPEED
    reset_settle_steps: int = RESET_SETTLE_STEPS
    
    def __post_init__(self) -> None:
        """Initialize defaults for mutable fields."""
        self.recurrent_cell = self.recurrent_cell.lower().strip()
        if self.recurrent_cell not in {"lstm", "gru"}:
            raise ValueError(f"Unsupported recurrent_cell: {self.recurrent_cell}")
        if self.episodes <= 0:
            raise ValueError("episodes must be greater than 0")
        if self.update_every <= 0:
            raise ValueError("update_every must be greater than 0")
        if self.epochs <= 0:
            raise ValueError("epochs must be greater than 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be greater than 0")
        if self.burn_in < 0:
            raise ValueError("burn_in must be non-negative")
        if self.burn_in >= self.sequence_length:
            raise ValueError("burn_in must be smaller than sequence_length")
        if self.sequence_stride <= 0:
            raise ValueError("sequence_stride must be greater than 0")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be greater than 0")
        if self.max_steering_angle <= 0.0:
            raise ValueError("max_steering_angle must be greater than 0")
        if self.max_speed <= self.min_speed:
            raise ValueError("max_speed must be greater than min_speed")
        if self.save_every < 0:
            raise ValueError("save_every must be non-negative")
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




# ============================================================================
# PROXIMAL POLICY OPTIMIZATION AGENT
# ============================================================================


class PPOAgent:
    """
    Proximal Policy Optimization (PPO) agent with recurrent actor-critic network.
    
    Maintains a shared recurrent feature encoder with separate policy and value heads.
    Optimizes the policy using the PPO algorithm: clipped surrogate objective for stability
    and shared critic loss for value estimation. Entropy regularization encourages exploration.
    
    Sections:
    - Model construction and device management
    - Checkpoint metadata and validation
    - Action bounds and policy distribution
    - Action sampling for rollouts
    - Trajectory processing and advantage computation
    - Policy evaluation and optimization
    """

    def __init__(self, obs_size: int, action_dim: int, config: Config):
        self.config = config
        self.device = self._get_device()
        self.action_dim = action_dim
        self.obs_size = obs_size
        self._build_model(self.config.recurrent_cell)
        print(f"[PPO] Using recurrent cell: {self.config.recurrent_cell.upper()}", flush=True)

    def _build_model(self, recurrent_cell: str) -> None:
        recurrent_cell = recurrent_cell.lower().strip()
        model_class = GRUActorCritic if recurrent_cell == "gru" else LSTMActorCritic
        self.model = model_class(self.obs_size, self.action_dim, self.config).to(self.device)
        self.actor = self.model.policy_head
        with torch.no_grad():
            if self.actor.bias is not None and self.action_dim > 1:
                # Mild low-speed bias: keep early rollouts controllable without freezing motion.
                self.actor.bias[1] = -0.8
        self.actor_log_std = nn.Parameter(torch.full((self.action_dim,), -0.5, dtype=torch.float32, device=self.device))
        params = list(self.model.parameters()) + [self.actor_log_std]
        self.optimizer = torch.optim.Adam(params, lr=self.config.learning_rate)

    def _get_device(self) -> torch.device:
        """Get appropriate device (CUDA when available, unless force_cpu is set)."""
        if self.config.force_cpu or not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device("cuda")

    def get_initial_state(self, batch_size: int = 1) -> RecurrentState:
        """Expose recurrent state initialization for rollouts."""
        return self.model.get_initial_state(batch_size, device=self.device)

    def _checkpoint_metadata(self) -> Dict[str, Any]:
        """Return architecture metadata needed to validate checkpoints."""
        return {
            "obs_size": self.obs_size,
            "action_dim": self.action_dim,
            "recurrent_cell": self.config.recurrent_cell,
            "architecture": {
                "hidden_size": self.config.hidden_size,
                "latent_size": self.config.latent_size,
                "lstm_hidden_size": self.config.lstm_hidden_size,
                "lstm_layers": self.config.lstm_layers,
                "lidar_sector_dim": self.config.lidar_sector_dim,
                "pose_goal_dim": self.config.pose_goal_dim,
                "imu_feature_dim": self.config.imu_feature_dim,
                "occupancy_grid_shape": (
                    tuple(self.config.occupancy_grid_shape)
                    if self.config.occupancy_grid_shape is not None
                    else None
                ),
            },
        }

    def _validate_checkpoint_metadata(self, checkpoint: Dict[str, Any]) -> None:
        """Fail fast when a checkpoint was trained with incompatible dimensions."""
        saved_obs_size = checkpoint.get("obs_size")
        if saved_obs_size is not None and int(saved_obs_size) != self.obs_size:
            raise ValueError(
                f"Checkpoint observation size {saved_obs_size} does not match current observation size {self.obs_size}."
            )

        saved_action_dim = checkpoint.get("action_dim")
        if saved_action_dim is not None and int(saved_action_dim) != self.action_dim:
            raise ValueError(
                f"Checkpoint action dim {saved_action_dim} does not match current action dim {self.action_dim}."
            )

        saved_architecture = checkpoint.get("architecture")
        if not isinstance(saved_architecture, dict):
            return

        current_architecture = self._checkpoint_metadata()["architecture"]
        mismatches: List[str] = []
        for key, expected_value in current_architecture.items():
            if key not in saved_architecture:
                continue
            actual_value = saved_architecture[key]
            if key == "occupancy_grid_shape" and actual_value is not None:
                actual_value = tuple(actual_value)
            if actual_value != expected_value:
                mismatches.append(f"{key}: checkpoint={actual_value}, current={expected_value}")

        if mismatches:
            mismatch_text = "; ".join(mismatches)
            raise ValueError(f"Checkpoint architecture is incompatible with the current configuration: {mismatch_text}")

    def _action_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get per-dimension action bounds."""
        low = torch.tensor(
            [-self.config.max_steering_angle, self.config.min_speed],
            dtype=torch.float32,
            device=self.device,
        )
        high = torch.tensor(
            [self.config.max_steering_angle, self.config.max_speed],
            dtype=torch.float32,
            device=self.device,
        )
        return low, high

    def _action_affine(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action bounds and affine transform parameters."""
        low, high = self._action_bounds()
        center = (high + low) / 2.0
        scale = (high - low) / 2.0
        return low, high, center, scale

    def _policy_stats(self, policy_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract mean and log-standard deviation from policy network output.
        
        Includes numerical stability measures: clamping log-std to valid bounds
        and replacing NaN/inf values with defaults.
        """
        policy_output = torch.nan_to_num(policy_output, nan=0.0, posinf=1.0, neginf=-1.0)
        safe_actor_log_std = torch.nan_to_num(self.actor_log_std, nan=-0.5, posinf=2.0, neginf=-5.0)
        safe_actor_log_std = safe_actor_log_std.clamp(-5.0, 2.0)
        log_std = safe_actor_log_std.expand_as(policy_output)
        std = torch.nan_to_num(log_std.exp(), nan=1.0, posinf=7.5, neginf=1e-3).clamp_min(1e-3)
        return policy_output, std

    def _squash_action(self, pre_tanh_action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Map pre-tanh actions to bounded environment actions."""
        low, high, center, scale = self._action_affine()
        squashed = torch.tanh(pre_tanh_action)
        action = squashed * scale + center
        eps = 1e-5
        action = torch.max(torch.min(action, high - eps), low + eps)
        return action, squashed

    def _log_prob_from_pre_tanh(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        pre_tanh_action: torch.Tensor,
        squashed_action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute stable log-probability for tanh-squashed Gaussian actions."""
        if squashed_action is None:
            squashed_action = torch.tanh(pre_tanh_action)
        _, _, _, scale = self._action_affine()
        eps = 1e-6
        dist = Normal(mean, std)
        log_prob = dist.log_prob(pre_tanh_action)
        log_prob -= torch.log(scale + eps)
        log_prob -= torch.log(1.0 - squashed_action.pow(2) + eps)
        log_prob = log_prob.sum(dim=-1)
        return torch.nan_to_num(log_prob, nan=0.0, posinf=0.0, neginf=0.0)

    def _inverse_squash_action(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert bounded actions back to pre-tanh space with safe clipping."""
        low, high, center, scale = self._action_affine()
        eps = 1e-6
        safe_action = torch.max(torch.min(action, high - 1e-5), low + 1e-5)
        squashed = ((safe_action - center) / (scale + eps)).clamp(-1.0 + eps, 1.0 - eps)
        pre_tanh = 0.5 * (torch.log1p(squashed) - torch.log1p(-squashed))
        pre_tanh = torch.nan_to_num(pre_tanh, nan=0.0, posinf=5.0, neginf=-5.0)
        return pre_tanh, squashed

    def _sample_action_and_log_prob(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        deterministic: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample bounded action and compute its stable log probability."""
        dist = Normal(mean, std)
        pre_tanh = mean if deterministic else dist.rsample()
        action, squashed = self._squash_action(pre_tanh)
        log_prob = self._log_prob_from_pre_tanh(mean, std, pre_tanh, squashed_action=squashed)
        return action, log_prob

    def select_action(
        self,
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        recurrent_state: Optional[RecurrentState] = None,
        done: bool = False,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor, RecurrentState]:
        """
        Sample an action from the policy for environment interaction.
        
        Returns action as numpy array and state value for advantage estimation,
        along with log probability for policy gradient computation.
        """
        done_mask = torch.tensor([float(done)], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            policy_output, state_value, next_state = self.model(
                obs,
                recurrent_state=recurrent_state,
                done_mask=done_mask,
            )
            mean, std = self._policy_stats(policy_output)
            action, log_prob = self._sample_action_and_log_prob(mean, std, deterministic=deterministic)
        return (
            action.squeeze(0).cpu().numpy(),
            log_prob.squeeze(0),
            state_value.squeeze(0),
            next_state,
        )

    def calculate_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        bootstrap_value: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation (GAE) and TD(λ) returns.

        GAE reduces the variance of Monte-Carlo advantage estimates (the old
        approach) by exponentially down-weighting longer-horizon TD errors with
        factor γλ.  λ=1 recovers Monte-Carlo; λ=0 gives pure 1-step TD.

        Returns:
            advantages: GAE estimates A(s_t, a_t) for each step.
            returns: TD(λ) targets V̂(s_t) = A_t + V(s_t) used for the critic.
        """
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        next_value = float(bootstrap_value)
        for t in reversed(range(T)):
            delta = rewards[t] + self.config.gamma * next_value - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * gae
            advantages[t] = gae
            next_value = float(values[t])
        returns = (advantages + values).astype(np.float32)
        return advantages.astype(np.float32), returns

    def _prepare_batch(
        self,
        trajectories: List[Dict[str, np.ndarray]],
    ) -> Dict[str, torch.Tensor]:
        """
        Pad variable-length trajectories into a fixed-size batch for GPU computation.
        
        Converts numpy arrays to tensors and pads sequences to the maximum length in the batch,
        with masks indicating valid (non-padded) elements.
        """
        obs_tensors = [torch.as_tensor(t["observations"], dtype=torch.float32, device=self.device) for t in trajectories]
        act_tensors = [torch.as_tensor(t["actions"], dtype=torch.float32, device=self.device) for t in trajectories]
        logp_tensors = [torch.as_tensor(t["log_probs"], dtype=torch.float32, device=self.device) for t in trajectories]
        ret_tensors = [torch.as_tensor(t["returns"], dtype=torch.float32, device=self.device) for t in trajectories]
        adv_tensors = [torch.as_tensor(t["advantages"], dtype=torch.float32, device=self.device) for t in trajectories]
        obs_tensors = [torch.nan_to_num(t, nan=0.0, posinf=1.0, neginf=-1.0) for t in obs_tensors]
        act_tensors = [torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0) for t in act_tensors]
        logp_tensors = [torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0) for t in logp_tensors]
        ret_tensors = [torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0) for t in ret_tensors]
        adv_tensors = [torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0) for t in adv_tensors]
        valid_masks = [
            torch.ones(len(t["returns"]), dtype=torch.float32, device=self.device)
            for t in trajectories
        ]
        reset_masks = []
        for trajectory in trajectories:
            mask = torch.zeros(len(trajectory["returns"]), dtype=torch.float32, device=self.device)
            if len(mask) > 0:
                mask[0] = 1.0
            reset_masks.append(mask)

        return {
            "observations": pad_sequence(obs_tensors, batch_first=True),
            "actions": pad_sequence(act_tensors, batch_first=True),
            "log_probs": pad_sequence(logp_tensors, batch_first=True),
            "returns": pad_sequence(ret_tensors, batch_first=True),
            "advantages": pad_sequence(adv_tensors, batch_first=True),
            "valid_mask": pad_sequence(valid_masks, batch_first=True),
            "done_mask": pad_sequence(reset_masks, batch_first=True),
        }

    def _split_trajectories(self, trajectories: List[Dict[str, np.ndarray]]) -> List[Dict[str, np.ndarray]]:
        """
        Split full episodes into overlapping shorter sequences for recurrent network updates.
        
        Uses a sliding window with configured stride to create training sequences,
        allowing the recurrent network to maintain state continuity.
        """
        chunked_trajectories: List[Dict[str, np.ndarray]] = []
        chunk_length = self.config.sequence_length
        chunk_stride = self.config.sequence_stride
        for trajectory in trajectories:
            total_length = len(trajectory["returns"])
            for start in range(0, total_length, chunk_stride):
                end = min(start + chunk_length, total_length)
                if end <= start:
                    continue
                chunked_trajectories.append({key: value[start:end] for key, value in trajectory.items()})
                if end >= total_length:
                    break
        return chunked_trajectories

    def _sanitize_trajectories(self, trajectories: List[Dict[str, np.ndarray]]) -> None:
        """Replace NaN and inf values in trajectory data with safe defaults."""
        low = np.array([-self.config.max_steering_angle, self.config.min_speed], dtype=np.float32)
        high = np.array([self.config.max_steering_angle, self.config.max_speed], dtype=np.float32)
        for trajectory in trajectories:
            trajectory["observations"] = np.nan_to_num(
                trajectory["observations"], nan=0.0, posinf=1.0, neginf=-1.0
            ).astype(np.float32)
            trajectory["actions"] = np.nan_to_num(
                trajectory["actions"], nan=0.0, posinf=0.0, neginf=0.0
            ).astype(np.float32)
            trajectory["actions"] = np.clip(trajectory["actions"], low + 1e-5, high - 1e-5)
            trajectory["log_probs"] = np.nan_to_num(
                trajectory["log_probs"], nan=0.0, posinf=0.0, neginf=0.0
            ).astype(np.float32)
            trajectory["returns"] = np.nan_to_num(
                trajectory["returns"], nan=0.0, posinf=0.0, neginf=0.0
            ).astype(np.float32)
            trajectory["advantages"] = np.nan_to_num(
                trajectory["advantages"], nan=0.0, posinf=0.0, neginf=0.0
            ).astype(np.float32)

    def _normalize_advantages(self, trajectories: List[Dict[str, np.ndarray]]) -> None:
        """Normalize advantages across all trajectories to zero mean and unit variance."""
        all_advantages = np.concatenate([t["advantages"] for t in trajectories], axis=0)
        adv_mean = float(all_advantages.mean())
        adv_std = float(all_advantages.std() + 1e-8)
        for trajectory in trajectories:
            trajectory["advantages"] = ((trajectory["advantages"] - adv_mean) / adv_std).astype(np.float32)

    @staticmethod
    def _sequence_loss_mask(valid_mask: torch.Tensor, burn_in: int) -> torch.Tensor:
        """
        Build a loss mask that ignores initial burn-in steps per sequence.

        Keeps at least one valid step active for very short sequences.
        """
        valid_lengths = valid_mask.sum(dim=1).to(dtype=torch.long)
        burn_tensor = torch.full_like(valid_lengths, burn_in)
        start_index = torch.minimum(burn_tensor, torch.clamp(valid_lengths - 1, min=0))
        time_index = torch.arange(valid_mask.shape[1], device=valid_mask.device).unsqueeze(0)
        return valid_mask * (time_index >= start_index.unsqueeze(1)).to(dtype=valid_mask.dtype)

    def _update_batch(self, batch: Dict[str, torch.Tensor]) -> None:
        """
        Perform a single PPO policy gradient update on one batch of data.
        
        Computes clipped surrogate objective for policy, smooth L1 loss for value estimation,
        and entropy regularization. Applies gradient clipping for stability.
        """
        log_probs_new, values, entropy = self.evaluate_sequences(
            batch["observations"],
            batch["actions"],
            batch["done_mask"],
        )
        if not (torch.isfinite(log_probs_new).all() and torch.isfinite(values).all() and torch.isfinite(entropy).all()):
            print("[PPO] WARNING: Skipping update batch due to non-finite policy evaluation.", flush=True)
            return

        valid_mask = batch["valid_mask"]
        learn_mask = self._sequence_loss_mask(valid_mask, self.config.burn_in)
        mask_bool = learn_mask > 0
        log_ratio = torch.nan_to_num(
            log_probs_new - batch["log_probs"],
            nan=0.0,
            posinf=20.0,
            neginf=-20.0,
        ).clamp(-20.0, 20.0)
        ratio = torch.exp(log_ratio)
        surrogate1 = ratio * batch["advantages"]
        surrogate2 = torch.clamp(ratio, 1 - self.config.epsilon, 1 + self.config.epsilon) * batch["advantages"]
        surrogate = torch.where(mask_bool, torch.min(surrogate1, surrogate2), torch.zeros_like(surrogate1))

        value_error = nn.functional.smooth_l1_loss(values, batch["returns"], reduction="none")
        value_error = torch.where(mask_bool, value_error, torch.zeros_like(value_error))
        entropy = torch.where(mask_bool, entropy, torch.zeros_like(entropy))

        valid_count = learn_mask.sum().clamp_min(1.0)
        policy_loss = -surrogate.sum() / valid_count
        value_loss = value_error.sum() / valid_count
        entropy_bonus = entropy.sum() / valid_count
        loss = policy_loss + 0.5 * value_loss - self.config.entropy_coef * entropy_bonus

        if not torch.isfinite(loss):
            print("[PPO] WARNING: Skipping update batch due to non-finite loss.", flush=True)
            return

        self.optimizer.zero_grad()
        loss.backward()
        gradients_finite = True
        for parameter in list(self.model.parameters()) + [self.actor_log_std]:
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                gradients_finite = False
                break
        if not gradients_finite:
            print("[PPO] WARNING: Skipping update batch due to non-finite gradients.", flush=True)
            self.optimizer.zero_grad()
            return

        nn.utils.clip_grad_norm_(list(self.model.parameters()) + [self.actor_log_std], max_norm=1.0)
        self.optimizer.step()
        with torch.no_grad():
            self.actor_log_std.data.copy_(
                torch.nan_to_num(self.actor_log_std.data, nan=-0.5, posinf=2.0, neginf=-5.0).clamp(-5.0, 2.0)
            )

    def evaluate_sequences(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        done_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate policy log probabilities, state values, and entropy over full sequences.
        
        Used during policy updates to compute the policy gradient and critic loss.
        """
        policy_output, state_values, _ = self.model(
            observations,
            recurrent_state=self.get_initial_state(observations.shape[0]),
            done_mask=done_mask,
        )
        mean, std = self._policy_stats(policy_output)
        safe_actions = torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
        pre_tanh_actions, squashed_actions = self._inverse_squash_action(safe_actions)
        log_probs = self._log_prob_from_pre_tanh(mean, std, pre_tanh_actions, squashed_actions)
        entropy_pre_tanh = Normal(mean, std).rsample()
        entropy = -self._log_prob_from_pre_tanh(mean, std, entropy_pre_tanh, torch.tanh(entropy_pre_tanh))
        return log_probs, state_values, entropy

    def update(self, trajectories: List[Dict[str, np.ndarray]]) -> None:
        """
        Update policy from a collection of complete episodes.
        
        Processes trajectories through sanitization, advantage normalization, and splitting
        into sequences. Then performs multiple epochs of SGD with batch updates.
        """
        if not trajectories:
            return

        self._sanitize_trajectories(trajectories)
        self._normalize_advantages(trajectories)

        trajectories = self._split_trajectories(trajectories)
        if not trajectories:
            return

        num_episodes = len(trajectories)
        for _ in range(self.config.epochs):
            indices = torch.randperm(num_episodes).tolist()
            for start in range(0, num_episodes, self.config.batch_size):
                batch_indices = indices[start : start + self.config.batch_size]
                batch = self._prepare_batch([trajectories[idx] for idx in batch_indices])
                self._update_batch(batch)

    def load_model(self, model_path: str) -> None:
        """
        Load a saved PPO checkpoint and restore the recurrent model.
        
        Validates checkpoint metadata and rebuilds the recurrent cell if needed.
        """
        checkpoint = _load_checkpoint(model_path, map_location=self.device)
        checkpoint_algorithm = str(checkpoint.get("algorithm", "ppo")).lower().strip()
        if checkpoint_algorithm != "ppo":
            raise ValueError(f"Checkpoint algorithm '{checkpoint_algorithm}' does not match PPO.")
        if "model" not in checkpoint:
            raise ValueError(
                "Checkpoint does not contain recurrent 'model' weights. "
                "Legacy feed-forward checkpoints are not compatible with this module."
            )
        self._validate_checkpoint_metadata(checkpoint)
        checkpoint_cell = str(checkpoint.get("recurrent_cell", self.config.recurrent_cell)).lower().strip()
        if checkpoint_cell not in {"lstm", "gru"}:
            raise ValueError(f"Unsupported recurrent_cell in checkpoint: {checkpoint_cell}")
        if checkpoint_cell != self.config.recurrent_cell:
            self.config.recurrent_cell = checkpoint_cell
            self._build_model(checkpoint_cell)
        print(f"[PPO] Loaded recurrent cell: {checkpoint_cell.upper()}", flush=True)
        self.model.load_state_dict(checkpoint["model"])
        if "actor_log_std" in checkpoint:
            self.actor_log_std.data.copy_(checkpoint["actor_log_std"].to(self.device))



# ============================================================================
# TRAINING LOOP
# ============================================================================


def train(config: Optional[Config] = None) -> None:
    """
    Main training loop integrating environment interaction, rollout collection, and policy updates.
    
    Collects on-policy experience by rolling out the current policy in the Webots simulator,
    accumulates returns and advantages, and performs periodic PPO updates. Maintains metrics
    for progress tracking and saves checkpoints of the best models.
    
    Args:
        config: Configuration object (uses defaults if None)
    """
    if config is None:
        config = Config()
    
    _init_supervisor()
    
    env = WebotsEnv(config)
    env.reset()
    run_id = Path(env.run_folder).name
    obs_size = env.observation_size
    action_dim = env.action_dim
    agent = PPOAgent(obs_size, action_dim, config)
    checkpoint_dir = _run_checkpoint_dir(run_id)
    final_model_path = _run_checkpoint_path(run_id, "final")
    print(
        f"[TRAIN][PPO] rnn={config.recurrent_cell.upper()} "
        f"weights_dir={checkpoint_dir} final={final_model_path}",
        flush=True,
    )
    
    print(
        f"[TRAIN][PPO] episodes={config.episodes} update_every={config.update_every} "
        f"obs={obs_size} act={action_dim} cell={config.recurrent_cell.upper()} "
        f"seq={config.sequence_length} burn_in={config.burn_in}",
        flush=True,
    )
    
    rollout_trajectories: List[Dict[str, np.ndarray]] = []
    best_reward = float('-inf')
    best_goal_reward = float('-inf')
    best_goal_episode: Optional[int] = None
    reward_window: List[float] = []
    success_window: List[float] = []
    goal_touch_window: List[float] = []
    collision_window: List[float] = []
    timeout_window: List[float] = []
    start_time = time.perf_counter()
    metrics_logger = MetricsLogger(env.run_folder, algorithm="ppo")
    
    for episode in range(config.episodes):
        obs, _ = env.reset()
        done = False
        episode_step = 0
        episode_observations: List[np.ndarray] = []
        episode_actions: List[np.ndarray] = []
        episode_log_probs: List[float] = []
        episode_rewards: List[float] = []
        episode_end_reason = "max_steps"
        episode_goal_reached = False
        episode_success = False
        recurrent_state = agent.get_initial_state(batch_size=1)
        prev_done = True
        
        while not done:
            action, log_prob, _, recurrent_state = agent.select_action(
                obs,
                recurrent_state=recurrent_state,
                done=prev_done,
            )
            
            obs_next, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            prev_done = done
            episode_step += 1
            episode_goal_reached = episode_goal_reached or bool(info.get("goal_reached", False))
            episode_success = bool(info.get("success", False))
            
            if done:
                if info.get("reset_reason") == "low_score":
                    episode_end_reason = "low_score"
                elif info.get("reset_reason") == "collision":
                    episode_end_reason = "collision"
                elif info.get("reset_reason") == "goal":
                    episode_end_reason = "goal"
                elif truncated:
                    episode_end_reason = "max_steps"
            
            episode_observations.append(obs)
            episode_actions.append(action)
            episode_log_probs.append(float(log_prob.item()))
            episode_rewards.append(reward)
            
            obs = obs_next
        
        # Compute per-step value estimates for the full episode, then GAE.
        episode_obs_array = np.array(episode_observations, dtype=np.float32)
        with torch.no_grad():
            _, episode_values, _ = agent.model(
                episode_obs_array,
                recurrent_state=agent.get_initial_state(batch_size=1),
                done_mask=np.concatenate(([1.0], np.zeros(len(episode_rewards) - 1, dtype=np.float32))),
            )
            episode_values_np = episode_values.squeeze(0).detach().cpu().numpy()

        # Bootstrap V(s_T) from the critic for truncated episodes so the
        # advantage estimate accounts for future rewards beyond the episode.
        bootstrap_value = 0.0
        if episode_end_reason == "max_steps":
            with torch.no_grad():
                _, bootstrap_state_value, _ = agent.model(
                    np.asarray(obs, dtype=np.float32),
                    recurrent_state=recurrent_state,
                    done_mask=np.array([0.0], dtype=np.float32),
                )
            bootstrap_value = float(bootstrap_state_value.squeeze(0).item())

        episode_advantages, episode_returns = agent.calculate_gae(
            np.array(episode_rewards, dtype=np.float32) * REWARD_SCALE,
            episode_values_np,
            bootstrap_value=bootstrap_value,
        )

        rollout_trajectories.append(
            {
                "observations": episode_obs_array,
                "actions": np.array(episode_actions, dtype=np.float32),
                "log_probs": np.array(episode_log_probs, dtype=np.float32),
                "returns": episode_returns,
                "advantages": episode_advantages,
            }
        )
        
        # PPO update every N episodes
        if (episode + 1) % config.update_every == 0:
            # Update policy
            agent.update(rollout_trajectories)
            
            # Clear buffers
            rollout_trajectories.clear()
        
        # Logging metadata for one concise line per episode.
        episode_reward_sum = sum(episode_rewards)
        reward_window.append(episode_reward_sum)
        success_window.append(1.0 if episode_success else 0.0)
        goal_touch_window.append(1.0 if episode_goal_reached else 0.0)
        collision_window.append(1.0 if episode_end_reason == "collision" else 0.0)
        timeout_window.append(1.0 if episode_end_reason == "max_steps" else 0.0)
        checkpoint_flags: List[str] = []
        
        if episode_end_reason == "goal":
            if episode_reward_sum > best_goal_reward:
                best_goal_reward = episode_reward_sum
                best_goal_episode = episode + 1
                env.robot.slam.save_episode(env.run_folder, episode + 1, episode_reward_sum)
                header = _make_checkpoint_header(best_goal_episode, best_goal_reward, True, 'ppo', asdict(config))
                header.update(agent._checkpoint_metadata())
                header.update({'model': agent.model.state_dict(), 'actor_log_std': agent.actor_log_std.detach().cpu()})
                _save_checkpoint_file(_CHECKPOINT_DIR, run_id, 'best', header)
                checkpoint_flags.append("best_goal")
        elif best_goal_episode is None and episode_reward_sum > best_reward:
            best_reward = episode_reward_sum
            env.robot.slam.save_episode(env.run_folder, episode + 1, episode_reward_sum)
            header = _make_checkpoint_header(episode + 1, best_reward, False, 'ppo', asdict(config))
            header.update(agent._checkpoint_metadata())
            header.update({'model': agent.model.state_dict(), 'actor_log_std': agent.actor_log_std.detach().cpu()})
            _save_checkpoint_file(_CHECKPOINT_DIR, run_id, 'best', header)
            checkpoint_flags.append("best")

        if config.save_every > 0 and (episode + 1) % config.save_every == 0:
            latest = _make_checkpoint_header(episode + 1, episode_reward_sum, episode_end_reason == "goal", 'ppo', asdict(config))
            latest.update(agent._checkpoint_metadata())
            latest.update({'model': agent.model.state_dict(), 'actor_log_std': agent.actor_log_std.detach().cpu()})
            _save_checkpoint_file(_CHECKPOINT_DIR, run_id, 'checkpoint', latest)
            checkpoint_flags.append("latest")

        rolling_reward = float(np.mean(reward_window[-10:]))
        rolling_success = float(np.mean(success_window[-10:]))
        rolling_goal_touch = float(np.mean(goal_touch_window[-10:]))
        rolling_collision = float(np.mean(collision_window[-10:]))
        rolling_timeout = float(np.mean(timeout_window[-10:]))
        elapsed = time.perf_counter() - start_time
        checkpoint_note = f" ckpt={'+'.join(checkpoint_flags)}" if checkpoint_flags else ""
        print(
            f"[TRAIN][PPO] ep={episode + 1:03d}/{config.episodes} "
            f"r={episode_reward_sum:8.2f} avg10={rolling_reward:8.2f} steps={episode_step:4d} "
            f"succ10={rolling_success:4.2f} touch10={rolling_goal_touch:4.2f} "
            f"col10={rolling_collision:4.2f} to10={rolling_timeout:4.2f} "
            f"min_d={env.min_episode_distance:5.2f} end={episode_end_reason} t={elapsed:7.1f}s{checkpoint_note}",
            flush=True,
        )
        metrics_logger.log(
            episode=episode + 1,
            reward=round(episode_reward_sum, 4),
            avg10=round(rolling_reward, 4),
            steps=episode_step,
            success=int(episode_success),
            goal_touched=int(episode_goal_reached),
            collision=int(episode_end_reason == "collision"),
            timeout=int(episode_end_reason == "max_steps"),
            min_dist=round(env.min_episode_distance, 4),
            end_reason=episode_end_reason,
            elapsed_s=round(elapsed, 1),
        )

    if rollout_trajectories:
        agent.update(rollout_trajectories)

    metrics_logger.close()
    print(f"[TRAIN][PPO] metrics saved to {metrics_logger.path}", flush=True)

    # Save final model
    final_reward = best_goal_reward if best_goal_episode is not None else best_reward
    header = _make_checkpoint_header('final', final_reward, best_goal_episode is not None, 'ppo', asdict(config))
    header.update(agent._checkpoint_metadata())
    header.update({'model': agent.model.state_dict(), 'actor_log_std': agent.actor_log_std.detach().cpu()})
    _save_checkpoint_file(_CHECKPOINT_DIR, run_id, 'final', header)
    elapsed = time.perf_counter() - start_time
    print(f"[TRAIN][PPO] final reward={final_reward:.2f} t={elapsed:7.1f}s", flush=True)

    # Cleanup
    env.robot.motors.stop()
    print("[TRAIN][PPO] done", flush=True)


if __name__ == "__main__":
    train()
