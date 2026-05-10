"""PPO training controller for ALTINO robot in Webots obstacle avoidance task."""
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any, Union
import numpy as np
import torch
from torch import nn
from torch.distributions import AffineTransform, Independent, Normal, TanhTransform, TransformedDistribution
from torch.nn.utils.rnn import pad_sequence

RecurrentState = Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from controllers.RNN import GRUActorCritic, LSTMActorCritic
from controllers.Webots.webots_env import WebotsEnv, _init_supervisor

_CONTROLLER_DIR = Path(__file__).resolve().parent
_CHECKPOINT_DIR = _CONTROLLER_DIR / "checkpoints"


def _checkpoint_path(filename: str) -> str:
    """Return a checkpoint path pinned to the PPO controller directory."""
    return str(_CONTROLLER_DIR / filename)


def _dated_checkpoint_path(run_id: str, filename: str) -> str:
    """Return a checkpoint path inside the timestamped checkpoint folder."""
    checkpoint_dir = _CHECKPOINT_DIR / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return str(checkpoint_dir / filename)


def _load_checkpoint(path: str, map_location: Union[str, torch.device]) -> Dict[str, Any]:
    """Load local controller checkpoints without relying on PyTorch's changing default."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Training and environment hyperparameters."""
    
    # Training
    episodes: int = 500
    update_every: int = 5  # PPO update frequency (episodes)
    epochs: int = 4  # Optimization epochs per update
    batch_size: int = 64
    save_every: int = 100  # Save latest checkpoint every N episodes (0 disables)
    
    # PPO Agent
    gamma: float = 0.99  # Discount factor
    epsilon: float = 0.2  # PPO clip parameter
    learning_rate: float = 1e-4
    entropy_coef: float = 0.02  # Entropy regularization
    hidden_size: int = 128  # Network hidden layer size
    latent_size: int = 128  # Encoder output size before the recurrent core
    lstm_hidden_size: int = 128  # Hidden size of the recurrent core
    lstm_layers: int = 1
    recurrent_cell: str = "gru" # "lstm" or "gru"
    lidar_sector_dim: int = 16
    pose_goal_dim: int = 7  # [x, y] + [sin/cos heading, sin/cos goal_error, norm dist]
    imu_feature_dim: int = 10  # accel(3) + gyro(3) + quaternion(4)
    occupancy_grid_shape: Optional[Tuple[int, ...]] = None  # Optional CNN shape, e.g. (1, 16, 16)
    
    # Environment
    max_steps: int = 2000  # Max steps per episode
    collision_threshold: float = 0.1  # LiDAR distance threshold for collision
    low_score_threshold: float = -800.0  # Episode reset threshold
    collision_penalty: float = -20.0  # Penalty when collision happens
    progress_reward_scale: float = 3.0  # Scale for distance-progress reward
    distance_reward_scale: float = 2.0  # Dense reward for being closer to the goal than the start state
    heading_reward_scale: float = 0.5  # Bonus when facing toward the goal
    safety_reward_scale: float = 0.2  # Encourages keeping distance from obstacles
    motion_reward_scale: float = 0.05  # Bonus for moving forward to avoid stop-policy collapse
    new_best_distance_bonus: float = 1.0  # Bonus when reaching a new closest distance to goal
    step_penalty: float = -0.01  # Small per-step penalty to encourage efficiency
    endpoint: Tuple[float, float] = (2.0, 0.0)  # Goal location
    goal_threshold: float = 0.1  # Radius around goal considered reached
    goal_stop_speed_threshold: float = 0.1  # Speed limit for counting goal success
    goal_stop_bonus: float = 120.0  # Extra reward for stopping at the goal
    goal_hold_reward: float = 10.0  # Reward per timestep while inside goal threshold
    goal_speed_penalty: float = -60.0  # Penalty for still moving inside the goal region
    goal_overshoot_penalty: float = -50.0  # Penalty for driving past the goal region
    reference_distance: Optional[float] = None  # Start-to-goal distance, filled in at init

    enable_slam: bool = True
    profile_slam: bool = False
    slam_profile_interval: int = 500
    save_slam_plots: bool = False
    force_cpu: bool = True  # Force CPU-only training even if CUDA is available
    
    # Robot Control
    max_steering_angle: float = 0.9
    min_speed: float = 0.0
    start_position: Optional[List[float]] = None  # [x, y, z]
    start_rotation: Optional[List[float]] = None  # [x, y, z, w]
    start_position_noise: float = 0.03  # Random position jitter at reset
    start_yaw_noise: float = 0.2  # Random yaw jitter at reset
    
    # Motor/Sensor Config
    max_speed: float = 10.0
    reset_settle_steps: int = 10  # Steps to wait for physics to settle after reset
    
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
            self.start_position = [-2.0, 0.0, 0.02]
        if self.start_rotation is None:
            self.start_rotation = [0.0, 0.0, 1.0, 0.0]
        if self.reference_distance is None:
            start_xy = np.array(self.start_position[:2], dtype=np.float32)
            endpoint_xy = np.array(self.endpoint, dtype=np.float32)
            self.reference_distance = float(np.linalg.norm(start_xy - endpoint_xy))




# ============================================================================
# PPO AGENT
# ============================================================================

class PPOAgent:
    """Proximal Policy Optimization agent with a recurrent actor-critic core."""

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

    def _policy_stats(self, policy_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return numerically stable mean and standard deviation for the policy."""
        policy_output = torch.nan_to_num(policy_output, nan=0.0, posinf=1.0, neginf=-1.0)
        safe_actor_log_std = torch.nan_to_num(self.actor_log_std, nan=-0.5, posinf=2.0, neginf=-5.0)
        safe_actor_log_std = safe_actor_log_std.clamp(-5.0, 2.0)
        log_std = safe_actor_log_std.expand_as(policy_output)
        std = torch.nan_to_num(log_std.exp(), nan=1.0, posinf=7.5, neginf=1e-3).clamp_min(1e-3)
        return policy_output, std

    def _action_distribution(self, mean: torch.Tensor, std: torch.Tensor) -> TransformedDistribution:
        """Construct the bounded Gaussian policy used for action selection and PPO updates."""
        low, high = self._action_bounds()
        center = (high + low) / 2.0
        scale = (high - low) / 2.0
        base_dist = Independent(Normal(mean, std), 1)
        return TransformedDistribution(
            base_dist,
            [TanhTransform(cache_size=1), AffineTransform(loc=center, scale=scale)],
        )

    def _entropy_estimate(self, action_distribution: TransformedDistribution) -> torch.Tensor:
        """Estimate the entropy of the bounded policy with a reparameterized sample."""
        action_sample = action_distribution.rsample()
        return -action_distribution.log_prob(action_sample)

    def _squash_action(self, pre_tanh_action: torch.Tensor) -> torch.Tensor:
        """Map unconstrained actions to environment action bounds."""
        low, high = self._action_bounds()
        center = (high + low) / 2.0
        scale = (high - low) / 2.0
        squashed = torch.tanh(pre_tanh_action)
        action = squashed * scale + center
        eps = 1e-5
        return torch.max(torch.min(action, high - eps), low + eps)

    def select_action(
        self,
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        recurrent_state: Optional[RecurrentState] = None,
        done: bool = False,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor, RecurrentState]:
        """Sample a single action and advance the recurrent state."""
        done_mask = torch.tensor([float(done)], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            policy_output, state_value, next_state = self.model(
                obs,
                recurrent_state=recurrent_state,
                done_mask=done_mask,
            )
            mean, std = self._policy_stats(policy_output)
            action_distribution = self._action_distribution(mean, std)
            action = self._squash_action(mean) if deterministic else action_distribution.rsample()
            log_prob = action_distribution.log_prob(action)
        return (
            action.squeeze(0).cpu().numpy(),
            log_prob.squeeze(0),
            state_value.squeeze(0),
            next_state,
        )

    def calculate_returns(self, rewards: np.ndarray) -> np.ndarray:
        """Calculate cumulative discounted returns."""
        returns = np.zeros(len(rewards), dtype=np.float32)
        discounted_return = 0.0
        for t in reversed(range(len(rewards))):
            discounted_return = rewards[t] + self.config.gamma * discounted_return
            returns[t] = discounted_return
        return returns

    def _prepare_batch(
        self,
        trajectories: List[Dict[str, np.ndarray]],
    ) -> Dict[str, torch.Tensor]:
        """Pad variable-length episode trajectories to a batch of sequences."""
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

    def evaluate_sequences(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        done_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate PPO quantities over full recurrent sequences."""
        policy_output, state_values, _ = self.model(
            observations,
            recurrent_state=self.get_initial_state(observations.shape[0]),
            done_mask=done_mask,
        )
        mean, std = self._policy_stats(policy_output)
        action_distribution = self._action_distribution(mean, std)
        log_probs = action_distribution.log_prob(actions)
        entropy = self._entropy_estimate(action_distribution)
        return log_probs, state_values, entropy

    def update(self, trajectories: List[Dict[str, np.ndarray]]) -> None:
        """Update the recurrent PPO policy from complete episodes."""
        if not trajectories:
            return

        for trajectory in trajectories:
            trajectory["observations"] = np.nan_to_num(
                trajectory["observations"], nan=0.0, posinf=1.0, neginf=-1.0
            ).astype(np.float32)
            trajectory["actions"] = np.nan_to_num(
                trajectory["actions"], nan=0.0, posinf=0.0, neginf=0.0
            ).astype(np.float32)
            trajectory["log_probs"] = np.nan_to_num(
                trajectory["log_probs"], nan=0.0, posinf=0.0, neginf=0.0
            ).astype(np.float32)
            trajectory["returns"] = np.nan_to_num(
                trajectory["returns"], nan=0.0, posinf=0.0, neginf=0.0
            ).astype(np.float32)
            trajectory["advantages"] = np.nan_to_num(
                trajectory["advantages"], nan=0.0, posinf=0.0, neginf=0.0
            ).astype(np.float32)

        all_advantages = np.concatenate([t["advantages"] for t in trajectories], axis=0)
        adv_mean = float(all_advantages.mean())
        adv_std = float(all_advantages.std() + 1e-8)
        for trajectory in trajectories:
            trajectory["advantages"] = ((trajectory["advantages"] - adv_mean) / adv_std).astype(np.float32)

        num_episodes = len(trajectories)
        for _ in range(self.config.epochs):
            indices = torch.randperm(num_episodes).tolist()
            for start in range(0, num_episodes, self.config.batch_size):
                batch_indices = indices[start : start + self.config.batch_size]
                batch = self._prepare_batch([trajectories[idx] for idx in batch_indices])

                log_probs_new, values, entropy = self.evaluate_sequences(
                    batch["observations"],
                    batch["actions"],
                    batch["done_mask"],
                )

                valid_mask = batch["valid_mask"]
                mask_bool = valid_mask > 0
                log_ratio = torch.nan_to_num(
                    log_probs_new - batch["log_probs"],
                    nan=0.0,
                    posinf=20.0,
                    neginf=-20.0,
                ).clamp(-20.0, 20.0)
                ratio = torch.exp(log_ratio)
                surrogate1 = ratio * batch["advantages"]
                surrogate2 = (
                    torch.clamp(ratio, 1 - self.config.epsilon, 1 + self.config.epsilon)
                    * batch["advantages"]
                )
                surrogate = torch.min(surrogate1, surrogate2)
                surrogate = torch.where(mask_bool, surrogate, torch.zeros_like(surrogate))

                value_error = (values - batch["returns"]) ** 2
                value_error = torch.where(mask_bool, value_error, torch.zeros_like(value_error))

                entropy = torch.where(mask_bool, entropy, torch.zeros_like(entropy))
                valid_count = valid_mask.sum().clamp_min(1.0)
                policy_loss = -surrogate.sum() / valid_count
                value_loss = value_error.sum() / valid_count
                entropy_bonus = entropy.sum() / valid_count

                loss = policy_loss + 0.5 * value_loss - self.config.entropy_coef * entropy_bonus
                if not torch.isfinite(loss):
                    print("[PPO] WARNING: Skipping update batch due to non-finite loss.", flush=True)
                    continue
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
                    continue
                nn.utils.clip_grad_norm_(list(self.model.parameters()) + [self.actor_log_std], max_norm=1.0)
                self.optimizer.step()
                with torch.no_grad():
                    self.actor_log_std.data.copy_(
                        torch.nan_to_num(self.actor_log_std.data, nan=-0.5, posinf=2.0, neginf=-5.0).clamp(-5.0, 2.0)
                    )

    def load_model(self, model_path: str) -> None:
        """Load saved recurrent model weights."""
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
# TRAINING
# ============================================================================

def train(config: Optional[Config] = None) -> None:
    """Run PPO training loop.
    
    Args:
        config: Configuration object (uses defaults if None)
    """
    if config is None:
        config = Config()
    
    # Initialize Webots supervisor
    _init_supervisor()
    
    # Create environment and agent
    env = WebotsEnv(config)
    env.reset()
    run_id = Path(env.run_folder).name
    obs_size = env.observation_size
    action_dim = env.action_dim
    agent = PPOAgent(obs_size, action_dim, config)
    checkpoint_dir = f"checkpoints/{run_id}"
    final_model_path = "final_model.pth"
    print(
        f"[TRAIN][PPO] rnn={config.recurrent_cell.upper()} "
        f"weights_dir={checkpoint_dir} final={final_model_path}",
        flush=True,
    )
    
    print(
        f"[TRAIN][PPO] episodes={config.episodes} update_every={config.update_every} "
        f"obs={obs_size} act={action_dim} cell={config.recurrent_cell.upper()}",
        flush=True,
    )
    
    # Training buffers
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
    
    # Training loop
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
            # Select action from the current policy for every transition so the
            # rollout stays on-policy for PPO updates.
            action, log_prob, _, recurrent_state = agent.select_action(
                obs,
                recurrent_state=recurrent_state,
                done=prev_done,
            )
            
            # Step environment
            obs_next, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            prev_done = done
            episode_step += 1
            episode_goal_reached = episode_goal_reached or bool(info.get("goal_reached", False))
            episode_success = bool(info.get("success", False))
            
            # Track termination reason
            if done:
                if info.get("reset_reason") == "low_score":
                    episode_end_reason = "low_score"
                elif info.get("reset_reason") == "collision":
                    episode_end_reason = "collision"
                elif info.get("reset_reason") == "goal":
                    episode_end_reason = "goal"
                elif truncated:
                    episode_end_reason = "max_steps"
            
            # Accumulate
            episode_observations.append(obs)
            episode_actions.append(action)
            episode_log_probs.append(float(log_prob.item()))
            episode_rewards.append(reward)
            
            obs = obs_next
        
        # Build returns and advantages per episode so reward signals do not
        # leak across episode boundaries.
        episode_returns = agent.calculate_returns(np.array(episode_rewards, dtype=np.float32))
        episode_obs_array = np.array(episode_observations, dtype=np.float32)
        with torch.no_grad():
            _, episode_values, _ = agent.model(
                episode_obs_array,
                recurrent_state=agent.get_initial_state(batch_size=1),
                done_mask=np.concatenate(([1.0], np.zeros(len(episode_rewards) - 1, dtype=np.float32))),
            )
            episode_values = episode_values.squeeze(0).detach().cpu().numpy()

        episode_advantages = episode_returns - episode_values

        rollout_trajectories.append(
            {
                "observations": episode_obs_array,
                "actions": np.array(episode_actions, dtype=np.float32),
                "log_probs": np.array(episode_log_probs, dtype=np.float32),
                "returns": episode_returns.astype(np.float32),
                "advantages": episode_advantages.astype(np.float32),
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
                checkpoint = {
                    'model': agent.model.state_dict(),
                    'actor_log_std': agent.actor_log_std.detach().cpu(),
                    'episode': best_goal_episode,
                    'reward': best_goal_reward,
                    'goal_episode': True,
                    'algorithm': 'ppo',
                    'config': asdict(config),
                }
                checkpoint.update(agent._checkpoint_metadata())
                torch.save(checkpoint, _dated_checkpoint_path(run_id, 'best_model.pth'))
                checkpoint_flags.append("best_goal")
        elif best_goal_episode is None and episode_reward_sum > best_reward:
            best_reward = episode_reward_sum
            env.robot.slam.save_episode(env.run_folder, episode + 1, episode_reward_sum)
            checkpoint = {
                'model': agent.model.state_dict(),
                'actor_log_std': agent.actor_log_std.detach().cpu(),
                'episode': episode + 1,
                'reward': best_reward,
                'goal_episode': False,
                'algorithm': 'ppo',
                'config': asdict(config),
            }
            checkpoint.update(agent._checkpoint_metadata())
            torch.save(checkpoint, _dated_checkpoint_path(run_id, 'best_model.pth'))
            checkpoint_flags.append("best")

        if config.save_every > 0 and (episode + 1) % config.save_every == 0:
            latest_checkpoint = {
                'model': agent.model.state_dict(),
                'actor_log_std': agent.actor_log_std.detach().cpu(),
                'episode': episode + 1,
                'reward': episode_reward_sum,
                'goal_episode': episode_end_reason == "goal",
                'algorithm': 'ppo',
                'config': asdict(config),
            }
            latest_checkpoint.update(agent._checkpoint_metadata())
            torch.save(latest_checkpoint, _dated_checkpoint_path(run_id, 'latest_model.pth'))
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

    if rollout_trajectories:
        agent.update(rollout_trajectories)
    
    # Save final model
    final_reward = best_goal_reward if best_goal_episode is not None else best_reward
    checkpoint = {
        'model': agent.model.state_dict(),
        'actor_log_std': agent.actor_log_std.detach().cpu(),
        'episode': 'final',
        'reward': final_reward,
        'goal_episode': best_goal_episode is not None,
        'algorithm': 'ppo',
        'config': asdict(config),
    }
    checkpoint.update(agent._checkpoint_metadata())
    torch.save(checkpoint, _checkpoint_path('final_model.pth'))
    elapsed = time.perf_counter() - start_time
    print(f"[TRAIN][PPO] final reward={final_reward:.2f} t={elapsed:7.1f}s", flush=True)

    # Cleanup
    env.robot.motors.stop()
    print("[TRAIN][PPO] done", flush=True)


if __name__ == "__main__":
    train()
