"""Soft Actor-Critic controller for the ALTINO Webots task."""
from __future__ import annotations

import sys, time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from controllers.Webots import WebotsEnv, _init_supervisor
from controllers.RNN import GRUActorCritic, LSTMActorCritic
from controllers.common.reward_defaults import (
    COLLISION_THRESHOLD, COLLISION_PENALTY, DISTANCE_REWARD_SCALE,
    ENABLE_SLAM, ENDPOINT, FORCE_CPU, GOAL_HOLD_REWARD,
    GOAL_STOP_SPEED_THRESHOLD, GOAL_OVERSHOOT_PENALTY, GOAL_SPEED_PENALTY,
    GOAL_STOP_BONUS, GOAL_THRESHOLD, GOAL_SUCCESS_REWARD,
    HEADING_REWARD_SCALE, IMU_FEATURE_DIM, LIDAR_SECTOR_DIM,
    LOW_SCORE_THRESHOLD, MAX_SPEED, MAX_STEERING_ANGLE, MAX_STEPS,
    MIN_SPEED, MOTION_REWARD_SCALE, SLOW_SPEED_PENALTY, SLOW_SPEED_THRESHOLD,
    HIGH_SPEED_THRESHOLD, HIGH_SPEED_BONUS, NEW_BEST_DISTANCE_BONUS,
    OCCUPANCY_GRID_SHAPE, POSE_GOAL_DIM, PROGRESS_REWARD_SCALE,
    PROFILE_SLAM, RESET_SETTLE_STEPS, REWARD_SCALE, SAFETY_REWARD_SCALE,
    SAVE_SLAM_PLOTS, SLAM_PROFILE_INTERVAL, START_POSITION,
    START_POSITION_NOISE, START_ROTATION, START_YAW_NOISE, STEP_PENALTY,
)
from controllers.common.training_defaults import RecurrentDefaults, SACDefaults
from controllers.common.checkpoints import (
    run_checkpoint_dir as _run_checkpoint_dir,
    run_checkpoint_path as _run_checkpoint_path,
    load_checkpoint as _load_checkpoint,
    save_checkpoint_file as _save_checkpoint_file,
)
from controllers.common.metrics_logger import MetricsLogger

_CONTROLLER_DIR = Path(__file__).resolve().parent
_CHECKPOINT_DIR = _CONTROLLER_DIR / "checkpoints"



@dataclass
class Config:
    episodes: int = SACDefaults.episodes
    update_after_steps: int = SACDefaults.update_after_steps
    updates_per_step: int = SACDefaults.updates_per_step
    gradient_steps_per_episode: int = SACDefaults.gradient_steps_per_episode
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
    slow_speed_threshold: float = SLOW_SPEED_THRESHOLD
    slow_speed_penalty: float = SLOW_SPEED_PENALTY
    high_speed_threshold: float = HIGH_SPEED_THRESHOLD
    high_speed_bonus: float = HIGH_SPEED_BONUS
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
    lstm_hidden_size: int = 0
    lstm_layers: int = 0
    latent_size: int = 0

    def __post_init__(self):
        self.recurrent_cell = self.recurrent_cell.lower().strip()
        assert self.recurrent_cell in {"gru", "lstm"}, f"Unsupported recurrent_cell: {self.recurrent_cell}"
        if self.recurrent_hidden_size is None:
            self.recurrent_hidden_size = self.hidden_size
        self.lstm_hidden_size = self.recurrent_hidden_size
        self.lstm_layers = self.recurrent_layers
        self.latent_size = self.hidden_size
        if self.start_position is None:
            self.start_position = list(START_POSITION)
        if self.start_rotation is None:
            self.start_rotation = list(START_ROTATION)
        if self.reference_distance is None:
            start_xy = np.array(self.start_position[:2], dtype=np.float32)
            endpoint_xy = np.array(self.endpoint, dtype=np.float32)
            self.reference_distance = float(np.linalg.norm(start_xy - endpoint_xy))


class SequenceReplayBuffer:
    def __init__(self, obs_size: int, action_dim: int, config: Config) -> None:
        """Initialise fixed-capacity circular replay buffer."""
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
        """Return number of stored sequences."""
        return self.size

    def _store_window(self, obs_w, act_w, rew_w, next_w, done_w) -> None:
        """Store one sliding window at current buffer position."""
        length = obs_w.shape[0]
        idx = self.pos
        self.obs[idx, :length] = obs_w
        self.actions[idx, :length] = act_w
        self.rewards[idx, :length, 0] = rew_w
        self.next_obs[idx, :length] = next_w
        self.dones[idx, :length, 0] = done_w.astype(np.float32)
        self.valid_mask[idx, :length] = 1.0
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_episode(self, ep_obs, ep_act, ep_rew, ep_next, ep_done) -> None:
        """Split episode into sliding windows and add to buffer."""
        if not ep_obs:
            return
        obs = np.asarray(ep_obs, dtype=np.float32)
        actions = np.asarray(ep_act, dtype=np.float32)
        rewards = np.asarray(ep_rew, dtype=np.float32)
        next_obs = np.asarray(ep_next, dtype=np.float32)
        dones = np.asarray(ep_done, dtype=np.bool_)
        total = obs.shape[0]
        start = 0
        while start < total:
            end = min(start + self.seq_len, total)
            self._store_window(obs[start:end], actions[start:end], rewards[start:end], next_obs[start:end], dones[start:end])
            if end >= total:
                break
            start += self.stride

    def can_sample(self, batch_size: int, min_sequences: int) -> bool:
        """Check if buffer has enough sequences for training."""
        return self.size >= max(batch_size, min_sequences)

    def sample(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """Sample random batch of sequences from buffer."""
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs[indices], dtype=torch.float32, device=device),
            "actions": torch.as_tensor(self.actions[indices], dtype=torch.float32, device=device),
            "rewards": torch.as_tensor(self.rewards[indices], dtype=torch.float32, device=device),
            "next_obs": torch.as_tensor(self.next_obs[indices], dtype=torch.float32, device=device),
            "dones": torch.as_tensor(self.dones[indices], dtype=torch.float32, device=device),
            "valid_mask": torch.as_tensor(self.valid_mask[indices], dtype=torch.float32, device=device),
        }


class SACActor(nn.Module):
    def __init__(self, obs_size: int, action_dim: int, config: Config, encoder_cls):
        """Initialise Gaussian actor with shared RNN encoder."""
        super().__init__()
        self.encoder = encoder_cls(obs_size, action_dim, config)
        self.mean = nn.Linear(self.encoder.recurrent_hidden_size, action_dim)
        self.log_std = nn.Linear(self.encoder.recurrent_hidden_size, action_dim)

    def get_initial_state(self, batch_size: int, device=None):
        """Return zeroed recurrent state."""
        return self.encoder.get_initial_state(batch_size, device)

    def forward(self, obs, recurrent_state=None, done_mask=None):
        """Encode observation and return Gaussian policy parameters."""
        features, next_state = self.encoder.encode_only(obs, recurrent_state, done_mask)
        return self.mean(features), self.log_std(features), next_state


class SACQNet(nn.Module):
    def __init__(self, obs_size: int, action_dim: int, config: Config, encoder_cls):
        """Initialise Q-network with shared RNN encoder."""
        super().__init__()
        self.encoder = encoder_cls(obs_size, action_dim, config)
        self.head = nn.Sequential(
            nn.Linear(self.encoder.recurrent_hidden_size + action_dim, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 1),
        )

    def forward(self, obs, action, recurrent_state=None, done_mask=None):
        """Encode observation-action pair and return Q-value."""
        features, next_state = self.encoder.encode_only(obs, recurrent_state, done_mask)
        if features.ndim != action.ndim:
            if features.ndim == 2 and action.ndim == 3 and action.shape[1] == 1:
                action = action.squeeze(1)
            elif features.ndim == 3 and action.ndim == 2:
                action = action.unsqueeze(1)
        return self.head(torch.cat([features, action], dim=-1)), next_state


class SACAgent:
    def __init__(self, obs_size: int, action_dim: int, config: Config) -> None:
        """Initialise SAC agent with actor, twin critics, and temperature."""
        self.config = config
        self.device = self._get_device()
        self.obs_size = obs_size
        self.action_dim = action_dim
        ec = LSTMActorCritic if config.recurrent_cell.lower().strip() == "lstm" else GRUActorCritic

        self.actor = SACActor(obs_size, action_dim, config, ec).to(self.device)
        with torch.no_grad():
            if self.actor.mean.bias is not None and action_dim > 1:
                self.actor.mean.bias[1] = -0.8
        self.q1 = SACQNet(obs_size, action_dim, config, ec).to(self.device)
        self.q2 = SACQNet(obs_size, action_dim, config, ec).to(self.device)
        self.target_q1 = SACQNet(obs_size, action_dim, config, ec).to(self.device)
        self.target_q2 = SACQNet(obs_size, action_dim, config, ec).to(self.device)
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
        """Select CUDA if available and not force_cpu, else CPU."""
        if self.config.force_cpu or not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device("cuda")

    def get_initial_state(self, batch_size: int = 1):
        """Return zeroed recurrent state."""
        return self.actor.get_initial_state(batch_size, self.device)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def _checkpoint_metadata(self):
        """Return architecture metadata for checkpoint validation."""
        return {"obs_size": self.obs_size, "action_dim": self.action_dim}

    def _validate_checkpoint_metadata(self, checkpoint):
        """Raise on incompatible checkpoint dimensions."""
        for key in ("obs_size", "action_dim"):
            saved = checkpoint.get(key)
            if saved is not None and int(saved) != getattr(self, key):
                raise ValueError(f"Checkpoint {key}={saved} != current {getattr(self, key)}")

    def _tensor_obs(self, obs: np.ndarray) -> torch.Tensor:
        """Convert flat numpy observation to batched tensor."""
        return torch.as_tensor(obs, dtype=torch.float32, device=self.device).view(1, -1)

    def _sample_policy(self, obs, recurrent_state=None, done_mask=None, deterministic=False):
        """Sample tanh-squashed action with log-prob from current policy."""
        mean, log_std, next_state = self.actor(obs, recurrent_state=recurrent_state, done_mask=done_mask)
        log_std = log_std.clamp(self.config.log_std_min, self.config.log_std_max)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        pre_tanh = mean if deterministic else dist.rsample()
        tanh_action = torch.tanh(pre_tanh)
        log_prob = dist.log_prob(pre_tanh)
        log_prob -= torch.log(self.action_scale + 1e-6)
        log_prob -= torch.log(1.0 - tanh_action.pow(2) + 1e-6)
        return self._squash_action(tanh_action), log_prob.sum(dim=-1, keepdim=True), next_state

    def _squash_action(self, action: torch.Tensor) -> torch.Tensor:
        """Scale tanh output to environment action bounds."""
        return action * self.action_scale + self.action_center

    def select_action(self, obs: np.ndarray, recurrent_state=None, done=False, deterministic=False):
        """Select action for environment interaction."""
        done_mask = torch.tensor([float(done)], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action, _, next_state = self._sample_policy(self._tensor_obs(obs), recurrent_state, done_mask, deterministic)
        return action.squeeze().cpu().numpy(), next_state

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        """Polyak-averaged target network update."""
        tau = self.config.tau
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)

    @staticmethod
    def _sequence_loss_mask(valid_mask: torch.Tensor, burn_in: int) -> torch.Tensor:
        """Build loss mask ignoring burn-in steps."""
        valid_lengths = valid_mask.sum(dim=1).to(dtype=torch.long)
        start_index = torch.minimum(torch.full_like(valid_lengths, burn_in), torch.clamp(valid_lengths - 1, min=0))
        return valid_mask * (torch.arange(valid_mask.shape[1], device=valid_mask.device).unsqueeze(0) >= start_index.unsqueeze(1)).to(dtype=valid_mask.dtype)

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute mean over valid (non-padded) elements."""
        return (values * mask.unsqueeze(-1).to(dtype=values.dtype)).sum() / mask.sum().clamp_min(1.0)

    def _ensure_time_dim(self, t: torch.Tensor) -> torch.Tensor:
        """Unsqueeze rank-2 tensors to rank-3 for time-dimension consistency."""
        return t.unsqueeze(1) if t.ndim == 2 else t

    def _build_update_masks(self, batch):
        """Create done-mask tensors for obs/next_obs sequences."""
        valid_mask = batch["valid_mask"]
        learn_mask = self._sequence_loss_mask(valid_mask, min(self.config.burn_in, batch["obs"].shape[1] - 1))
        done_flags = batch["dones"].squeeze(-1)
        done_mask_obs = torch.zeros_like(done_flags)
        done_mask_obs[:, 0] = 1.0
        if done_flags.shape[1] > 1:
            done_mask_obs[:, 1:] = done_flags[:, :-1]
        done_mask_next = torch.cat([torch.ones_like(done_flags[:, :1]), done_flags], dim=1)
        return learn_mask, done_mask_obs, done_mask_next

    def update(self, batch):
        """Single SAC update: critics, actor, alpha, and soft-targets."""
        if batch["obs"].shape[1] <= 0:
            return None

        learn_mask, done_mask_obs, done_mask_next = self._build_update_masks(batch)
        scaled_rewards = batch["rewards"] * REWARD_SCALE

        with torch.no_grad():
            cat_obs = torch.cat([batch["obs"][:, :1], batch["next_obs"]], dim=1)
            na, nlp, _ = self._sample_policy(cat_obs, done_mask=done_mask_next, deterministic=False)
            na = self._ensure_time_dim(na)[:, 1:]
            nlp = self._ensure_time_dim(nlp)[:, 1:]
            ta = torch.cat([torch.zeros_like(batch["actions"][:, :1]), na], dim=1)
            tq1, _ = self.target_q1(cat_obs, ta, done_mask=done_mask_next)
            tq2, _ = self.target_q2(cat_obs, ta, done_mask=done_mask_next)
            tq = self._ensure_time_dim(torch.min(tq1, tq2))[:, 1:]
            tq = tq - self.alpha.detach() * nlp
            target_q = scaled_rewards + (1.0 - batch["dones"]) * self.config.gamma * tq

        cq1, _ = self.q1(batch["obs"], batch["actions"], done_mask=done_mask_obs)
        cq2, _ = self.q2(batch["obs"], batch["actions"], done_mask=done_mask_obs)
        critic_loss = self._masked_mean((self._ensure_time_dim(cq1) - self._ensure_time_dim(target_q)).pow(2), learn_mask)
        critic_loss += self._masked_mean((self._ensure_time_dim(cq2) - self._ensure_time_dim(target_q)).pow(2), learn_mask)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), max_norm=1.0)
        self.critic_optimizer.step()

        new_action, log_prob, _ = self._sample_policy(batch["obs"], done_mask=done_mask_obs, deterministic=False)
        q1_pi, _ = self.q1(batch["obs"], new_action, done_mask=done_mask_obs)
        q2_pi, _ = self.q2(batch["obs"], new_action, done_mask=done_mask_obs)
        actor_loss = self._masked_mean(self.alpha.detach() * self._ensure_time_dim(log_prob) - torch.min(self._ensure_time_dim(q1_pi), self._ensure_time_dim(q2_pi)), learn_mask)

        alpha_loss = torch.tensor(0.0, device=self.device)
        if self.config.auto_entropy_tuning:
            alpha_loss = -self._masked_mean(self.log_alpha * (self._ensure_time_dim(log_prob) + self.target_entropy).detach(), learn_mask)

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

    def checkpoint(self, episode, reward):
        """Return full model state dict for saving."""
        ckpt = {
            "episode": episode, "reward": reward, "algorithm": "sac",
            "actor": self.actor.state_dict(), "critic1": self.q1.state_dict(), "critic2": self.q2.state_dict(),
            "target_critic1": self.target_q1.state_dict(), "target_critic2": self.target_q2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(), "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(), "log_alpha": self.log_alpha.detach().cpu(),
        }
        ckpt.update(self._checkpoint_metadata())
        ckpt["config"] = asdict(self.config)
        return ckpt

    def save(self, path, episode, reward):
        """Persist checkpoint to disk."""
        torch.save(self.checkpoint(episode, reward), path)

    def load(self, path):
        """Load SAC checkpoint and restore all network weights."""
        checkpoint = _load_checkpoint(path, self.device)
        algo = str(checkpoint.get("algorithm", "sac")).lower().strip()
        assert algo == "sac", f"Checkpoint algorithm '{algo}' does not match SAC."
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


def train(config=None):
    """Main training loop: collect experience, populate replay, and update agent."""
    if config is None:
        config = Config()

    _init_supervisor()
    env = WebotsEnv(config)
    env.reset()
    run_id = Path(env.run_folder).name
    agent = SACAgent(env.observation_size, env.action_dim, config)
    replay = SequenceReplayBuffer(env.observation_size, env.action_dim, config)
    checkpoint_dir = _run_checkpoint_dir(_CHECKPOINT_DIR, run_id)
    final_model_path = _run_checkpoint_path(_CHECKPOINT_DIR, run_id, "final")
    print(f"[TRAIN][SAC] rnn={config.recurrent_cell.upper()} weights_dir={checkpoint_dir} final={final_model_path}", flush=True)
    print(f"[TRAIN][SAC] replay=on cap={config.replay_capacity} seq={config.sequence_length} stride={config.sequence_stride} batch={config.replay_batch_size}", flush=True)

    total_steps = 0
    best_reward = float("-inf")
    best_goal_reward = float("-inf")
    best_goal_episode = None
    rew_w, suc_w, gol_w, col_w, to_w = [], [], [], [], []
    start_time = time.perf_counter()
    metrics_logger = MetricsLogger(env.run_folder, algorithm="sac")

    for episode in range(config.episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        ep_end_reason = "max_steps"
        ep_obs, ep_act, ep_rew, ep_next, ep_done = [], [], [], [], []
        ep_goal = ep_success = False
        ep_speeds = []
        actor_state = agent.get_initial_state(batch_size=1)
        prev_done = True

        while not done:
            action, actor_state = agent.select_action(obs, actor_state, done=prev_done, deterministic=False)
            next_obs, reward, terminated, truncated, info = env.step(action)
            transition_done = bool(terminated)
            ep_obs.append(np.asarray(obs, dtype=np.float32))
            ep_act.append(np.asarray(action, dtype=np.float32))
            ep_rew.append(float(reward))
            ep_next.append(np.asarray(next_obs, dtype=np.float32))
            ep_done.append(transition_done)
            obs = next_obs
            episode_reward += reward
            ep_speeds.append(float(info.get("speed_norm", 0.0)))
            ep_goal = ep_goal or bool(info.get("goal_reached", False))
            ep_success = bool(info.get("success", False))
            done = transition_done
            prev_done = done
            total_steps += 1
            if done:
                reason = info.get("reset_reason", "")
                ep_end_reason = reason if reason else ("max_steps" if truncated else ep_end_reason)

        replay.add_episode(ep_obs, ep_act, ep_rew, ep_next, ep_done)

        if total_steps >= config.update_after_steps and replay.can_sample(config.replay_batch_size, config.min_replay_sequences):
            for _ in range(config.gradient_steps_per_episode):
                agent.update(replay.sample(config.replay_batch_size, agent.device))

        rew_w.append(episode_reward)
        suc_w.append(1.0 if ep_success else 0.0)
        gol_w.append(1.0 if ep_goal else 0.0)
        col_w.append(1.0 if ep_end_reason == "collision" else 0.0)
        to_w.append(1.0 if ep_end_reason == "max_steps" else 0.0)
        ckpt_flags = []

        if ep_end_reason == "goal" and episode_reward > best_goal_reward:
            best_goal_reward = episode_reward
            best_goal_episode = episode + 1
            env.robot.slam.save_episode(env.run_folder, episode + 1, episode_reward)
            ckpt = agent.checkpoint(best_goal_episode, best_goal_reward)
            ckpt["goal_episode"] = True
            _save_checkpoint_file(_CHECKPOINT_DIR, run_id, "best", ckpt)
            ckpt_flags.append("best_goal")
        elif best_goal_episode is None and episode_reward > best_reward:
            best_reward = episode_reward
            env.robot.slam.save_episode(env.run_folder, episode + 1, episode_reward)
            ckpt = agent.checkpoint(episode + 1, best_reward)
            ckpt["goal_episode"] = False
            _save_checkpoint_file(_CHECKPOINT_DIR, run_id, "best", ckpt)
            ckpt_flags.append("best")

        if config.save_every > 0 and (episode + 1) % config.save_every == 0:
            ckpt = agent.checkpoint(episode + 1, episode_reward)
            ckpt["goal_episode"] = ep_end_reason == "goal"
            _save_checkpoint_file(_CHECKPOINT_DIR, run_id, "checkpoint", ckpt)
            ckpt_flags.append("latest")

        r10 = float(np.mean(rew_w[-10:]))
        s10 = float(np.mean(suc_w[-10:]))
        g10 = float(np.mean(gol_w[-10:]))
        c10 = float(np.mean(col_w[-10:]))
        t10 = float(np.mean(to_w[-10:]))
        avg_spd = float(np.mean(ep_speeds)) * config.max_speed if ep_speeds else 0.0
        elapsed = time.perf_counter() - start_time
        ckpt_note = f" ckpt={'+'.join(ckpt_flags)}" if ckpt_flags else ""
        print(f"[TRAIN][SAC] ep={episode + 1:03d}/{config.episodes} r={episode_reward:8.2f} avg10={r10:8.2f} steps={env.current_step:4d} succ10={s10:4.2f} touch10={g10:4.2f} col10={c10:4.2f} to10={t10:4.2f} min_d={env.min_episode_distance:5.2f} avg_spd={avg_spd:4.2f}m/s end={ep_end_reason} replay={len(replay):4d} t={elapsed:7.1f}s{ckpt_note}", flush=True)
        metrics_logger.log(episode=episode + 1, reward=round(episode_reward, 4), avg10=round(r10, 4), steps=env.current_step,
                          success=int(ep_success), goal_touched=int(ep_goal), collision=int(ep_end_reason == "collision"),
                          timeout=int(ep_end_reason == "max_steps"), min_dist=round(env.min_episode_distance, 4),
                          avg_speed_ms=round(avg_spd, 3), end_reason=ep_end_reason, replay_size=len(replay),
                          elapsed_s=round(elapsed, 1))

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
