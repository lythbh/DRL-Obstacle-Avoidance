"""Soft Actor-Critic controller for the ALTINO Webots task."""
from __future__ import annotations

import sys, time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from controllers.Webots import WebotsEnv, _init_supervisor
from controllers.RNN import GRUActorCritic, LSTMActorCritic
import controllers.common.defaults as d
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
    episodes: int = d.SACDefaults.episodes
    update_after_steps: int = d.SACDefaults.update_after_steps
    updates_per_step: int = d.SACDefaults.updates_per_step
    gradient_steps_per_episode: int = d.SACDefaults.gradient_steps_per_episode
    save_every: int = d.SACDefaults.save_every
    gamma: float = d.SACDefaults.gamma
    tau: float = d.SACDefaults.tau
    actor_lr: float = d.SACDefaults.actor_lr
    critic_lr: float = d.SACDefaults.critic_lr
    alpha_lr: float = d.SACDefaults.alpha_lr
    initial_alpha: float = d.SACDefaults.initial_alpha
    auto_entropy_tuning: bool = d.SACDefaults.auto_entropy_tuning
    target_entropy_scale: float = d.SACDefaults.target_entropy_scale
    hidden_size: int = d.SACDefaults.hidden_size
    latent_size: int = d.SACDefaults.latent_size
    recurrent_cell: str = d.SACDefaults.recurrent_cell
    recurrent_hidden_size: Optional[int] = d.SACDefaults.recurrent_hidden_size
    recurrent_layers: int = d.SACDefaults.recurrent_layers
    lstm_hidden_size: int = d.SACDefaults.lstm_hidden_size
    lstm_layers: int = d.SACDefaults.lstm_layers
    log_std_min: float = d.SACDefaults.log_std_min
    log_std_max: float = d.SACDefaults.log_std_max
    sequence_length: int = d.RecurrentDefaults.sequence_length
    burn_in: int = d.RecurrentDefaults.burn_in
    sequence_stride: int = d.RecurrentDefaults.sequence_stride
    replay_capacity: int = d.SACDefaults.replay_capacity
    replay_batch_size: int = d.SACDefaults.replay_batch_size
    min_replay_sequences: int = d.SACDefaults.min_replay_sequences
    lidar_sector_dim: int = d.ENV_LIDAR_SECTOR_DIM
    pose_goal_dim: int = d.ENV_POSE_GOAL_DIM
    imu_feature_dim: int = d.ENV_IMU_FEATURE_DIM
    occupancy_grid_shape: Optional[Tuple[int, ...]] = d.ENV_OCCUPANCY_GRID_SHAPE
    max_steps: int = d.ENV_MAX_STEPS
    collision_threshold: float = d.ENV_COLLISION_THRESHOLD
    low_score_threshold: float = d.ENV_LOW_SCORE_THRESHOLD
    collision_penalty: float = d.REW_COLLISION_PENALTY
    progress_reward_scale: float = d.REW_PROGRESS_SCALE
    distance_reward_scale: float = d.REW_DISTANCE_SCALE
    heading_reward_scale: float = d.REW_HEADING_SCALE
    safety_reward_scale: float = d.REW_SAFETY_SCALE
    motion_reward_scale: float = d.REW_MOTION_SCALE
    slow_speed_threshold: float = d.REW_SLOW_SPEED_THRESHOLD
    slow_speed_penalty: float = d.REW_SLOW_SPEED_PENALTY
    high_speed_threshold: float = d.REW_HIGH_SPEED_THRESHOLD
    high_speed_bonus: float = d.REW_HIGH_SPEED_BONUS
    new_best_distance_bonus: float = d.REW_NEW_BEST_DISTANCE_BONUS
    step_penalty: float = d.REW_STEP_PENALTY
    endpoint: Tuple[float, float] = d.ENV_ENDPOINT
    goal_threshold: float = d.ENV_GOAL_THRESHOLD
    goal_stop_speed_threshold: float = d.ENV_GOAL_STOP_SPEED_THRESHOLD
    goal_success_reward: float = d.REW_GOAL_SUCCESS
    goal_stop_bonus: float = d.REW_GOAL_STOP_BONUS
    goal_hold_reward: float = d.REW_GOAL_HOLD
    goal_speed_penalty: float = d.REW_GOAL_SPEED_PENALTY
    goal_overshoot_penalty: float = d.REW_GOAL_OVERSHOOT_PENALTY
    reference_distance: Optional[float] = None
    enable_slam: bool = d.SLAM_ENABLE
    profile_slam: bool = d.SLAM_PROFILE
    slam_profile_interval: int = d.SLAM_PROFILE_INTERVAL
    save_slam_plots: bool = d.SLAM_SAVE_PLOTS
    force_cpu: bool = d.SLAM_FORCE_CPU
    max_steering_angle: float = d.ENV_MAX_STEERING_ANGLE
    min_speed: float = d.ENV_MIN_SPEED
    start_position: Optional[List[float]] = None
    start_rotation: Optional[List[float]] = None
    start_position_noise: float = d.ENV_START_POSITION_NOISE
    start_yaw_noise: float = d.ENV_START_YAW_NOISE
    max_speed: float = d.ENV_MAX_SPEED
    reset_settle_steps: int = d.ENV_RESET_SETTLE_STEPS

    def __post_init__(self):
        self.recurrent_cell = self.recurrent_cell.lower().strip()
        assert self.recurrent_cell in {"gru", "lstm"}, f"Unsupported recurrent_cell: {self.recurrent_cell}"
        if self.recurrent_hidden_size is None:
            self.recurrent_hidden_size = self.hidden_size
        if self.start_position is None:
            self.start_position = list(d.ENV_START_POSITION)
        if self.start_rotation is None:
            self.start_rotation = list(d.ENV_START_ROTATION)
        if self.reference_distance is None:
            start_xy = np.array(self.start_position[:2], dtype=np.float32)
            endpoint_xy = np.array(self.endpoint, dtype=np.float32)
            self.reference_distance = float(np.linalg.norm(start_xy - endpoint_xy))


class SequenceReplayBuffer:
    def __init__(self, obs_size: int, action_dim: int, config: Config) -> None:
        """Initialize replay buffer with fixed capacity for storing episode sequences."""
        self.capacity = config.replay_capacity
        self.seq_len = config.sequence_length
        self.buffer: List[Dict[str, np.ndarray]] = []
        self.pos = 0

    def __len__(self) -> int:
        """Return the current number of sequences in the replay buffer."""
        return len(self.buffer)

    def add_episode(self, ep_obs, ep_act, ep_rew, ep_next, ep_done) -> None:
        """Add an episode as fixed-length sequences to the replay buffer, cycling when at capacity."""
        if not ep_obs:
            return
        obs = np.asarray(ep_obs, dtype=np.float32)
        actions = np.asarray(ep_act, dtype=np.float32)
        rewards = np.asarray(ep_rew, dtype=np.float32).reshape(-1, 1)
        next_obs = np.asarray(ep_next, dtype=np.float32)
        dones = np.asarray(ep_done, dtype=np.float32).reshape(-1, 1)
        total = obs.shape[0]
        for start in range(0, total, self.seq_len):
            end = min(start + self.seq_len, total)
            entry = {
                "obs": obs[start:end].copy(),
                "actions": actions[start:end].copy(),
                "rewards": rewards[start:end].copy(),
                "next_obs": next_obs[start:end].copy(),
                "dones": dones[start:end].copy(),
                "valid_mask": np.ones(end - start, dtype=np.float32),
            }
            if len(self.buffer) < self.capacity:
                self.buffer.append(entry)
            else:
                self.buffer[self.pos] = entry
            self.pos = (self.pos + 1) % self.capacity

    def can_sample(self, batch_size: int, min_sequences: int) -> bool:
        """Check if buffer has enough sequences to sample a batch."""
        return len(self.buffer) >= max(batch_size, min_sequences)

    def sample(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """Sample random sequences from buffer, padding to uniform length."""
        indices = np.random.randint(0, len(self.buffer), size=batch_size)
        keys = ["obs", "actions", "rewards", "next_obs", "dones", "valid_mask"]
        result = {}
        for k in keys:
            tensors = []
            for i in indices:
                arr = self.buffer[i][k]
                t = torch.as_tensor(arr, dtype=torch.float32, device=device)
                if len(t) < self.seq_len:
                    pad = [(0, self.seq_len - len(t))] + [(0, 0)] * (t.ndim - 1)
                    flat_pads = [p for pad_dim in reversed(pad) for p in pad_dim]
                    t = nn.functional.pad(t, flat_pads)
                tensors.append(t)
            result[k] = torch.stack(tensors)
        return result


class SACAgent:
    def __init__(self, obs_size: int, action_dim: int, config: Config) -> None:
        """Initialize SAC agent with actor, dual critic networks, and entropy regularization."""
        self.config = config
        self.device = self._get_device()
        self.obs_size = obs_size
        self.action_dim = action_dim
        encoder_cls = LSTMActorCritic if config.recurrent_cell.lower().strip() == "lstm" else GRUActorCritic

        def _make_encoder():
            return encoder_cls(obs_size, action_dim, config).to(self.device)

        self.actor_enc = _make_encoder()
        self.actor_mean = nn.Linear(self.actor_enc.recurrent_hidden_size, action_dim).to(self.device)
        self.actor_log_std_head = nn.Linear(self.actor_enc.recurrent_hidden_size, action_dim).to(self.device)

        def _make_critic():
            enc = _make_encoder()
            head = nn.Sequential(
                nn.Linear(enc.recurrent_hidden_size + action_dim, config.hidden_size), nn.ReLU(),
                nn.Linear(config.hidden_size, config.hidden_size), nn.ReLU(),
                nn.Linear(config.hidden_size, 1),
            ).to(self.device)
            return enc, head

        self.q1_enc, self.q1_head = _make_critic()
        self.q2_enc, self.q2_head = _make_critic()
        self.target_q1_enc, self.target_q1_head = _make_critic()
        self.target_q2_enc, self.target_q2_head = _make_critic()
        self.target_q1_enc.load_state_dict(self.q1_enc.state_dict())
        self.target_q1_head.load_state_dict(self.q1_head.state_dict())
        self.target_q2_enc.load_state_dict(self.q2_enc.state_dict())
        self.target_q2_head.load_state_dict(self.q2_head.state_dict())

        self.actor_optimizer = torch.optim.Adam(
                    list(self.actor_enc.parameters()) + list(self.actor_mean.parameters()) + list(self.actor_log_std_head.parameters()),
                    lr=config.actor_lr,
                )
        self.critic_optimizer = torch.optim.Adam(
            list(self.q1_enc.parameters()) + list(self.q1_head.parameters()) +
            list(self.q2_enc.parameters()) + list(self.q2_head.parameters()),
            lr=config.critic_lr,
        )
        self.log_alpha = torch.tensor(np.log(config.initial_alpha), dtype=torch.float32, device=self.device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.alpha_lr)
        self.target_entropy = -float(action_dim) * float(config.target_entropy_scale)

        self.action_low = torch.tensor([-config.max_steering_angle, config.min_speed], dtype=torch.float32, device=self.device)
        self.action_high = torch.tensor([config.max_steering_angle, config.max_speed], dtype=torch.float32, device=self.device)
        self.action_center = (self.action_high + self.action_low) / 2.0
        self.action_scale = (self.action_high - self.action_low) / 2.0

    def _get_device(self) -> torch.device:
        """Determine whether to use CPU or CUDA GPU for training."""
        if self.config.force_cpu or not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device("cuda")

    def get_initial_state(self, batch_size: int = 1):
        """Get initial hidden state for the recurrent actor network."""
        return self.actor_enc.get_initial_state(batch_size, self.device)

    @property
    def alpha(self) -> torch.Tensor:
        """Get entropy regularization coefficient (exponential of log_alpha)."""
        return self.log_alpha.exp()

    def _checkpoint_metadata(self):
        """Return observation and action dimensions for checkpoint validation."""
        return {"obs_size": self.obs_size, "action_dim": self.action_dim}

    def _validate_checkpoint_metadata(self, checkpoint):
        """Verify checkpoint observation and action dimensions match current agent."""
        for key in ("obs_size", "action_dim"):
            saved = checkpoint.get(key)
            if saved is not None and int(saved) != getattr(self, key):
                raise ValueError(f"Checkpoint {key}={saved} != current {getattr(self, key)}")

    def _tensor_obs(self, obs: np.ndarray) -> torch.Tensor:
        """Convert numpy observation to device tensor with batch dimension."""
        return torch.as_tensor(obs, dtype=torch.float32, device=self.device).view(1, -1)

    def _sample_policy(self, obs, recurrent_state=None, done_mask=None, deterministic=False):
        """Sample action from squashed normal distribution with proper log probability."""
        features, next_state = self.actor_enc.encode_only(obs, recurrent_state=recurrent_state, done_mask=done_mask)
        mean = self.actor_mean(features)
        log_std = self.actor_log_std_head(features).clamp(self.config.log_std_min, self.config.log_std_max)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        pre_tanh = mean if deterministic else dist.rsample()
        tanh_action = torch.tanh(pre_tanh)
        log_prob = dist.log_prob(pre_tanh)
        log_prob -= torch.log(self.action_scale + 1e-6)
        log_prob -= torch.log(1.0 - tanh_action.pow(2) + 1e-6)
        return tanh_action * self.action_scale + self.action_center, log_prob.sum(dim=-1, keepdim=True), next_state

    def select_action(self, obs: np.ndarray, recurrent_state=None, done=False, deterministic=False):
        """Select action given observation and recurrent state for environment execution."""
        done_mask = torch.tensor([float(done)], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action, _, next_state = self._sample_policy(self._tensor_obs(obs), recurrent_state, done_mask, deterministic)
        return action.squeeze().cpu().numpy(), next_state

    def _soft_update(self, source_enc, source_head, target_enc, target_head) -> float:
        """Perform soft update of target networks using EMA; return L2 magnitude of parameter changes."""
        tau = self.config.tau
        total_change_sq = 0.0
        for tp, sp in zip(target_enc.parameters(), source_enc.parameters()):
            delta = sp.data * tau - tp.data * tau
            total_change_sq += float(delta.norm(2).item() ** 2)
            tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)
        for tp, sp in zip(target_head.parameters(), source_head.parameters()):
            delta = sp.data * tau - tp.data * tau
            total_change_sq += float(delta.norm(2).item() ** 2)
            tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)
        import math
        return float(math.sqrt(total_change_sq))

    @staticmethod
    def _sequence_loss_mask(valid_mask: torch.Tensor, burn_in: int) -> torch.Tensor:
        """Create learning mask that excludes burn-in steps and invalid positions from gradient computation."""
        valid_lengths = valid_mask.sum(dim=1).to(dtype=torch.long)
        start_index = torch.minimum(torch.full_like(valid_lengths, burn_in), torch.clamp(valid_lengths - 1, min=0))
        return valid_mask * (torch.arange(valid_mask.shape[1], device=valid_mask.device).unsqueeze(0) >= start_index.unsqueeze(1)).to(dtype=valid_mask.dtype)

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute mean of values, ignoring masked (invalid) positions."""
        return (values * mask.unsqueeze(-1).to(dtype=values.dtype)).sum() / mask.sum().clamp_min(1.0)

    def _critic_forward(self, q_enc, q_head, obs, action, done_mask=None):
        """Forward pass through critic encoder and head, handling shape mismatches."""
        features, next_state = q_enc.encode_only(obs, None, done_mask)
        if features.ndim != action.ndim:
            if features.ndim == 2 and action.ndim == 3 and action.shape[1] == 1:
                action = action.squeeze(1)
            elif features.ndim == 3 and action.ndim == 2:
                action = action.unsqueeze(1)
        return q_head(torch.cat([features, action], dim=-1)), next_state

    def update(self, batch):
        """Perform SAC training update, returning a dict with all loss components, gradient norms, and diagnostics."""
        if batch["obs"].shape[1] <= 0:
            return None

        valid_mask = batch["valid_mask"]
        learn_mask = self._sequence_loss_mask(valid_mask, min(self.config.burn_in, batch["obs"].shape[1] - 1))
        done_flags = batch["dones"].squeeze(-1)
        done_mask_obs = torch.zeros_like(done_flags)
        done_mask_obs[:, 0] = 1.0
        if done_flags.shape[1] > 1:
            done_mask_obs[:, 1:] = done_flags[:, :-1]
        done_mask_next = torch.cat([torch.ones_like(done_flags[:, :1]), done_flags], dim=1)

        scaled_rewards = batch["rewards"] * d.REW_SCALE

        with torch.no_grad():
            cat_obs = torch.cat([batch["obs"][:, :1], batch["next_obs"]], dim=1)
            na, nlp, _ = self._sample_policy(cat_obs, done_mask=done_mask_next, deterministic=False)
            na = na[:, 1:]
            nlp = nlp[:, 1:]
            ta = torch.cat([torch.zeros_like(batch["actions"][:, :1]), na], dim=1)
            tq1, _ = self._critic_forward(self.target_q1_enc, self.target_q1_head, cat_obs, ta, done_mask=done_mask_next)
            tq2, _ = self._critic_forward(self.target_q2_enc, self.target_q2_head, cat_obs, ta, done_mask=done_mask_next)
            tq = torch.min(tq1, tq2)[:, 1:]
            tq = tq - self.alpha.detach() * nlp
            target_q = scaled_rewards + (1.0 - batch["dones"]) * self.config.gamma * tq

        cq1, _ = self._critic_forward(self.q1_enc, self.q1_head, batch["obs"], batch["actions"], done_mask=done_mask_obs)
        cq2, _ = self._critic_forward(self.q2_enc, self.q2_head, batch["obs"], batch["actions"], done_mask=done_mask_obs)
        critic_loss = self._masked_mean((cq1 - target_q).pow(2), learn_mask)
        critic_loss += self._masked_mean((cq2 - target_q).pow(2), learn_mask)

        td_error = self._masked_mean(torch.abs(cq1 - target_q), learn_mask)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_params = list(self.q1_enc.parameters()) + list(self.q1_head.parameters()) + list(self.q2_enc.parameters()) + list(self.q2_head.parameters())
        grad_norm_critic = MetricsLogger.compute_grad_norm(critic_params)
        nn.utils.clip_grad_norm_(critic_params, max_norm=1.0)
        self.critic_optimizer.step()

        new_action, log_prob, _ = self._sample_policy(batch["obs"], done_mask=done_mask_obs, deterministic=False)
        q1_pi, _ = self._critic_forward(self.q1_enc, self.q1_head, batch["obs"], new_action, done_mask=done_mask_obs)
        q2_pi, _ = self._critic_forward(self.q2_enc, self.q2_head, batch["obs"], new_action, done_mask=done_mask_obs)
        actor_loss = self._masked_mean(self.alpha.detach() * log_prob - torch.min(q1_pi, q2_pi), learn_mask)
        policy_entropy = self._masked_mean(-log_prob, learn_mask)

        alpha_loss = torch.tensor(0.0, device=self.device)
        if self.config.auto_entropy_tuning:
            alpha_loss = -self._masked_mean(self.log_alpha * (log_prob + self.target_entropy).detach(), learn_mask)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_params = list(self.actor_enc.parameters()) + list(self.actor_mean.parameters()) + list(self.actor_log_std_head.parameters())
        grad_norm_actor = MetricsLogger.compute_grad_norm(actor_params)
        nn.utils.clip_grad_norm_(actor_params, max_norm=1.0)
        self.actor_optimizer.step()

        if self.config.auto_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        target_update_mag = self._soft_update(self.q1_enc, self.q1_head, self.target_q1_enc, self.target_q1_head)
        target_update_mag += self._soft_update(self.q2_enc, self.q2_head, self.target_q2_enc, self.target_q2_head)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.item()),
            "policy_entropy": float(policy_entropy.item()),
            "value_residual": float(td_error.item()),
            "grad_norm_actor": round(grad_norm_actor, 6),
            "grad_norm_critic": round(grad_norm_critic, 6),
            "target_update_magnitude": round(target_update_mag, 6),
            "lr_actor": round(float(self.actor_optimizer.param_groups[0]["lr"]), 8),
            "lr_critic": round(float(self.critic_optimizer.param_groups[0]["lr"]), 8),
        }

    def checkpoint(self, episode, reward):
        """Create checkpoint dictionary with all network weights, optimizers, and training state."""
        ckpt = {
            "episode": episode, "reward": reward, "algorithm": "sac",
            "actor_enc": self.actor_enc.state_dict(),
            "actor_mean": self.actor_mean.state_dict(),
            "actor_log_std": self.actor_log_std_head.state_dict(),
            "critic1_enc": self.q1_enc.state_dict(), "critic1_head": self.q1_head.state_dict(),
            "critic2_enc": self.q2_enc.state_dict(), "critic2_head": self.q2_head.state_dict(),
            "target_critic1_enc": self.target_q1_enc.state_dict(), "target_critic1_head": self.target_q1_head.state_dict(),
            "target_critic2_enc": self.target_q2_enc.state_dict(), "target_critic2_head": self.target_q2_head.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
        }
        ckpt.update(self._checkpoint_metadata())
        ckpt["config"] = asdict(self.config)
        return ckpt

    def save(self, path, episode, reward):
        """Save checkpoint to disk."""
        torch.save(self.checkpoint(episode, reward), path)

    def load(self, path):
        """Load checkpoint from disk and restore all network weights and optimizer states."""
        checkpoint = _load_checkpoint(path, self.device)
        algo = str(checkpoint.get("algorithm", "sac")).lower().strip()
        assert algo == "sac", f"Checkpoint algorithm '{algo}' does not match SAC."
        self._validate_checkpoint_metadata(checkpoint)
        self.actor_enc.load_state_dict(checkpoint["actor_enc"])
        self.actor_mean.load_state_dict(checkpoint["actor_mean"])
        self.actor_log_std_head.load_state_dict(checkpoint["actor_log_std"])
        self.q1_enc.load_state_dict(checkpoint["critic1_enc"])
        self.q1_head.load_state_dict(checkpoint["critic1_head"])
        self.q2_enc.load_state_dict(checkpoint["critic2_enc"])
        self.q2_head.load_state_dict(checkpoint["critic2_head"])
        self.target_q1_enc.load_state_dict(checkpoint.get("target_critic1_enc", checkpoint["critic1_enc"]))
        self.target_q1_head.load_state_dict(checkpoint.get("target_critic1_head", checkpoint["critic1_head"]))
        self.target_q2_enc.load_state_dict(checkpoint.get("target_critic2_enc", checkpoint["critic2_enc"]))
        self.target_q2_head.load_state_dict(checkpoint.get("target_critic2_head", checkpoint["critic2_head"]))
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        if self.config.auto_entropy_tuning and "alpha_optimizer" in checkpoint:
            self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
        if "log_alpha" in checkpoint:
            self.log_alpha.data.copy_(checkpoint["log_alpha"].to(self.device))
        return checkpoint


def train(config=None):
    """Main training loop: collect episodes, sample from replay buffer, perform SAC updates, and save checkpoints."""
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
    metrics_logger.log_hyperparams(asdict(config), recurrent_cell=config.recurrent_cell,
                                   obs_size=env.observation_size, action_dim=env.action_dim)

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

        all_update_metrics = []
        if total_steps >= config.update_after_steps and replay.can_sample(config.replay_batch_size, config.min_replay_sequences):
            for _ in range(config.gradient_steps_per_episode):
                upd = agent.update(replay.sample(config.replay_batch_size, agent.device))
                if upd is not None:
                    all_update_metrics.append(upd)
                    metrics_logger.log_update(
                        global_step=total_steps, episode=episode + 1,
                        recurrent_cell=config.recurrent_cell,
                        **upd,
                    )

        act_stats = MetricsLogger.compute_action_stats(ep_act)
        obs_stats = MetricsLogger.compute_obs_stats(ep_obs)
        ep_val_residual = 0.0
        agg_upd = MetricsLogger.aggregate_update_metrics(all_update_metrics)

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

        metrics_logger.log_episode(
            episode=episode + 1,
            global_step=total_steps,
            reward=round(episode_reward, 4),
            avg10=round(r10, 4),
            length=env.current_step,
            success=int(ep_success),
            goal_touched=int(ep_goal),
            collision=int(ep_end_reason == "collision"),
            timeout=int(ep_end_reason == "max_steps"),
            min_dist=round(env.min_episode_distance, 4),
            avg_speed_ms=round(avg_spd, 3),
            end_reason=ep_end_reason,
            elapsed_s=round(elapsed, 1),
            recurrent_cell=config.recurrent_cell,
            replay_buffer_size=len(replay),
            **act_stats,
            **obs_stats,
            **agg_upd,
        )

    metrics_logger.close()
    print(f"[TRAIN][SAC] metrics saved to {metrics_logger.path}", flush=True)
    print(f"[TRAIN][SAC] updates saved to {metrics_logger.update_path}", flush=True)
    print(f"[TRAIN][SAC] hyperparams saved to {metrics_logger.hyperparams_path}", flush=True)
    final_reward = best_goal_reward if best_goal_episode is not None else best_reward
    agent.save(final_model_path, "final", final_reward)
    elapsed = time.perf_counter() - start_time
    print(f"[TRAIN][SAC] final reward={final_reward:.2f} t={elapsed:7.1f}s", flush=True)
    env.robot.stop()
    print("[TRAIN][SAC] done", flush=True)


if __name__ == "__main__":
    train()





