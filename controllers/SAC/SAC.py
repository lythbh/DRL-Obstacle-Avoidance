"""Soft Actor-Critic controller for the ALTINO Webots task."""

from __future__ import annotations

import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from controllers.Webots import WebotsEnv, _init_supervisor

_CONTROLLER_DIR = Path(__file__).resolve().parent
_CHECKPOINT_DIR = _CONTROLLER_DIR / "checkpoints"


def _checkpoint_path(filename: str) -> str:
    """Return a checkpoint path pinned to the SAC controller directory."""
    return str(_CONTROLLER_DIR / filename)


def _dated_checkpoint_path(run_id: str, filename: str) -> str:
    """Return a checkpoint path inside the timestamped checkpoint folder."""
    checkpoint_dir = _CHECKPOINT_DIR / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return str(checkpoint_dir / filename)


def _load_checkpoint(path: str, map_location: torch.device) -> Dict[str, Any]:
    """Load local controller checkpoints without relying on PyTorch's changing default."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


@dataclass
class Config:
    episodes: int = 500
    batch_size: int = 64
    replay_size: int = 200_000
    update_after_steps: int = 1_000
    updates_per_step: int = 1
    save_every: int = 100
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    initial_alpha: float = 0.2
    auto_entropy_tuning: bool = True
    hidden_size: int = 128
    recurrent_cell: str = "gru"
    recurrent_hidden_size: Optional[int] = None
    recurrent_layers: int = 1
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    sequence_length: int = 16
    burn_in: int = 4

    lidar_sector_dim: int = 16
    pose_goal_dim: int = 7
    imu_feature_dim: int = 10
    occupancy_grid_shape: Optional[Tuple[int, ...]] = None

    max_steps: int = 4000
    collision_threshold: float = 0.1
    low_score_threshold: float = -800.0
    collision_penalty: float = -20.0
    progress_reward_scale: float = 3.0
    distance_reward_scale: float = 2.0
    heading_reward_scale: float = 0.5
    safety_reward_scale: float = 0.2
    motion_reward_scale: float = 0.05
    new_best_distance_bonus: float = 1.0
    step_penalty: float = -0.01
    endpoint: Tuple[float, float] = (2.0, 0.0)
    goal_threshold: float = 0.1
    goal_stop_bonus: float = 120.0
    goal_hold_reward: float = 10.0
    goal_speed_penalty: float = -60.0
    goal_overshoot_penalty: float = -50.0
    goal_score_threshold: float = 5000.0
    reference_distance: Optional[float] = None

    max_steering_angle: float = 0.9
    min_speed: float = 0.0
    start_position: Optional[List[float]] = None
    start_rotation: Optional[List[float]] = None
    start_position_noise: float = 0.03
    start_yaw_noise: float = 0.2
    reset_settle_steps: int = 10
    max_speed: float = 10.0

    def __post_init__(self) -> None:
        self.recurrent_cell = self.recurrent_cell.lower().strip()
        if self.recurrent_cell not in {"ff", "gru", "lstm"}:
            raise ValueError(f"Unsupported recurrent_cell: {self.recurrent_cell}")
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be > 0")
        if self.burn_in < 0:
            raise ValueError("burn_in must be >= 0")
        if self.burn_in >= self.sequence_length:
            self.burn_in = max(self.sequence_length - 1, 0)
        if self.recurrent_hidden_size is None:
            self.recurrent_hidden_size = self.hidden_size
        if self.start_position is None:
            self.start_position = [-2.0, 0.0, 0.02]
        if self.start_rotation is None:
            self.start_rotation = [0.0, 0.0, 1.0, 0.0]
        if self.reference_distance is None:
            start_xy = np.array(self.start_position[:2], dtype=np.float32)
            endpoint_xy = np.array(self.endpoint, dtype=np.float32)
            self.reference_distance = float(np.linalg.norm(start_xy - endpoint_xy))


class SequenceReplayBuffer:
    def __init__(self, obs_size: int, action_dim: int, capacity: int, sequence_length: int) -> None:
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.obs_size = obs_size
        self.action_dim = action_dim
        self.episodes: deque[Dict[str, np.ndarray]] = deque()
        self.current_episode: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = []
        self.size = 0

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool) -> None:
        transition = (
            np.asarray(obs, dtype=np.float32),
            np.asarray(action, dtype=np.float32),
            float(reward),
            np.asarray(next_obs, dtype=np.float32),
            bool(done),
        )
        self.current_episode.append(transition)
        if done:
            self._commit_episode()

    def _commit_episode(self) -> None:
        if not self.current_episode:
            return
        obs = np.stack([t[0] for t in self.current_episode], axis=0)
        actions = np.stack([t[1] for t in self.current_episode], axis=0)
        rewards = np.array([t[2] for t in self.current_episode], dtype=np.float32).reshape(-1, 1)
        next_obs = np.stack([t[3] for t in self.current_episode], axis=0)
        dones = np.array([t[4] for t in self.current_episode], dtype=np.float32).reshape(-1, 1)
        self.episodes.append(
            {
                "obs": obs,
                "actions": actions,
                "rewards": rewards,
                "next_obs": next_obs,
                "dones": dones,
            }
        )
        self.size += len(obs)
        self.current_episode = []
        self._trim()

    def _trim(self) -> None:
        while self.size > self.capacity and self.episodes:
            removed = self.episodes.popleft()
            self.size -= len(removed["obs"])

    def __len__(self) -> int:
        return self.size + len(self.current_episode)

    def _current_episode_dict(self) -> Optional[Dict[str, np.ndarray]]:
        if len(self.current_episode) < self.sequence_length:
            return None
        obs = np.stack([t[0] for t in self.current_episode], axis=0)
        actions = np.stack([t[1] for t in self.current_episode], axis=0)
        rewards = np.array([t[2] for t in self.current_episode], dtype=np.float32).reshape(-1, 1)
        next_obs = np.stack([t[3] for t in self.current_episode], axis=0)
        dones = np.array([t[4] for t in self.current_episode], dtype=np.float32).reshape(-1, 1)
        return {
            "obs": obs,
            "actions": actions,
            "rewards": rewards,
            "next_obs": next_obs,
            "dones": dones,
        }

    def _eligible_episodes(self) -> List[Dict[str, np.ndarray]]:
        eligible = [ep for ep in self.episodes if len(ep["obs"]) >= self.sequence_length]
        current = self._current_episode_dict()
        if current is not None:
            eligible.append(current)
        return eligible

    def sample(self, batch_size: int, device: torch.device) -> Optional[Dict[str, torch.Tensor]]:
        eligible = self._eligible_episodes()
        if not eligible:
            return None
        obs_batch: List[np.ndarray] = []
        actions_batch: List[np.ndarray] = []
        rewards_batch: List[np.ndarray] = []
        next_obs_batch: List[np.ndarray] = []
        dones_batch: List[np.ndarray] = []

        for _ in range(batch_size):
            episode = eligible[np.random.randint(0, len(eligible))]
            ep_len = len(episode["obs"])
            start = np.random.randint(0, ep_len - self.sequence_length + 1)
            end = start + self.sequence_length
            obs_batch.append(episode["obs"][start:end])
            actions_batch.append(episode["actions"][start:end])
            rewards_batch.append(episode["rewards"][start:end])
            next_obs_batch.append(episode["next_obs"][start:end])
            dones_batch.append(episode["dones"][start:end])

        return {
            "obs": torch.as_tensor(np.stack(obs_batch, axis=0), dtype=torch.float32, device=device),
            "actions": torch.as_tensor(np.stack(actions_batch, axis=0), dtype=torch.float32, device=device),
            "rewards": torch.as_tensor(np.stack(rewards_batch, axis=0), dtype=torch.float32, device=device),
            "next_obs": torch.as_tensor(np.stack(next_obs_batch, axis=0), dtype=torch.float32, device=device),
            "dones": torch.as_tensor(np.stack(dones_batch, axis=0), dtype=torch.float32, device=device),
        }


class RecurrentEncoder(nn.Module):
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
            self.core = None

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
        if done_mask is None:
            return None
        mask = torch.as_tensor(done_mask, dtype=torch.float32, device=device)
        if mask.ndim == 0:
            mask = mask.view(1, 1)
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
        return self.head(torch.cat([features, action], dim=-1)), next_state


class SACAgent:
    def __init__(self, obs_size: int, action_dim: int, config: Config) -> None:
        self.config = config
        self.device = self._get_device()
        self.obs_size = obs_size
        self.action_dim = action_dim

        self.actor = GaussianActor(obs_size, action_dim, config).to(self.device)
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
        self.target_entropy = -float(action_dim)

        self.action_low = torch.tensor([-config.max_steering_angle, config.min_speed], dtype=torch.float32, device=self.device)
        self.action_high = torch.tensor([config.max_steering_angle, config.max_speed], dtype=torch.float32, device=self.device)
        self.action_center = (self.action_high + self.action_low) / 2.0
        self.action_scale = (self.action_high - self.action_low) / 2.0

        self.replay_buffer = SequenceReplayBuffer(
            obs_size,
            action_dim,
            config.replay_size,
            config.sequence_length,
        )
        print(f"[SAC] Using recurrent cell: {self.config.recurrent_cell.upper()}", flush=True)

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

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

    def _tensor_obs(self, obs: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(obs, dtype=torch.float32, device=self.device).view(1, -1)

    def _squash_action(self, action: torch.Tensor) -> torch.Tensor:
        return action * self.action_scale + self.action_center

    def _sample_policy(
        self,
        obs: torch.Tensor,
        recurrent_state: Optional[Any] = None,
        done_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Any]]:
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
        tau = self.config.tau
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)

    def update(self) -> Optional[Dict[str, float]]:
        if len(self.replay_buffer) < self.config.batch_size:
            return None

        batch = self.replay_buffer.sample(self.config.batch_size, self.device)
        if batch is None:
            return None

        burn_in = min(self.config.burn_in, self.config.sequence_length - 1)
        learn_slice = slice(burn_in, None)
        done_flags = batch["dones"].squeeze(-1)
        done_mask_obs = torch.zeros_like(done_flags)
        done_mask_obs[:, 0] = 1.0
        done_mask_obs[:, 1:] = done_flags[:, :-1]
        done_mask_next = done_flags.clone()
        done_mask_next[:, 0] = 1.0

        def _ensure_time_dim(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.unsqueeze(1) if tensor.ndim == 2 else tensor

        with torch.no_grad():
            next_action, next_log_prob, _ = self._sample_policy(
                batch["next_obs"],
                done_mask=done_mask_next,
                deterministic=False,
            )
            target_q1, _ = self.target_q1(batch["next_obs"], next_action, done_mask=done_mask_next)
            target_q2, _ = self.target_q2(batch["next_obs"], next_action, done_mask=done_mask_next)
            target_q = torch.min(target_q1, target_q2) - self.alpha.detach() * next_log_prob
            target_q = batch["rewards"] + (1.0 - batch["dones"]) * self.config.gamma * target_q

        current_q1, _ = self.q1(batch["obs"], batch["actions"], done_mask=done_mask_obs)
        current_q2, _ = self.q2(batch["obs"], batch["actions"], done_mask=done_mask_obs)
        target_q = _ensure_time_dim(target_q)
        current_q1 = _ensure_time_dim(current_q1)
        current_q2 = _ensure_time_dim(current_q2)

        critic_loss = (
            F.mse_loss(current_q1[:, learn_slice], target_q[:, learn_slice])
            + F.mse_loss(current_q2[:, learn_slice], target_q[:, learn_slice])
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), max_norm=1.0)
        self.critic_optimizer.step()

        new_action, log_prob, _ = self._sample_policy(
            batch["obs"],
            done_mask=done_mask_obs,
            deterministic=False,
        )
        q1_pi, _ = self.q1(batch["obs"], new_action, done_mask=done_mask_obs)
        q2_pi, _ = self.q2(batch["obs"], new_action, done_mask=done_mask_obs)
        log_prob = _ensure_time_dim(log_prob)
        q1_pi = _ensure_time_dim(q1_pi)
        q2_pi = _ensure_time_dim(q2_pi)
        actor_loss = (self.alpha.detach() * log_prob - torch.min(q1_pi, q2_pi))
        actor_loss = actor_loss[:, learn_slice].mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        alpha_loss = torch.tensor(0.0, device=self.device)
        if self.config.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob[:, learn_slice] + self.target_entropy).detach()).mean()
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
        torch.save(self.checkpoint(episode, reward), path)

    def load(self, path: str) -> Dict[str, Any]:
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


def train(config: Optional[Config] = None) -> None:
    if config is None:
        config = Config()

    _init_supervisor()
    env = WebotsEnv(config)
    env.reset()
    run_id = Path(env.run_folder).name
    agent = SACAgent(env.observation_size, env.action_dim, config)

    print(
        f"[TRAIN][SAC] episodes={config.episodes} "
        f"obs={env.observation_size} act={env.action_dim} cell={config.recurrent_cell.upper()} "
        f"seq={config.sequence_length} burn_in={config.burn_in}",
        flush=True,
    )

    total_steps = 0
    best_reward = float("-inf")
    best_goal_reward = float("-inf")
    best_goal_episode: Optional[int] = None
    reward_window: List[float] = []
    start_time = time.perf_counter()

    for episode in range(config.episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        episode_end_reason = "max_steps"
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
            agent.replay_buffer.add(obs, action, reward, next_obs, terminated or truncated)

            obs = next_obs
            episode_reward += reward
            done = terminated or truncated
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

            if total_steps >= config.update_after_steps:
                for _ in range(config.updates_per_step):
                    agent.update()

        reward_window.append(episode_reward)
        if (episode + 1) % 10 == 0 or episode_end_reason == "goal" or episode == config.episodes - 1:
            rolling_reward = float(np.mean(reward_window[-10:]))
            elapsed = time.perf_counter() - start_time
            print(
                f"[TRAIN][SAC] ep={episode + 1:03d}/{config.episodes} "
                f"r={episode_reward:8.2f} avg10={rolling_reward:8.2f} steps={env.current_step:4d} "
                f"min_d={env.min_episode_distance:5.2f} end={episode_end_reason} t={elapsed:7.1f}s",
                flush=True,
            )

        if episode_end_reason == "goal":
            if episode_reward > best_goal_reward:
                best_goal_reward = episode_reward
                best_goal_episode = episode + 1
                env.robot.slam.save_episode(env.run_folder, episode + 1, episode_reward)
                checkpoint = agent.checkpoint(best_goal_episode, best_goal_reward)
                checkpoint["goal_episode"] = True
                torch.save(checkpoint, _dated_checkpoint_path(run_id, "best_model.pth"))
                print(f"[CKPT][SAC] goal ep={best_goal_episode:03d} r={best_goal_reward:.2f}", flush=True)
        elif best_goal_episode is None and episode_reward > best_reward:
            best_reward = episode_reward
            env.robot.slam.save_episode(env.run_folder, episode + 1, episode_reward)
            checkpoint = agent.checkpoint(episode + 1, best_reward)
            checkpoint["goal_episode"] = False
            torch.save(checkpoint, _dated_checkpoint_path(run_id, "best_model.pth"))
            print(f"[CKPT][SAC] best ep={episode + 1:03d} r={best_reward:.2f}", flush=True)

        if config.save_every > 0 and (episode + 1) % config.save_every == 0:
            latest_checkpoint = agent.checkpoint(episode + 1, episode_reward)
            latest_checkpoint["goal_episode"] = episode_end_reason == "goal"
            torch.save(latest_checkpoint, _dated_checkpoint_path(run_id, "latest_model.pth"))
            print(f"[CKPT][SAC] latest ep={episode + 1:03d} r={episode_reward:.2f}", flush=True)

    final_reward = best_goal_reward if best_goal_episode is not None else best_reward
    agent.save(_checkpoint_path("final_model.pth"), "final", final_reward)
    elapsed = time.perf_counter() - start_time
    print(f"[TRAIN][SAC] final reward={final_reward:.2f} t={elapsed:7.1f}s", flush=True)

    env.robot.motors.stop()
    print("[TRAIN][SAC] done", flush=True)


if __name__ == "__main__":
    train()
