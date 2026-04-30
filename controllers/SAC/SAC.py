"""Soft Actor-Critic controller for the ALTINO Webots task."""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from controllers.Webots import WebotsEnv, _init_supervisor


@dataclass
class Config:
    episodes: int = 500
    batch_size: int = 256
    replay_size: int = 200_000
    warmup_steps: int = 2_000
    update_after_steps: int = 1_000
    updates_per_step: int = 1
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    initial_alpha: float = 0.2
    auto_entropy_tuning: bool = True
    hidden_size: int = 256
    recurrent_cell: str = "gru"
    recurrent_hidden_size: Optional[int] = None
    recurrent_layers: int = 1
    log_std_min: float = -5.0
    log_std_max: float = 2.0

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


class ReplayBuffer:
    def __init__(self, obs_size: int, action_dim: int, capacity: int) -> None:
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_size), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_size), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.index = 0
        self.size = 0

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool) -> None:
        self.obs[self.index] = np.asarray(obs, dtype=np.float32)
        self.actions[self.index] = np.asarray(action, dtype=np.float32)
        self.rewards[self.index, 0] = float(reward)
        self.next_obs[self.index] = np.asarray(next_obs, dtype=np.float32)
        self.dones[self.index, 0] = 1.0 if done else 0.0
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def __len__(self) -> int:
        return self.size

    def sample(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs[indices], dtype=torch.float32, device=device),
            "actions": torch.as_tensor(self.actions[indices], dtype=torch.float32, device=device),
            "rewards": torch.as_tensor(self.rewards[indices], dtype=torch.float32, device=device),
            "next_obs": torch.as_tensor(self.next_obs[indices], dtype=torch.float32, device=device),
            "dones": torch.as_tensor(self.dones[indices], dtype=torch.float32, device=device),
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

    def _apply_done_mask(self, state: Any, done_mask: Optional[torch.Tensor]) -> Any:
        if state is None or done_mask is None:
            return state
        mask = done_mask.to(dtype=torch.float32)
        if mask.ndim == 0:
            mask = mask.view(1)
        mask = 1.0 - mask.view(1, -1, 1)
        if self.cell == "lstm":
            h, c = state
            return h * mask, c * mask
        return state * mask

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
        if done_mask is not None:
            state = self._apply_done_mask(state, torch.as_tensor(done_mask, dtype=torch.float32, device=latent.device))
        core_out, next_state = self.core(latent, state)
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

        self.replay_buffer = ReplayBuffer(obs_size, action_dim, config.replay_size)

    def _get_device(self) -> torch.device:
        return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

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
        log_prob = dist.log_prob(pre_tanh) - torch.log(1.0 - tanh_action.pow(2) + 1e-6)
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

    def select_random_action(self) -> np.ndarray:
        return np.random.uniform(low=self.action_low.cpu().numpy(), high=self.action_high.cpu().numpy()).astype(np.float32)

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        tau = self.config.tau
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)

    def update(self) -> Optional[Dict[str, float]]:
        if len(self.replay_buffer) < self.config.batch_size:
            return None

        batch = self.replay_buffer.sample(self.config.batch_size, self.device)

        with torch.no_grad():
            next_action, next_log_prob, _ = self._sample_policy(batch["next_obs"], deterministic=False)
            target_q1, _ = self.target_q1(batch["next_obs"], next_action)
            target_q2, _ = self.target_q2(batch["next_obs"], next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha.detach() * next_log_prob
            target_q = batch["rewards"] + (1.0 - batch["dones"]) * self.config.gamma * target_q

        current_q1, _ = self.q1(batch["obs"], batch["actions"])
        current_q2, _ = self.q2(batch["obs"], batch["actions"])
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), max_norm=1.0)
        self.critic_optimizer.step()

        new_action, log_prob, _ = self._sample_policy(batch["obs"], deterministic=False)
        q1_pi, _ = self.q1(batch["obs"], new_action)
        q2_pi, _ = self.q2(batch["obs"], new_action)
        actor_loss = (self.alpha.detach() * log_prob - torch.min(q1_pi, q2_pi)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        alpha_loss = torch.tensor(0.0, device=self.device)
        if self.config.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
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

    def checkpoint(self, episode: int, reward: float) -> Dict[str, Any]:
        checkpoint = {
            "episode": episode,
            "reward": reward,
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

    def save(self, path: str, episode: int, reward: float) -> None:
        torch.save(self.checkpoint(episode, reward), path)

    def load(self, path: str) -> Dict[str, Any]:
        checkpoint = torch.load(path, map_location=self.device)
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
    obs, _ = env.reset()
    agent = SACAgent(env.observation_size, env.action_dim, config)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    repo_root = Path(__file__).resolve().parents[2]
    run_dir = repo_root / "controllers" / "SAC" / "checkpoints" / ts
    os.makedirs(run_dir, exist_ok=True)

    total_steps = 0
    best_reward = float("-inf")

    for episode in range(config.episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        actor_state = agent.get_initial_state(batch_size=1)
        prev_done = True

        while not done:
            if total_steps < config.warmup_steps:
                action = agent.select_random_action()
            else:
                action, actor_state = agent.select_action(
                    obs,
                    recurrent_state=actor_state,
                    done=prev_done,
                    deterministic=False,
                )

            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.replay_buffer.add(obs, action, reward, next_obs, terminated)

            obs = next_obs
            episode_reward += reward
            done = terminated or truncated
            prev_done = done
            total_steps += 1

            if total_steps >= config.update_after_steps:
                for _ in range(config.updates_per_step):
                    agent.update()

        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(str(run_dir / "best_model.pth"), episode, episode_reward)

        if (episode + 1) % 10 == 0:
            agent.save(str(run_dir / "latest_model.pth"), episode, episode_reward)
            print(f"[SAC] Episode {episode + 1:4d} | Reward {episode_reward:8.2f} | Best {best_reward:8.2f}", flush=True)

    agent.save(str(run_dir / "final_model.pth"), config.episodes, best_reward)


if __name__ == "__main__":
    train()
