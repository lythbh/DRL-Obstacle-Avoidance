"""SACAgent extracted from SAC.py."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import nn

import controllers.common.SAC_defaults as d
from controllers.common.metrics_logger import MetricsLogger
from controllers.SAC.critics import build_critic_pair
from controllers.RNN import GRUActorCritic, LSTMActorCritic


class RunningMeanStd:
    def __init__(self, shape: int, epsilon: float = 1e-4) -> None:
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon
        self.epsilon = epsilon

    def update(self, x: np.ndarray) -> None:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        self.mean = new_mean.astype(np.float32)
        self.var = (M2 / tot_count).astype(np.float32)
        self.count = float(tot_count)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


class SACAgent:
    def __init__(self, obs_size: int, action_dim: int, config: Any) -> None:
        """Initialize SAC agent with actor, dual critic networks, and entropy regularization."""
        self.config = config
        self.device = self._get_device()
        self.obs_size = obs_size
        self.action_dim = action_dim
        self._build_actor_encoder()
        self._build_critics()

        self.actor_optimizer = torch.optim.Adam(self._actor_params, lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self._critic_params, lr=config.critic_lr)
        self.log_alpha = torch.tensor(np.log(config.initial_alpha), dtype=torch.float32, device=self.device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.alpha_lr)
        self.target_entropy = -float(action_dim) * float(config.target_entropy_scale)

        self.action_low = torch.tensor([-config.max_steering_angle, config.min_speed], dtype=torch.float32, device=self.device)
        self.action_high = torch.tensor([config.max_steering_angle, config.max_speed], dtype=torch.float32, device=self.device)
        self.action_center = (self.action_high + self.action_low) / 2.0
        self.action_scale = (self.action_high - self.action_low) / 2.0
        self.obs_rms = RunningMeanStd(obs_size)
        self._target_update_counter = 0

    @property
    def _actor_params(self):
        return (
            list(self.actor_enc.parameters())
            + list(self.actor_mean.parameters())
            + list(self.actor_log_std_head.parameters())
        )

    @property
    def _critic_params(self):
        return (
            list(self.q1_enc.parameters())
            + list(self.q1_head.parameters())
            + list(self.q2_enc.parameters())
            + list(self.q2_head.parameters())
        )

    def _build_actor_encoder(self):
        """Build actor encoder network and output heads."""
        encoder_cls = LSTMActorCritic if self.config.recurrent_cell.lower().strip() == "lstm" else GRUActorCritic
        self.actor_enc = encoder_cls(self.obs_size, self.action_dim, self.config).to(self.device)
        self.actor_mean = nn.Linear(
            self.actor_enc.recurrent_hidden_size, self.action_dim
        ).to(self.device)
        self.actor_log_std_head = nn.Linear(
            self.actor_enc.recurrent_hidden_size, self.action_dim
        ).to(self.device)

    def _create_critic_pair(self):
        return build_critic_pair(self.obs_size, self.action_dim, self.config, self.device)

    def _build_critics(self):
        """Build dual Q-networks and their target networks."""
        self.q1_enc, self.q1_head = self._create_critic_pair()
        self.q2_enc, self.q2_head = self._create_critic_pair()
        self.target_q1_enc, self.target_q1_head = self._create_critic_pair()
        self.target_q2_enc, self.target_q2_head = self._create_critic_pair()
        self.target_q1_enc.load_state_dict(self.q1_enc.state_dict())
        self.target_q1_head.load_state_dict(self.q1_head.state_dict())
        self.target_q2_enc.load_state_dict(self.q2_enc.state_dict())
        self.target_q2_head.load_state_dict(self.q2_head.state_dict())

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
        return self.log_alpha.exp().clamp(min=1e-4)

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
        done_mask = torch.tensor([float(done)], dtype=torch.float32, device=self.device)
        obs_norm = self.obs_rms.normalize(np.asarray(obs, dtype=np.float32))
        with torch.no_grad():
            action, _, next_state = self._sample_policy(self._tensor_obs(obs_norm), recurrent_state, done_mask, deterministic)
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

    def _critic_forward(self, q_enc, q_head, obs, action, recurrent_state=None, done_mask=None):
        """Forward pass through critic encoder and head, handling shape mismatches."""
        features, next_state = q_enc.encode_only(obs, recurrent_state=recurrent_state, done_mask=done_mask)
        if features.ndim != action.ndim:
            if features.ndim == 2 and action.ndim == 3 and action.shape[1] == 1:
                action = action.squeeze(1)
            elif features.ndim == 3 and action.ndim == 2:
                action = action.unsqueeze(1)
        q_val = q_head(torch.cat([features, action], dim=-1))
        return q_val, next_state

    def _prepare_done_masks(self, batch):
        """Compute done masks for observation and next-observation alignment."""
        done_flags = batch["dones"].squeeze(-1)
        sequence_start = batch.get("sequence_start")
        if sequence_start is None:
            sequence_start = torch.ones(done_flags.shape[0], dtype=done_flags.dtype, device=done_flags.device)
        else:
            sequence_start = sequence_start.to(device=done_flags.device, dtype=done_flags.dtype).view(-1)
        done_mask_obs = torch.zeros_like(done_flags)
        done_mask_obs[:, 0] = sequence_start
        if done_flags.shape[1] > 1:
            done_mask_obs[:, 1:] = done_flags[:, :-1]
        done_mask_next = torch.zeros_like(done_flags)
        done_mask_next[:, 0] = sequence_start
        if done_flags.shape[1] > 1:
            done_mask_next[:, 1:] = done_flags[:, :-1]
        return done_mask_obs, done_mask_next

    def _normalize_obs_batch(self, batch):
        """Normalize observation tensors in-place using running statistics."""
        obs_mean_t = torch.as_tensor(self.obs_rms.mean, dtype=torch.float32, device=batch["obs"].device)
        obs_std_t = torch.sqrt(torch.as_tensor(self.obs_rms.var, dtype=torch.float32, device=batch["obs"].device)) + 1e-8
        batch["obs"] = (batch["obs"] - obs_mean_t) / obs_std_t
        batch["next_obs"] = (batch["next_obs"] - obs_mean_t) / obs_std_t

    def _compute_target_q(self, batch, recurrent_state=None):
        """Compute target Q-values using target networks (no gradient tracking)."""
        scaled_rewards = batch["rewards"] * d.REW_SCALE
        with torch.no_grad():
            na, nlp, _ = self._sample_policy(batch["next_obs"], recurrent_state=recurrent_state, deterministic=False)
            tq1, _ = self._critic_forward(self.target_q1_enc, self.target_q1_head, batch["next_obs"], na, recurrent_state=recurrent_state)
            tq2, _ = self._critic_forward(self.target_q2_enc, self.target_q2_head, batch["next_obs"], na, recurrent_state=recurrent_state)
            tq = torch.min(tq1, tq2)
            tq = tq - self.alpha.detach() * nlp
            target_q = scaled_rewards + (1.0 - batch["dones"]) * self.config.gamma * tq
        return target_q

    def _update_critic(self, batch, target_q, learn_mask, done_mask_obs, recurrent_state=None):
        """Update both Q-networks, return loss, TD error, and gradient norm."""
        cq1, _ = self._critic_forward(self.q1_enc, self.q1_head, batch["obs"], batch["actions"], recurrent_state=recurrent_state, done_mask=done_mask_obs)
        cq2, _ = self._critic_forward(self.q2_enc, self.q2_head, batch["obs"], batch["actions"], recurrent_state=recurrent_state, done_mask=done_mask_obs)
        critic_loss = self._masked_mean(nn.functional.smooth_l1_loss(cq1, target_q, reduction='none'), learn_mask)
        critic_loss += self._masked_mean(nn.functional.smooth_l1_loss(cq2, target_q, reduction='none'), learn_mask)
        td_error = self._masked_mean(torch.abs(cq1 - target_q), learn_mask)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        grad_norm_critic = MetricsLogger.compute_grad_norm(self._critic_params)
        nn.utils.clip_grad_norm_(self._critic_params, max_norm=1.0)
        self.critic_optimizer.step()

        return critic_loss, td_error, grad_norm_critic

    def _update_actor(self, batch, learn_mask, done_mask_obs, recurrent_state=None):
        """Update actor network, return log_prob, entropy, loss, and gradient norm."""
        new_action, log_prob, _ = self._sample_policy(batch["obs"], recurrent_state=recurrent_state, done_mask=done_mask_obs, deterministic=False)
        q1_pi, _ = self._critic_forward(self.q1_enc, self.q1_head, batch["obs"], new_action, recurrent_state=recurrent_state, done_mask=done_mask_obs)
        q2_pi, _ = self._critic_forward(self.q2_enc, self.q2_head, batch["obs"], new_action, recurrent_state=recurrent_state, done_mask=done_mask_obs)
        actor_loss = self._masked_mean(self.alpha.detach() * log_prob - torch.min(q1_pi, q2_pi), learn_mask)
        policy_entropy = self._masked_mean(-log_prob, learn_mask)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        grad_norm_actor = MetricsLogger.compute_grad_norm(self._actor_params)
        nn.utils.clip_grad_norm_(self._actor_params, max_norm=1.0)
        self.actor_optimizer.step()

        return log_prob, policy_entropy, actor_loss, grad_norm_actor

    def _update_alpha(self, log_prob, learn_mask):
        """Update entropy coefficient alpha using automatic entropy tuning."""
        if not self.config.auto_entropy_tuning:
            return torch.tensor(0.0, device=self.device)
        alpha_loss = -self._masked_mean(self.log_alpha * (log_prob + self.target_entropy).detach(), learn_mask)
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        nn.utils.clip_grad_norm_([self.log_alpha], max_norm=1.0)
        self.alpha_optimizer.step()
        return alpha_loss

    def _update_target_networks(self):
        """Soft-update target Q-networks at the configured interval."""
        self._target_update_counter += 1
        if self._target_update_counter % self.config.target_update_interval != 0:
            return 0.0
        mag = self._soft_update(self.q1_enc, self.q1_head, self.target_q1_enc, self.target_q1_head)
        mag += self._soft_update(self.q2_enc, self.q2_head, self.target_q2_enc, self.target_q2_head)
        return mag

    def update(self, batch):
        """Perform SAC training update, returning a dict with all loss components, gradient norms, and diagnostics."""
        if batch["obs"].shape[1] <= 0:
            return None

        valid_mask = batch["valid_mask"]
        learn_mask = self._sequence_loss_mask(valid_mask, min(self.config.burn_in, batch["obs"].shape[1] - 1))
        done_mask_obs, _ = self._prepare_done_masks(batch)
        recurrent_state = batch.get("init_state")

        self._normalize_obs_batch(batch)
        target_q = self._compute_target_q(batch, recurrent_state=recurrent_state)

        critic_loss, td_error, grad_norm_critic = self._update_critic(batch, target_q, learn_mask, done_mask_obs, recurrent_state=recurrent_state)
        log_prob, policy_entropy, actor_loss, grad_norm_actor = self._update_actor(batch, learn_mask, done_mask_obs, recurrent_state=recurrent_state)
        alpha_loss = self._update_alpha(log_prob, learn_mask)
        target_update_mag = self._update_target_networks()

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.item()),
            "entropy_coef": float(self.alpha.item()),
            "policy_entropy": float(policy_entropy.item()),
            "td_error": float(td_error.item()),
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
        ckpt["obs_rms_mean"] = self.obs_rms.mean.copy()
        ckpt["obs_rms_var"] = self.obs_rms.var.copy()
        ckpt["obs_rms_count"] = self.obs_rms.count
        ckpt.update(self._checkpoint_metadata())
        ckpt["config"] = self.config.__dict__ if hasattr(self.config, "__dict__") else self.config
        return ckpt

    def save(self, path, episode, reward):
        """Save checkpoint to disk."""
        torch.save(self.checkpoint(episode, reward), path)

    def load(self, path):
        """Load checkpoint from disk and restore all network weights and optimizer states."""
        from controllers.common.checkpoints import load_checkpoint as _load_checkpoint

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
        if "obs_rms_mean" in checkpoint:
            self.obs_rms.mean = checkpoint["obs_rms_mean"].copy()
            self.obs_rms.var = checkpoint["obs_rms_var"].copy()
            self.obs_rms.count = float(checkpoint["obs_rms_count"])
        return checkpoint
