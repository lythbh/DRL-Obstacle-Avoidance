"""PPO actor-critic agent implementation."""

from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import multiprocessing, nn


class PPOAgent:
    """Proximal Policy Optimization agent."""

    def __init__(self, obs_size: int, n_actions: int, config: Any):
        self.config = config
        self.device = self._get_device()

        self.actor = self._build_actor(obs_size, n_actions)
        self.critic = self._build_critic(obs_size)

        params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.optimizer = torch.optim.Adam(params, lr=config.learning_rate)

    def _get_device(self) -> torch.device:
        """Get appropriate device (GPU or CPU)."""
        is_fork = multiprocessing.get_start_method(allow_none=True) == "fork"
        cuda_available = torch.cuda.is_available() and not is_fork
        requested_device = str(self.config.train_device).strip().lower()

        if requested_device == "cpu":
            return torch.device("cpu")

        if requested_device == "cuda":
            if cuda_available:
                return torch.device("cuda:0")
            print("[PPO] WARNING: train_device='cuda' requested but CUDA is unavailable, using CPU.")
            return torch.device("cpu")

        if requested_device not in ("", "auto"):
            print(f"[PPO] WARNING: Unknown train_device '{self.config.train_device}', using auto.")

        # For this tiny policy network, CPU can be faster due to GPU kernel launch overhead.
        if cuda_available and (self.config.hidden_size >= 256 or self.config.batch_size >= 2048):
            return torch.device("cuda:0")
        return torch.device("cpu")

    def _build_actor(self, obs_size: int, n_actions: int) -> nn.Module:
        """Build actor (policy) network."""
        return nn.Sequential(
            nn.Linear(obs_size, self.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size, n_actions),
            nn.Softmax(dim=-1),
        ).to(self.device)

    def _build_critic(self, obs_size: int) -> nn.Module:
        """Build critic (value) network."""
        return nn.Sequential(
            nn.Linear(obs_size, self.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size, 1),
        ).to(self.device)

    def select_action(self, obs: np.ndarray) -> Tuple[int, float]:
        """Select action using current policy."""
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            probs = self.actor(obs_tensor)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item())

    def calculate_returns(self, rewards: np.ndarray, bootstrap_value: float = 0.0) -> np.ndarray:
        """Calculate cumulative discounted returns with optional bootstrap value."""
        returns = np.zeros(len(rewards), dtype=np.float32)
        g_return = float(bootstrap_value)
        for t in reversed(range(len(rewards))):
            g_return = rewards[t] + self.config.gamma * g_return
            returns[t] = g_return
        return returns

    def update(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        log_probs_old: Union[np.ndarray, List[float]],
        returns: np.ndarray,
        advantages: np.ndarray,
        entropy_coef: Optional[float] = None,
    ) -> None:
        """Update policy using collected rollout."""
        obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        act_tensor = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        logp_old_tensor = torch.as_tensor(log_probs_old, dtype=torch.float32, device=self.device)
        ret_tensor = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        adv_tensor = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        entropy_weight = self.config.entropy_coef if entropy_coef is None else float(entropy_coef)

        dataset_size = len(observations)

        for _ in range(self.config.epochs):
            indices = torch.randperm(dataset_size)

            for start in range(0, dataset_size, self.config.batch_size):
                batch_idx = indices[start : start + self.config.batch_size]

                obs_batch = obs_tensor[batch_idx]
                act_batch = act_tensor[batch_idx]
                logp_old_batch = logp_old_tensor[batch_idx]
                ret_batch = ret_tensor[batch_idx]
                adv_batch = adv_tensor[batch_idx]

                probs = self.actor(obs_batch)
                dist = torch.distributions.Categorical(probs)
                log_probs_new = dist.log_prob(act_batch)
                entropy = dist.entropy().mean()

                ratio = torch.exp(log_probs_new - logp_old_batch)
                surrogate1 = ratio * adv_batch
                surrogate2 = torch.clamp(ratio, 1 - self.config.epsilon, 1 + self.config.epsilon) * adv_batch
                policy_loss = -torch.min(surrogate1, surrogate2).mean()

                values = self.critic(obs_batch).squeeze(-1)
                value_loss = nn.MSELoss()(values, ret_batch)

                loss = policy_loss + 0.5 * value_loss - entropy_weight * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()
