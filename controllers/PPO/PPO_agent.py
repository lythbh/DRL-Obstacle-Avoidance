"""
PPO agent, sequence utilities, and checkpoint helpers.

LLM level: 4 - LLM wrote the majority of the starting code, but we have since iterated on it a lot.
"""

import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, Generator, cast

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn.utils.rnn import pad_sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from controllers.RNN import GRUActorCritic, LSTMActorCritic, RecurrentState
from controllers.PPO.PPO_config import Config
from controllers.PPO.PPO_feedforward import FeedForwardActorCritic
from controllers.common.checkpoints import (
    load_checkpoint,
    make_checkpoint_header as _make_checkpoint_header,
    save_checkpoint_file as _save_checkpoint_file,
)
from controllers.common.metrics_logger import MetricsLogger

_CONTROLLER_DIR = Path(__file__).resolve().parent
_CHECKPOINT_DIR = _CONTROLLER_DIR / "checkpoints"


def _sequence_loss_mask(valid_mask: torch.Tensor, burn_in: int) -> torch.Tensor:
    """
    Creates a mask for sequence loss calculation.
    
    Parameters
    ----------
    valid_mask : torch.Tensor
        Mask indicating which elements are valid.
    burn_in : int
        Number of steps to ignore at the beginning of the sequence.
    
    Returns
    -------
    torch.Tensor
        Mask for sequence loss calculation.
    """
    valid_lengths = valid_mask.sum(dim=1).to(dtype=torch.long)
    start_index = torch.minimum(torch.full_like(valid_lengths, burn_in), torch.clamp(valid_lengths - 1, min=0))
    
    return valid_mask * (torch.arange(valid_mask.shape[1], device=valid_mask.device).unsqueeze(0) >= start_index.unsqueeze(1)).to(dtype=valid_mask.dtype)


def _split_sequences(episodes, seq_len, stride) -> Generator[dict, None, None]:
    """
    Splits episodes into sequences of a given length and stride.
    
    Parameters
    ----------
    episodes : list[dict]
        List of episodes.
    seq_len : int
        Length of the sequences.
    stride : int
        Stride between sequences.
    
    Returns
    -------
    Generator[dict, None, None]
        Generator of sequences.
    """
    for ep in episodes:
        total = len(ep["returns"])
        for start in range(0, total, stride):
            end = min(start + seq_len, total)
            if end > start:
                yield {k: v[start:end] for k, v in ep.items()}


class PPOAgent:
    def __init__(self, obs_size: int, action_dim: int, config: Config) -> None:
        """
        Initialize the PPO agent.
        
        Parameters
        ----------
        obs_size : int
            Size of the observation space.
        action_dim : int
            Dimension of the action space.
        config : Config
            Configuration for the PPO agent.
        """
        self.config = config
        self.device = self._get_device()
        self.action_dim = action_dim
        self.obs_size = obs_size
        self.action_low = torch.tensor([-config.max_steering_angle, config.min_speed], dtype=torch.float32, device=self.device)
        self.action_high = torch.tensor([config.max_steering_angle, config.max_speed], dtype=torch.float32, device=self.device)
        self.action_center = (self.action_high + self.action_low) / 2.0
        self.action_scale = (self.action_high - self.action_low) / 2.0
        self._build_model(config.recurrent_cell)
        print(f"[PPO] Using architecture: {config.recurrent_cell.upper()}", flush=True)


    def _build_model(self, recurrent_cell: str) -> None:
        """
        Builds the model based on the recurrent cell type.
        
        Parameters
        ----------
        recurrent_cell : str
            Type of recurrent cell to use.
        """
        recurrent_cell = recurrent_cell.lower().strip()
        if recurrent_cell == "none":
            model_class = FeedForwardActorCritic
        else:
            model_class = GRUActorCritic if recurrent_cell == "gru" else LSTMActorCritic
        
        self.model = model_class(self.obs_size, self.action_dim, self.config).to(self.device)
        self.actor = self.model.policy_head
        self.critic = self.model.value_head
        self.actor_log_std = nn.Parameter(torch.full((self.action_dim,), -0.5, dtype=torch.float32, device=self.device))
        params = list(self.model.parameters()) + [self.actor_log_std]
        self.optimizer = torch.optim.Adam(params, lr=self.config.learning_rate)


    def _get_device(self) -> torch.device:
        """
        Get the device to use for training.
        
        Returns
        -------
        torch.device
            Device to use for training.
        """
        if self.config.force_cpu or not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device("cuda")


    def get_initial_state(self, batch_size: int = 1) -> Optional[RecurrentState]:
        """
        Get the initial state for the recurrent model.
        
        Parameters
        ----------
        batch_size : int, optional
            Batch size, by default 1.
        
        Returns
        -------
        Optional[RecurrentState]
            Initial state for the recurrent model.
        """
        if self.config.recurrent_cell == "none":
            return self.model.get_initial_state(batch_size)
        
        return cast(Any, self.model).get_initial_state(batch_size, device=self.device)


    def _sample_action(self, policy_output, deterministic=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy.
        
        Parameters
        ----------
        policy_output : torch.Tensor
            Output of the policy network.
        deterministic : bool, optional
            Whether to sample deterministically, by default False.
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple of the action and the log probability of the action.
        """
        mean = policy_output
        std = self.actor_log_std.expand_as(policy_output).exp().clamp_min(1e-3)
        dist = Normal(mean, std)
        
        pre_tanh = mean if deterministic else dist.rsample()
        action_tanh = torch.tanh(pre_tanh)
        
        action = action_tanh * self.action_scale + self.action_center
        eps = 1e-5
        action = torch.clamp(action, self.action_low + eps, self.action_high - eps)
        log_prob = dist.log_prob(pre_tanh)
        log_prob -= torch.log(self.action_scale + 1e-6)
        log_prob -= torch.log(1.0 - action_tanh.pow(2) + 1e-6)
        
        return action, log_prob.sum(dim=-1)


    def select_action(self, obs, recurrent_state=None, done=False, deterministic=False) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor, Optional[RecurrentState]]:
        """
        Select an action from the policy.
        
        Parameters
        ----------
        obs : np.ndarray
            Observation.
        recurrent_state : Optional[RecurrentState], optional
            Recurrent state, by default None.
        done : bool, optional
            Whether the episode is done, by default False.
        deterministic : bool, optional
            Whether to sample deterministically, by default False.
        
        Returns
        -------
        Tuple[np.ndarray, torch.Tensor, torch.Tensor, Optional[RecurrentState]]
            Tuple of the action, the log probability of the action, the state value, and the next recurrent state.
        """
        done_mask = torch.tensor([float(done)], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            if self.config.recurrent_cell == "none":
                policy_output, state_value, next_state = self.model(obs)
            else:
                policy_output, state_value, next_state = self.model(obs, recurrent_state=recurrent_state, done_mask=done_mask)
            
            action, log_prob = self._sample_action(policy_output, deterministic=deterministic)
        
        return (
            action.squeeze(0).cpu().numpy(),
            log_prob.squeeze(0),
            state_value.squeeze(0),
            next_state,
        )


    def calculate_gae(self, rewards, values, bootstrap_value=0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generalized Advantage Estimation (GAE) calculation.
        
        Parameters
        ----------
        rewards : np.ndarray
            Rewards.
        values : np.ndarray
            State values.
        bootstrap_value : float, optional
            Bootstrap value, by default 0.0.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple of the advantages and the returns.
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
        
        return advantages.astype(np.float32), (advantages + values).astype(np.float32)


    def _prepare_batch(self, trajectories) -> Dict[str, torch.Tensor]:
        """
        Prepare a batch of trajectories for training.
        
        Parameters
        ----------
        trajectories : List[Dict[str, np.ndarray]]
            List of trajectories.
        
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of batched trajectories.
        """
        pad_keys = ["observations", "actions", "log_probs", "returns", "advantages"]
        result = {}
        for key in pad_keys:
            result[key] = pad_sequence(
                [torch.as_tensor(t[key], dtype=torch.float32, device=self.device) for t in trajectories],
                batch_first=True,
            )
        
        valid_masks = [torch.ones(len(t["returns"]), dtype=torch.float32, device=self.device) for t in trajectories]
        result["valid_mask"] = pad_sequence(valid_masks, batch_first=True)
        reset_masks = []
        
        for t in trajectories:
            mask = torch.zeros(len(t["returns"]), dtype=torch.float32, device=self.device)
            if len(mask) > 0:
                mask[0] = 1.0
            reset_masks.append(mask)
        
        result["done_mask"] = pad_sequence(reset_masks, batch_first=True)
        
        return result


    def _sanitize_trajectories(self, trajectories) -> None:
        """
        Sanitize trajectories by replacing NaN and inf values with zeros.
        
        Parameters
        ----------
        trajectories : List[Dict[str, np.ndarray]]
            List of trajectories.
        """
        low = np.array([-self.config.max_steering_angle, self.config.min_speed], dtype=np.float32)
        high = np.array([self.config.max_steering_angle, self.config.max_speed], dtype=np.float32)
        for t in trajectories:
            t["observations"] = np.nan_to_num(t["observations"], nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)
            t["actions"] = np.clip(np.nan_to_num(t["actions"], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32), low + 1e-5, high - 1e-5)
            t["log_probs"] = np.nan_to_num(t["log_probs"], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            t["returns"] = np.nan_to_num(t["returns"], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            t["advantages"] = np.nan_to_num(t["advantages"], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


    def _normalize_advantages(self, trajectories) -> None:
        """
        Normalize advantages by subtracting the mean and dividing by the standard deviation.
        
        Parameters
        ----------
        trajectories : List[Dict[str, np.ndarray]]
            List of trajectories.
        """
        all_adv = np.concatenate([t["advantages"] for t in trajectories], axis=0)
        adv_mean = float(all_adv.mean())
        adv_std = float(all_adv.std() + 1e-8)
        
        for t in trajectories:
            t["advantages"] = ((t["advantages"] - adv_mean) / adv_std).astype(np.float32)
            t["advantages"] = np.clip(t["advantages"], -5.0, 5.0)


    def _update_batch(self, batch) -> Optional[Dict[str, float]]:
        """
        Update the batch by computing the loss and performing a gradient step.
        
        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Batch of data.

        Returns
        -------
        Optional[Dict[str, float]]
            Dictionary of loss values.
        """
        log_probs_new, values, entropy = self.evaluate_sequences(
            batch["observations"], batch["actions"], batch["done_mask"],
        )
        
        if not (torch.isfinite(log_probs_new).all() and torch.isfinite(values).all() and torch.isfinite(entropy).all()):
            return None
        
        valid_mask = batch["valid_mask"]
        learn_mask = _sequence_loss_mask(valid_mask, self.config.burn_in)
        mask_bool = learn_mask > 0
        log_ratio = torch.nan_to_num(log_probs_new - batch["log_probs"], nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
        ratio = torch.exp(log_ratio)
        surr1 = ratio * batch["advantages"]
        surr2 = torch.clamp(ratio, 1 - self.config.epsilon, 1 + self.config.epsilon) * batch["advantages"]
        surrogate = torch.where(mask_bool, torch.min(surr1, surr2), torch.zeros_like(surr1))
        value_error = nn.functional.smooth_l1_loss(values, batch["returns"], reduction="none")
        entropy_term = torch.where(mask_bool, entropy, torch.zeros_like(entropy))
        valid_count = learn_mask.sum().clamp_min(1.0)
        loss = (-surrogate.sum() + 0.5 * torch.where(mask_bool, value_error, torch.zeros_like(value_error)).sum() - self.config.entropy_coef * entropy_term.sum()) / valid_count
        
        if not torch.isfinite(loss):
            return None

        with torch.no_grad():
            actor_loss_val = float(surrogate.sum() / valid_count)
            critic_loss_val = float(value_error.sum() / valid_count)
            entropy_val = float(entropy_term.sum() / valid_count)
            value_residual_val = float(torch.abs(values - batch["returns"])[mask_bool].mean().item())
            approx_kl = float((log_probs_new - batch["log_probs"])[mask_bool].mean().item())

        self.optimizer.zero_grad()
        loss.backward()
        for p in list(self.model.parameters()) + [self.actor_log_std]:
            if p.grad is not None and not torch.isfinite(p.grad).all():
                self.optimizer.zero_grad()
                return None

        rnn_attr = "gru" if hasattr(self.model, "gru") else ("lstm" if hasattr(self.model, "lstm") else None)
        rnn_clip = list(getattr(self.model, rnn_attr).parameters()) if rnn_attr else []

        actor_params = [self.actor.weight, self.actor.bias, self.actor_log_std]
        critic_params = [self.critic.weight, self.critic.bias]

        grad_norm_actor = MetricsLogger.compute_grad_norm(actor_params)
        grad_norm_critic = MetricsLogger.compute_grad_norm(critic_params)
        grad_norm_rnn = MetricsLogger.compute_grad_norm(rnn_clip) if rnn_clip else 0.0

        actor_clip = list(self.actor.parameters()) + [self.actor_log_std]
        critic_clip = list(self.critic.parameters())
        encoder_clip = [p for n, p in self.model.named_parameters()
                        if "policy_head" not in n and "value_head" not in n and (rnn_attr is None or rnn_attr not in n)]
        nn.utils.clip_grad_norm_(actor_clip, max_norm=0.5)
        nn.utils.clip_grad_norm_(critic_clip, max_norm=5.0)

        if rnn_clip:
            nn.utils.clip_grad_norm_(rnn_clip, max_norm=1.0)
        
        nn.utils.clip_grad_norm_(encoder_clip, max_norm=0.5)
        self.optimizer.step()
        
        with torch.no_grad():
            self.actor_log_std.data.copy_(torch.nan_to_num(self.actor_log_std.data, nan=-0.5, posinf=2.0, neginf=-5.0).clamp(-5.0, 2.0))

        lr = float(self.optimizer.param_groups[0]["lr"])
        
        return {
            "actor_loss": round(actor_loss_val, 6),
            "critic_loss": round(critic_loss_val, 6),
            "policy_entropy": round(entropy_val, 6),
            "entropy_coef": round(self.config.entropy_coef, 6),
            "value_residual": round(value_residual_val, 6),
            "approx_kl": round(approx_kl, 6),
            "grad_norm_actor": round(grad_norm_actor, 6),
            "grad_norm_critic": round(grad_norm_critic, 6),
            "grad_norm_rnn": round(grad_norm_rnn, 6),
            "lr_actor": lr,
        }


    def evaluate_sequences(self, observations, actions, done_mask) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate the sequences of observations and actions with the current policy by computing the log probabilities, state values, and entropies.
        
        Parameters
        ----------
        observations : torch.Tensor
            Observations.
        actions : torch.Tensor
            Actions.
        done_mask : torch.Tensor
            Done mask.
        
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple of log probabilities, state values, and entropies.
        """
        if self.config.recurrent_cell == "none":
            policy_output, state_values, _ = self.model(observations)
        else:
            policy_output, state_values, _ = self.model(observations, recurrent_state=self.get_initial_state(observations.shape[0]), done_mask=done_mask)
        
        mean = policy_output
        std = self.actor_log_std.expand_as(policy_output).exp().clamp_min(1e-3)
        dist = Normal(mean, std)
        eps = 1e-6
        safe_action = torch.clamp(actions, self.action_low + 1e-5, self.action_high - 1e-5)
        squashed = ((safe_action - self.action_center) / (self.action_scale + eps)).clamp(-1.0 + eps, 1.0 - eps)
        pre_tanh = 0.5 * (torch.log1p(squashed) - torch.log1p(-squashed))
        action_tanh = torch.tanh(pre_tanh)
        log_prob = dist.log_prob(pre_tanh)
        log_prob -= torch.log(self.action_scale + 1e-6)
        log_prob -= torch.log(1.0 - action_tanh.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        flat_entropy = 0.5 * self.action_dim * (1.0 + math.log(2.0 * math.pi)) + self.actor_log_std.sum()
        entropy = flat_entropy.expand(observations.shape[0], observations.shape[1])
        
        return log_prob, state_values, entropy


    def update(self, trajectories) -> list[dict[str, float]]:
        """
        Update the policy and value function using the given trajectories.
        
        Parameters
        ----------
        trajectories : list[Trajectory]
            List of trajectories.
        
        Returns
        -------
        list[dict[str, float]]
            List of metrics.
        """
        if not trajectories:
            return []
        
        self._sanitize_trajectories(trajectories)
        self._normalize_advantages(trajectories)
        
        trajectories = list(_split_sequences(trajectories, self.config.sequence_length, self.config.sequence_stride))
        if not trajectories:
            return []
        
        update_metrics = []
        num = len(trajectories)
        for epoch in range(self.config.epochs):
            indices = torch.randperm(num).tolist()
            for start in range(0, num, self.config.batch_size):
                batch_indices = indices[start: start + self.config.batch_size]
                batch = self._prepare_batch([trajectories[i] for i in batch_indices])
                metrics = self._update_batch(batch)
                if metrics is not None:
                    update_metrics.append(metrics)
        
        return update_metrics

    def load_model(self, model_path: str) -> None:
        """
        Load the model from the given path.
        
        Parameters
        ----------
        model_path : str
            Path to the model file.
        """
        checkpoint = load_checkpoint(model_path, map_location=self.device)
        algo = str(checkpoint.get("algorithm", "ppo")).lower().strip()
        assert algo == "ppo", f"Checkpoint algorithm '{algo}' does not match PPO."
        assert "model" in checkpoint, "Checkpoint does not contain recurrent 'model' weights."
        
        for key in ("obs_size", "action_dim"):
            saved = checkpoint.get(key)
            if saved is not None and int(saved) != getattr(self, key):
                raise ValueError(f"Checkpoint {key}={saved} != current {getattr(self, key)}")
        
        cell = str(checkpoint.get("recurrent_cell", self.config.recurrent_cell)).lower().strip()
        cell = {"mlp": "none", "feedforward": "none", "ff": "none"}.get(cell, cell)
        assert cell in {"none", "lstm", "gru"}, f"Unsupported recurrent_cell in checkpoint: {cell}"
        
        if cell != self.config.recurrent_cell:
            raise ValueError(
                f"Checkpoint recurrent_cell='{cell}' does not match configured '{self.config.recurrent_cell}'. "
                f"Use --arch {cell} or provide a {self.config.recurrent_cell} checkpoint."
            )
        
        print(f"[PPO] Loaded architecture: {cell.upper()}", flush=True)
        
        self.model.load_state_dict(checkpoint["model"])
        if "actor_log_std" in checkpoint:
            self.actor_log_std.data.copy_(checkpoint["actor_log_std"].to(self.device))


def _save_checkpoint(agent, episode, reward, is_goal, prefix, run_id) -> None:
    """
    Save the agent's model and optimizer state to a checkpoint file.
    
    Parameters
    ----------
    agent : PPOAgent
        Agent to save.
    episode : int
        Episode number.
    reward : float
        Episode reward.
    is_goal : bool
        Whether the episode reached the goal.
    prefix : str
        Prefix for the checkpoint file.
    run_id : str
        Run ID.
    """
    header = _make_checkpoint_header(episode, reward, is_goal, "ppo", asdict(agent.config))
    header["obs_size"] = agent.obs_size
    header["action_dim"] = agent.action_dim
    header["recurrent_cell"] = agent.config.recurrent_cell
    header["model"] = agent.model.state_dict()
    header["actor_log_std"] = agent.actor_log_std.detach().cpu()
    _save_checkpoint_file(_CHECKPOINT_DIR, run_id, prefix, header)
