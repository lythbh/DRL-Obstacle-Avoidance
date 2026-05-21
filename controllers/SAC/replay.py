"""Replay buffer utilities for SAC controller."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F


def _clone_state(state):
    if state is None:
        return None
    if isinstance(state, tuple):
        return tuple(_clone_state(part) for part in state)
    tensor = state.detach().to("cpu") if torch.is_tensor(state) else torch.as_tensor(state)
    return tensor.contiguous().numpy().copy()


def _stack_state_batch(states, device: torch.device):
    first = states[0]
    if isinstance(first, tuple):
        return tuple(_stack_state_batch([state[i] for state in states], device) for i in range(len(first)))
    stacked = np.concatenate([np.asarray(state, dtype=np.float32) for state in states], axis=1)
    return torch.as_tensor(stacked, dtype=torch.float32, device=device).contiguous()


class SequenceReplayBuffer:
    def __init__(self, obs_size: int, action_dim: int, config: Any) -> None:
        """Initialize replay buffer with fixed capacity for storing episode sequences."""
        self.capacity = config.replay_capacity
        self.seq_len = config.sequence_length
        self.seq_stride = int(getattr(config, "sequence_stride", self.seq_len))
        self.burn_in = int(getattr(config, "burn_in", 0))
        self.recurrent_cell = str(getattr(config, "recurrent_cell", "gru")).lower().strip()
        self.recurrent_layers = int(getattr(config, "recurrent_layers", getattr(config, "lstm_layers", 1)))
        self.recurrent_hidden_size = int(getattr(config, "recurrent_hidden_size", getattr(config, "lstm_hidden_size", 1)))
        self.buffer: List[Dict[str, np.ndarray]] = []
        self.pos = 0

    def __len__(self) -> int:
        """Return the current number of sequences in the replay buffer."""
        return len(self.buffer)

    def _zero_state(self):
        shape = (self.recurrent_layers, 1, self.recurrent_hidden_size)
        zeros = np.zeros(shape, dtype=np.float32)
        if self.recurrent_cell == "lstm":
            return (zeros.copy(), zeros.copy())
        return zeros

    def add_episode(self, ep_obs, ep_act, ep_rew, ep_next, ep_done, ep_states=None, ep_critic_states=None) -> None:
        """Add an episode as fixed-length sequences to the replay buffer, cycling when at capacity."""
        if not ep_obs:
            return
        obs = np.asarray(ep_obs, dtype=np.float32)
        actions = np.asarray(ep_act, dtype=np.float32)
        rewards = np.asarray(ep_rew, dtype=np.float32).reshape(-1, 1)
        next_obs = np.asarray(ep_next, dtype=np.float32)
        dones = np.asarray(ep_done, dtype=np.float32).reshape(-1, 1)
        if ep_states is None:
            ep_states = [self._zero_state() for _ in range(len(obs))]
        if ep_critic_states is None:
            ep_critic_states = [self._zero_state() for _ in range(len(obs))]
        total = obs.shape[0]
        for start in range(0, total, self.seq_stride):
            end = min(start + self.seq_len, total)
            entry = {
                "obs": obs[start:end].copy(),
                "actions": actions[start:end].copy(),
                "rewards": rewards[start:end].copy(),
                "next_obs": next_obs[start:end].copy(),
                "dones": dones[start:end].copy(),
                "valid_mask": np.ones(end - start, dtype=np.float32),
                "init_state": _clone_state(ep_states[start]),
                "critic_init_state": _clone_state(ep_critic_states[start]),
                "sequence_start": np.float32(1.0 if (start == 0 or bool(dones[start - 1])) else 0.0),
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
        result: Dict[str, torch.Tensor] = {}
        for k in keys:
            tensors = []
            for i in indices:
                arr = self.buffer[i][k]
                t = torch.as_tensor(arr, dtype=torch.float32, device=device)
                if len(t) < self.seq_len:
                    # build pad widths for F.pad: last two values are for the first dim
                    pad = [(0, self.seq_len - len(t))] + [(0, 0)] * (t.ndim - 1)
                    # F.pad expects a flat reversed list
                    flat_pads = [p for pad_dim in reversed(pad) for p in pad_dim]
                    t = F.pad(t, flat_pads)
                tensors.append(t)
            result[k] = torch.stack(tensors)
        result["sequence_start"] = torch.as_tensor([self.buffer[i]["sequence_start"] for i in indices], dtype=torch.float32, device=device)
        result["init_state"] = _stack_state_batch([self.buffer[i]["init_state"] for i in indices], device)
        result["critic_init_state"] = _stack_state_batch([self.buffer[i]["critic_init_state"] for i in indices], device)
        return result
