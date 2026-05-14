"""LSTM actor-critic used by shared RL controllers."""

from typing import Optional, Tuple

import torch
from torch import nn

from .base import RecurrentActorCriticBase


class LSTMActorCritic(RecurrentActorCriticBase):
    """Actor-critic network with lightweight feature branches and an LSTM core."""

    def __init__(self, obs_size: int, action_dim: int, config) -> None:
        super().__init__(obs_size, action_dim, config)
        self.lstm = nn.LSTM(
            input_size=config.latent_size,
            hidden_size=self.recurrent_hidden_size,
            num_layers=self.recurrent_layers,
            batch_first=True,
        )

    def get_initial_state(self, batch_size: int, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if device is None:
            device = next(self.parameters()).device
        state_shape = (self.recurrent_layers, batch_size, self.recurrent_hidden_size)
        return (torch.zeros(state_shape, device=device), torch.zeros(state_shape, device=device))

    def _run_recurrent(self, latent, recurrent_state, mask, batch_size, seq_len):
        h_t, c_t = recurrent_state
        outputs = []
        for t in range(seq_len):
            if mask is not None:
                keep = (1.0 - mask[:, t]).view(1, batch_size, 1)
                h_t = h_t * keep
                c_t = c_t * keep
            step_output, (h_t, c_t) = self.lstm(latent[:, t : t + 1], (h_t, c_t))
            outputs.append(step_output)
        return torch.cat(outputs, dim=1), (h_t, c_t)
