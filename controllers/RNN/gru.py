"""GRU actor-critic used by shared RL controllers."""

from typing import Optional
from torch import nn

import torch

from .base import RecurrentActorCriticBase


class GRUActorCritic(RecurrentActorCriticBase):
    """Actor-critic network with lightweight feature branches and a GRU core."""

    def __init__(self, obs_size: int, action_dim: int, config) -> None:
        super().__init__(obs_size, action_dim, config)
        self.gru = nn.GRU(
            input_size=config.latent_size,
            hidden_size=self.recurrent_hidden_size,
            num_layers=self.recurrent_layers,
            batch_first=True,
        )

    def get_initial_state(self, batch_size: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """Initialize GRU hidden state with zeros."""
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros((self.recurrent_layers, batch_size, self.recurrent_hidden_size), device=device)

    def _run_recurrent(self, latent, recurrent_state, mask, batch_size, seq_len):
        """Run GRU one timestep at a time, resetting hidden state when done_mask indicates episode end."""
        h_t = recurrent_state
        outputs = []
        for t in range(seq_len):
            if mask is not None:
                keep = (1.0 - mask[:, t]).view(1, batch_size, 1)
                h_t = h_t * keep
            step_output, h_t = self.gru(latent[:, t : t + 1], h_t)
            outputs.append(step_output)
        return torch.cat(outputs, dim=1), h_t
