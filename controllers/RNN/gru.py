"""GRU actor-critic used by shared RL controllers."""

from typing import List, Optional, Tuple

import torch
from torch import nn

from .base import RecurrentActorCriticBase

RecurrentState = torch.Tensor


class GRUActorCritic(RecurrentActorCriticBase):
    """Actor-critic network with lightweight feature branches and a GRU core."""

    def __init__(self, obs_size: int, action_dim: int, config) -> None:
        """Initialise shared base and add a GRU recurrent cell."""
        super().__init__(obs_size, action_dim, config)
        self.gru = nn.GRU(
            input_size=config.latent_size,
            hidden_size=self.recurrent_hidden_size,
            num_layers=self.recurrent_layers,
            batch_first=True,
        )

    def get_initial_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
    ) -> RecurrentState:
        """Return zeroed single hidden-state tensor [L, B, H] for GRU."""
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros((self.recurrent_layers, batch_size, self.recurrent_hidden_size), device=device)

    def _run_recurrent(
        self,
        latent: torch.Tensor,
        recurrent_state: RecurrentState,
        mask: Optional[torch.Tensor],
        batch_size: int,
        seq_len: int,
    ) -> Tuple[torch.Tensor, RecurrentState]:
        """GRU forward pass with per-step done-mask hidden-state reset."""
        h_t = recurrent_state
        if mask is None or not bool((mask[:, 1:] > 0).any().item()):
            if mask is not None:
                keep = (1.0 - mask[:, 0]).view(1, batch_size, 1)
                h_t = h_t * keep
            recurrent_features, h_t = self.gru(latent, h_t)
        else:
            outputs: List[torch.Tensor] = []
            for t in range(seq_len):
                if mask is not None:
                    keep = (1.0 - mask[:, t]).view(1, batch_size, 1)
                    h_t = h_t * keep
                step_output, h_t = self.gru(latent[:, t : t + 1], h_t)
                outputs.append(step_output)
            recurrent_features = torch.cat(outputs, dim=1)
        return recurrent_features, h_t
