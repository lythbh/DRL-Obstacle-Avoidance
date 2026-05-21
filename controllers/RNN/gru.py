"""GRU actor-critic used by shared RL controllers."""

from typing import Optional
from torch import nn

import torch

from .base import RecurrentActorCriticBase


def _init_gru_weights(gru: nn.GRU) -> None:
    """Initialize GRU with orthogonal recurrent weights and moderate update-gate bias.

    Orthogonal initialization of recurrent weights ensures stable gradient flow
    through time.  The update-gate bias is set to -1.0 to lower z from ~0.5 to
    ~0.28, giving the GRU longer temporal memory (retains 72% of state per step
    vs 50% with default init).
    """
    hidden = gru.hidden_size
    with torch.no_grad():
        for layer in range(gru.num_layers):
            whh = getattr(gru, f"weight_hh_l{layer}")
            bih = getattr(gru, f"bias_ih_l{layer}")

            for gate_start in (0, hidden, 2 * hidden):
                nn.init.orthogonal_(whh.data[gate_start : gate_start + hidden])

            bih.data[hidden : 2 * hidden] = -1.0


class GRUActorCritic(RecurrentActorCriticBase):
    """Actor-critic network with lightweight feature branches and a GRU core."""

    def __init__(self, obs_size: int, action_dim: int, config, use_heads: bool = True) -> None:
        super().__init__(obs_size, action_dim, config, use_heads=use_heads)
        self.gru = nn.GRU(
            input_size=config.latent_size,
            hidden_size=self.recurrent_hidden_size,
            num_layers=self.recurrent_layers,
            batch_first=True,
        )
        _init_gru_weights(self.gru)

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
