"""LSTM actor-critic used by shared RL controllers."""

from typing import Optional, Tuple

import torch
from torch import nn

from .base import RecurrentActorCriticBase


def _init_lstm_weights(lstm: nn.LSTM) -> None:
    """Initialize LSTM with orthogonal recurrent weights and forget-gate bias = 1.0.

    The forget-gate bias is set to 1.0 (standard practice) to prevent vanishing
    gradients and ensure long-term memory.  Recurrent weights use orthogonal init.
    """
    hidden = lstm.hidden_size
    with torch.no_grad():
        for layer in range(lstm.num_layers):
            whh = getattr(lstm, f"weight_hh_l{layer}")
            bhh = getattr(lstm, f"bias_hh_l{layer}")
            bih = getattr(lstm, f"bias_ih_l{layer}")

            for gate_start in (0, hidden, 2 * hidden, 3 * hidden):
                nn.init.orthogonal_(whh.data[gate_start : gate_start + hidden])

            bih.data[hidden : 2 * hidden] = 1.0
            bhh.data[hidden : 2 * hidden] = 1.0


class LSTMActorCritic(RecurrentActorCriticBase):
    """Actor-critic network with lightweight feature branches and an LSTM core."""

    def __init__(self, obs_size: int, action_dim: int, config, use_heads: bool = True) -> None:
        super().__init__(obs_size, action_dim, config, use_heads=use_heads)
        self.lstm = nn.LSTM(
            input_size=config.latent_size,
            hidden_size=self.recurrent_hidden_size,
            num_layers=self.recurrent_layers,
            batch_first=True,
        )
        _init_lstm_weights(self.lstm)

    def get_initial_state(self, batch_size: int, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize LSTM hidden and cell states with zeros."""
        if device is None:
            device = next(self.parameters()).device
        state_shape = (self.recurrent_layers, batch_size, self.recurrent_hidden_size)
        return (torch.zeros(state_shape, device=device), torch.zeros(state_shape, device=device))

    def _run_recurrent(self, latent, recurrent_state, mask, batch_size, seq_len):
        """Run LSTM one timestep at a time, resetting hidden/cell states when done_mask indicates episode end."""
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
