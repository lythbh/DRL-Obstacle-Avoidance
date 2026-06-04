"""
LSTM actor-critic used by shared RL controllers.

LLM level: 1 - LLM helped with stability issues encountered during training.
"""

from typing import Optional, Tuple
import torch
from torch import nn


from .base import RecurrentActorCriticBase


class LSTMActorCritic(RecurrentActorCriticBase):
    """Actor-critic network with lightweight feature branches and an LSTM core."""

    def __init__(self, obs_size: int, action_dim: int, config) -> None:
        """
        Initialize LSTM recurrent core on top of base feature encoders.
        
        Parameters
        ----------
        obs_size : int
            Size of observation vector.
        action_dim : int
            Size of action vector.
        config : Config
            Configuration object.
        """
        super().__init__(obs_size, action_dim, config)
        self.lstm = nn.LSTM(
            input_size=config.latent_size,
            hidden_size=self.recurrent_hidden_size,
            num_layers=self.recurrent_layers,
            batch_first=True,
        )


    def get_initial_state(self, batch_size: int, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize LSTM hidden and cell states with zeros.
        
        Parameters
        ---------
        batch_size : int
            Batch size.
        device : torch.device, optional
            Device to initialize states on. If None, uses device of model parameters.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Initial hidden and cell states.
        """
        if device is None:
            device = next(self.parameters()).device
        
        state_shape = (self.recurrent_layers, batch_size, self.recurrent_hidden_size)
        
        return (torch.zeros(state_shape, device=device), torch.zeros(state_shape, device=device))


    def _run_recurrent(self, latent, recurrent_state, mask, batch_size, seq_len) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Run LSTM one timestep at a time, resetting hidden/cell states when done_mask indicates episode end.
        
        Parameters
        ---------
        latent : torch.Tensor
            Latent state tensor of shape (batch_size, seq_len, latent_size).
        recurrent_state : Tuple[torch.Tensor, torch.Tensor]
            Tuple of hidden and cell states of shape (recurrent_layers, batch_size, recurrent_hidden_size).
        mask : torch.Tensor
            Done mask tensor of shape (batch_size, seq_len).
        batch_size : int
            Batch size.
        seq_len : int
            Sequence length.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, seq_len, recurrent_hidden_size).
        Tuple[torch.Tensor, torch.Tensor]
            Next hidden and cell states of shape (recurrent_layers, batch_size, recurrent_hidden_size).
        """
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
