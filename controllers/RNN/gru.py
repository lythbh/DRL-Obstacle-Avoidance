"""
GRU actor-critic.

LLM level: 1 - LLM Debuged an error with the initialization of nn.GRU
               and output of _run_reccurent.
"""

from typing import Optional, Tuple
from torch import nn
import torch


from .base import RecurrentActorCriticBase


class GRUActorCritic(RecurrentActorCriticBase):
    """
    Actor-critic network with lightweight feature branches and a GRU core.
    """

    def __init__(self, obs_size: int, action_dim: int, config) -> None:
        """
        Initialize GRU recurrent core on top of base feature encoders.
        
        Parameters
        ---------
        obs_size : int
            Observation space size.
        action_dim : int
            Action space size.
        config : Config
            Configuration object.
        """
        super().__init__(obs_size, action_dim, config)
        self.gru = nn.GRU(
            input_size=config.latent_size,
            hidden_size=self.recurrent_hidden_size,
            num_layers=self.recurrent_layers,
            batch_first=True,
        )


    def get_initial_state(self, batch_size: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Initialize GRU hidden state with zeros.
        
        Parameters
        ---------
        batch_size : int
            Batch size.
        device : torch.device or None
            Device to initialize the state on. If None, uses the device of the model parameters.

        Returns
        -------
        torch.Tensor
            Initial hidden state tensor of shape (batch_size, latent_size).
        """
        if device is None:
            device = next(self.parameters()).device
        
        return torch.zeros((self.recurrent_layers, batch_size, self.recurrent_hidden_size), device=device)


    def _run_recurrent(self, latent, recurrent_state, mask, batch_size, seq_len) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run GRU one timestep at a time, resetting hidden state when done_mask indicates episode end.
        
        Parameters
        ---------
        latent : torch.Tensor
            Latent tensor of shape (batch_size, seq_len, latent_size).
        recurrent_state : torch.Tensor
            Recurrent state tensor of shape (batch_size, latent_size).
        mask : torch.Tensor
            Done mask tensor of shape (batch_size, seq_len).
        batch_size : int
            Batch size.
        seq_len : int
            Sequence length.

        Returns
        -------
        torch.Tensor
            Recurrent features tensor of shape (batch_size, seq_len, latent_size).
        torch.Tensor
            Next recurrent state tensor of shape (batch_size, latent_size).
        """
        h_t = recurrent_state
        outputs = []
        for t in range(seq_len):
            if mask is not None:
                keep = (1.0 - mask[:, t]).view(1, batch_size, 1)
                h_t = h_t * keep
            
            step_output, h_t = self.gru(latent[:, t : t + 1], h_t)
            outputs.append(step_output)
        
        return torch.cat(outputs, dim=1), h_t
