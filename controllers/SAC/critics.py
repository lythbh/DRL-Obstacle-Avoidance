import torch
from torch import nn
import controllers.common.SAC_defaults as d


def build_critic_head(recurrent_hidden_size: int, action_dim: int, config: d.SACDefaults, device: torch.device):
    """Construct a Q-value head that takes [recurrent_features, action] and outputs Q(s,a)."""
    head = nn.Sequential(
        nn.Linear(recurrent_hidden_size + action_dim, config.hidden_size),
        nn.ReLU(),
        nn.Linear(config.hidden_size, config.hidden_size),
        nn.ReLU(),
        nn.Linear(config.hidden_size, 1),
    ).to(device)
    return head
