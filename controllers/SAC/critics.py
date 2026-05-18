import torch
from torch import nn
from controllers.RNN import GRUActorCritic, LSTMActorCritic
import controllers.common.SAC_defaults as d

def build_critic_pair(obs_size: int, action_dim: int, config: d.SACDefaults, device: torch.device):
    """Construct a recurrent critic encoder (obs only) and Q-value head."""
    encoder_cls = LSTMActorCritic if config.recurrent_cell.lower().strip() == "lstm" else GRUActorCritic
    # Base network only encodes observations
    enc = encoder_cls(obs_size, action_dim, config, use_heads=False).to(device)
    
    # Head processes [RNN_features, Action]
    head = nn.Sequential(
        nn.Linear(enc.recurrent_hidden_size + action_dim, config.hidden_size),
        nn.ReLU(),
        nn.Linear(config.hidden_size, config.hidden_size),
        nn.ReLU(),
        nn.Linear(config.hidden_size, 1),
    ).to(device)
    
    return enc, head