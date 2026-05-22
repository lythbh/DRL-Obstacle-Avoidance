"""Shared training utilities used by PPO, SAC, and future RL controllers."""

import torch


def sequence_loss_mask(valid_mask: torch.Tensor, burn_in: int) -> torch.Tensor:
    """Create learning mask that excludes burn-in steps from gradient computation."""
    valid_lengths = valid_mask.sum(dim=1).to(dtype=torch.long)
    start_index = torch.minimum(torch.full_like(valid_lengths, burn_in), torch.clamp(valid_lengths - 1, min=0))
    return valid_mask * (torch.arange(valid_mask.shape[1], device=valid_mask.device).unsqueeze(0) >= start_index.unsqueeze(1)).to(dtype=valid_mask.dtype)
