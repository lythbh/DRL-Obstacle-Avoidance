"""Seed configuration for reproducible training across all frameworks."""

import random
import numpy as np
import torch


SEED = 42


def set_all_seeds(seed: int = SEED) -> None:
    """Set random seeds for all frameworks to ensure reproducibility.
    
    
    Args:
        seed: Random seed value to use (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # For CUDA determinism (if using GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # This can impact performance but ensures determinism
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
