"""Reproducible seed helpers shared across all controllers."""

import random
import numpy as np
import torch


SEED = 42


def set_all_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # deterministic mode costs some throughput but guarantees reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
