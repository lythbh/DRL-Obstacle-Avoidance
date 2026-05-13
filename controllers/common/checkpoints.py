"""Shared checkpoint utilities used by controllers.

Provides small helpers for constructing checkpoint paths and loading
checkpoints consistently across controllers. Callers should pass their
controller directory (usually `Path(__file__).resolve().parent`) so
paths remain pinned to the specific controller folder.
"""
from pathlib import Path
from typing import Any, Dict, Union

import torch


def checkpoint_path(controller_dir: Path, filename: str) -> str:
    """Return a checkpoint path pinned to the given controller directory."""
    return str(controller_dir / filename)


def run_checkpoint_dir(controller_checkpoints_dir: Path, run_id: str) -> Path:
    """Return the checkpoint folder for a training run and ensure it exists."""
    checkpoint_dir = controller_checkpoints_dir / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def run_checkpoint_path(controller_checkpoints_dir: Path, run_id: str, prefix: str, extension: str = "pth") -> str:
    """Return a checkpoint file path inside the run folder.

    The filename is formed as `{prefix}_{run_id}.{extension}`.
    """
    return str(run_checkpoint_dir(controller_checkpoints_dir, run_id) / f"{prefix}_{run_id}.{extension}")


def load_checkpoint(path: str, map_location: Union[str, torch.device]) -> Dict[str, Any]:
    """Load a PyTorch checkpoint while handling older PyTorch signatures.

    Some versions of `torch.load` accept `weights_only` while others do not;
    this helper tries the modern signature first and falls back to the simpler
    call when necessary.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def make_checkpoint_header(episode: Any, reward: float, goal_episode: bool, algorithm: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Construct a minimal, consistent checkpoint header used by controllers.

    This header can be extended by per-algorithm code with model weights
    and any algorithm-specific fields.
    """
    return {
        "episode": episode,
        "reward": reward,
        "goal_episode": goal_episode,
        "algorithm": algorithm,
        "config": config,
    }


def save_checkpoint_file(controller_checkpoints_dir: Path, run_id: str, prefix: str, checkpoint: Dict[str, Any]) -> str:
    """Save `checkpoint` under a run-specific filename and return the path.

    Uses `run_checkpoint_path` to build the filename and persists with
    `torch.save`.
    """
    path = run_checkpoint_path(controller_checkpoints_dir, run_id, prefix)
    torch.save(checkpoint, path)
    return path
