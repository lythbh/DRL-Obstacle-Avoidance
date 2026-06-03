"""
Shared checkpoint utilities.

Provides small helpers for constructing checkpoint paths and loading
checkpoints consistently across controllers. Callers should pass their
controller directory (usually `Path(__file__).resolve().parent`) so
paths remain pinned to the specific controller folder.

LLM level: 4 - LLM generated most of logic, minor improvements and complete functional test by us.
"""

from pathlib import Path
from typing import Any, Dict, Union
import torch


def run_checkpoint_dir(controller_checkpoints_dir: Path, run_id: str) -> Path:
    """
    Return the checkpoint folder for a training run and ensure it exists.

    Parameters
    ----------
    controller_checkpoints_dir : Path
        The controller's checkpoints directory.
    run_id : str
        The run ID.

    Returns
    -------
    Path
        The checkpoint folder for the run.

    """
    checkpoint_dir = controller_checkpoints_dir / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    return checkpoint_dir


def run_checkpoint_path(controller_checkpoints_dir: Path, run_id: str, prefix: str, extension: str = "pth") -> str:
    """
    Checkpoint file path inside the run folder with the filename formed as `{prefix}_{run_id}.{extension}`.

    Parameters
    ----------
    controller_checkpoints_dir : Path
        The controller's checkpoints directory.
    run_id : str
        The run ID.
    prefix : str
        The checkpoint prefix.
    extension : str, optional
        The checkpoint file extension, by default "pth".

    Returns
    -------
    str
        The checkpoint file path.
    """
    return str(run_checkpoint_dir(controller_checkpoints_dir, run_id) / f"{prefix}_{run_id}.{extension}")


def load_checkpoint(path: str, map_location: Union[str, torch.device]) -> Dict[str, Any]:
    """
    Loads the modern signature with `weights_only=False` if available, falling back to the simpler call if not.

    Parameters
    ----------
    path : str
        The checkpoint file path.
    map_location : Union[str, torch.device]
        The map location to pass to `torch.load`.

    Returns
    -------
    Dict[str, Any]
        The loaded checkpoint.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def make_checkpoint_header(episode: Any, reward: float, goal_episode: bool, algorithm: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Construct a minimal, consistent checkpoint header used by controllers.
    
    Parameters
    ----------
    episode : Any
        The episode number.
    reward : float
        The reward.
    goal_episode : bool
        Whether the episode was a goal episode.
    algorithm : str
        The algorithm name.
    config : Dict[str, Any]
        The configuration.

    Returns
    -------
    Dict[str, Any]
        The checkpoint header.
    """
    return {
        "episode": episode,
        "reward": reward,
        "goal_episode": goal_episode,
        "algorithm": algorithm,
        "config": config,
    }


def save_checkpoint_file(controller_checkpoints_dir: Path, run_id: str, prefix: str, checkpoint: Dict[str, Any]) -> str:
    """
    Save `checkpoint` under a run-specific filename and return the path.
    Uses `run_checkpoint_path` to build the filename and persists with `torch.save`.

    Parameters
    ----------
    controller_checkpoints_dir : Path
        The controller's checkpoints directory.
    run_id : str
        The run ID.
    prefix : str
        The checkpoint prefix.
    checkpoint : Dict[str, Any]
        The checkpoint to save.

    Returns
    -------
    str
        The checkpoint file path.
    """
    path = run_checkpoint_path(controller_checkpoints_dir, run_id, prefix)
    torch.save(checkpoint, path)
    
    return path
