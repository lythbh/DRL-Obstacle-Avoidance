"""
Structured CSV metrics logger for PPO.

Writes per-run CSV files for hyperparameters, per-episode metrics, and
per-update metrics so runs can be analyzed after training.

LLM level: 4 - LLM generated most of logic, minor improvements and complete functional test by us.
"""

from __future__ import annotations
import math
import os
from typing import Any, Dict, List, Sequence
import csv
import numpy as np
import torch


ACTIONSTATS = Dict[str, float]
OBSSTATS = Dict[str, float]


class MetricsLogger:
    """
    Appends structured rows to CSV files in plots.

    Episode-level metrics go to `ppo_episodes.csv`, per-update
    metrics go to `ppo_updates.csv`, and static hyperparameters
    go to `ppo_hyperparams.csv`.
    """

    _EPISODE_FIELDNAMES: List[str] = [
        "episode",
        "global_step",
        "reward",
        "avg10",
        "length",
        "success",
        "goal_touched",
        "collision",
        "timeout",
        "min_dist",
        "avg_speed_ms",
        "end_reason",
        "elapsed_s",
        "act0_mean", "act0_std", "act0_min", "act0_max",
        "act1_mean", "act1_std", "act1_min", "act1_max",
        "obs_mean", "obs_std", "obs_min", "obs_max",
        "actor_loss", "critic_loss", "policy_entropy", "entropy_coef",
        "approx_kl",
        "value_residual", "grad_norm_actor", "grad_norm_critic", "grad_norm_rnn",
        "lr_actor",
        "recurrent_cell",
    ]

    _UPDATE_FIELDNAMES: List[str] = [
        "global_step",
        "episode",
        "actor_loss",
        "critic_loss",
        "policy_entropy",
        "entropy_coef",
        "approx_kl",
        "value_residual",
        "grad_norm_actor",
        "grad_norm_critic",
        "grad_norm_rnn",
        "lr_actor",
        "recurrent_cell",
    ]

    _HYPERPARAM_FIELDNAMES: List[str] = [
        "algorithm",
        "recurrent_cell",
        "hidden_size",
        "latent_size",
        "lstm_hidden_size",
        "lstm_layers",
        "sequence_length",
        "burn_in",
        "sequence_stride",
        "gamma",
        "gae_lambda",
        "epsilon",
        "learning_rate",
        "entropy_coef",
        "epochs",
        "batch_size",
        "update_every",
        "obs_size",
        "action_dim",
    ]

    def __init__(self, run_folder: str, algorithm: str) -> None:
        """
        Initialize the CSV writers for a training run.

        Parameters
        ----------
        run_folder : str
            The folder where the CSV files will be written.
        algorithm : str
            The algorithm name used to prefix the CSV filenames.
        """
        self._algo = algorithm.lower().strip()
        self._ep_path = os.path.join(run_folder, f"{self._algo}_episodes.csv")
        self._up_path = os.path.join(run_folder, f"{self._algo}_updates.csv")
        self._hp_path = os.path.join(run_folder, f"{self._algo}_hyperparams.csv")

        self._ep_file = open(self._ep_path, "w", newline="", buffering=1)
        self._up_file = open(self._up_path, "w", newline="", buffering=1)
        self._hp_file = open(self._hp_path, "w", newline="", buffering=1)

        self._ep_writer = csv.DictWriter(self._ep_file, fieldnames=self._EPISODE_FIELDNAMES, extrasaction="ignore")
        self._up_writer = csv.DictWriter(self._up_file, fieldnames=self._UPDATE_FIELDNAMES, extrasaction="ignore")
        self._hp_writer = csv.DictWriter(self._hp_file, fieldnames=self._HYPERPARAM_FIELDNAMES, extrasaction="ignore")

        self._ep_writer.writeheader()
        self._up_writer.writeheader()
        self._hp_writer.writeheader()

    def log_hyperparams(self, config: Dict[str, Any], **extra: Any) -> None:
        """
        Write one row of static hyperparameters to the hyperparams CSV.

        Parameters
        ----------
        config : Dict[str, Any]
            The main configuration dictionary.
        **extra : Any
            Additional hyperparameters that override entries in config.
        """
        row: Dict[str, Any] = {}
        row["algorithm"] = self._algo
        for field in self._HYPERPARAM_FIELDNAMES:
            if field == "algorithm":
                continue
            val = extra.get(field) if field in extra else config.get(field)
            
            if isinstance(val, bool):
                val = int(val)
            row[field] = val if val is not None else ""
        
        self._hp_writer.writerow(row)

    def log_episode(self, **kwargs: Any) -> None:
        """
        Write one episode-level row to the episodes CSV.

        Parameters
        ----------
        **kwargs : Any
            Episode metrics keyed by the episode CSV field names.
        """
        row: Dict[str, Any] = {}
        for field in self._EPISODE_FIELDNAMES:
            val = kwargs.get(field)
            row[field] = _fmt(val)
        
        self._ep_writer.writerow(row)

    def log_update(self, **kwargs: Any) -> None:
        """
        Write one update-level row to the updates CSV.

        Parameters
        ----------
        **kwargs : Any
            Update metrics keyed by the update CSV field names.
        """
        row: Dict[str, Any] = {}
        for field in self._UPDATE_FIELDNAMES:
            val = kwargs.get(field)
            row[field] = _fmt(val)
        
        self._up_writer.writerow(row)

    def close(self) -> None:
        """
        Flush and close all CSV file handles.

        Returns
        -------
        None
            This method closes the open file handles in place.
        """
        for fh in (self._ep_file, self._up_file, self._hp_file):
            fh.flush()
            fh.close()

    @property
    def path(self) -> str:
        return self._ep_path

    @property
    def update_path(self) -> str:
        return self._up_path

    @property
    def hyperparams_path(self) -> str:
        return self._hp_path

    @staticmethod
    def compute_action_stats(actions: Sequence[np.ndarray]) -> ACTIONSTATS:
        """
        Compute per-dimension action statistics for a 2D action buffer.

        Parameters
        ----------
        actions : Sequence[np.ndarray]
            A sequence of 2D action vectors.

        Returns
        -------
        ACTIONSTATS
            A dict with keys for the mean, standard deviation, minimum,
            and maximum of each action dimension.
        """
        if not actions:
            return _empty_action_stats()
        
        arr = np.stack(actions, axis=0).astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
        
        stats: ACTIONSTATS = {}
        for d in range(arr.shape[1]):
            col = arr[:, d]
            stats[f"act{d}_mean"] = float(np.mean(col))
            stats[f"act{d}_std"] = float(np.std(col))
            stats[f"act{d}_min"] = float(np.min(col))
            stats[f"act{d}_max"] = float(np.max(col))
        
        return stats

    @staticmethod
    def compute_obs_stats(observations: Sequence[np.ndarray]) -> ACTIONSTATS:
        """
        Compute aggregate observation statistics across all dimensions.

        Parameters
        ----------
        observations : Sequence[np.ndarray]
            A sequence of observation vectors.

        Returns
        -------
        ACTIONSTATS
            A dict with keys for the mean, standard deviation, minimum,
            and maximum over all observation values.
        """
        if not observations:
            return _empty_obs_stats()
        
        arr = np.stack(observations, axis=0).astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
        return {
            "obs_mean": float(np.mean(arr)),
            "obs_std": float(np.std(arr)),
            "obs_min": float(np.min(arr)),
            "obs_max": float(np.max(arr)),
        }

    @staticmethod
    def compute_grad_norm(parameters: Sequence) -> float:
        """
        Compute the total L2 gradient norm over parameters.

        Parameters
        ----------
        parameters : Sequence
            Model parameters whose gradients will be inspected.

        Returns
        -------
        float
            The total gradient norm, or 0.0 if no finite gradients exist.
        """
        total = 0.0
        has = False
        for p in parameters:
            if p.grad is None:
                continue   
            g = p.grad.detach()

            if not torch.isfinite(g).all():
                continue
            total += float(g.data.norm(2).item() ** 2)
            has = True
            
        return float(math.sqrt(total)) if has else 0.0

    @staticmethod
    def aggregate_update_metrics(update_list: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Average a list of per-update metric dicts into one summary dict.

        Parameters
        ----------
        update_list : List[Dict[str, Any]]
            A list of metric dictionaries to average.

        Returns
        -------
        Dict[str, float]
            A dictionary containing the mean of each numeric key.
        """
        if not update_list:
            return {}
        
        accum: Dict[str, List[float]] = {}
        for upd in update_list:
            for k, v in upd.items():
                if not _is_numeric(v):
                    continue
                accum.setdefault(k, []).append(float(v))

        return {k: float(np.mean(lst)) for k, lst in accum.items()}


def _fmt(val: Any) -> Any:
    """
    Format a value for CSV output.

    Parameters
    ----------
    val : Any
        The value to format.

    Returns
    -------
    Any
        A CSV-safe value with floats rounded and missing values blank.
    """
    if val is None:
        return ""
    
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return 0.0
        return round(val, 6)
    
    if _is_numeric(val):
        return val
    
    if isinstance(val, str):
        return val
    
    return str(val)


def _is_numeric(val: Any) -> bool:
    """
    Check whether a value is numeric and not boolean.

    Parameters
    ----------
    val : Any
        The value to inspect.

    Returns
    -------
    bool
        True if the value is a numeric scalar, otherwise False.
    """
    return isinstance(val, (int, float, np.floating, np.integer)) and not isinstance(val, bool)


def _empty_action_stats() -> ACTIONSTATS:
    """
    Build a zero-filled action statistics dictionary.

    Returns
    -------
    ACTIONSTATS
        A dictionary with zero values for both action dimensions.
    """
    empty: ACTIONSTATS = {}
    for d in range(2):
        empty[f"act{d}_mean"] = 0.0
        empty[f"act{d}_std"] = 0.0
        empty[f"act{d}_min"] = 0.0
        empty[f"act{d}_max"] = 0.0
    
    return empty


def _empty_obs_stats() -> OBSSTATS:
    """
    Build a zero-filled observation statistics dictionary.

    Returns
    -------
    OBSSTATS
        A dictionary with zero values for observation aggregates.
    """
    return {"obs_mean": 0.0, "obs_std": 0.0, "obs_min": 0.0, "obs_max": 0.0}


