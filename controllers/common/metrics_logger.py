"""Structured CSV metrics logger for PPO, SAC, and future RL controllers.

Writes three CSV files per run to the run folder so that learning curves,
diagnostic analysis, and hyperparameter comparisons can be performed
without re-running the simulation:

- ``{algorithm}_hyperparams.csv``  -- static run configuration
- ``{algorithm}_episodes.csv``     -- per-episode metrics with action/obs stats
- ``{algorithm}_updates.csv``      -- per-training-update metrics (losses,
  entropy, gradient norms, learning rates, target-network drift)

Usage::

    logger = MetricsLogger(run_folder, algorithm="ppo")
    logger.log_hyperparams(config_dict, recurrent_cell="gru")
    ...
    act_stats = MetricsLogger.compute_action_stats(ep_act_array)
    obs_stats = MetricsLogger.compute_obs_stats(ep_obs_array)
    logger.log_episode(
        episode=1, global_step=400, reward=-12.3, length=400,
        success=True, goal_touched=False, collision=False, timeout=True,
        min_dist=0.45, avg_speed_ms=1.2, end_reason="max_steps",
        elapsed_s=120.0, action_stats=act_stats, obs_stats=obs_stats,
        recurrent_cell="gru",
        update_metrics={...},   # mean actor/critic loss, entropy, grad norms
        replay_buffer_size=128,
    )
    logger.log_update(global_step=410, episode=2, **update_losses)
    logger.close()
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Sequence

import csv
import numpy as np
import torch


ActionStats = Dict[str, float]
ObsStats = Dict[str, float]


class MetricsLogger:
    """Appends structured rows to CSV files in *run_folder*.

    Episode-level metrics go to ``{algorithm}_episodes.csv``, per-update
    metrics go to ``{algorithm}_updates.csv``, and static hyperparameters
    go to ``{algorithm}_hyperparams.csv``.
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
        "value_residual", "grad_norm_actor", "grad_norm_critic",
        "lr_actor", "lr_critic",
        "alpha", "alpha_loss",
        "target_update_magnitude",
        "replay_buffer_size",
        "recurrent_cell",
    ]

    _UPDATE_FIELDNAMES: List[str] = [
        "global_step",
        "episode",
        "actor_loss",
        "critic_loss",
        "policy_entropy",
        "entropy_coef",
        "value_residual",
        "grad_norm_actor",
        "grad_norm_critic",
        "lr_actor",
        "lr_critic",
        "alpha",
        "alpha_loss",
        "target_update_magnitude",
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
        "actor_lr",
        "critic_lr",
        "alpha_lr",
        "initial_alpha",
        "auto_entropy_tuning",
        "target_entropy_scale",
        "tau",
        "replay_capacity",
        "replay_batch_size",
        "min_replay_sequences",
        "update_after_steps",
        "log_std_min",
        "log_std_max",
        "obs_size",
        "action_dim",
    ]

    def __init__(self, run_folder: str, algorithm: str) -> None:
        self._algo = algorithm.lower().strip()
        self._ep_path = os.path.join(run_folder, f"{self._algo}_episodes.csv")
        self._up_path = os.path.join(run_folder, f"{self._algo}_updates.csv")
        self._hp_path = os.path.join(run_folder, f"{self._algo}_hyperparams.csv")

        self._ep_file = open(self._ep_path, "w", newline="", buffering=1)
        self._up_file = open(self._up_path, "w", newline="", buffering=1)
        self._hp_file = open(self._hp_path, "w", newline="", buffering=1)

        self._ep_writer = csv.DictWriter(self._ep_file, fieldnames=self._EPISODE_FIELDNAMES,
                                         extrasaction="ignore")
        self._up_writer = csv.DictWriter(self._up_file, fieldnames=self._UPDATE_FIELDNAMES,
                                         extrasaction="ignore")
        self._hp_writer = csv.DictWriter(self._hp_file, fieldnames=self._HYPERPARAM_FIELDNAMES,
                                         extrasaction="ignore")

        self._ep_writer.writeheader()
        self._up_writer.writeheader()
        self._hp_writer.writeheader()

    def log_hyperparams(self, config: Dict[str, Any], **extra: Any) -> None:
        """Write one row of static hyperparameters to the hyperparams CSV."""
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
        """Write one episode-level row to the episodes CSV."""
        row: Dict[str, Any] = {}
        for field in self._EPISODE_FIELDNAMES:
            val = kwargs.get(field)
            row[field] = _fmt(val)
        self._ep_writer.writerow(row)

    def log_update(self, **kwargs: Any) -> None:
        """Write one update-level row to the updates CSV."""
        row: Dict[str, Any] = {}
        for field in self._UPDATE_FIELDNAMES:
            val = kwargs.get(field)
            row[field] = _fmt(val)
        self._up_writer.writerow(row)

    def close(self) -> None:
        """Flush and close all CSV file handles."""
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
    def compute_action_stats(
        actions: Sequence[np.ndarray],
    ) -> ActionStats:
        """Compute per-dimension (mean, std, min, max) for a 2D action buffer.

        *actions* should be a sequence of (2,) numpy arrays. Returns a dict
        with keys ``act0_mean``, ``act0_std``, ``act0_min``, ``act0_max``,
        ``act1_mean``, ``act1_std``, ``act1_min``, ``act1_max``.
        """
        if not actions:
            return _empty_action_stats()
        arr = np.stack(actions, axis=0).astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
        stats: ActionStats = {}
        for d in range(arr.shape[1]):
            col = arr[:, d]
            stats[f"act{d}_mean"] = float(np.mean(col))
            stats[f"act{d}_std"] = float(np.std(col))
            stats[f"act{d}_min"] = float(np.min(col))
            stats[f"act{d}_max"] = float(np.max(col))
        return stats

    @staticmethod
    def compute_obs_stats(
        observations: Sequence[np.ndarray],
    ) -> ObsStats:
        """Compute aggregate (mean, std, min, max) across all observation dims.

        *observations* should be a sequence of (obs_dim,) numpy arrays.
        Returns a dict with keys ``obs_mean``, ``obs_std``, ``obs_min``,
        ``obs_max``.
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
    def compute_grad_norm(
        parameters: Sequence,
    ) -> float:
        """Compute the total L2 gradient norm over *parameters*.

        Only parameters with non-``None`` and finite gradients contribute.
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
    def compute_value_residual(
        values: np.ndarray,
        returns: np.ndarray,
    ) -> float:
        """Compute mean absolute error between predicted values and returns.

        This reflects how well the value function tracks empirical returns.
        """
        v = np.asarray(values, dtype=np.float32)
        r = np.asarray(returns, dtype=np.float32)
        return float(np.mean(np.abs(v - r)))

    @staticmethod
    def compute_td_error(
        rewards: np.ndarray,
        values: np.ndarray,
        next_values: np.ndarray,
        gamma: float,
    ) -> float:
        """Compute mean absolute TD-error across a trajectory.

        TD(t) = r(t) + gamma * V(t+1) - V(t), where V(T+1) = 0 for terminal.
        """
        r = np.asarray(rewards, dtype=np.float32)
        v = np.asarray(values, dtype=np.float32)
        nv = np.asarray(next_values, dtype=np.float32)
        td = r + gamma * nv - v
        return float(np.mean(np.abs(td)))

    @staticmethod
    def aggregate_update_metrics(
        update_list: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Average a list of per-update metric dicts into one summary dict.

        Each element of *update_list* is a dict with numeric values. The
        returned dict contains the **mean** of each key seen across the list.
        Non-numeric or missing values are skipped.
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
    """Round floats, stringify non-scalars, leave None as empty string."""
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
    return isinstance(val, (int, float, np.floating, np.integer)) and not isinstance(val, bool)


def _empty_action_stats() -> ActionStats:
    empty: ActionStats = {}
    for d in range(2):
        empty[f"act{d}_mean"] = 0.0
        empty[f"act{d}_std"] = 0.0
        empty[f"act{d}_min"] = 0.0
        empty[f"act{d}_max"] = 0.0
    return empty


def _empty_obs_stats() -> ObsStats:
    return {"obs_mean": 0.0, "obs_std": 0.0, "obs_min": 0.0, "obs_max": 0.0}


