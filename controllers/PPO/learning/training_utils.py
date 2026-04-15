"""Training-loop utility helpers for PPO controller."""

from typing import Dict, List, Tuple
import time

import numpy as np


def resolve_episode_end_reason(info: Dict[str, object], truncated: bool) -> str:
    """Map environment reset info to normalized episode end reason labels."""
    reason = info.get("reset_reason")
    if reason in ("collision", "near_collision"):
        return "collision"
    if reason == "goal_reached":
        return "goal_reached"
    if reason == "stagnation":
        return "stagnation"
    if reason == "low_score":
        return "low_score"
    if truncated:
        return "max_steps"
    return "max_steps"


class TrajectoryBuffer:
    """Stores rollout data between PPO updates."""

    def __init__(self) -> None:
        self.observations: List[np.ndarray] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.returns: List[float] = []
        self.advantages: List[float] = []

    def add_episode(
        self,
        episode_obs_array: np.ndarray,
        episode_actions: List[int],
        episode_log_probs: List[float],
        episode_returns: np.ndarray,
        episode_advantages: np.ndarray,
    ) -> None:
        """Append one episode worth of trajectory data."""
        self.observations.extend(episode_obs_array)
        self.actions.extend(episode_actions)
        self.log_probs.extend(episode_log_probs)
        self.returns.extend(np.asarray(episode_returns, dtype=np.float32).reshape(-1).tolist())
        self.advantages.extend(np.asarray(episode_advantages, dtype=np.float32).reshape(-1).tolist())

    def to_arrays(self) -> Tuple[np.ndarray, np.ndarray, List[float], np.ndarray, np.ndarray]:
        """Return buffered rollout data as arrays suitable for PPO updates."""
        obs_array = np.array(self.observations, dtype=np.float32)
        actions_array = np.array(self.actions, dtype=np.int64)
        returns_array = np.array(self.returns, dtype=np.float32)
        advantages_array = np.array(self.advantages, dtype=np.float32)
        return obs_array, actions_array, list(self.log_probs), returns_array, advantages_array

    def clear(self) -> None:
        """Clear buffered rollout data."""
        self.observations.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.returns.clear()
        self.advantages.clear()


class RollingDiagnostics:
    """Tracks rolling training diagnostics and throughput metrics."""

    def __init__(self, window_size: int):
        self.window_size = max(1, int(window_size))
        self.recent_end_reasons: List[str] = []
        self.recent_rewards: List[float] = []
        self.recent_min_dists: List[float] = []
        self.total_sim_steps = 0
        self.train_start_time = time.perf_counter()

    def record_episode(
        self,
        end_reason: str,
        reward_sum: float,
        min_distance: float,
        episode_steps: int,
    ) -> None:
        """Add one episode to rolling diagnostics."""
        self.recent_end_reasons.append(end_reason)
        self.recent_rewards.append(float(reward_sum))
        self.recent_min_dists.append(float(min_distance))
        if len(self.recent_end_reasons) > self.window_size:
            self.recent_end_reasons.pop(0)
            self.recent_rewards.pop(0)
            self.recent_min_dists.pop(0)

        self.total_sim_steps += int(episode_steps)

    def should_emit(self, episode_index: int) -> bool:
        """Return True when diagnostics should be printed for this episode index."""
        return (episode_index + 1) % self.window_size == 0

    def summary(self, timestep_ms: int) -> Dict[str, float]:
        """Compute current rolling diagnostics and throughput summary."""
        collision_rate = (
            sum(reason == "collision" for reason in self.recent_end_reasons) / self.window_size
        )
        goal_rate = (
            sum(reason == "goal_reached" for reason in self.recent_end_reasons) / self.window_size
        )
        stagnation_rate = (
            sum(reason == "stagnation" for reason in self.recent_end_reasons) / self.window_size
        )
        avg_recent_reward = float(np.mean(self.recent_rewards))
        avg_recent_min_dist = float(np.mean(self.recent_min_dists))

        elapsed = max(time.perf_counter() - self.train_start_time, 1e-6)
        steps_per_sec = self.total_sim_steps / elapsed
        realtime_factor = ((self.total_sim_steps * timestep_ms) / 1000.0) / elapsed

        return {
            "collision_rate": collision_rate,
            "goal_rate": goal_rate,
            "stagnation_rate": stagnation_rate,
            "avg_reward": avg_recent_reward,
            "avg_min_dist": avg_recent_min_dist,
            "steps_per_sec": steps_per_sec,
            "realtime_factor": realtime_factor,
            "total_sim_steps": float(self.total_sim_steps),
        }
