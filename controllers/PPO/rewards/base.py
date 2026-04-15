"""Core reward computation for PPO training."""

from typing import Optional, Tuple

import numpy as np


class RewardComputer:
    """Computes rewards for the obstacle avoidance task."""

    def __init__(
        self,
        endpoint: np.ndarray,
        reference_distance: float,
        collision_reward: float = -40.0,
        goal_reward: float = 200.0,
        progress_scale: float = 5.0,
        progress_away_multiplier: float = 1.7,
        distance_penalty_scale: float = 0.08,
        orbit_distance_threshold: float = 1.2,
        orbit_progress_tolerance: float = 0.01,
        orbit_penalty: float = 0.25,
        no_progress_penalty: float = 0.05,
        heading_reward_scale: float = 0.15,
        goal_radius: float = 0.3,
        step_penalty: float = -0.01,
    ):
        self.endpoint = np.array(endpoint, dtype=np.float32)
        self.reference_distance = float(reference_distance)
        self.collision_reward = collision_reward
        self.goal_reward = goal_reward
        self.progress_scale = progress_scale
        self.progress_away_multiplier = progress_away_multiplier
        self.distance_penalty_scale = distance_penalty_scale
        self.orbit_distance_threshold = orbit_distance_threshold
        self.orbit_progress_tolerance = orbit_progress_tolerance
        self.orbit_penalty = orbit_penalty
        self.no_progress_penalty = no_progress_penalty
        self.heading_reward_scale = heading_reward_scale
        self.goal_radius = goal_radius
        self.step_penalty = step_penalty
        self.best_time = np.inf

    def compute(
        self,
        collision: bool,
        current_pos: np.ndarray,
        current_step: int,
        prev_distance: Optional[float],
        goal_error: float,
    ) -> Tuple[float, Optional[float]]:
        """Compute reward for current state."""
        if collision:
            return self.collision_reward, None

        distance_to_end = float(np.linalg.norm(current_pos - self.endpoint))

        if distance_to_end < self.goal_radius:
            if current_step < self.best_time:
                self.best_time = current_step
                return self.goal_reward, distance_to_end
            return 0.5 * self.goal_reward, distance_to_end

        progress_reward = 0.0
        delta = 0.0
        if prev_distance is not None:
            delta = float(prev_distance - distance_to_end)
            if delta >= 0.0:
                progress_reward = delta * self.progress_scale
            else:
                progress_reward = delta * self.progress_scale * self.progress_away_multiplier

        heading_reward = float(np.cos(goal_error)) * self.heading_reward_scale
        if prev_distance is not None and delta < self.orbit_progress_tolerance:
            heading_reward *= 0.2

        distance_penalty = -self.distance_penalty_scale * (
            distance_to_end / max(self.reference_distance, 1e-6)
        )

        orbit_penalty = 0.0
        if (
            prev_distance is not None
            and distance_to_end < self.orbit_distance_threshold
            and abs(delta) < self.orbit_progress_tolerance
        ):
            orbit_penalty = -self.orbit_penalty

        no_progress_penalty = 0.0
        if prev_distance is not None and abs(delta) < self.orbit_progress_tolerance:
            no_progress_penalty = -self.no_progress_penalty

        return (
            progress_reward
            + heading_reward
            + distance_penalty
            + orbit_penalty
            + no_progress_penalty
            + self.step_penalty
        ), distance_to_end
