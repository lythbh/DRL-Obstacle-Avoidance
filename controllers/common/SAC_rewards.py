"""SAC-specific reward computation for obstacle avoidance task."""

from typing import Optional, Tuple

import numpy as np

from controllers.common.SAC_defaults import (
    REW_COLLISION_PENALTY,
    REW_GOAL_SUCCESS,
)


class SACRewardComputer:
    """Computes rewards for the obstacle avoidance task (SAC variant).

    Per-step reward:  -time_penalty + progress + alignment - danger
    where:
        time_penalty   = 0.01  (efficiency pressure)
        progress       = (prev_distance - current_distance) * 2.0
        alignment      = speed_norm * cos(goal_error) * 0.1
        danger         = 0  (if min_lidar >= 0.3)
                       = speed_norm * ((0.3 - min_lidar) / 0.3)^2  (if min_lidar < 0.3)
    """

    def __init__(
        self,
        endpoint: Tuple[float, float] = (2.0, 0.0),
        reference_distance: float = 4.0,
        collision_penalty: float = REW_COLLISION_PENALTY,
        goal_threshold: float = 0.3,
        goal_success_reward: float = REW_GOAL_SUCCESS,
        **kwargs,
    ) -> None:
        self.endpoint = np.array(endpoint, dtype=np.float32)
        self.collision_penalty = float(collision_penalty)
        self.goal_threshold = float(goal_threshold)
        self.goal_success_reward = float(goal_success_reward)

    def compute(
        self,
        collision: bool,
        current_pos: np.ndarray,
        prev_distance: Optional[float],
        goal_error: float,
        min_lidar_norm: float,
        speed_norm: float,
        reached_new_best_distance: bool,
    ) -> Tuple[float, Optional[float]]:
        if collision:
            return self.collision_penalty, None

        distance_to_end = float(np.linalg.norm(current_pos - self.endpoint))

        # 1. Time penalty — efficiency pressure to reach goal quickly
        reward = -0.01

        # 2. Progress toward goal — dominant signal
        if prev_distance is not None:
            reward += (float(prev_distance) - distance_to_end) * 2.0

        # 3. Heading alignment — small bonus for facing the goal while moving
        reward += speed_norm * float(np.cos(goal_error)) * 0.1

        # 4. Danger penalty — only when very close to obstacles, scales with speed
        if min_lidar_norm < 0.3:
            danger_zone = (0.3 - min_lidar_norm) / 0.3
            reward -= 1.0 * (danger_zone ** 2) * speed_norm

        if distance_to_end < self.goal_threshold:
            return reward + self.goal_success_reward, distance_to_end

        return reward, distance_to_end
