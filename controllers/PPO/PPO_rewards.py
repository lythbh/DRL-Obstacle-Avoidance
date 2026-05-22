"""PPO-specific reward computation for obstacle avoidance task."""

from typing import Optional, Tuple

import numpy as np

from controllers.PPO.PPO_defaults import (
    REW_COLLISION_PENALTY,
    REW_DISTANCE_SCALE,
    REW_GOAL_HOLD,
    REW_GOAL_SUCCESS,
    REW_HEADING_SCALE,
    REW_HIGH_SPEED_BONUS,
    REW_HIGH_SPEED_THRESHOLD,
    REW_MOTION_SCALE,
    REW_NEW_BEST_DISTANCE_BONUS,
    REW_PROGRESS_SCALE,
    REW_SAFETY_SCALE,
    REW_SLOW_SPEED_PENALTY,
    REW_SLOW_SPEED_THRESHOLD,
    REW_STEP_PENALTY,
)


class PPORewardComputer:
    """Computes rewards for the obstacle avoidance task (PPO variant)."""

    def __init__(
        self,
        endpoint: Tuple[float, float] = (2.0, 0.0),
        reference_distance: float = 4.0,
        collision_penalty: float = REW_COLLISION_PENALTY,
        progress_reward_scale: float = REW_PROGRESS_SCALE,
        distance_reward_scale: float = REW_DISTANCE_SCALE,
        heading_reward_scale: float = REW_HEADING_SCALE,
        safety_reward_scale: float = REW_SAFETY_SCALE,
        motion_reward_scale: float = REW_MOTION_SCALE,
        slow_speed_threshold: float = REW_SLOW_SPEED_THRESHOLD,
        slow_speed_penalty: float = REW_SLOW_SPEED_PENALTY,
        high_speed_threshold: float = REW_HIGH_SPEED_THRESHOLD,
        high_speed_bonus: float = REW_HIGH_SPEED_BONUS,
        new_best_distance_bonus: float = REW_NEW_BEST_DISTANCE_BONUS,
        step_penalty: float = REW_STEP_PENALTY,
        goal_threshold: float = 0.3,
        goal_success_reward: float = REW_GOAL_SUCCESS,
        goal_hold_reward: float = REW_GOAL_HOLD,
    ) -> None:
        """Initialize reward computer with environment and tuning parameters."""
        self.endpoint = np.array(endpoint, dtype=np.float32)
        self.reference_distance = float(reference_distance)
        self.collision_reward = float(collision_penalty)
        self.progress_scale = float(progress_reward_scale)
        self.distance_reward_scale = float(distance_reward_scale)
        self.heading_reward_scale = float(heading_reward_scale)
        self.safety_reward_scale = float(safety_reward_scale)
        self.motion_reward_scale = float(motion_reward_scale)
        self.slow_speed_threshold = float(slow_speed_threshold)
        self.slow_speed_penalty = float(slow_speed_penalty)
        self.high_speed_threshold = float(high_speed_threshold)
        self.high_speed_bonus = float(high_speed_bonus)
        self.new_best_distance_bonus = float(new_best_distance_bonus)
        self.step_penalty = float(step_penalty)
        self.goal_threshold = float(goal_threshold)
        self.goal_success_reward = float(goal_success_reward)
        self.goal_hold_reward = float(goal_hold_reward)

    def compute(
        self,
        collision: bool,
        current_pos: np.ndarray,
        current_step: int,
        prev_distance: Optional[float],
        goal_error: float,
        min_lidar_norm: float,
        speed_norm: float,
        reached_new_best_distance: bool,
        accel: np.ndarray,
    ) -> Tuple[float, Optional[float]]:
        """Compute reward from collision, progress, heading, safety, speed, and goal bonus components."""
        if collision:
            return self.collision_reward, None

        distance_to_end = float(np.linalg.norm(current_pos - self.endpoint))

        if distance_to_end < self.goal_threshold:
            return self.goal_success_reward +  self.goal_hold_reward, distance_to_end

        progress = 0.0
        if prev_distance is not None:
            delta = float(prev_distance - distance_to_end)
            proximity_factor = 1.0 - (distance_to_end / max(self.reference_distance, 1e-6))
            progress = delta * self.progress_scale * proximity_factor

        distance_ratio = float(np.clip(distance_to_end / max(self.reference_distance, 1e-6), 0.0, 2.0))
        distance_penalty = -distance_ratio * self.distance_reward_scale
        heading_alignment = float(np.cos(goal_error))
        heading_reward = heading_alignment * self.heading_reward_scale
        safety_penalty = -(1.0 - float(np.clip(min_lidar_norm, 0.0, 1.0))) * self.safety_reward_scale
        motion_reward = float(np.clip(speed_norm, 0.0, 1.0)) * self.motion_reward_scale
        slow_penalty = self.slow_speed_penalty if speed_norm < self.slow_speed_threshold else 0.0
        high_speed_reward = self.high_speed_bonus if speed_norm > self.high_speed_threshold else 0.0
        new_best_bonus = self.new_best_distance_bonus if reached_new_best_distance else 0.0

        return (
            progress
            + distance_penalty
            + heading_reward
            + safety_penalty
            + motion_reward
            + slow_penalty
            + high_speed_reward
            + new_best_bonus
            + self.step_penalty
        ), distance_to_end