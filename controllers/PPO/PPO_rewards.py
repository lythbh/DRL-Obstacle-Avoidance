"""
PPO-specific reward computation for obstacle avoidance task.

LLM level: 3 - LLM generated the skeleton, we added the reward logic.
"""

from typing import Optional, Tuple
import numpy as np
from controllers.PPO.PPO_defaults import (
    ENV_ENDPOINT,
    ENV_GOAL_THRESHOLD,
    ENV_REFERENCE_DISTANCE,
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
    """
    Computes rewards for the obstacle avoidance task.

    Attributes
    ----------
    endpoint : np.ndarray
        The goal position in the environment.
    reference_distance : float
        The reference distance for distance-based rewards.
    collision_reward : float
        The penalty for colliding with an obstacle.
    progress_scale : float
        The scaling factor for progress rewards.
    distance_reward_scale : float
        The scaling factor for distance rewards.
    heading_reward_scale : float
        The scaling factor for heading rewards.
    safety_reward_scale : float
        The scaling factor for safety rewards.
    motion_reward_scale : float
        The scaling factor for motion rewards.
    slow_speed_threshold : float
        The speed threshold below which a slow speed penalty is applied.
    slow_speed_penalty : float
        The penalty for moving too slowly.
    high_speed_threshold : float
        The speed threshold above which a high speed bonus is applied.
    high_speed_bonus : float
        The bonus for moving at high speed.
    new_best_distance_bonus : float
        The bonus for achieving a new best distance.
    step_penalty : float
        The penalty for each step taken.
    goal_success_reward : float
        The reward for successfully reaching the goal
    """


    def __init__(
        self,
        endpoint: Tuple[float, float] = ENV_ENDPOINT,
        reference_distance: float = ENV_REFERENCE_DISTANCE,
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
        goal_success_reward: float = REW_GOAL_SUCCESS,
        goal_hold_reward: float = REW_GOAL_HOLD,
        goal_threshold: float = ENV_GOAL_THRESHOLD,
    ) -> None:
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
        prev_distance: Optional[float],
        goal_error: float,
        min_lidar_norm: float,
        speed_norm: float,
        reached_new_best_distance: bool,
    ) -> Tuple[float, Optional[float]]:
        """
        Compute reward from collision, progress, heading, safety, speed, and goal bonus components.
        
        Parameters
        ----------
        collision : bool
            Whether the robot has collided with an obstacle.
        current_pos : np.ndarray
            Current position of the robot.
        prev_distance : Optional[float]
            Previous distance to the goal.
        goal_error : float
            Angle between the robot's heading and the goal direction.
        min_lidar_norm : float
            Minimum normalized LiDAR reading.
        speed_norm : float
            Normalized speed of the robot.
        reached_new_best_distance : bool
            Whether the robot has reached a new best distance to the goal.

        Returns
        -------
        Tuple[float, Optional[float]]
            The computed reward and the new distance to the goal.
        """
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
