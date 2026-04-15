"""Webots environment wrapper for ALTINO PPO training."""

from typing import Any, Dict, Optional, Tuple

import numpy as np

from controller import Supervisor  # pyright: ignore[reportMissingImports]

from config import Config
from environment.observation_builder import build_observation, goal_geometry
from environment.robot_driver import AltinoDriver
from rewards.base import RewardComputer
from rewards.penalties import calculate_clearance_penalty


class WebotsEnv:
    """Webots simulation environment for ALTINO obstacle avoidance."""

    def __init__(self, supervisor: Supervisor, config: Config):
        """Initialize environment with config."""
        self.config = config
        self.robot = AltinoDriver(supervisor, config)
        self.timestep = self.robot.timestep

        self.headlights = self.robot.get_device("headlights")
        self.backlights = self.robot.get_device("backlights")

        self.reward_computer = RewardComputer(
            np.array(config.endpoint, dtype=np.float32),
            reference_distance=float(config.reference_distance if config.reference_distance is not None else 1.0),
            collision_reward=config.collision_penalty,
            goal_reward=config.goal_reward,
            progress_scale=config.progress_reward_scale,
            progress_away_multiplier=config.progress_away_multiplier,
            distance_penalty_scale=config.distance_penalty_scale,
            orbit_distance_threshold=config.orbit_distance_threshold,
            orbit_progress_tolerance=config.orbit_progress_tolerance,
            orbit_penalty=config.orbit_penalty,
            no_progress_penalty=config.no_progress_penalty,
            heading_reward_scale=config.heading_reward_scale,
            goal_radius=config.goal_radius,
            step_penalty=config.step_penalty,
        )

        self.goal_marker_node = None
        self._ensure_goal_marker()

        self.current_step = 0
        self.episode_reward = 0.0
        self.current_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.current_heading = 0.0
        self.current_distance = float("inf")
        self.min_episode_distance = float("inf")
        self.best_distance = float("inf")
        self.steps_without_progress = 0
        self.collision = False
        self.prev_distance: Optional[float] = None

    def _ensure_goal_marker(self) -> None:
        """Create (if needed) and place a visible marker at the configured goal."""
        if not self.config.visualize_goal:
            return

        try:
            supervisor = self.robot.supervisor
            marker = supervisor.getFromDef("PPO_GOAL_MARKER")

            if marker is None:
                root = supervisor.getRoot()
                if root is None:
                    return
                children = root.getField("children")
                if children is None:
                    return

                marker_node = (
                    "DEF PPO_GOAL_MARKER Transform { "
                    "translation 0 0 0 "
                    "children [ "
                    "Shape { "
                    "appearance PBRAppearance { baseColor 0 1 0 roughness 0.2 metalness 0 } "
                    "geometry Sphere { radius 0.12 } "
                    "} "
                    "] "
                    "}"
                )
                children.importMFNodeFromString(-1, marker_node)
                marker = supervisor.getFromDef("PPO_GOAL_MARKER")

            self.goal_marker_node = marker
            self._place_goal_marker()
            print(
                f"[PPO] Goal marker at ({self.reward_computer.endpoint[0]:.2f}, "
                f"{self.reward_computer.endpoint[1]:.2f})"
            )
        except Exception as e:
            print(f"[PPO] WARNING: Could not create goal marker: {e}")

    def _place_goal_marker(self) -> None:
        """Move goal marker to current endpoint coordinates."""
        if self.goal_marker_node is None:
            return

        try:
            translation_field = self.goal_marker_node.getField("translation")
            if translation_field is None:
                return
            x = float(self.reward_computer.endpoint[0])
            y = float(self.reward_computer.endpoint[1])
            z = float(self.config.goal_marker_height)
            translation_field.setSFVec3f([x, y, z])
        except Exception as e:
            print(f"[PPO] WARNING: Could not place goal marker: {e}")

    def _reset_episode_state(self) -> None:
        """Reset internal episode state."""
        self.current_step = 0
        self.prev_distance = None
        self.episode_reward = 0.0
        self.current_heading = 0.0
        self.current_distance = float("inf")
        self.min_episode_distance = float("inf")
        self.best_distance = float("inf")
        self.steps_without_progress = 0
        self.collision = False

    def _goal_geometry(self, pos: np.ndarray, heading: float) -> Tuple[float, float]:
        """Compute goal distance and heading error."""
        return goal_geometry(
            pos,
            heading,
            self.reward_computer.endpoint,
            self.config.heading_frame_offset,
        )

    def _build_observation(self, lidar: np.ndarray, pos: np.ndarray, heading: float) -> np.ndarray:
        """Build compact observation vector with goal-direction context."""
        ref_dist = float(self.config.reference_distance if self.config.reference_distance is not None else 1.0)
        return build_observation(
            lidar,
            pos,
            heading,
            self.reward_computer.endpoint,
            ref_dist,
            self.config.heading_frame_offset,
        )

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and return initial observation."""
        self.robot.motors.stop()
        self._place_goal_marker()

        self.robot.reset_position()
        self._reset_episode_state()

        for _ in range(self.config.reset_settle_steps):
            self.robot.step(self.timestep)

        lidar, pos, heading, collision = self.robot.read_sensors()
        self.current_pos = pos
        self.current_heading = heading
        self.collision = collision
        self.current_distance, _ = self._goal_geometry(pos, heading)
        self.min_episode_distance = self.current_distance
        self.best_distance = self.current_distance
        self.steps_without_progress = 0
        self.reward_computer.reference_distance = max(self.current_distance, 1e-6)

        observation = self._build_observation(lidar, pos, heading)
        return observation, {}

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step environment."""
        assert self.config.actions is not None, "Actions not initialized"
        steering, speed = self.config.actions[action_idx]
        self.robot.set_steering(steering)
        self.robot.set_speed(speed)
        for _ in range(max(1, int(self.config.action_repeat))):
            self.robot.step(self.timestep)
            self.current_step += 1
            if self.current_step >= self.config.max_steps:
                break

        lidar, pos, heading, collision = self.robot.read_sensors()
        self.current_pos = pos
        self.current_heading = heading
        self.collision = collision
        min_lidar_norm = float(np.min(lidar)) if lidar.size > 0 else 1.0
        self.current_distance, goal_error = self._goal_geometry(pos, heading)
        if self.current_distance + 1e-6 < self.min_episode_distance:
            self.min_episode_distance = self.current_distance
        if self.current_distance < (self.best_distance - self.config.stagnation_progress_delta):
            self.best_distance = self.current_distance
            self.steps_without_progress = 0
        else:
            self.steps_without_progress += 1

        reward, new_distance = self.reward_computer.compute(
            collision,
            self.current_pos,
            self.current_step,
            self.prev_distance,
            goal_error,
        )

        reward += calculate_clearance_penalty(
            min_lidar_norm,
            self.config.clearance_target_norm,
            self.config.clearance_penalty_scale,
        )

        self.prev_distance = new_distance
        self.episode_reward += reward

        terminated = collision
        truncated = self.current_step >= self.config.max_steps
        info: Dict[str, Any] = {}

        if collision:
            info["reset_reason"] = "collision"
        elif min_lidar_norm < self.config.near_collision_norm:
            terminated = True
            info["reset_reason"] = "near_collision"
            reward += self.config.near_collision_penalty
            self.episode_reward += self.config.near_collision_penalty
        elif self.current_distance < self.config.goal_radius:
            terminated = True
            info["reset_reason"] = "goal_reached"
        elif self.steps_without_progress >= self.config.stagnation_patience_steps:
            terminated = True
            info["reset_reason"] = "stagnation"
            reward += self.config.stagnation_penalty
            self.episode_reward += self.config.stagnation_penalty
        elif self.episode_reward <= self.config.low_score_threshold:
            terminated = True
            info["reset_reason"] = "low_score"

        if terminated or truncated:
            self.robot.motors.stop()

        observation = self._build_observation(lidar, pos, heading)
        return observation, reward, terminated, truncated, info
