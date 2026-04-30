"""Webots simulation stack for the ALTINO robot."""

import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np

from controller import Supervisor  # pyright: ignore[reportMissingImports]

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from controllers.SLAM.lidar_preprocessing import LiDARPreprocessor, LiDARFeatures  # type: ignore
    from controllers.SLAM.imu_filter import IMUProcessor, IMUState                       # type: ignore
    from controllers.SLAM.iekf_backend import IEKFBackend                                # type: ignore
    from controllers.SLAM.slam_map import SLAMMap                                        # type: ignore
    _SLAM_AVAILABLE = True
    print("[PPO] SLAM sensor modules loaded OK", flush=True)
except ImportError as _slam_err:
    _SLAM_AVAILABLE = False
    print(f"[PPO] WARNING: SLAM modules not importable ({_slam_err}). "
          "Falling back to basic sensor processing.", flush=True)

if TYPE_CHECKING:
    from controllers.PPO.PPO import Config


_supervisor: Optional[Supervisor] = None


def _init_supervisor() -> None:
    """Initialize global Supervisor instance for field access."""
    global _supervisor
    _supervisor = Supervisor()


class MotorController:
    """Manages steering and wheel motors."""

    def __init__(self, supervisor: Supervisor):
        self.supervisor: Any = supervisor
        self.left_steer: Any = supervisor.getDevice('left_steer')
        self.right_steer: Any = supervisor.getDevice('right_steer')
        self._init_steering()

        self.wheels: List[Any] = [
            supervisor.getDevice('left_front_wheel'),
            supervisor.getDevice('right_front_wheel'),
            supervisor.getDevice('left_rear_wheel'),
            supervisor.getDevice('right_rear_wheel'),
        ]
        self._init_wheels()

    def _init_steering(self) -> None:
        for motor in [self.left_steer, self.right_steer]:
            motor.setPosition(0.0)
            motor.setVelocity(1.0)

    def _init_wheels(self) -> None:
        for motor in self.wheels:
            motor.setPosition(float('inf'))
            motor.setVelocity(0.0)

    def set_steering(self, angle: float) -> None:
        self.left_steer.setPosition(angle)
        self.right_steer.setPosition(angle)

    def set_speed(self, speed: float) -> None:
        for motor in self.wheels:
            motor.setVelocity(speed)

    def stop(self) -> None:
        self.set_steering(0.0)
        self.set_speed(0.0)


class SensorReader:
    """Manages LiDAR, GPS, accelerometer, and gyroscope sensors."""

    def __init__(self, supervisor: Supervisor, timestep: int, collision_threshold: float):
        self.supervisor: Any = supervisor
        self.timestep = timestep
        self.collision_threshold = collision_threshold

        self.lidar: Any = supervisor.getDevice("lidar")
        self.lidar.enable(timestep)
        self.lidar_max_range = self.lidar.getMaxRange()

        self.gps: Any = supervisor.getDevice("gps")
        self.gps.enable(timestep)

        self.accelerometer: Any = supervisor.getDevice("accelerometer")
        self.accelerometer.enable(timestep)

        try:
            self.gyro: Any = supervisor.getDevice("gyro")
            self.gyro.enable(timestep)
            self._has_gyro = True
        except Exception:
            self.gyro = None
            self._has_gyro = False
            print("[PPO] WARNING: 'gyro' device not found – gyro readings will be zero.", flush=True)

    def read_observation(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
        raw_ranges = np.array(self.lidar.getRangeImage(), dtype=np.float32)

        gps_values = self.gps.getValues()
        position = np.array([gps_values[0], gps_values[1]], dtype=np.float32)

        accel_values = self.accelerometer.getValues()
        accel = np.array([accel_values[0], accel_values[1], accel_values[2]], dtype=np.float32)

        if self._has_gyro:
            gyro_values = self.gyro.getValues()
            gyro = np.array([gyro_values[0], gyro_values[1], gyro_values[2]], dtype=np.float32)
        else:
            gyro = np.zeros(3, dtype=np.float32)

        valid = raw_ranges[raw_ranges > 0.01]
        collision = bool(len(valid) > 0 and float(valid.min()) < self.collision_threshold)

        return raw_ranges, position, accel, gyro, collision


class SLAMProcessor:
    """IMU filtering + IEKF heading + occupancy map for the controller."""

    WHEEL_RADIUS: float = 0.033
    N_SECTORS: int = 16

    def __init__(self, lidar_max_range: float, lidar_fov: float, dt: float, goal: Tuple[float, float] = (0.0, 0.0)) -> None:
        self._dt = dt
        self._lidar_max_range = lidar_max_range
        self._lidar_fov = lidar_fov
        self._lidar_angles: Optional[np.ndarray] = None
        self._goal = goal

        if _SLAM_AVAILABLE:
            self.imu_proc: Any = IMUProcessor(dt=dt)
            self.iekf: Any = IEKFBackend()
            self.slam_map: Any = SLAMMap(map_resolution=0.05)
        else:
            self.imu_proc = None
            self.iekf = None
            self.slam_map = None

    def reset(self, init_pos: np.ndarray, init_heading: float) -> None:
        if not _SLAM_AVAILABLE or self.imu_proc is None:
            return
        self.imu_proc.reset()
        self.iekf = IEKFBackend(
            init_pos=init_pos.astype(np.float64),
            init_heading=float(init_heading),
        )

    def reset_map(self) -> None:
        if _SLAM_AVAILABLE:
            self.slam_map = SLAMMap(map_resolution=0.05)

    def save_episode(self, run_folder: str, episode: int, reward: float = 0.0) -> None:
        if not _SLAM_AVAILABLE or self.slam_map is None:
            return
        path = os.path.join(run_folder, f"episode_{episode:04d}_reward_{reward:.0f}.png")
        self.slam_map.save_plot(path, goal=self._goal)

    def sector_lidar(self, raw_ranges: np.ndarray) -> np.ndarray:
        valid = np.where((raw_ranges > 0.01) & np.isfinite(raw_ranges), raw_ranges, self._lidar_max_range)
        remainder = len(valid) % self.N_SECTORS
        if remainder:
            valid = np.concatenate(
                [valid, np.full(self.N_SECTORS - remainder, self._lidar_max_range, dtype=np.float32)]
            )
        sectors = valid.reshape(self.N_SECTORS, -1).min(axis=1)
        return np.clip(sectors / self._lidar_max_range, 0.0, 1.0).astype(np.float32)

    def _scan_to_world(self, raw_ranges: np.ndarray, pos: np.ndarray, heading: float) -> np.ndarray:
        n = len(raw_ranges)
        if self._lidar_angles is None or len(self._lidar_angles) != n:
            self._lidar_angles = np.linspace(
                start=-self._lidar_fov / 2.0,
                stop=self._lidar_fov / 2.0,
                num=n,
                dtype=np.float32,
            )
        mask = (raw_ranges > 0.01) & np.isfinite(raw_ranges)
        r = raw_ranges[mask]
        lidar_angles = self._lidar_angles
        assert lidar_angles is not None
        a = lidar_angles[mask]
        c, s = float(np.cos(heading)), float(np.sin(heading))
        xl = r * np.cos(a)
        yl = r * np.sin(a)
        xw = c * xl - s * yl + pos[0]
        yw = s * xl + c * yl + pos[1]
        return np.column_stack([xw, yw]).astype(np.float32)

    def process(
        self,
        raw_ranges: np.ndarray,
        accel: np.ndarray,
        gyro: np.ndarray,
        cmd_speed_rads: float,
        gps_pos: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Any]:
        lidar_sectors = self.sector_lidar(raw_ranges)

        if not _SLAM_AVAILABLE or self.imu_proc is None or self.iekf is None or self.slam_map is None:
            return lidar_sectors, None

        imu_state = self.imu_proc.step(gyro, accel)
        cmd_speed_ms = cmd_speed_rads * self.WHEEL_RADIUS
        self.iekf.propagate_odom(cmd_speed_ms, float(gyro[2]), self._dt)

        heading = self.iekf.state.heading
        pos = gps_pos if gps_pos is not None else self.iekf.state.position
        pts_2d = self._scan_to_world(raw_ranges, pos, heading)
        self.slam_map.try_add_keyframe(
            float(pos[0]), float(pos[1]), heading, scan_points=pts_2d
        )

        return lidar_sectors, imu_state


class AltinoDriver:
    """High-level robot control interface."""

    def __init__(self, config: "Config"):
        global _supervisor
        assert _supervisor is not None, "Supervisor not initialized. Call _init_supervisor() first."
        self.supervisor = _supervisor
        self.config = config
        self.timestep = int(self.supervisor.getBasicTimeStep())
        self._dt = self.timestep / 1000.0

        self.motors = MotorController(self.supervisor)
        self.sensors = SensorReader(self.supervisor, self.timestep, config.collision_threshold)
        self.slam = SLAMProcessor(
            lidar_max_range=self.sensors.lidar_max_range,
            lidar_fov=self.sensors.lidar.getFov(),
            dt=self._dt,
            goal=config.endpoint,
        )
        self._cmd_speed_rads = 0.0

        try:
            self.altino_node = self.supervisor.getFromDef('ALTINO')
            self.translation_field = self.altino_node.getField('translation')
            self.rotation_field = self.altino_node.getField('rotation')
        except Exception as e:
            print(f"[PPO] ERROR: Failed to get ALTINO node: {e}")
            self.altino_node = None
            self.translation_field = None
            self.rotation_field = None

    def set_steering(self, angle: float) -> None:
        self.motors.set_steering(angle)

    def set_speed(self, speed: float) -> None:
        self.motors.set_speed(speed)
        self._cmd_speed_rads = speed

    def get_device(self, name: str):
        return self.supervisor.getDevice(name)

    def step(self, timestep: int) -> int:
        return self.supervisor.step(timestep)

    def _get_heading(self) -> float:
        if self.rotation_field is None:
            return 0.0
        rotation = self.rotation_field.getSFRotation()
        if rotation is None or len(rotation) < 4:
            return 0.0
        x, y, z, angle = map(float, rotation)
        axis_norm = np.sqrt(x * x + y * y + z * z)
        if axis_norm < 1e-8:
            return 0.0
        x /= axis_norm
        y /= axis_norm
        z /= axis_norm
        c = float(np.cos(angle))
        s = float(np.sin(angle))
        one_c = 1.0 - c
        r00 = c + x * x * one_c
        r10 = z * s + y * x * one_c
        yaw = float(np.arctan2(r10, r00))
        return float(np.arctan2(np.sin(yaw), np.cos(yaw)))

    def reset_slam(self) -> None:
        gps_vals = self.sensors.gps.getValues()
        init_pos = np.array([gps_vals[0], gps_vals[1]], dtype=np.float32)
        self.slam.reset(init_pos, self._get_heading())
        self._cmd_speed_rads = 0.0

    def read_sensors(self) -> Tuple[np.ndarray, np.ndarray, float, Any, bool]:
        raw_ranges, pos, accel, gyro, collision = self.sensors.read_observation()
        lidar_sectors, imu_state = self.slam.process(
            raw_ranges, accel, gyro, self._cmd_speed_rads, gps_pos=pos
        )
        heading = (self.slam.iekf.state.heading
                   if (_SLAM_AVAILABLE and self.slam.iekf is not None)
                   else self._get_heading())
        return lidar_sectors, pos, heading, imu_state, collision

    def reset_position(self) -> None:
        if self.translation_field is not None and self.rotation_field is not None:
            start_position_values = self.config.start_position or [-2.0, 0.0, 0.02]
            start_rotation_values = self.config.start_rotation or [0.0, 0.0, 1.0, 0.0]

            start_position = np.array(start_position_values, dtype=np.float32)
            if self.config.start_position_noise > 0.0:
                start_position[:2] += np.random.uniform(
                    -self.config.start_position_noise,
                    self.config.start_position_noise,
                    size=2,
                ).astype(np.float32)

            start_rotation = list(start_rotation_values)
            if self.config.start_yaw_noise > 0.0:
                start_rotation[3] = float(
                    start_rotation[3]
                    + np.random.uniform(-self.config.start_yaw_noise, self.config.start_yaw_noise)
                )

            self.translation_field.setSFVec3f(start_position.tolist())
            self.rotation_field.setSFRotation(start_rotation)
            self.supervisor.simulationResetPhysics()
        else:
            print("[PPO] WARNING: Cannot reset - ALTINO node not accessible!")


class RewardComputer:
    """Computes rewards for the obstacle avoidance task."""

    def __init__(
        self,
        endpoint: np.ndarray,
        reference_distance: float,
        collision_reward: float = -40.0,
        progress_scale: float = 3.0,
        distance_reward_scale: float = 2.0,
        heading_reward_scale: float = 0.08,
        safety_reward_scale: float = 0.2,
        motion_reward_scale: float = 0.05,
        new_best_distance_bonus: float = 1.0,
        step_penalty: float = -0.01,
        goal_threshold: float = 0.8,
        goal_stop_bonus: float = 120.0,
        goal_hold_reward: float = 10.0,
        goal_speed_penalty: float = -60.0,
        goal_overshoot_penalty: float = -50.0,
    ):
        self.endpoint = np.array(endpoint, dtype=np.float32)
        self.reference_distance = float(reference_distance)
        self.collision_reward = collision_reward
        self.progress_scale = progress_scale
        self.distance_reward_scale = distance_reward_scale
        self.heading_reward_scale = heading_reward_scale
        self.safety_reward_scale = safety_reward_scale
        self.motion_reward_scale = motion_reward_scale
        self.new_best_distance_bonus = new_best_distance_bonus
        self.step_penalty = step_penalty
        self.goal_threshold = float(goal_threshold)
        self.goal_stop_bonus = goal_stop_bonus
        self.goal_hold_reward = goal_hold_reward
        self.goal_speed_penalty = goal_speed_penalty
        self.goal_overshoot_penalty = goal_overshoot_penalty
        self.best_time = np.inf

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
        if collision:
            return self.collision_reward, None

        distance_to_end = float(np.linalg.norm(current_pos - self.endpoint))

        if distance_to_end < self.goal_threshold:
            if speed_norm <= 0.1:
                return 200.0 + self.goal_stop_bonus + self.goal_hold_reward, distance_to_end
            return self.goal_speed_penalty * speed_norm + self.goal_hold_reward, distance_to_end

        progress = 0.0
        if prev_distance is not None:
            delta = float(prev_distance - distance_to_end)
            progress = delta * self.progress_scale if delta >= 0.0 else delta * (0.25 * self.progress_scale)

        proximity = max(0.0, (self.reference_distance - distance_to_end) / max(self.reference_distance, 1e-6))
        proximity_reward = proximity * self.distance_reward_scale
        heading_alignment = max(0.0, float(np.cos(goal_error)))
        heading_reward = heading_alignment * self.heading_reward_scale
        safety_reward = float(np.clip(min_lidar_norm, 0.0, 1.0)) * self.safety_reward_scale
        motion_reward = float(np.clip(speed_norm, 0.0, 1.0)) * self.motion_reward_scale
        new_best_bonus = self.new_best_distance_bonus if reached_new_best_distance else 0.0

        accel_magnitude = float(np.linalg.norm(accel))
        accel_penalty = 0.0
        if distance_to_end < 1.5 and distance_to_end >= self.goal_threshold:
            accel_penalty = -0.05 * accel_magnitude

        return (
            progress
            + proximity_reward
            + heading_reward
            + safety_reward
            + motion_reward
            + new_best_bonus
            + accel_penalty
            + self.step_penalty
        ), distance_to_end


class WebotsEnv:
    """Webots simulation environment for ALTINO obstacle avoidance."""

    def __init__(self, config: "Config"):
        self.config = config
        self.robot = AltinoDriver(config)
        self.timestep = self.robot.timestep

        self.headlights = self.robot.get_device("headlights")
        self.backlights = self.robot.get_device("backlights")

        self.reward_computer = RewardComputer(
            np.array(config.endpoint, dtype=np.float32),
            reference_distance=float(config.reference_distance if config.reference_distance is not None else 1.0),
            collision_reward=config.collision_penalty,
            progress_scale=config.progress_reward_scale,
            distance_reward_scale=config.distance_reward_scale,
            heading_reward_scale=config.heading_reward_scale,
            safety_reward_scale=config.safety_reward_scale,
            motion_reward_scale=config.motion_reward_scale,
            new_best_distance_bonus=config.new_best_distance_bonus,
            step_penalty=config.step_penalty,
            goal_threshold=config.goal_threshold,
            goal_stop_bonus=config.goal_stop_bonus,
            goal_hold_reward=config.goal_hold_reward,
            goal_speed_penalty=config.goal_speed_penalty,
            goal_overshoot_penalty=config.goal_overshoot_penalty,
        )

        self.current_step = 0
        self.episode_reward = 0.0
        self.current_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.current_heading = 0.0
        self.current_distance = float("inf")
        self.min_episode_distance = float("inf")
        self.collision = False
        self.prev_distance: Optional[float] = None
        self.was_in_goal: bool = False

        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        _repo_root = Path(__file__).parent.parent.parent
        self.run_folder = str(_repo_root / "plots" / ts)
        os.makedirs(self.run_folder, exist_ok=True)
        self._episode_count = 0
        print(f"[PPO] SLAM maps will be saved to: {self.run_folder}", flush=True)

    def _reset_episode_state(self) -> None:
        self.current_step = 0
        self.prev_distance = None
        self.episode_reward = 0.0
        self.current_heading = 0.0
        self.current_distance = float("inf")
        self.min_episode_distance = float("inf")
        self.collision = False
        self.was_in_goal = False

    def _goal_geometry(self, pos: np.ndarray, heading: float) -> Tuple[float, float]:
        goal_vec = np.array(self.config.endpoint, dtype=np.float32) - pos
        goal_distance = float(np.linalg.norm(goal_vec))
        goal_direction = float(np.arctan2(goal_vec[1], goal_vec[0]))
        goal_error = float(np.arctan2(np.sin(goal_direction - heading), np.cos(goal_direction - heading)))
        return goal_distance, goal_error

    def _build_observation(
        self,
        lidar_sectors: np.ndarray,
        pos: np.ndarray,
        heading: float,
        imu_state: Any,
    ) -> np.ndarray:
        goal_distance, goal_error = self._goal_geometry(pos, heading)
        ref_dist = float(self.config.reference_distance if self.config.reference_distance is not None else 1.0)

        direction_features = np.array([
            np.sin(heading),
            np.cos(heading),
            np.sin(goal_error),
            np.cos(goal_error),
            goal_distance / max(ref_dist, 1e-6),
        ], dtype=np.float32)

        if _SLAM_AVAILABLE and imu_state is not None:
            accel_norm = np.clip(imu_state.accel_body, -10.0, 10.0) / 10.0
            gyro_norm = np.clip(imu_state.gyro_body, -np.pi, np.pi) / np.pi
            quat = imu_state.quaternion.astype(np.float32)
        else:
            accel_norm = np.zeros(3, dtype=np.float32)
            gyro_norm = np.zeros(3, dtype=np.float32)
            quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        observation = np.concatenate([
            lidar_sectors,
            pos,
            direction_features,
            accel_norm,
            gyro_norm,
            quat,
        ]).astype(np.float32)
        return np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.robot.slam.reset_map()
        self._episode_count += 1
        self.robot.motors.stop()
        self.robot.reset_position()
        self._reset_episode_state()

        for _ in range(self.config.reset_settle_steps):
            self.robot.step(self.timestep)

        self.robot.reset_slam()

        lidar_sectors, pos, heading, imu_state, collision = self.robot.read_sensors()
        self.current_pos = pos
        self.current_heading = heading
        self.collision = collision
        self.current_distance, _ = self._goal_geometry(pos, heading)
        self.min_episode_distance = self.current_distance

        observation = self._build_observation(lidar_sectors, pos, heading, imu_state)
        return observation, {}

    def step(self, action: Union[np.ndarray, List[float], Tuple[float, float]]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_arr.size != 2:
            raise ValueError(f"Expected action with 2 elements [steering, speed], got shape {action_arr.shape}")
        steering = float(np.clip(action_arr[0], -self.config.max_steering_angle, self.config.max_steering_angle))
        speed = float(np.clip(action_arr[1], self.config.min_speed, self.config.max_speed))
        self.robot.set_steering(steering)
        self.robot.set_speed(speed)
        self.robot.step(self.timestep)
        self.current_step += 1

        lidar_sectors, pos, heading, imu_state, collision = self.robot.read_sensors()
        self.current_pos = pos
        self.current_heading = heading
        self.collision = collision
        self.current_distance, goal_error = self._goal_geometry(pos, heading)
        reached_new_best_distance = self.current_distance + 1e-6 < self.min_episode_distance
        if reached_new_best_distance:
            self.min_episode_distance = self.current_distance

        min_lidar_norm = float(lidar_sectors.min())
        speed_norm = float(speed / max(self.config.max_speed, 1e-6))
        goal_reached = self.current_distance < self.config.goal_threshold

        accel_for_reward = (imu_state.accel_body
                            if (_SLAM_AVAILABLE and imu_state is not None)
                            else np.zeros(3, dtype=np.float32))

        terminated = False
        truncated = self.current_step >= self.config.max_steps
        info: Dict[str, Any] = {}

        reward, new_distance = self.reward_computer.compute(
            collision,
            self.current_pos,
            self.current_step,
            self.prev_distance,
            goal_error,
            min_lidar_norm,
            speed_norm,
            reached_new_best_distance,
            accel_for_reward,
        )
        self.prev_distance = new_distance
        self.episode_reward += reward

        if self.was_in_goal and not goal_reached and speed_norm > 0.05:
            reward += self.config.goal_overshoot_penalty
            self.episode_reward += self.config.goal_overshoot_penalty

        goal_stopped = goal_reached and speed_norm <= 0.1
        if goal_stopped:
            terminated = True
            info["reset_reason"] = "goal"
        elif goal_reached:
            penalty = self.config.goal_speed_penalty * speed_norm
            reward += penalty
            self.episode_reward += penalty

        if self.episode_reward >= self.config.goal_score_threshold:
            terminated = True
            info["reset_reason"] = "goal"

        self.was_in_goal = goal_reached

        if collision:
            terminated = True
            info["reset_reason"] = "collision"
        elif self.episode_reward <= self.config.low_score_threshold:
            terminated = True
            info["reset_reason"] = "low_score"

        if terminated or truncated:
            self.robot.motors.stop()

        observation = self._build_observation(lidar_sectors, pos, heading, imu_state)
        return observation, reward, terminated, truncated, info