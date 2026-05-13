"""Webots simulation stack for the ALTINO robot."""

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from controller import Supervisor  # pyright: ignore[reportMissingImports]
from controllers.common.reward_defaults import (
    COLLISION_PENALTY,
    DISTANCE_REWARD_SCALE,
    GOAL_HOLD_REWARD,
    GOAL_OVERSHOOT_PENALTY,
    GOAL_SPEED_PENALTY,
    GOAL_STOP_BONUS,
    GOAL_SUCCESS_REWARD,
    HEADING_REWARD_SCALE,
    MOTION_REWARD_SCALE,
    NEW_BEST_DISTANCE_BONUS,
    PROGRESS_REWARD_SCALE,
    SAFETY_REWARD_SCALE,
    SLOW_SPEED_PENALTY,
    SLOW_SPEED_THRESHOLD,
    HIGH_SPEED_THRESHOLD,
    HIGH_SPEED_BONUS,
    STEP_PENALTY,
)

_SLAM_IMPORT_ERROR: Optional[Exception] = None
try:
    from controllers.SLAM.imu_filter import IMUProcessor, IMUState                       # type: ignore
    from controllers.SLAM.iekf_backend import IEKFBackend                                # type: ignore
    from controllers.SLAM.slam_map import SLAMMap                                        # type: ignore
    _SLAM_AVAILABLE = True
except ImportError as _slam_err:
    _SLAM_AVAILABLE = False
    _SLAM_IMPORT_ERROR = _slam_err

_SLAM_STATUS_REPORTED = False


def _report_slam_status() -> None:
    global _SLAM_STATUS_REPORTED
    if _SLAM_STATUS_REPORTED:
        return
    _SLAM_STATUS_REPORTED = True
    if not _SLAM_AVAILABLE:
        print(
            f"[ENV] WARNING: SLAM modules unavailable ({_SLAM_IMPORT_ERROR}); using basic sensor processing.",
            flush=True,
        )

_supervisor: Optional[Supervisor] = None


def _init_supervisor() -> None:
    global _supervisor
    supervisor = Supervisor()
    supervisor.simulationSetMode(Supervisor.SIMULATION_MODE_FAST)
    _supervisor = supervisor


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
            print("[ENV] WARNING: gyro device not found; using zeros.", flush=True)

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

    def __init__(
        self,
        lidar_max_range: float,
        lidar_fov: float,
        dt: float,
        goal: Tuple[float, float] = (0.0, 0.0),
        lidar_sector_dim: int = N_SECTORS,
        enabled: bool = True,
        profile_enabled: bool = False,
        profile_interval: int = 500,
        save_episodes: bool = False,
    ) -> None:
        self._dt = dt
        self._lidar_max_range = lidar_max_range
        self._lidar_max_range_inv = 1.0 / max(lidar_max_range, 1e-6)
        self._lidar_fov = lidar_fov
        self._lidar_angles: Optional[np.ndarray] = None
        self._goal = goal
        self.n_sectors = int(lidar_sector_dim)
        self.enabled = bool(enabled) and _SLAM_AVAILABLE
        self.profile_enabled = bool(profile_enabled) and self.enabled
        self.profile_interval = max(1, int(profile_interval))
        self.save_episodes = bool(save_episodes) and self.enabled
        self._profile_step_count = 0
        self._profile_process_time = 0.0
        self._profile_keyframe_time = 0.0
        self._profile_reset_time = 0.0
        self._profile_save_time = 0.0
        self._profile_keyframes_added = 0
        self._profile_reset_calls = 0
        self._profile_save_calls = 0
        if self.n_sectors <= 0:
            raise ValueError(f"lidar_sector_dim must be positive, got {lidar_sector_dim}.")

        if self.enabled:
            self.imu_proc: Any = IMUProcessor(dt=dt)
            self.iekf: Any = IEKFBackend()
            self.slam_map: Any = SLAMMap(map_resolution=0.05)
        else:
            self.imu_proc = None
            self.iekf = None
            self.slam_map = None

    def _report_profile(self, label: str) -> None:
        if not self.profile_enabled or self._profile_step_count <= 0:
            return
        avg_process_ms = (self._profile_process_time / max(self._profile_step_count, 1)) * 1000.0
        avg_keyframe_ms = (self._profile_keyframe_time / max(self._profile_step_count, 1)) * 1000.0
        avg_reset_ms = (self._profile_reset_time / max(self._profile_reset_calls, 1)) * 1000.0
        avg_save_ms = (self._profile_save_time / max(self._profile_save_calls, 1)) * 1000.0
        print(
            f"[SLAM][PROFILE] {label} steps={self._profile_step_count} "
            f"avg_process={avg_process_ms:.3f}ms avg_keyframe={avg_keyframe_ms:.3f}ms "
            f"avg_reset={avg_reset_ms:.3f}ms avg_save={avg_save_ms:.3f}ms "
            f"keyframes={self._profile_keyframes_added} enabled={self.enabled}",
            flush=True,
        )
        self._profile_step_count = 0
        self._profile_process_time = 0.0
        self._profile_keyframe_time = 0.0
        self._profile_reset_time = 0.0
        self._profile_save_time = 0.0
        self._profile_keyframes_added = 0
        self._profile_reset_calls = 0
        self._profile_save_calls = 0

    def reset(self, init_pos: np.ndarray, init_heading: float) -> None:
        if not self.enabled or self.imu_proc is None:
            return
        start = time.perf_counter()
        self.imu_proc.reset()
        self.iekf = IEKFBackend(
            init_pos=init_pos.astype(np.float64),
            init_heading=float(init_heading),
        )
        if self.profile_enabled:
            self._profile_reset_time += time.perf_counter() - start
            self._profile_reset_calls += 1

    def reset_map(self) -> None:
        if self.enabled:
            start = time.perf_counter()
            self.slam_map = SLAMMap(map_resolution=0.05)
            if self.profile_enabled:
                self._profile_reset_time += time.perf_counter() - start
                self._profile_reset_calls += 1

    def save_episode(self, run_folder: str, episode: int, reward: float = 0.0) -> None:
        if not self.save_episodes or self.slam_map is None:
            return
        start = time.perf_counter()
        path = os.path.join(run_folder, f"episode_{episode:04d}_reward_{reward:.0f}.png")
        self.slam_map.save_plot(path, goal=self._goal)
        if self.profile_enabled:
            self._profile_save_time += time.perf_counter() - start
            self._profile_save_calls += 1

    def sector_lidar(self, raw_ranges: np.ndarray) -> np.ndarray:
        valid = np.where((raw_ranges > 0.01) & np.isfinite(raw_ranges), raw_ranges, self._lidar_max_range)
        remainder = len(valid) % self.n_sectors
        if remainder:
            valid = np.concatenate(
                [valid, np.full(self.n_sectors - remainder, self._lidar_max_range, dtype=np.float32)]
            )
        sectors = valid.reshape(self.n_sectors, -1).min(axis=1)
        return np.clip(sectors * self._lidar_max_range_inv, 0.0, 1.0).astype(np.float32)

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

        if not self.enabled or self.imu_proc is None or self.iekf is None or self.slam_map is None:
            return lidar_sectors, None

        process_start = time.perf_counter()
        imu_state = self.imu_proc.step(gyro, accel)
        cmd_speed_ms = cmd_speed_rads * self.WHEEL_RADIUS
        self.iekf.propagate_odom(cmd_speed_ms, float(gyro[2]), self._dt)

        heading = self.iekf.state.heading
        pos = gps_pos if gps_pos is not None else self.iekf.state.position
        pts_2d = self._scan_to_world(raw_ranges, pos, heading)
        keyframe_start = time.perf_counter()
        new_keyframe = self.slam_map.try_add_keyframe(
            float(pos[0]), float(pos[1]), heading, scan_points=pts_2d
        )
        if new_keyframe is not None:
            self._profile_keyframes_added += 1
        if self.profile_enabled:
            self._profile_process_time += time.perf_counter() - process_start
            self._profile_keyframe_time += time.perf_counter() - keyframe_start
            self._profile_step_count += 1
            if self._profile_step_count % self.profile_interval == 0:
                self._report_profile(f"interval={self.profile_interval}")

        return lidar_sectors, imu_state


class AltinoDriver:
    """High-level robot control interface."""

    def __init__(self, config: Any):
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
            lidar_sector_dim=config.lidar_sector_dim,
            enabled=bool(getattr(config, "enable_slam", True)),
            profile_enabled=bool(getattr(config, "profile_slam", False)),
            profile_interval=int(getattr(config, "slam_profile_interval", 500)),
            save_episodes=bool(getattr(config, "save_slam_plots", True)),
        )
        self._cmd_speed_rads = 0.0

        try:
            self.altino_node = self.supervisor.getFromDef('ALTINO')
            self.translation_field = self.altino_node.getField('translation')
            self.rotation_field = self.altino_node.getField('rotation')
        except Exception as e:
            print(f"[ENV] ERROR: Failed to get ALTINO node: {e}")
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
            print("[ENV] WARNING: cannot reset; ALTINO node not accessible!", flush=True)


class RewardComputer:
    """Computes rewards for the obstacle avoidance task."""

    def __init__(
        self,
        endpoint: np.ndarray,
        reference_distance: float,
        collision_reward: float = COLLISION_PENALTY,
        progress_scale: float = PROGRESS_REWARD_SCALE,
        distance_reward_scale: float = DISTANCE_REWARD_SCALE,
        heading_reward_scale: float = HEADING_REWARD_SCALE,
        safety_reward_scale: float = SAFETY_REWARD_SCALE,
        motion_reward_scale: float = MOTION_REWARD_SCALE,
        slow_speed_threshold: float = SLOW_SPEED_THRESHOLD,
        slow_speed_penalty: float = SLOW_SPEED_PENALTY,
        high_speed_threshold: float = HIGH_SPEED_THRESHOLD,
        high_speed_bonus: float = HIGH_SPEED_BONUS,
        new_best_distance_bonus: float = NEW_BEST_DISTANCE_BONUS,
        step_penalty: float = STEP_PENALTY,
        goal_threshold: float = 0.8,
        goal_stop_speed_threshold: float = 0.1,
        goal_success_reward: float = GOAL_SUCCESS_REWARD,
        goal_stop_bonus: float = GOAL_STOP_BONUS,
        goal_hold_reward: float = GOAL_HOLD_REWARD,
        goal_speed_penalty: float = GOAL_SPEED_PENALTY,
        goal_overshoot_penalty: float = GOAL_OVERSHOOT_PENALTY,
    ):
        self.endpoint = np.array(endpoint, dtype=np.float32)
        self.reference_distance = float(reference_distance)
        self.collision_reward = collision_reward
        self.progress_scale = progress_scale
        self.distance_reward_scale = distance_reward_scale
        self.heading_reward_scale = heading_reward_scale
        self.safety_reward_scale = safety_reward_scale
        self.motion_reward_scale = motion_reward_scale
        self.slow_speed_threshold = float(slow_speed_threshold)
        self.slow_speed_penalty = float(slow_speed_penalty)
        self.high_speed_threshold = float(high_speed_threshold)
        self.high_speed_bonus = float(high_speed_bonus)
        self.new_best_distance_bonus = new_best_distance_bonus
        self.step_penalty = step_penalty
        self.goal_threshold = float(goal_threshold)
        self.goal_stop_speed_threshold = float(goal_stop_speed_threshold)
        self.goal_success_reward = goal_success_reward
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
            if speed_norm <= self.goal_stop_speed_threshold:
                return self.goal_success_reward + self.goal_stop_bonus + self.goal_hold_reward, distance_to_end
            return self.goal_speed_penalty * speed_norm + self.goal_hold_reward, distance_to_end

        progress = 0.0
        if prev_distance is not None:
            delta = float(prev_distance - distance_to_end)
            progress = delta * self.progress_scale if delta >= 0.0 else delta * (0.25 * self.progress_scale)

        distance_ratio = float(np.clip(distance_to_end / max(self.reference_distance, 1e-6), 0.0, 2.0))
        distance_penalty = -distance_ratio * self.distance_reward_scale
        heading_alignment = float(np.cos(goal_error))
        heading_reward = heading_alignment * self.heading_reward_scale
        safety_penalty = -(1.0 - float(np.clip(min_lidar_norm, 0.0, 1.0))) * self.safety_reward_scale
        motion_reward = float(np.clip(speed_norm, 0.0, 1.0)) * self.motion_reward_scale
        slow_penalty = self.slow_speed_penalty if speed_norm < self.slow_speed_threshold else 0.0
        high_speed_reward = self.high_speed_bonus if speed_norm > self.high_speed_threshold else 0.0
        new_best_bonus = self.new_best_distance_bonus if reached_new_best_distance else 0.0

        accel_magnitude = float(np.linalg.norm(accel))
        accel_penalty = 0.0
        if distance_to_end < 1.5 and distance_to_end >= self.goal_threshold:
            accel_penalty = -0.05 * accel_magnitude

        return (
            progress
            + distance_penalty
            + heading_reward
            + safety_penalty
            + motion_reward
            + slow_penalty
            + high_speed_reward
            + new_best_bonus
            + accel_penalty
            + self.step_penalty
        ), distance_to_end


class WebotsEnv:
    """Webots simulation environment for ALTINO obstacle avoidance."""

    def __init__(self, config: Any):
        _report_slam_status()
        self.config = config
        self.action_dim = 2
        self._lidar_sector_dim = int(config.lidar_sector_dim)
        self._pose_goal_dim = int(config.pose_goal_dim)
        self._imu_feature_dim = int(config.imu_feature_dim)
        if self._lidar_sector_dim <= 0:
            raise ValueError(f"lidar_sector_dim must be positive, got {config.lidar_sector_dim}.")
        if self._pose_goal_dim != 7:
            raise ValueError(f"pose_goal_dim must be 7 for the active observation schema, got {config.pose_goal_dim}.")
        if self._imu_feature_dim != 10:
            raise ValueError(f"imu_feature_dim must be 10 for the active observation schema, got {config.imu_feature_dim}.")
        self._occupancy_grid_shape = self._normalize_occupancy_grid_shape(config.occupancy_grid_shape)
        self._occupancy_grid_size = (
            int(np.prod(self._occupancy_grid_shape)) if self._occupancy_grid_shape is not None else 0
        )
        self.observation_size = (
            self._lidar_sector_dim
            + self._pose_goal_dim
            + self._imu_feature_dim
            + self._occupancy_grid_size
        )
        self._endpoint = np.array(config.endpoint, dtype=np.float32)
        self._reference_distance = float(config.reference_distance if config.reference_distance is not None else 1.0)
        self.robot = AltinoDriver(config)
        self.timestep = self.robot.timestep

        self.headlights = self.robot.get_device("headlights")
        self.backlights = self.robot.get_device("backlights")

        self.reward_computer = RewardComputer(
            self._endpoint,
            reference_distance=self._reference_distance,
            collision_reward=config.collision_penalty,
            progress_scale=config.progress_reward_scale,
            distance_reward_scale=config.distance_reward_scale,
            heading_reward_scale=config.heading_reward_scale,
            safety_reward_scale=config.safety_reward_scale,
            motion_reward_scale=config.motion_reward_scale,
            slow_speed_threshold=float(getattr(config, "slow_speed_threshold", SLOW_SPEED_THRESHOLD)),
            slow_speed_penalty=float(getattr(config, "slow_speed_penalty", SLOW_SPEED_PENALTY)),
            high_speed_threshold=float(getattr(config, "high_speed_threshold", HIGH_SPEED_THRESHOLD)),
            high_speed_bonus=float(getattr(config, "high_speed_bonus", HIGH_SPEED_BONUS)),
            new_best_distance_bonus=config.new_best_distance_bonus,
            step_penalty=config.step_penalty,
            goal_threshold=config.goal_threshold,
            goal_stop_speed_threshold=float(getattr(config, "goal_stop_speed_threshold", 0.1)),
            goal_success_reward=config.goal_success_reward,
            goal_stop_bonus=config.goal_stop_bonus,
            goal_hold_reward=config.goal_hold_reward,
            goal_speed_penalty=config.goal_speed_penalty,
            goal_overshoot_penalty=config.goal_overshoot_penalty,
        )

        self.current_step = 0
        self.episode_reward = 0.0
        self.current_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.current_heading = 0.0
        self.current_speed_norm = 0.0
        self.current_distance = float("inf")
        self.min_episode_distance = float("inf")
        self.collision = False
        self.prev_distance: Optional[float] = None
        self.was_in_goal: bool = False
        self.last_min_lidar_norm: float = 1.0

        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        _repo_root = Path(__file__).parent.parent.parent
        self.run_folder = str(_repo_root / "plots" / ts)
        os.makedirs(self.run_folder, exist_ok=True)
        self._episode_count = 0

    @staticmethod
    def _normalize_occupancy_grid_shape(shape: Optional[Tuple[int, ...]]) -> Optional[Tuple[int, ...]]:
        if shape is None:
            return None
        grid_shape = tuple(int(dim) for dim in shape)
        if len(grid_shape) not in {2, 3}:
            raise ValueError("occupancy_grid_shape must be (H, W) or (1, H, W).")
        if any(dim <= 0 for dim in grid_shape):
            raise ValueError(f"occupancy_grid_shape dimensions must be positive, got {shape}.")
        if len(grid_shape) == 3 and grid_shape[0] != 1:
            raise ValueError("WebotsEnv emits a single occupancy channel; use shape (1, H, W).")
        return grid_shape

    def _reset_episode_state(self) -> None:
        self.current_step = 0
        self.prev_distance = None
        self.episode_reward = 0.0
        self.current_heading = 0.0
        self.current_distance = float("inf")
        self.min_episode_distance = float("inf")
        self.collision = False
        self.was_in_goal = False
        self.last_min_lidar_norm = 1.0

    def _goal_geometry(self, pos: np.ndarray, heading: float) -> Tuple[float, float]:
        goal_vec = self._endpoint - pos
        goal_distance = float(np.linalg.norm(goal_vec))
        goal_direction = float(np.arctan2(goal_vec[1], goal_vec[0]))
        goal_error = float(np.arctan2(np.sin(goal_direction - heading), np.cos(goal_direction - heading)))
        return goal_distance, goal_error

    def _occupancy_grid_observation(self) -> np.ndarray:
        if self._occupancy_grid_shape is None:
            return np.empty((0,), dtype=np.float32)

        if len(self._occupancy_grid_shape) == 2:
            height, width = self._occupancy_grid_shape
        else:
            _, height, width = self._occupancy_grid_shape

        log_odds: Optional[np.ndarray] = None
        if _SLAM_AVAILABLE and self.robot.slam.slam_map is not None:
            log_odds = self.robot.slam.slam_map.occ_map.log_odds

        if log_odds is None:
            grid_2d = np.full((height, width), 0.5, dtype=np.float32)
        else:
            row_idx = np.linspace(0, log_odds.shape[0] - 1, height).astype(np.int64)
            col_idx = np.linspace(0, log_odds.shape[1] - 1, width).astype(np.int64)
            sampled_log_odds = log_odds[np.ix_(row_idx, col_idx)]
            grid_2d = (1.0 / (1.0 + np.exp(-sampled_log_odds))).astype(np.float32)

        grid_2d = np.nan_to_num(np.clip(grid_2d, 0.0, 1.0), nan=0.5, posinf=1.0, neginf=0.0)
        if len(self._occupancy_grid_shape) == 3:
            return grid_2d.reshape(1, height, width).reshape(-1)  # [1,H,W] -> flat grid tail for the RNN.
        return grid_2d.reshape(-1)  # [H,W] -> flat grid tail for the RNN.

    def _build_observation(
        self,
        lidar_sectors: np.ndarray,
        pos: np.ndarray,
        heading: float,
        imu_state: Any,
    ) -> np.ndarray:
        goal_distance, goal_error = self._goal_geometry(pos, heading)
        ref_dist = self._reference_distance

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

        grid_features = self._occupancy_grid_observation()
        observation = np.concatenate([
            lidar_sectors,
            pos,
            direction_features,
            accel_norm,
            gyro_norm,
            quat,
            grid_features,
        ]).astype(np.float32)
        if observation.size != self.observation_size:
            raise RuntimeError(
                f"Observation size {observation.size} does not match configured size {self.observation_size}."
            )
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
        self.current_speed_norm = 0.0
        self.current_distance, _ = self._goal_geometry(pos, heading)
        self.min_episode_distance = self.current_distance
        self.last_min_lidar_norm = float(lidar_sectors.min())

        observation = self._build_observation(lidar_sectors, pos, heading, imu_state)
        return observation, {}

    def step(self, action: Union[np.ndarray, List[float], Tuple[float, float]]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_arr.size != 2:
            raise ValueError(f"Expected action with 2 elements [steering, speed], got shape {action_arr.shape}")
        steering = float(np.clip(action_arr[0], -self.config.max_steering_angle, self.config.max_steering_angle))
        requested_speed = float(np.clip(action_arr[1], self.config.min_speed, self.config.max_speed))
        steering_norm = abs(steering) / max(self.config.max_steering_angle, 1e-6)
        obstacle_factor = float(np.clip((self.last_min_lidar_norm - 0.05) / 0.95, 0.45, 1.0))
        steering_factor = float(np.clip(1.0 - 0.55 * steering_norm, 0.45, 1.0))
        adaptive_speed_cap = self.config.max_speed * obstacle_factor * steering_factor
        speed = float(np.clip(min(requested_speed, adaptive_speed_cap), self.config.min_speed, self.config.max_speed))
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
        self.last_min_lidar_norm = min_lidar_norm
        speed_norm = float(speed / max(self.config.max_speed, 1e-6))
        self.current_speed_norm = speed_norm
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

        goal_stopped = goal_reached and speed_norm <= float(getattr(self.config, "goal_stop_speed_threshold", 0.1))
        if goal_stopped:
            terminated = True
            info["reset_reason"] = "goal"

        self.was_in_goal = goal_reached

        info["goal_reached"] = goal_reached
        info["goal_stopped"] = goal_stopped
        info["success"] = goal_stopped
        info["speed_norm"] = speed_norm
        info["distance_to_goal"] = self.current_distance

        if collision:
            terminated = True
            info["reset_reason"] = "collision"
            info["success"] = False
        elif self.episode_reward <= self.config.low_score_threshold:
            terminated = True
            info["reset_reason"] = "low_score"
            info["success"] = False

        if truncated and not terminated:
            distance_ratio = float(np.clip(self.current_distance / max(self._reference_distance, 1e-6), 0.0, 2.0))
            timeout_penalty = -10.0 - 20.0 * distance_ratio
            reward += timeout_penalty
            self.episode_reward += timeout_penalty
            info["reset_reason"] = "max_steps"
            info["success"] = False

        if terminated or truncated:
            self.robot.motors.stop()

        observation = self._build_observation(lidar_sectors, pos, heading, imu_state)
        return observation, reward, terminated, truncated, info
