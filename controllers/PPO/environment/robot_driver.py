"""Webots hardware abstraction for the ALTINO robot."""

from typing import Any, Optional, Tuple

import numpy as np

from controller import Supervisor  # pyright: ignore[reportMissingImports]


class MotorController:
    """Manages steering and wheel motors."""

    def __init__(self, supervisor: Supervisor):
        """Initialize motor devices."""
        self.supervisor = supervisor

        self.left_steer = supervisor.getDevice("left_steer")
        self.right_steer = supervisor.getDevice("right_steer")
        self._init_steering()

        self.wheels = [
            supervisor.getDevice("left_front_wheel"),
            supervisor.getDevice("right_front_wheel"),
            supervisor.getDevice("left_rear_wheel"),
            supervisor.getDevice("right_rear_wheel"),
        ]
        self._init_wheels()

    def _init_steering(self) -> None:
        """Initialize steering motors."""
        for motor in [self.left_steer, self.right_steer]:
            motor.setPosition(0.0)
            motor.setVelocity(1.0)

    def _init_wheels(self) -> None:
        """Initialize wheel motors."""
        for motor in self.wheels:
            motor.setPosition(float("inf"))
            motor.setVelocity(0.0)

    def set_steering(self, angle: float) -> None:
        """Set steering angle (radians)."""
        self.left_steer.setPosition(angle)
        self.right_steer.setPosition(angle)

    def set_speed(self, speed: float) -> None:
        """Set wheel velocity."""
        for motor in self.wheels:
            motor.setVelocity(speed)

    def stop(self) -> None:
        """Stop all motors."""
        self.set_steering(0.0)
        self.set_speed(0.0)


class SensorReader:
    """Manages LiDAR and GPS sensors."""

    def __init__(self, supervisor: Supervisor, timestep: int, collision_threshold: float, lidar_bins: int):
        """Initialize sensors."""
        self.supervisor = supervisor
        self.timestep = timestep
        self.collision_threshold = collision_threshold
        self.lidar_bins = max(8, int(lidar_bins))

        self.lidar = supervisor.getDevice("lidar")
        self.lidar.enable(timestep)
        self.lidar_max_range = self.lidar.getMaxRange()
        self.lidar_inv_max_range = 1.0 / max(float(self.lidar_max_range), 1e-6)
        self._pool_starts: Optional[np.ndarray] = None
        self._pool_raw_size = 0

        self.gps = supervisor.getDevice("gps")
        self.gps.enable(timestep)

    def _pool_lidar_ranges(self, range_array: np.ndarray) -> np.ndarray:
        """Min-pool raw LiDAR beams into a fixed number of bins."""
        if range_array.size <= self.lidar_bins:
            return range_array

        if self._pool_starts is None or self._pool_raw_size != range_array.size:
            edges = np.linspace(0, range_array.size, self.lidar_bins + 1, dtype=np.int32)
            self._pool_starts = edges[:-1]
            self._pool_raw_size = int(range_array.size)

        assert self._pool_starts is not None
        return np.minimum.reduceat(range_array, self._pool_starts)[: self.lidar_bins]

    def read_observation(self) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Read LiDAR and GPS data."""
        range_array = np.asarray(self.lidar.getRangeImage(), dtype=np.float32)
        pooled_ranges = self._pool_lidar_ranges(range_array)
        lidar_data = np.clip(pooled_ranges, 0.0, self.lidar_max_range) * self.lidar_inv_max_range

        gps_values = self.gps.getValues()
        position = np.asarray(gps_values[:2], dtype=np.float32)

        min_range = float(np.min(range_array)) if range_array.size > 0 else self.lidar_max_range
        collision = min_range < self.collision_threshold

        return lidar_data, position, collision


class AltinoDriver:
    """High-level robot control interface."""

    def __init__(self, supervisor: Supervisor, config: Any):
        """Initialize robot with config and supervisor."""
        self.supervisor = supervisor
        self.config = config
        self.timestep = int(self.supervisor.getBasicTimeStep())  # type: ignore[union-attr]

        self.motors = MotorController(self.supervisor)  # type: ignore[arg-type]
        self.sensors = SensorReader(
            self.supervisor,
            self.timestep,
            config.collision_threshold,
            config.lidar_bins,
        )  # type: ignore[arg-type]

        try:
            self.altino_node = self.supervisor.getFromDef("ALTINO")  # type: ignore[union-attr]
            self.translation_field = self.altino_node.getField("translation")
            self.rotation_field = self.altino_node.getField("rotation")
        except Exception as e:
            print(f"[PPO] ERROR: Failed to get ALTINO node: {e}")
            self.altino_node = None
            self.translation_field = None
            self.rotation_field = None

    def set_steering(self, angle: float) -> None:
        """Set steering angle."""
        self.motors.set_steering(angle)

    def set_speed(self, speed: float) -> None:
        """Set velocity."""
        self.motors.set_speed(speed)

    def get_device(self, name: str):
        """Get a named device from supervisor."""
        return self.supervisor.getDevice(name)  # type: ignore[union-attr]

    def step(self, timestep: int) -> int:
        """Step simulation."""
        return self.supervisor.step(timestep)  # type: ignore[union-attr]

    def _get_heading(self) -> float:
        """Estimate robot yaw in world frame from rotation field."""
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

    def read_sensors(self) -> Tuple[np.ndarray, np.ndarray, float, bool]:
        """Read LiDAR, GPS, heading and collision status."""
        lidar, pos, collision = self.sensors.read_observation()
        heading = self._get_heading()
        return lidar, pos, heading, collision

    def reset_position(self) -> None:
        """Reset robot to start position and orientation."""
        if self.translation_field is not None and self.rotation_field is not None:
            start_position_values = self.config.start_position or [0.05, -0.1, 0.02]
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

            self.translation_field.setSFVec3f(start_position.tolist())  # type: ignore[arg-type]
            self.rotation_field.setSFRotation(start_rotation)  # type: ignore[arg-type]
            self.supervisor.simulationResetPhysics()  # type: ignore[union-attr]
        else:
            print("[PPO] WARNING: Cannot reset - ALTINO node not accessible!")
