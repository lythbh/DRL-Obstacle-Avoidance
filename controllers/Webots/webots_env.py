"""Webots simulation stack for the ALTINO robot."""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from controller import Supervisor  # pyright: ignore[reportMissingImports]

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
    """Report SLAM availability status once at startup."""
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
    """Initialize Webots Supervisor and set fast simulation mode."""
    global _supervisor
    supervisor = Supervisor()
    supervisor.simulationSetMode(Supervisor.SIMULATION_MODE_FAST)
    _supervisor = supervisor


class SLAMProcessor:
    """IMU filtering + IEKF heading + optional occupancy map for visualization."""

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
        profile_interval: int = 500,
        save_episodes: bool = False,
    ) -> None:
        """Initialize SLAM processor with IMU filtering, IEKF backend, and occupancy mapping."""
        self._dt = dt
        self._lidar_max_range = lidar_max_range
        self._lidar_max_range_inv = 1.0 / max(lidar_max_range, 1e-6)
        self._lidar_fov = lidar_fov
        self._lidar_angles: Optional[np.ndarray] = None
        self._goal = goal
        self.n_sectors = int(lidar_sector_dim)
        self.enabled = bool(enabled) and _SLAM_AVAILABLE
        self.save_episodes = bool(save_episodes) and self.enabled
        if self.n_sectors <= 0:
            raise ValueError(f"lidar_sector_dim must be positive, got {lidar_sector_dim}.")

        if self.enabled:
            self.imu_proc: Any = IMUProcessor(dt=dt)
            self.iekf: Any = IEKFBackend()
            self.slam_map: Any = SLAMMap(map_resolution=0.05) if self.save_episodes else None
        else:
            self.imu_proc = None
            self.iekf = None
            self.slam_map = None

    def reset(self, init_pos: np.ndarray, init_heading: float) -> None:
        """Reset IMU processor and IEKF backend with initial position and heading."""
        if not self.enabled or self.imu_proc is None:
            return
        self.imu_proc.reset()
        self.iekf = IEKFBackend(
            init_pos=init_pos.astype(np.float64),
            init_heading=float(init_heading),
        )

    def reset_map(self) -> None:
        """Clear occupancy grid map and reset SLAM state."""
        if self.save_episodes:
            self.slam_map = SLAMMap(map_resolution=0.05)

    def save_episode(self, run_folder: str, episode: int, reward: float = 0.0) -> None:
        """Save occupancy map visualization for the episode."""
        if not self.save_episodes or self.slam_map is None:
            return
        path = os.path.join(run_folder, f"episode_{episode:04d}_reward_{reward:.0f}.png")
        self.slam_map.save_plot(path, goal=self._goal)

    def sector_lidar(self, raw_ranges: np.ndarray) -> np.ndarray:
        """Bin raw lidar ranges into equal sectors and normalize by max range."""
        valid = np.where((raw_ranges > 0.01) & np.isfinite(raw_ranges), raw_ranges, self._lidar_max_range)
        remainder = len(valid) % self.n_sectors
        if remainder:
            valid = np.concatenate(
                [valid, np.full(self.n_sectors - remainder, self._lidar_max_range, dtype=np.float32)]
            )
        sectors = valid.reshape(self.n_sectors, -1).min(axis=1)
        return np.clip(sectors * self._lidar_max_range_inv, 0.0, 1.0).astype(np.float32)

    def _scan_to_world(self, raw_ranges: np.ndarray, pos: np.ndarray, heading: float) -> np.ndarray:
        """Transform lidar scan points from robot frame to world frame using position and heading."""
        n = len(raw_ranges)
        if self._lidar_angles is None or len(self._lidar_angles) != n:
            self._lidar_angles = np.linspace(
                -self._lidar_fov / 2.0, self._lidar_fov / 2.0, n, dtype=np.float32,
            )
        mask = (raw_ranges > 0.01) & np.isfinite(raw_ranges)
        r = raw_ranges[mask]
        assert self._lidar_angles is not None
        a = self._lidar_angles[mask]
        c, s = float(np.cos(heading)), float(np.sin(heading))
        xl = r * np.cos(a);  yl = r * np.sin(a)
        xw = c * xl - s * yl + pos[0];  yw = s * xl + c * yl + pos[1]
        return np.column_stack([xw, yw]).astype(np.float32)

    def process(
        self,
        raw_ranges: np.ndarray,
        accel: np.ndarray,
        gyro: np.ndarray,
        cmd_speed_rads: float,
        gps_pos: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Any]:
        """Process sensor data: compute lidar sectors, update IMU/IEKF, add keyframes to occupancy map."""
        lidar_sectors = self.sector_lidar(raw_ranges)

        if not self.enabled or self.imu_proc is None or self.iekf is None:
            return lidar_sectors, None

        imu_state = self.imu_proc.step(gyro, accel)
        self.iekf.propagate_odom(cmd_speed_rads * self.WHEEL_RADIUS, float(gyro[2]), self._dt)

        if self.slam_map is not None:
            pos = gps_pos if gps_pos is not None else self.iekf.state.position
            pts_2d = self._scan_to_world(raw_ranges, pos, self.iekf.state.heading)
            self.slam_map.try_add_keyframe(float(pos[0]), float(pos[1]), self.iekf.state.heading, scan_points=pts_2d)

        return lidar_sectors, imu_state


class AltinoDriver:
    """High-level robot control interface."""

    def __init__(self, config: Any):
        """Initialize ALTINO robot driver with steering, wheels, sensors, and SLAM processor."""
        global _supervisor
        assert _supervisor is not None, "Supervisor not initialized. Call _init_supervisor() first."
        self.supervisor = _supervisor
        self.config = config
        self.timestep = int(self.supervisor.getBasicTimeStep())
        self._dt = self.timestep / 1000.0

        self.left_steer = self.supervisor.getDevice('left_steer')
        self.right_steer = self.supervisor.getDevice('right_steer')
        for motor in [self.left_steer, self.right_steer]:
            motor.setPosition(0.0)
            motor.setVelocity(1.0)

        self.wheels = [
            self.supervisor.getDevice('left_front_wheel'),
            self.supervisor.getDevice('right_front_wheel'),
            self.supervisor.getDevice('left_rear_wheel'),
            self.supervisor.getDevice('right_rear_wheel'),
        ]
        for motor in self.wheels:
            motor.setPosition(float('inf'))
            motor.setVelocity(0.0)

        self.lidar = self.supervisor.getDevice("lidar")
        self.lidar.enable(self.timestep)
        self.lidar_max_range = self.lidar.getMaxRange()

        self.gps = self.supervisor.getDevice("gps")
        self.gps.enable(self.timestep)

        self.accelerometer = self.supervisor.getDevice("accelerometer")
        self.accelerometer.enable(self.timestep)

        try:
            self.gyro = self.supervisor.getDevice("gyro")
            self.gyro.enable(self.timestep)
            self._has_gyro = True
        except Exception:
            self.gyro = None
            self._has_gyro = False
            print("[ENV] WARNING: gyro device not found; using zeros.", flush=True)

        self._cmd_speed_rads = 0.0

        self.slam = SLAMProcessor(
            lidar_max_range=self.lidar_max_range,
            lidar_fov=self.lidar.getFov(),
            dt=self._dt,
            goal=config.endpoint,
            lidar_sector_dim=config.lidar_sector_dim,
            enabled=bool(getattr(config, "enable_slam", True)),
            profile_interval=int(getattr(config, "slam_profile_interval", 500)),
            save_episodes=bool(getattr(config, "save_slam_plots", True)),
        )

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
        """Set steering angle for both left and right wheels."""
        self.left_steer.setPosition(angle)
        self.right_steer.setPosition(angle)

    def set_speed(self, speed: float) -> None:
        """Set velocity target for all four wheels (front and rear)."""
        for motor in self.wheels:
            motor.setVelocity(speed)

    def stop(self) -> None:
        """Stop the robot by setting steering to zero and speed to zero."""
        self.set_steering(0.0)
        self.set_speed(0.0)

    def step(self, timestep: int) -> int:
        """Advance simulation by one timestep."""
        return self.supervisor.step(timestep)

    def _get_heading(self) -> float:
        """Extract yaw angle from robot's rotation matrix representation."""
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
        """Reset SLAM state with current GPS position and computed heading."""
        gps_vals = self.gps.getValues()
        init_pos = np.array([gps_vals[0], gps_vals[1]], dtype=np.float32)
        self.slam.reset(init_pos, self._get_heading())

    def read_sensors(self) -> Tuple[np.ndarray, np.ndarray, float, Any, bool]:
        """Read all robot sensors: lidar, GPS, accelerometer, gyro; process through SLAM."""
        raw_ranges = np.array(self.lidar.getRangeImage(), dtype=np.float32)
        gps_values = self.gps.getValues()
        pos = np.array([gps_values[0], gps_values[1]], dtype=np.float32)
        accel_values = self.accelerometer.getValues()
        accel = np.array([accel_values[0], accel_values[1], accel_values[2]], dtype=np.float32)
        if self._has_gyro and self.gyro is not None:
            gyro_values = self.gyro.getValues()
            gyro = np.array([gyro_values[0], gyro_values[1], gyro_values[2]], dtype=np.float32)
        else:
            gyro = np.zeros(3, dtype=np.float32)
        valid = raw_ranges[raw_ranges > 0.01]
        collision = bool(len(valid) > 0 and float(valid.min()) < self.config.collision_threshold)

        lidar_sectors, imu_state = self.slam.process(raw_ranges, accel, gyro, self._cmd_speed_rads, gps_pos=pos)
        heading = (self.slam.iekf.state.heading
                   if (_SLAM_AVAILABLE and self.slam.iekf is not None)
                   else self._get_heading())
        return lidar_sectors, pos, heading, imu_state, collision

    _OBSTACLE_DEFS: List[Tuple[str, float]] = [
        ("OBS_BARREL",   0.0),
        ("OBS_CONE",     0.0),
        ("OBS_FOOTBALL", 0.1),
        ("OBS_BOTTLE",   0.0),
        ("OBS_CHAIR",    0.0),
        ("OBS_BEER_1",   0.0),
        ("OBS_BEER_2",   0.0),
        ("OBS_CYL_1",    0.15),
        ("OBS_CYL_2",    0.1),
        ("OBS_CYL_3",    0.15),
        ("OBS_CYL_4",    0.15),
        ("OBS_CYL_5",    0.15),
        ("OBS_BOX_1",    0.15),
        ("OBS_BOX_2",    0.15),
        ("OBS_BOX_3",    0.15),
    ]

    def randomize_goal(self, y_range: float = 1.5) -> np.ndarray:
        """Randomise goal y-position and move the barrier wall gap to match."""
        goal_x = float(self.config.endpoint[0])
        goal_y = float(np.random.uniform(-y_range, y_range))
        wall_x = goal_x - 0.5
        half_span = 1.55

        top = self.supervisor.getFromDef("BARRIER_TOP")
        bot = self.supervisor.getFromDef("BARRIER_BOTTOM")
        if top is not None:
            top.getField("translation").setSFVec3f([wall_x, goal_y + half_span, 0.25])
        if bot is not None:
            bot.getField("translation").setSFVec3f([wall_x, goal_y - half_span, 0.25])

        return np.array([goal_x, goal_y], dtype=np.float32)

    def randomize_obstacles(self) -> None:
        """Move each obstacle into the travel corridor with 75 % probability."""
        start_xy = np.array(self.config.start_position[:2], dtype=np.float32)
        goal_xy  = np.array(self.config.endpoint, dtype=np.float32)
        placed: List[np.ndarray] = []

        for def_name, z in self._OBSTACLE_DEFS:
            node = self.supervisor.getFromDef(def_name)
            if node is None:
                continue
            field = node.getField("translation")
            for _ in range(60):
                if np.random.random() < 0.75:
                    x = float(np.random.uniform(-1.5, 2.3))
                    y = float(np.random.uniform(-1.2, 1.2))
                else:
                    x = float(np.random.uniform(-2.2, 2.2))
                    y = float(np.random.uniform(-2.2, 2.2))
                pos = np.array([x, y])
                if np.linalg.norm(pos - start_xy) < 1.0:
                    continue
                if np.linalg.norm(pos - goal_xy) < 1.0:
                    continue
                if any(np.linalg.norm(pos - p) < 0.45 for p in placed):
                    continue
                field.setSFVec3f([x, y, z])
                placed.append(pos)
                break

    def reset_position(self) -> None:
        """Reset robot to start position and rotation with added noise."""
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


class WebotsEnv:
    """Webots simulation environment for ALTINO obstacle avoidance."""

    def __init__(self, config: Any, reward_computer: Any):
        """Initialize Webots environment with robot, reward computer, and observation builders."""
        _report_slam_status()
        self.config = config
        self.action_dim = 2
        self._lidar_sector_dim = int(config.lidar_sector_dim)
        self._pose_goal_dim = int(config.pose_goal_dim)
        self._imu_feature_dim = int(config.imu_feature_dim)
        if self._lidar_sector_dim <= 0:
            raise ValueError(f"lidar_sector_dim must be positive, got {config.lidar_sector_dim}.")
        if self._pose_goal_dim != 5:
            raise ValueError(f"pose_goal_dim must be 5 for the active observation schema, got {config.pose_goal_dim}.")
        if self._imu_feature_dim != 10:
            raise ValueError(f"imu_feature_dim must be 10 for the active observation schema, got {config.imu_feature_dim}.")

        self._occupancy_grid_shape = None
        if config.occupancy_grid_shape is not None:
            grid_shape = tuple(int(dim) for dim in config.occupancy_grid_shape)
            if len(grid_shape) not in {2, 3} or any(dim <= 0 for dim in grid_shape):
                raise ValueError(f"Invalid occupancy_grid_shape {config.occupancy_grid_shape}.")
            if len(grid_shape) == 3 and grid_shape[0] != 1:
                raise ValueError("WebotsEnv emits a single occupancy channel; use shape (1, H, W).")
            self._occupancy_grid_shape = grid_shape

        self._occupancy_grid_size = (
            int(np.prod(self._occupancy_grid_shape)) if self._occupancy_grid_shape is not None else 0
        )
        self.observation_size = (
            self._lidar_sector_dim + self._pose_goal_dim + self._imu_feature_dim + self._occupancy_grid_size
        )
        self._endpoint = np.array(config.endpoint, dtype=np.float32)
        self._reference_distance = float(config.reference_distance if config.reference_distance is not None else 1.0)
        self.robot = AltinoDriver(config)
        self.timestep = self.robot.timestep

        self.headlights = self.robot.supervisor.getDevice("headlights")
        self.backlights = self.robot.supervisor.getDevice("backlights")

        self.reward_computer = reward_computer

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
        """Compute distance and direction error to goal from current position and heading."""
        goal_vec = self._endpoint - pos
        goal_distance = float(np.linalg.norm(goal_vec))
        goal_direction = float(np.arctan2(goal_vec[1], goal_vec[0]))
        goal_error = float(np.arctan2(np.sin(goal_direction - heading), np.cos(goal_direction - heading)))
        return goal_distance, goal_error

    def _occupancy_grid_observation(self) -> np.ndarray:
        """Extract occupancy grid features from SLAM map or generate default empty grid."""
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
            return grid_2d.reshape(1, height, width).reshape(-1)
        return grid_2d.reshape(-1)

    def _build_observation(self, lidar_sectors, pos, heading, imu_state):
        """Combine lidar, position, heading, IMU, and occupancy grid into observation vector."""
        goal_distance, goal_error = self._goal_geometry(pos, heading)

        direction_features = np.array([
            np.sin(heading), np.cos(heading),
            np.sin(goal_error), np.cos(goal_error),
            goal_distance / max(self._reference_distance, 1e-6),
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
            direction_features,
            accel_norm,
            gyro_norm,
            quat,
            self._occupancy_grid_observation(),
        ]).astype(np.float32)
        if observation.size != self.observation_size:
            raise RuntimeError(
                f"Observation size {observation.size} does not match configured size {self.observation_size}."
            )
        return np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to start state, return initial observation and info."""
        self.robot.slam.reset_map()
        self._episode_count += 1
        self.robot.stop()

        if getattr(self.config, "randomize_goal", False):
            y_range = float(getattr(self.config, "goal_y_range", 1.5))
            new_goal = self.robot.randomize_goal(y_range=y_range)
            self._endpoint = new_goal
            self.reward_computer.endpoint = new_goal
            self.robot.slam._goal = tuple(new_goal.tolist())
            start_xy = np.array(self.config.start_position[:2], dtype=np.float32)
            self._reference_distance = float(np.linalg.norm(start_xy - new_goal))

        self.robot.randomize_obstacles()
        self.robot.reset_position()

        self.current_step = 0
        self.prev_distance = None
        self.episode_reward = 0.0
        self.current_heading = 0.0
        self.current_distance = float("inf")
        self.min_episode_distance = float("inf")
        self.collision = False
        self.was_in_goal = False
        self.last_min_lidar_norm = 1.0

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
        """Execute action, compute reward, check termination, and return observation and info."""
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
            collision, self.current_pos, self.current_step, self.prev_distance,
            goal_error, min_lidar_norm, speed_norm, reached_new_best_distance, accel_for_reward,
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
            self.robot.stop()

        observation = self._build_observation(lidar_sectors, pos, heading, imu_state)
        return observation, reward, terminated, truncated, info
