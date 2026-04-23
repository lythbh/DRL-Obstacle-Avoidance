"""PPO training controller for ALTINO robot in Webots obstacle avoidance task."""
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any, Union
import numpy as np
import torch
from torch import multiprocessing, nn
from torch.nn.utils.rnn import pad_sequence
from torch.distributions import Independent, Normal

from controller import Supervisor  # pyright: ignore[reportMissingImports]

# ── SLAM sensor-processing modules ──────────────────────────────────────────
# Controllers live one level up from controllers/PPO/
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from lidar_preprocessing import LiDARPreprocessor, LiDARFeatures  # type: ignore
    from imu_filter import IMUProcessor, IMUState                       # type: ignore
    from iekf_backend import IEKFBackend                                # type: ignore
    from slam_map import SLAMMap                                        # type: ignore
    _SLAM_AVAILABLE = True
    print("[PPO] SLAM sensor modules loaded OK", flush=True)
except ImportError as _slam_err:
    _SLAM_AVAILABLE = False
    print(f"[PPO] WARNING: SLAM modules not importable ({_slam_err}). "
          "Falling back to basic sensor processing.", flush=True)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Training and environment hyperparameters."""
    
    # Training
    episodes: int = 500
    update_every: int = 5  # PPO update frequency (episodes)
    epochs: int = 4  # Optimization epochs per update
    batch_size: int = 64
    
    # PPO Agent
    gamma: float = 0.99  # Discount factor
    epsilon: float = 0.2  # PPO clip parameter
    learning_rate: float = 1e-4
    entropy_coef: float = 0.02  # Entropy regularization
    hidden_size: int = 128  # Network hidden layer size
    latent_size: int = 128  # Encoder output size before the recurrent core
    lstm_hidden_size: int = 128  # Hidden size of the recurrent core
    lstm_layers: int = 1
    lidar_sector_dim: int = 16
    pose_goal_dim: int = 7  # [x, y] + [sin/cos heading, sin/cos goal_error, norm dist]
    imu_feature_dim: int = 10  # accel(3) + gyro(3) + quaternion(4)
    occupancy_grid_shape: Optional[Tuple[int, ...]] = None  # Optional CNN shape, e.g. (1, 16, 16)
    
    # Environment
    max_steps: int = 4000  # Max steps per episode
    collision_threshold: float = 0.1  # LiDAR distance threshold for collision
    low_score_threshold: float = -800.0  # Episode reset threshold
    collision_penalty: float = -20.0  # Penalty when collision happens
    progress_reward_scale: float = 3.0  # Scale for distance-progress reward
    distance_reward_scale: float = 2.0  # Dense reward for being closer to the goal than the start state
    heading_reward_scale: float = 0.5  # Bonus when facing toward the goal
    safety_reward_scale: float = 0.2  # Encourages keeping distance from obstacles
    motion_reward_scale: float = 0.05  # Bonus for moving forward to avoid stop-policy collapse
    new_best_distance_bonus: float = 1.0  # Bonus when reaching a new closest distance to goal
    step_penalty: float = -0.01  # Small per-step penalty to encourage efficiency
    endpoint: Tuple[float, float] = (2.0, 0.0)  # Goal location
    goal_threshold: float = 0.1  # Radius around goal considered reached
    goal_stop_bonus: float = 120.0  # Extra reward for stopping at the goal
    goal_hold_reward: float = 10.0  # Reward per timestep while inside goal threshold
    goal_speed_penalty: float = -60.0  # Penalty for still moving inside the goal region
    goal_overshoot_penalty: float = -50.0  # Penalty for driving past the goal region
    goal_score_threshold: float = 5000.0  # End episode when total reward reaches this threshold
    reference_distance: Optional[float] = None  # Start-to-goal distance, filled in at init
    
    # Robot Control
    max_steering_angle: float = 0.9
    min_speed: float = 0.0
    start_position: Optional[List[float]] = None  # [x, y, z]
    start_rotation: Optional[List[float]] = None  # [x, y, z, w]
    start_position_noise: float = 0.03  # Random position jitter at reset
    start_yaw_noise: float = 0.2  # Random yaw jitter at reset
    episode_warmup_steps: int = 12  # Random exploration steps after reset
    
    # Motor/Sensor Config
    max_speed: float = 10.0
    reset_settle_steps: int = 10  # Steps to wait for physics to settle after reset
    
    def __post_init__(self) -> None:
        """Initialize defaults for mutable fields."""
        if self.start_position is None:
            self.start_position = [-2.0, 0.0, 0.02]
        if self.start_rotation is None:
            self.start_rotation = [0.0, 0.0, 1.0, 0.0]
        if self.reference_distance is None:
            start_xy = np.array(self.start_position[:2], dtype=np.float32)
            endpoint_xy = np.array(self.endpoint, dtype=np.float32)
            self.reference_distance = float(np.linalg.norm(start_xy - endpoint_xy))


# Global supervisor instance
_supervisor: Optional[Supervisor] = None

def _init_supervisor() -> None:
    """Initialize global Supervisor instance for field access."""
    global _supervisor
    _supervisor = Supervisor()



# ============================================================================
# HARDWARE CONTROL LAYER
# ============================================================================

class MotorController:
    """Manages steering and wheel motors."""
    
    def __init__(self, supervisor: Supervisor):
        """Initialize motor devices."""
        self.supervisor = supervisor
        
        # Steering motors
        self.left_steer = supervisor.getDevice('left_steer')
        self.right_steer = supervisor.getDevice('right_steer')
        self._init_steering()
        
        # Wheel motors
        self.wheels = [
            supervisor.getDevice('left_front_wheel'),
            supervisor.getDevice('right_front_wheel'),
            supervisor.getDevice('left_rear_wheel'),
            supervisor.getDevice('right_rear_wheel'),
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
            motor.setPosition(float('inf'))
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
    """Manages LiDAR, GPS, accelerometer, and gyroscope sensors."""

    def __init__(self, supervisor: Supervisor, timestep: int, collision_threshold: float):
        """Initialize sensors."""
        self.supervisor = supervisor
        self.timestep = timestep
        self.collision_threshold = collision_threshold

        # LiDAR
        self.lidar = supervisor.getDevice("lidar")
        self.lidar.enable(timestep)
        self.lidar_max_range = self.lidar.getMaxRange()

        # GPS
        self.gps = supervisor.getDevice("gps")
        self.gps.enable(timestep)

        # Accelerometer
        self.accelerometer = supervisor.getDevice("accelerometer")
        self.accelerometer.enable(timestep)

        # Gyroscope (needed for IMU filtering and IEKF odometry)
        try:
            self.gyro = supervisor.getDevice("gyro")
            self.gyro.enable(timestep)
            self._has_gyro = True
        except Exception:
            self.gyro = None
            self._has_gyro = False
            print("[PPO] WARNING: 'gyro' device not found – gyro readings will be zero.", flush=True)

    def read_observation(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
        """Read raw sensor data (unnormalized for SLAM preprocessing).

        Returns:
            raw_ranges:  Raw LiDAR range image in metres (not normalized)
            position:    GPS [x, y] in world coordinates
            accel:       Accelerometer [x, y, z] in m/s²
            gyro:        Gyroscope [x, y, z] in rad/s (zeros if unavailable)
            collision:   True when any valid range is below collision_threshold
        """
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

        # Collision: ignore zero/invalid returns (below min_range)
        valid = raw_ranges[raw_ranges > 0.01]
        collision = bool(len(valid) > 0 and float(valid.min()) < self.collision_threshold)

        return raw_ranges, position, accel, gyro, collision


class SLAMProcessor:
    """
    IMU filtering + IEKF heading + occupancy map for the PPO controller.

    Per-timestep pipeline (hot path, kept fast):
      1. Sector LiDAR (vectorised numpy) – 16 sector min-ranges for the obs.
      2. IMUProcessor (Madgwick + EKF)   – filtered quaternion, accel, gyro.
      3. IEKF propagate_odom             – heading from wheel odometry + gyro.
      4. SLAMMap keyframe update         – occupancy grid + point cloud (async,
         only when the robot has moved enough; ~30 cm or 8.5° threshold).

    LiDARPreprocessor (Python curvature loop) and IEKF measurement update
    (O(N_pts × N_map) matching) are omitted from the hot path for speed.
    """

    WHEEL_RADIUS: float = 0.033   # metres – ALTINO wheel radius
    N_SECTORS:    int   = 16      # angular sectors in the observation vector

    def __init__(self, lidar_max_range: float, lidar_fov: float, dt: float,
                 goal: Tuple[float, float] = (0.0, 0.0)) -> None:
        self._dt = dt
        self._lidar_max_range = lidar_max_range
        self._lidar_fov = lidar_fov
        self._lidar_angles: Optional[np.ndarray] = None  # built lazily
        self._goal = goal

        if _SLAM_AVAILABLE:
            self.imu_proc = IMUProcessor(dt=dt)
            self.iekf     = IEKFBackend()
            self.slam_map = SLAMMap(map_resolution=0.05)
        else:
            self.imu_proc  = None   # type: ignore[assignment]
            self.iekf      = None   # type: ignore[assignment]
            self.slam_map  = None   # type: ignore[assignment]

    # ── Episode management ───────────────────────────────────────────────────

    def reset(self, init_pos: np.ndarray, init_heading: float) -> None:
        """Reset IMU/IEKF filters at episode start with current ground-truth pose."""
        if not _SLAM_AVAILABLE:
            return
        self.imu_proc.reset()
        self.iekf = IEKFBackend(
            init_pos=init_pos.astype(np.float64),
            init_heading=float(init_heading),
        )

    def reset_map(self) -> None:
        """Discard the current episode map and start a fresh one."""
        if _SLAM_AVAILABLE:
            self.slam_map = SLAMMap(map_resolution=0.05)

    def save_episode(self, run_folder: str, episode: int, reward: float = 0.0) -> None:
        """Save occupancy map + trajectory plot for one episode."""
        if not _SLAM_AVAILABLE or self.slam_map is None:
            return
        path = os.path.join(run_folder, f"episode_{episode:04d}_reward_{reward:.0f}.png")
        self.slam_map.save_plot(path, goal=self._goal)

    # ── LiDAR helpers ────────────────────────────────────────────────────────

    def sector_lidar(self, raw_ranges: np.ndarray) -> np.ndarray:
        """16-sector min-range, normalised to [0, 1]. Fully vectorised."""
        valid = np.where((raw_ranges > 0.01) & np.isfinite(raw_ranges), raw_ranges, self._lidar_max_range)
        remainder = len(valid) % self.N_SECTORS
        if remainder:
            valid = np.concatenate(
                [valid, np.full(self.N_SECTORS - remainder, self._lidar_max_range, dtype=np.float32)]
            )
        sectors = valid.reshape(self.N_SECTORS, -1).min(axis=1)
        return np.clip(sectors / self._lidar_max_range, 0.0, 1.0).astype(np.float32)

    def _scan_to_world(self, raw_ranges: np.ndarray, pos: np.ndarray, heading: float) -> np.ndarray:
        """Project valid LiDAR returns to 2-D world-frame points."""
        n = len(raw_ranges)
        if self._lidar_angles is None or len(self._lidar_angles) != n:
            self._lidar_angles = np.linspace(
                -self._lidar_fov / 2.0, self._lidar_fov / 2.0, n, dtype=np.float32
            )
        mask = (raw_ranges > 0.01) & np.isfinite(raw_ranges)
        r = raw_ranges[mask]
        a = self._lidar_angles[mask]
        c, s = float(np.cos(heading)), float(np.sin(heading))
        xl = r * np.cos(a)
        yl = r * np.sin(a)
        xw = c * xl - s * yl + pos[0]
        yw = s * xl + c * yl + pos[1]
        return np.column_stack([xw, yw]).astype(np.float32)

    # ── Main per-step call ───────────────────────────────────────────────────

    def process(
        self,
        raw_ranges: np.ndarray,
        accel: np.ndarray,
        gyro: np.ndarray,
        cmd_speed_rads: float,
        gps_pos: np.ndarray = None,
    ) -> Tuple[np.ndarray, Any]:
        """Run one sensor-processing step.

        Returns:
            lidar_sectors: (16,) normalised sector min-ranges
            imu_state:     filtered IMUState (or None if SLAM unavailable)
        """
        lidar_sectors = self.sector_lidar(raw_ranges)

        if not _SLAM_AVAILABLE:
            return lidar_sectors, None

        # ── 1. IMU filter ────────────────────────────────────────────────────
        imu_state = self.imu_proc.step(gyro, accel)

        # ── 2. IEKF odometry propagation (heading only; GPS used for position) ─
        cmd_speed_ms = cmd_speed_rads * self.WHEEL_RADIUS
        self.iekf.propagate_odom(cmd_speed_ms, float(gyro[2]), self._dt)

        # ── 3. Map update: use GPS position if available, else IEKF ──────────
        heading = self.iekf.state.heading
        pos = gps_pos if gps_pos is not None else self.iekf.state.position
        pts_2d  = self._scan_to_world(raw_ranges, pos, heading)
        self.slam_map.try_add_keyframe(
            float(pos[0]), float(pos[1]), heading, scan_points=pts_2d
        )

        return lidar_sectors, imu_state


class AltinoDriver:
    """High-level robot control interface."""

    def __init__(self, config: Config):
        """Initialize robot with config."""
        global _supervisor
        assert _supervisor is not None, "Supervisor not initialized. Call _init_supervisor() first."
        self.supervisor = _supervisor
        self.config = config
        self.timestep = int(self.supervisor.getBasicTimeStep())  # type: ignore[union-attr]
        self._dt: float = self.timestep / 1000.0

        # Hardware components
        self.motors = MotorController(self.supervisor)  # type: ignore[arg-type]
        self.sensors = SensorReader(
            self.supervisor,
            self.timestep,
            config.collision_threshold,
        )  # type: ignore[arg-type]

        # SLAM sensor-processing pipeline
        self.slam = SLAMProcessor(
            lidar_max_range=self.sensors.lidar_max_range,
            lidar_fov=self.sensors.lidar.getFov(),
            dt=self._dt,
            goal=tuple(config.endpoint),
        )
        # Commanded wheel speed (rad/s) – updated by set_speed(); used for IEKF odometry
        self._cmd_speed_rads: float = 0.0

        # Position reset
        try:
            self.altino_node = self.supervisor.getFromDef('ALTINO')  # type: ignore[union-attr]
            self.translation_field = self.altino_node.getField('translation')
            self.rotation_field = self.altino_node.getField('rotation')
        except Exception as e:
            print(f"[PPO] ERROR: Failed to get ALTINO node: {e}")
            self.altino_node = None
            self.translation_field = None
            self.rotation_field = None
    
    def set_steering(self, angle: float) -> None:
        """Set steering angle."""
        self.motors.set_steering(angle)

    def set_speed(self, speed: float) -> None:
        """Set velocity and record it for IEKF wheel-odometry propagation."""
        self.motors.set_speed(speed)
        self._cmd_speed_rads = speed

    def get_device(self, name: str):
        """Get a named device from supervisor."""
        return self.supervisor.getDevice(name)  # type: ignore[union-attr]

    def step(self, timestep: int) -> int:
        """Step simulation."""
        return self.supervisor.step(timestep)  # type: ignore[union-attr]

    def _get_heading(self) -> float:
        """Estimate robot yaw from the Webots rotation field (axis-angle → yaw)."""
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
        """Seed SLAM filters from GPS + Webots heading after physics settle."""
        gps_vals = self.sensors.gps.getValues()
        init_pos = np.array([gps_vals[0], gps_vals[1]], dtype=np.float32)
        self.slam.reset(init_pos, self._get_heading())
        self._cmd_speed_rads = 0.0

    def read_sensors(self) -> Tuple[np.ndarray, np.ndarray, float, Any, bool]:
        """Read and process all sensor data for one timestep.

        Returns:
            lidar_sectors: (16,) normalised sector min-ranges
            position:      GPS [x, y] in world frame
            heading:       Yaw in radians (IEKF if available, else Webots field)
            imu_state:     Filtered IMU state (quaternion, accel_body, gyro_body)
            collision:     True when an obstacle is within collision_threshold
        """
        raw_ranges, pos, accel, gyro, collision = self.sensors.read_observation()

        lidar_sectors, imu_state = self.slam.process(
            raw_ranges, accel, gyro, self._cmd_speed_rads, gps_pos=pos
        )

        heading = (self.slam.iekf.state.heading
                   if (_SLAM_AVAILABLE and self.slam.iekf is not None)
                   else self._get_heading())

        return lidar_sectors, pos, heading, imu_state, collision
    
    def reset_position(self) -> None:
        """Reset robot to start position and orientation."""
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

            self.translation_field.setSFVec3f(start_position.tolist())  # type: ignore[arg-type]
            self.rotation_field.setSFRotation(start_rotation)  # type: ignore[arg-type]
            self.supervisor.simulationResetPhysics()  # type: ignore[union-attr]
        else:
            print("[PPO] WARNING: Cannot reset - ALTINO node not accessible!")



# ============================================================================
# ENVIRONMENT
# ============================================================================

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
        """Initialize reward computer.
        
        Args:
            endpoint: Goal position [x, y]
            collision_reward: Penalty for collision
        """
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
        """Compute reward for current state.
        
        Args:
            collision: Whether collision occurred
            current_pos: Current [x, y] position
            current_step: Current step in episode
            prev_distance: Distance from previous step (None on first step)
            goal_error: Heading error to goal
            min_lidar_norm: Minimum normalized LiDAR distance
            speed_norm: Normalized current speed
            reached_new_best_distance: Whether reached new closest distance
            accel: Accelerometer [x, y, z] values
        
        Returns:
            reward: Scalar reward
            new_distance: Distance to endpoint (for next step)
        """
        if collision:
            return self.collision_reward, None
        
        distance_to_end = float(np.linalg.norm(current_pos - self.endpoint))
        
        if distance_to_end < self.goal_threshold:
            if speed_norm <= 0.1:
                return 200.0 + self.goal_stop_bonus + self.goal_hold_reward, distance_to_end
            return self.goal_speed_penalty * speed_norm + self.goal_hold_reward, distance_to_end

        # Progress reward
        progress = 0.0
        if prev_distance is not None:
            delta = float(prev_distance - distance_to_end)
            progress = delta * self.progress_scale if delta >= 0.0 else delta * (0.25 * self.progress_scale)

        # Dense goal-proximity reward, normalized against the start distance.
        proximity = max(0.0, (self.reference_distance - distance_to_end) / max(self.reference_distance, 1e-6))
        proximity_reward = proximity * self.distance_reward_scale
        heading_alignment = max(0.0, float(np.cos(goal_error)))
        heading_reward = heading_alignment * self.heading_reward_scale
        safety_reward = float(np.clip(min_lidar_norm, 0.0, 1.0)) * self.safety_reward_scale
        motion_reward = float(np.clip(speed_norm, 0.0, 1.0)) * self.motion_reward_scale
        new_best_bonus = self.new_best_distance_bonus if reached_new_best_distance else 0.0

        # Penalty for high acceleration when approaching goal
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
           # + approach_speed_penalty
            + accel_penalty
            + self.step_penalty
        ), distance_to_end


class WebotsEnv:
    """Webots simulation environment for ALTINO obstacle avoidance."""
    
    def __init__(self, config: Config):
        """Initialize environment with config.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.robot = AltinoDriver(config)
        self.timestep = self.robot.timestep
        
        # Lights
        self.headlights = self.robot.get_device("headlights")
        self.backlights = self.robot.get_device("backlights")
        
        # Reward computation
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
        
        # Episode state
        self.current_step = 0
        self.episode_reward = 0.0
        self.current_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.current_heading = 0.0
        self.current_distance = float("inf")
        self.min_episode_distance = float("inf")
        self.collision = False
        self.prev_distance: Optional[float] = None
        self.was_in_goal: bool = False

        # Per-run output folder for SLAM maps (DRL-Obstacle-Avoidance/plots/<timestamp>/)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        _repo_root = Path(__file__).parent.parent.parent
        self.run_folder = str(_repo_root / "plots" / ts)
        os.makedirs(self.run_folder, exist_ok=True)
        self._episode_count = 0
        print(f"[PPO] SLAM maps will be saved to: {self.run_folder}", flush=True)
    
    def _reset_episode_state(self) -> None:
        """Reset internal episode state."""
        self.current_step = 0
        self.prev_distance = None
        self.episode_reward = 0.0
        self.current_heading = 0.0
        self.current_distance = float("inf")
        self.min_episode_distance = float("inf")
        self.collision = False
        self.was_in_goal = False

    def _goal_geometry(self, pos: np.ndarray, heading: float) -> Tuple[float, float]:
        """Compute goal distance and heading error."""
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
        """Build a 33-D observation vector.

        Layout:
          [0:16]  lidar_sectors – 16 angular-sector min-ranges, normalised [0,1]
          [16:18] GPS position  – world-frame [x, y]
          [18:23] direction     – sin/cos heading, sin/cos goal_error, norm dist
          [23:26] accel_body    – filtered accelerometer (÷10, clipped)
          [26:29] gyro_body     – filtered gyroscope (÷π, clipped)
          [29:33] quaternion    – orientation [w, x, y, z]
        """
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
            gyro_norm  = np.clip(imu_state.gyro_body,  -np.pi, np.pi) / np.pi
            quat       = imu_state.quaternion.astype(np.float32)
        else:
            accel_norm = np.zeros(3, dtype=np.float32)
            gyro_norm  = np.zeros(3, dtype=np.float32)
            quat       = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        observation = np.concatenate([
            lidar_sectors,      # (16,)
            pos,                # (2,)
            direction_features, # (5,)
            accel_norm,         # (3,)
            gyro_norm,          # (3,)
            quat,               # (4,)
        ]).astype(np.float32)  # total 33
        return np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)
    
    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and return initial observation.

        Returns:
            observation: Initial SLAM-processed sensor observation
            info:        Empty info dict
        """
        # Clear the map for the new episode (saving is done in train() on new records only)
        self.robot.slam.reset_map()
        self._episode_count += 1

        # Stop motors before reset
        self.robot.motors.stop()

        # Reset robot position
        self.robot.reset_position()
        self._reset_episode_state()

        # Settle physics
        for _ in range(self.config.reset_settle_steps):
            self.robot.step(self.timestep)

        # Re-seed SLAM filters from ground-truth pose after settling
        self.robot.reset_slam()

        # Get initial observation
        lidar_sectors, pos, heading, imu_state, collision = self.robot.read_sensors()
        self.current_pos = pos
        self.current_heading = heading
        self.collision = collision
        self.current_distance, _ = self._goal_geometry(pos, heading)
        self.min_episode_distance = self.current_distance

        observation = self._build_observation(lidar_sectors, pos, heading, imu_state)
        return observation, {}
    
    def step(self, action: Union[np.ndarray, List[float], Tuple[float, float]]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step environment.
        
        Args:
            action: Continuous action [steering, speed]
        
        Returns:
            observation: Current observation
            reward: Step reward
            terminated: Whether episode ended
            truncated: Whether max steps reached
            info: Info dict with metadata
        """
        # Apply action
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_arr.size != 2:
            raise ValueError(f"Expected action with 2 elements [steering, speed], got shape {action_arr.shape}")
        steering = float(np.clip(action_arr[0], -self.config.max_steering_angle, self.config.max_steering_angle))
        speed = float(np.clip(action_arr[1], self.config.min_speed, self.config.max_speed))
        self.robot.set_steering(steering)
        self.robot.set_speed(speed)  # also updates _cmd_speed_rads for IEKF
        self.robot.step(self.timestep)
        self.current_step += 1

        # Sense (SLAM-processed)
        lidar_sectors, pos, heading, imu_state, collision = self.robot.read_sensors()
        self.current_pos = pos
        self.current_heading = heading
        self.collision = collision
        self.current_distance, goal_error = self._goal_geometry(pos, heading)
        reached_new_best_distance = self.current_distance + 1e-6 < self.min_episode_distance
        if reached_new_best_distance:
            self.min_episode_distance = self.current_distance

        # Sectors are already normalised [0,1]; min = closest obstacle reading
        min_lidar_norm = float(lidar_sectors.min())
        speed_norm = float(speed / max(self.config.max_speed, 1e-6))
        goal_reached = self.current_distance < self.config.goal_threshold

        accel_for_reward = (imu_state.accel_body
                            if (_SLAM_AVAILABLE and imu_state is not None)
                            else np.zeros(3, dtype=np.float32))

        # Termination bookkeeping
        terminated = False
        truncated = self.current_step >= self.config.max_steps
        info: Dict[str, Any] = {}

        # Compute reward
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

        # Overshoot penalty
        if self.was_in_goal and not goal_reached and speed_norm > 0.05:
            reward += self.config.goal_overshoot_penalty
            self.episode_reward += self.config.goal_overshoot_penalty

        # Goal stop success
        goal_stopped = goal_reached and speed_norm <= 0.1
        if goal_stopped:
            terminated = True
            info["reset_reason"] = "goal"
        elif goal_reached:
            penalty = self.config.goal_speed_penalty * speed_norm
            reward += penalty
            self.episode_reward += penalty

        # Secondary goal stop condition based on accumulated score.
        if self.episode_reward >= self.config.goal_score_threshold:
            terminated = True
            info["reset_reason"] = "goal"

        self.was_in_goal = goal_reached

        # Collision should take precedence over low-score bookkeeping.
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


# ============================================================================
# PPO AGENT
# ============================================================================

class SLAMActorCritic(nn.Module):
    """Actor-critic network with lightweight SLAM feature branches and an LSTM core."""

    def __init__(
        self,
        obs_size: int,
        action_dim: int,
        config: Config,
        action_space: str = "continuous",
    ) -> None:
        super().__init__()
        self.obs_size = obs_size
        self.action_dim = action_dim
        self.action_space = action_space
        self.grid_shape = config.occupancy_grid_shape
        self.obstacle_dim = config.lidar_sector_dim
        self.pose_goal_dim = config.pose_goal_dim
        self.imu_dim = config.imu_feature_dim
        self.structured_obs_dim = self.obstacle_dim + self.pose_goal_dim + self.imu_dim
        if obs_size < self.structured_obs_dim:
            raise ValueError(
                f"Observation size {obs_size} is smaller than the structured feature layout "
                f"({self.structured_obs_dim})."
            )

        branch_latent_dim = max(config.latent_size // 2, 32)
        self.obstacle_encoder = nn.Sequential(
            nn.Linear(self.obstacle_dim, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, branch_latent_dim),
            nn.ReLU(),
        )
        self.pose_goal_encoder = nn.Sequential(
            nn.Linear(self.pose_goal_dim, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, branch_latent_dim),
            nn.ReLU(),
        )
        self.imu_encoder = nn.Sequential(
            nn.Linear(self.imu_dim, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, branch_latent_dim),
            nn.ReLU(),
        )

        grid_latent_dim = config.latent_size
        self.grid_encoder_cnn: Optional[nn.Module]
        self.grid_encoder_mlp: Optional[nn.Module]
        self.grid_feature_dim = max(obs_size - self.structured_obs_dim, 0)
        if self.grid_shape is not None:
            if len(self.grid_shape) == 2:
                in_channels = 1
            elif len(self.grid_shape) == 3:
                in_channels = self.grid_shape[0]
            else:
                raise ValueError(
                    "occupancy_grid_shape must be (H, W) or (C, H, W) when using CNN encoding."
                )
            self.grid_encoder_cnn = nn.Sequential(
                nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(32, grid_latent_dim),
                nn.ReLU(),
            )
            self.grid_encoder_mlp = None
        elif self.grid_feature_dim > 0:
            self.grid_encoder_cnn = None
            self.grid_encoder_mlp = nn.Sequential(
                nn.Linear(self.grid_feature_dim, config.hidden_size),
                nn.ReLU(),
                nn.Linear(config.hidden_size, grid_latent_dim),
                nn.ReLU(),
            )
        else:
            self.grid_encoder_cnn = None
            self.grid_encoder_mlp = None
            grid_latent_dim = 0

        fusion_input_dim = 3 * branch_latent_dim + grid_latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(fusion_input_dim, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.latent_size),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=config.latent_size,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_layers,
            batch_first=True,
        )
        self.policy_head = nn.Linear(config.lstm_hidden_size, action_dim)
        self.value_head = nn.Linear(config.lstm_hidden_size, 1)

    def get_initial_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return zero-initialized hidden and cell state."""
        if device is None:
            device = next(self.parameters()).device
        h0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=device)
        c0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=device)
        return h0, c0

    def _infer_batch_time(
        self,
        tensor: torch.Tensor,
        recurrent_state: Optional[Tuple[torch.Tensor, torch.Tensor]],
        done_mask: Optional[torch.Tensor],
    ) -> Tuple[int, int]:
        """Infer batch/time layout for 1D, 2D, and 3D+ observations."""
        if tensor.ndim == 1:
            return 1, 1
        if tensor.ndim == 2:
            if recurrent_state is not None and recurrent_state[0].shape[1] == tensor.shape[0]:
                return tensor.shape[0], 1
            if done_mask is not None and done_mask.ndim == 1 and done_mask.shape[0] == tensor.shape[0]:
                return 1, tensor.shape[0]
            return tensor.shape[0], 1
        return tensor.shape[0], tensor.shape[1]

    def _prepare_tensor_component(
        self,
        component: Union[np.ndarray, torch.Tensor],
        recurrent_state: Optional[Tuple[torch.Tensor, torch.Tensor]],
        done_mask: Optional[torch.Tensor],
        trailing_dims: int,
    ) -> torch.Tensor:
        """Convert an observation component to [B, T, ...]."""
        tensor = torch.as_tensor(component, dtype=torch.float32, device=next(self.parameters()).device)
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)
        batch_size, seq_len = self._infer_batch_time(tensor, recurrent_state, done_mask)
        if tensor.ndim == trailing_dims:
            return tensor.reshape(batch_size, seq_len, *tensor.shape[-trailing_dims:])
        if tensor.ndim == trailing_dims + 1:
            return tensor.reshape(batch_size, seq_len, *tensor.shape[-trailing_dims:])
        return tensor

    def _split_observation(
        self,
        observation: Union[np.ndarray, torch.Tensor, Dict[str, Union[np.ndarray, torch.Tensor]]],
        recurrent_state: Optional[Tuple[torch.Tensor, torch.Tensor]],
        done_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], int, int]:
        """Split observation into semantic SLAM feature groups."""
        if isinstance(observation, dict):
            obstacle_input = observation.get("obstacle_features", observation.get("lidar_sectors"))
            pose_goal_input = observation.get("pose_goal_features", observation.get("pose_goal"))
            imu_input = observation.get("imu_features", observation.get("imu"))
            if obstacle_input is None or pose_goal_input is None or imu_input is None:
                raise KeyError(
                    "Observation dict must contain obstacle/lidar, pose_goal, and imu feature groups."
                )
            grid_input = observation.get("grid", observation.get("occupancy_grid", observation.get("feature_map")))
            obstacle = self._prepare_tensor_component(
                obstacle_input, recurrent_state, done_mask, trailing_dims=1
            )
            pose_goal = self._prepare_tensor_component(
                pose_goal_input, recurrent_state, done_mask, trailing_dims=1
            )
            imu = self._prepare_tensor_component(
                imu_input, recurrent_state, done_mask, trailing_dims=1
            )
            batch_size, seq_len = obstacle.shape[:2]
            grid = None
            if grid_input is not None:
                grid_tensor = torch.as_tensor(grid_input, dtype=torch.float32, device=obstacle.device)
                grid_tensor = torch.nan_to_num(grid_tensor, nan=0.0, posinf=1.0, neginf=-1.0)
                if grid_tensor.ndim <= 3:
                    grid = self._prepare_tensor_component(grid_tensor, recurrent_state, done_mask, trailing_dims=1)
                elif grid_tensor.ndim == 4:
                    if self.grid_shape is not None and len(self.grid_shape) == 3:
                        grid = grid_tensor.reshape(batch_size, seq_len, *grid_tensor.shape[-3:])
                    else:
                        grid = grid_tensor.reshape(batch_size, seq_len, *grid_tensor.shape[-2:])
                else:
                    grid = grid_tensor.reshape(batch_size, seq_len, *grid_tensor.shape[-3:])
            return obstacle, pose_goal, imu, grid, batch_size, seq_len

        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=next(self.parameters()).device)
        obs_tensor = torch.nan_to_num(obs_tensor, nan=0.0, posinf=1.0, neginf=-1.0)
        batch_size, seq_len = self._infer_batch_time(obs_tensor, recurrent_state, done_mask)
        obs_tensor = obs_tensor.reshape(batch_size, seq_len, -1)
        obstacle_end = self.obstacle_dim
        pose_goal_end = obstacle_end + self.pose_goal_dim
        imu_end = pose_goal_end + self.imu_dim
        obstacle = obs_tensor[..., :obstacle_end]
        pose_goal = obs_tensor[..., obstacle_end:pose_goal_end]
        imu = obs_tensor[..., pose_goal_end:imu_end]
        grid = obs_tensor[..., imu_end:] if self.grid_feature_dim > 0 else None
        return obstacle, pose_goal, imu, grid, batch_size, seq_len

    def _encode_observation(
        self,
        observation: Union[np.ndarray, torch.Tensor, Dict[str, Union[np.ndarray, torch.Tensor]]],
        recurrent_state: Optional[Tuple[torch.Tensor, torch.Tensor]],
        done_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, int, int]:
        """Encode lightweight SLAM feature groups to latent features [B, T, D]."""
        obstacle, pose_goal, imu, grid, batch_size, seq_len = self._split_observation(
            observation,
            recurrent_state,
            done_mask,
        )
        obstacle_latent = self.obstacle_encoder(obstacle.reshape(batch_size * seq_len, -1))
        pose_goal_latent = self.pose_goal_encoder(pose_goal.reshape(batch_size * seq_len, -1))
        imu_latent = self.imu_encoder(imu.reshape(batch_size * seq_len, -1))
        encoded_parts = [obstacle_latent, pose_goal_latent, imu_latent]

        if grid is not None:
            if self.grid_encoder_cnn is not None:
                grid_flat = grid.reshape(batch_size * seq_len, *grid.shape[2:])
                if grid_flat.ndim == 3:
                    grid_flat = grid_flat.unsqueeze(1)
                grid_latent = self.grid_encoder_cnn(grid_flat)
            elif self.grid_encoder_mlp is not None:
                grid_latent = self.grid_encoder_mlp(grid.reshape(batch_size * seq_len, -1))
            else:
                grid_latent = grid.reshape(batch_size * seq_len, -1)
            encoded_parts.append(grid_latent)

        latent = torch.cat(encoded_parts, dim=-1)
        latent = self.encoder(latent).reshape(batch_size, seq_len, -1)
        return latent, batch_size, seq_len

    def _prepare_done_mask(
        self,
        done_mask: Optional[Union[np.ndarray, torch.Tensor]],
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Format done flags to [B, T] float mask."""
        if done_mask is None:
            return None
        mask = torch.as_tensor(done_mask, dtype=torch.float32, device=device)
        if mask.ndim == 0:
            mask = mask.view(1, 1)
        elif mask.ndim == 1:
            mask = mask.view(batch_size, seq_len)
        elif mask.ndim != 2:
            raise ValueError(f"done_mask must be scalar, [T], or [B, T], got shape {tuple(mask.shape)}")
        return mask

    def forward(
        self,
        observation: Union[np.ndarray, torch.Tensor, Dict[str, Union[np.ndarray, torch.Tensor]]],
        recurrent_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        done_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Run the encoder, recurrent core, and policy/value heads.

        Supports:
          - single-step inference with flat observations [D] or batched [B, D]
          - sequence training with [T, D] or [B, T, D]
          - dict observations with separate pose and grid inputs
        """
        latent, batch_size, seq_len = self._encode_observation(observation, recurrent_state, done_mask)
        device = latent.device
        if recurrent_state is None:
            recurrent_state = self.get_initial_state(batch_size, device=device)
        h_t, c_t = recurrent_state
        mask = self._prepare_done_mask(done_mask, batch_size, seq_len, device)

        outputs: List[torch.Tensor] = []
        for t in range(seq_len):
            if mask is not None:
                keep = (1.0 - mask[:, t]).view(1, batch_size, 1)
                h_t = h_t * keep
                c_t = c_t * keep
            lstm_out, (h_t, c_t) = self.lstm(latent[:, t : t + 1], (h_t, c_t))
            outputs.append(lstm_out)

        recurrent_features = torch.cat(outputs, dim=1)
        policy_output = torch.nan_to_num(self.policy_head(recurrent_features), nan=0.0, posinf=1.0, neginf=-1.0)
        state_value = torch.nan_to_num(
            self.value_head(recurrent_features).squeeze(-1),
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )

        if seq_len == 1:
            policy_output = policy_output[:, 0]
            state_value = state_value[:, 0]
        return policy_output, state_value, (h_t, c_t)


class PPOAgent:
    """Proximal Policy Optimization agent with a recurrent actor-critic core."""

    def __init__(self, obs_size: int, action_dim: int, config: Config):
        self.config = config
        self.device = self._get_device()
        self.action_dim = action_dim
        self.model = SLAMActorCritic(obs_size, action_dim, config).to(self.device)
        self.actor = self.model.policy_head
        self.critic = self.model.value_head
        self.actor_log_std = nn.Parameter(torch.full((action_dim,), -0.5, dtype=torch.float32, device=self.device))
        params = list(self.model.parameters()) + [self.actor_log_std]
        self.optimizer = torch.optim.Adam(params, lr=config.learning_rate)

    def _get_device(self) -> torch.device:
        """Get appropriate device (GPU or CPU)."""
        is_fork = multiprocessing.get_start_method(allow_none=True) == 'fork'
        if torch.cuda.is_available() and not is_fork:
            return torch.device("cuda:0")
        return torch.device("cpu")

    def get_initial_state(self, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Expose recurrent state initialization for rollouts."""
        return self.model.get_initial_state(batch_size, device=self.device)

    def _action_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get per-dimension action bounds."""
        low = torch.tensor(
            [-self.config.max_steering_angle, self.config.min_speed],
            dtype=torch.float32,
            device=self.device,
        )
        high = torch.tensor(
            [self.config.max_steering_angle, self.config.max_speed],
            dtype=torch.float32,
            device=self.device,
        )
        return low, high

    def _policy_stats(self, policy_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return stable mean, std, and base-distribution entropy."""
        policy_output = torch.nan_to_num(policy_output, nan=0.0, posinf=1.0, neginf=-1.0)
        safe_actor_log_std = torch.nan_to_num(self.actor_log_std, nan=-0.5, posinf=2.0, neginf=-5.0)
        safe_actor_log_std = safe_actor_log_std.clamp(-5.0, 2.0)
        log_std = safe_actor_log_std.expand_as(policy_output)
        std = torch.nan_to_num(log_std.exp(), nan=1.0, posinf=7.5, neginf=1e-3).clamp_min(1e-3)
        base_dist = Independent(Normal(policy_output, std), 1)
        entropy = torch.nan_to_num(base_dist.entropy(), nan=0.0, posinf=0.0, neginf=0.0)
        return policy_output, std, entropy

    def _squash_action(self, pre_tanh_action: torch.Tensor) -> torch.Tensor:
        """Map unconstrained actions to environment action bounds."""
        low, high = self._action_bounds()
        center = (high + low) / 2.0
        scale = (high - low) / 2.0
        squashed = torch.tanh(pre_tanh_action)
        action = squashed * scale + center
        eps = 1e-5
        return torch.max(torch.min(action, high - eps), low + eps)

    def _unsquash_action(self, action: torch.Tensor) -> torch.Tensor:
        """Map bounded actions back to pre-tanh space for stable log-prob evaluation."""
        low, high = self._action_bounds()
        center = (high + low) / 2.0
        scale = (high - low) / 2.0
        normalized = (action - center) / scale.clamp_min(1e-6)
        normalized = normalized.clamp(-1.0 + 1e-5, 1.0 - 1e-5)
        return 0.5 * (torch.log1p(normalized) - torch.log1p(-normalized))

    def _log_prob_from_action(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log-prob under a squashed Gaussian policy."""
        pre_tanh_action = self._unsquash_action(action)
        base_dist = Independent(Normal(mean, std), 1)
        base_log_prob = base_dist.log_prob(pre_tanh_action)
        correction = torch.log(1.0 - torch.tanh(pre_tanh_action).pow(2) + 1e-6).sum(dim=-1)
        log_prob = base_log_prob - correction
        return torch.nan_to_num(log_prob, nan=0.0, posinf=0.0, neginf=0.0)

    def select_action(
        self,
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        recurrent_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        done: bool = False,
    ) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Sample a single action and advance the recurrent state."""
        done_mask = torch.tensor([float(done)], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            policy_output, state_value, next_state = self.model(
                obs,
                recurrent_state=recurrent_state,
                done_mask=done_mask,
            )
            mean, std, _ = self._policy_stats(policy_output)
            noise = torch.randn_like(mean)
            pre_tanh_action = mean + std * noise
            action = self._squash_action(pre_tanh_action)
            log_prob = self._log_prob_from_action(mean, std, action)
        return (
            action.squeeze(0).cpu().numpy(),
            log_prob.squeeze(0),
            state_value.squeeze(0),
            next_state,
        )

    def calculate_returns(self, rewards: np.ndarray) -> np.ndarray:
        """Calculate cumulative discounted returns."""
        returns = np.zeros(len(rewards), dtype=np.float32)
        discounted_return = 0.0
        for t in reversed(range(len(rewards))):
            discounted_return = rewards[t] + self.config.gamma * discounted_return
            returns[t] = discounted_return
        return returns

    def _prepare_batch(
        self,
        trajectories: List[Dict[str, np.ndarray]],
    ) -> Dict[str, torch.Tensor]:
        """Pad variable-length episode trajectories to a batch of sequences."""
        obs_tensors = [torch.as_tensor(t["observations"], dtype=torch.float32, device=self.device) for t in trajectories]
        act_tensors = [torch.as_tensor(t["actions"], dtype=torch.float32, device=self.device) for t in trajectories]
        logp_tensors = [torch.as_tensor(t["log_probs"], dtype=torch.float32, device=self.device) for t in trajectories]
        ret_tensors = [torch.as_tensor(t["returns"], dtype=torch.float32, device=self.device) for t in trajectories]
        adv_tensors = [torch.as_tensor(t["advantages"], dtype=torch.float32, device=self.device) for t in trajectories]
        obs_tensors = [torch.nan_to_num(t, nan=0.0, posinf=1.0, neginf=-1.0) for t in obs_tensors]
        act_tensors = [torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0) for t in act_tensors]
        logp_tensors = [torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0) for t in logp_tensors]
        ret_tensors = [torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0) for t in ret_tensors]
        adv_tensors = [torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0) for t in adv_tensors]
        valid_masks = [
            torch.ones(len(t["returns"]), dtype=torch.float32, device=self.device)
            for t in trajectories
        ]
        reset_masks = []
        for trajectory in trajectories:
            mask = torch.zeros(len(trajectory["returns"]), dtype=torch.float32, device=self.device)
            if len(mask) > 0:
                mask[0] = 1.0
            reset_masks.append(mask)

        return {
            "observations": pad_sequence(obs_tensors, batch_first=True),
            "actions": pad_sequence(act_tensors, batch_first=True),
            "log_probs": pad_sequence(logp_tensors, batch_first=True),
            "returns": pad_sequence(ret_tensors, batch_first=True),
            "advantages": pad_sequence(adv_tensors, batch_first=True),
            "valid_mask": pad_sequence(valid_masks, batch_first=True),
            "done_mask": pad_sequence(reset_masks, batch_first=True),
        }

    def evaluate_sequences(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        done_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate PPO quantities over full recurrent sequences."""
        policy_output, state_values, _ = self.model(
            observations,
            recurrent_state=self.get_initial_state(observations.shape[0]),
            done_mask=done_mask,
        )
        mean, std, entropy = self._policy_stats(policy_output)
        log_probs = self._log_prob_from_action(mean, std, actions)
        return log_probs, state_values, entropy

    def update(self, trajectories: List[Dict[str, np.ndarray]]) -> None:
        """Update the recurrent PPO policy from complete episodes."""
        if not trajectories:
            return

        for trajectory in trajectories:
            trajectory["observations"] = np.nan_to_num(
                trajectory["observations"], nan=0.0, posinf=1.0, neginf=-1.0
            ).astype(np.float32)
            trajectory["actions"] = np.nan_to_num(
                trajectory["actions"], nan=0.0, posinf=0.0, neginf=0.0
            ).astype(np.float32)
            trajectory["log_probs"] = np.nan_to_num(
                trajectory["log_probs"], nan=0.0, posinf=0.0, neginf=0.0
            ).astype(np.float32)
            trajectory["returns"] = np.nan_to_num(
                trajectory["returns"], nan=0.0, posinf=0.0, neginf=0.0
            ).astype(np.float32)
            trajectory["advantages"] = np.nan_to_num(
                trajectory["advantages"], nan=0.0, posinf=0.0, neginf=0.0
            ).astype(np.float32)

        all_advantages = np.concatenate([t["advantages"] for t in trajectories], axis=0)
        adv_mean = float(all_advantages.mean())
        adv_std = float(all_advantages.std() + 1e-8)
        for trajectory in trajectories:
            trajectory["advantages"] = ((trajectory["advantages"] - adv_mean) / adv_std).astype(np.float32)

        num_episodes = len(trajectories)
        for _ in range(self.config.epochs):
            indices = torch.randperm(num_episodes).tolist()
            for start in range(0, num_episodes, self.config.batch_size):
                batch_indices = indices[start : start + self.config.batch_size]
                batch = self._prepare_batch([trajectories[idx] for idx in batch_indices])

                log_probs_new, values, entropy = self.evaluate_sequences(
                    batch["observations"],
                    batch["actions"],
                    batch["done_mask"],
                )

                valid_mask = batch["valid_mask"]
                mask_bool = valid_mask > 0
                log_ratio = torch.nan_to_num(
                    log_probs_new - batch["log_probs"],
                    nan=0.0,
                    posinf=20.0,
                    neginf=-20.0,
                ).clamp(-20.0, 20.0)
                ratio = torch.exp(log_ratio)
                surrogate1 = ratio * batch["advantages"]
                surrogate2 = (
                    torch.clamp(ratio, 1 - self.config.epsilon, 1 + self.config.epsilon)
                    * batch["advantages"]
                )
                surrogate = torch.min(surrogate1, surrogate2)
                surrogate = torch.where(mask_bool, surrogate, torch.zeros_like(surrogate))

                value_error = (values - batch["returns"]) ** 2
                value_error = torch.where(mask_bool, value_error, torch.zeros_like(value_error))

                entropy = torch.where(mask_bool, entropy, torch.zeros_like(entropy))
                valid_count = valid_mask.sum().clamp_min(1.0)
                policy_loss = -surrogate.sum() / valid_count
                value_loss = value_error.sum() / valid_count
                entropy_bonus = entropy.sum() / valid_count

                loss = policy_loss + 0.5 * value_loss - self.config.entropy_coef * entropy_bonus
                if not torch.isfinite(loss):
                    print("[PPO] WARNING: Skipping update batch due to non-finite loss.", flush=True)
                    continue
                self.optimizer.zero_grad()
                loss.backward()
                gradients_finite = True
                for parameter in list(self.model.parameters()) + [self.actor_log_std]:
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        gradients_finite = False
                        break
                if not gradients_finite:
                    print("[PPO] WARNING: Skipping update batch due to non-finite gradients.", flush=True)
                    self.optimizer.zero_grad()
                    continue
                nn.utils.clip_grad_norm_(list(self.model.parameters()) + [self.actor_log_std], max_norm=1.0)
                self.optimizer.step()
                with torch.no_grad():
                    self.actor_log_std.data.copy_(
                        torch.nan_to_num(self.actor_log_std.data, nan=-0.5, posinf=2.0, neginf=-5.0).clamp(-5.0, 2.0)
                    )

    def load_model(self, model_path: str) -> None:
        """Load saved recurrent model weights."""
        checkpoint = torch.load(model_path, map_location=self.device)
        if "model" not in checkpoint:
            raise ValueError(
                "Checkpoint does not contain recurrent 'model' weights. "
                "Legacy feed-forward checkpoints are not compatible with this module."
            )
        self.model.load_state_dict(checkpoint["model"])
        if "actor_log_std" in checkpoint:
            self.actor_log_std.data.copy_(checkpoint["actor_log_std"].to(self.device))



# ============================================================================
# TRAINING
# ============================================================================

def train(config: Optional[Config] = None) -> None:
    """Run PPO training loop.
    
    Args:
        config: Configuration object (uses defaults if None)
    """
    if config is None:
        config = Config()
    
    # Initialize Webots supervisor
    _init_supervisor()
    
    # Create environment and agent
    env = WebotsEnv(config)
    obs, _ = env.reset()
    obs_size = len(obs)
    action_dim = 2
    agent = PPOAgent(obs_size, action_dim, config)
    
    print(f"[TRAIN] Starting training: {config.episodes} episodes, "
          f"update every {config.update_every} episodes")
    print(f"[TRAIN] Observation size: {obs_size}, Action dims: {action_dim}")
    
    # Training buffers
    rollout_trajectories: List[Dict[str, np.ndarray]] = []
    best_reward = float('-inf')
    best_goal_reward = float('-inf')
    best_goal_episode: Optional[int] = None
    
    # Training loop
    for episode in range(config.episodes):
        obs, _ = env.reset()
        done = False
        episode_step = 0
        episode_observations: List[np.ndarray] = []
        episode_actions: List[np.ndarray] = []
        episode_log_probs: List[float] = []
        episode_rewards: List[float] = []
        episode_end_reason = "max_steps"
        recurrent_state = agent.get_initial_state(batch_size=1)
        prev_done = True
        
        while not done:
            # Select action from the current policy for every transition so the
            # rollout stays on-policy for PPO updates.
            action, log_prob, _, recurrent_state = agent.select_action(
                obs,
                recurrent_state=recurrent_state,
                done=prev_done,
            )
            
            # Step environment
            obs_next, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            prev_done = done
            episode_step += 1
            
            # Track termination reason
            if done:
                if info.get("reset_reason") == "low_score":
                    episode_end_reason = "low_score"
                elif info.get("reset_reason") == "collision":
                    episode_end_reason = "collision"
                elif info.get("reset_reason") == "goal":
                    episode_end_reason = "goal"
                elif truncated:
                    episode_end_reason = "max_steps"
            
            # Accumulate
            episode_observations.append(obs)
            episode_actions.append(action)
            episode_log_probs.append(float(log_prob.item()))
            episode_rewards.append(reward)
            
            obs = obs_next
        
        # Build returns and advantages per episode so reward signals do not
        # leak across episode boundaries.
        episode_returns = agent.calculate_returns(np.array(episode_rewards, dtype=np.float32))
        episode_obs_array = np.array(episode_observations, dtype=np.float32)
        with torch.no_grad():
            _, episode_values, _ = agent.model(
                episode_obs_array,
                recurrent_state=agent.get_initial_state(batch_size=1),
                done_mask=np.concatenate(([1.0], np.zeros(len(episode_rewards) - 1, dtype=np.float32))),
            )
            episode_values = episode_values.squeeze(0).detach().cpu().numpy()

        episode_advantages = episode_returns - episode_values

        rollout_trajectories.append(
            {
                "observations": episode_obs_array,
                "actions": np.array(episode_actions, dtype=np.float32),
                "log_probs": np.array(episode_log_probs, dtype=np.float32),
                "returns": episode_returns.astype(np.float32),
                "advantages": episode_advantages.astype(np.float32),
            }
        )
        
        # PPO update every N episodes
        if (episode + 1) % config.update_every == 0:
            # Update policy
            agent.update(rollout_trajectories)
            
            # Clear buffers
            rollout_trajectories.clear()
        
        # Logging
        episode_reward_sum = sum(episode_rewards)
        print(
            f"Episode {episode + 1:2d} | "
            f"Reward: {episode_reward_sum:8.2f} | "
            f"Steps: {env.current_step:4d} | "
            f"MinDist: {env.min_episode_distance:6.2f} | "
            f"LastDist: {env.current_distance:6.2f} | "
            f"End: {episode_end_reason}"
        )
        
        if episode_end_reason == "goal":
            if episode_reward_sum > best_goal_reward:
                best_goal_reward = episode_reward_sum
                best_goal_episode = episode + 1
                env.robot.slam.save_episode(env.run_folder, episode + 1, episode_reward_sum)
                torch.save({
                    'model': agent.model.state_dict(),
                    'actor_log_std': agent.actor_log_std.detach().cpu(),
                    'episode': best_goal_episode,
                    'reward': best_goal_reward,
                    'goal_episode': True
                }, 'best_model.pth')
                print(
                    f"[TRAIN] New best goal episode {best_goal_episode} "
                    f"with reward {best_goal_reward:.2f}, model saved."
                )
        elif best_goal_episode is None and episode_reward_sum > best_reward:
            best_reward = episode_reward_sum
            env.robot.slam.save_episode(env.run_folder, episode + 1, episode_reward_sum)
            torch.save({
                'model': agent.model.state_dict(),
                'actor_log_std': agent.actor_log_std.detach().cpu(),
                'episode': episode + 1,
                'reward': best_reward,
                'goal_episode': False
            }, 'best_model.pth')
            print(
                f"[TRAIN] New best episode {episode + 1} with reward {best_reward:.2f} "
                f"(no goal episode yet), model saved."
            )
        
        if episode_end_reason == "goal":
            break

    if rollout_trajectories:
        agent.update(rollout_trajectories)
    
    # Save final model
    torch.save({
        'model': agent.model.state_dict(),
        'actor_log_std': agent.actor_log_std.detach().cpu(),
        'episode': 'final',
        'reward': best_reward
    }, 'final_model.pth')
    print("[TRAIN] Final model saved.")

    # Cleanup
    env.robot.motors.stop()
    print("[TRAIN] Training complete. Robot stopped.")


if __name__ == "__main__":
    train()
