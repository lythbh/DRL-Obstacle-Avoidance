"""
CNN-LiDAR-SLAM  –  Webots controller

Integrates all modules from the paper into a single real-time controller
running on the ALTINO robot in Webots:

    LiDAR  ─► LiDARPreprocessor ─► feature vector ──┐
    IMU    ─► IMUProcessor       ─► filtered state ──┼─► CNN (object landmarks)
                                                     │
    IEKF backend ◄── geometric residuals ◄────────────┘
                 ◄── semantic  residuals ◄─── CNN landmarks
                 ─► SLAMMap (pose graph, occupancy grid)

Pipeline per timestep (Section III-A):
  1. Read sensors (LiDAR range image, InertialUnit, Accelerometer, Gyro)
  2. Preprocess LiDAR: curvature features + object candidates
  3. Filter IMU: Madgwick + EKF
  4. Assemble 26-D feature vector; push to CNN window
  5. CNN forward pass → 4 object centroids + radii (every N steps)
  6. IEKF propagate (IMU) + update (LiDAR geometric + CNN semantic)
  7. Record keyframe; optionally run pose-graph optimisation
  8. Visualise trajectory in Webots display (if available)
"""

from __future__ import annotations

import math
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, List, Optional, Tuple

import numpy as np
import torch

print("[CNN-LiDAR-SLAM] controller loading …", flush=True)

# ── Webots controller API ────────────────────────────────────────────────────
try:
    from controller import Robot, Supervisor, Lidar, InertialUnit, Accelerometer, Gyro, Display, GPS  # pyright: ignore[reportMissingImports]
    print("[CNN-LiDAR-SLAM] Webots controller API imported OK", flush=True)
except ImportError as _e:
    print(f"[CNN-LiDAR-SLAM] WARNING: {_e}", flush=True)
    Robot = Supervisor = InertialUnit = Accelerometer = Gyro = Display = GPS = Lidar = object  # type: ignore

# ── Project modules ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from controllers.SLAM.lidar_preprocessing import LiDARPreprocessor, LiDARFeatures
from controllers.SLAM.imu_filter import IMUProcessor, IMUState
from controllers.SLAM.cnn_model import CNNObjectDetector, build_model
from controllers.SLAM.iekf_backend import IEKFBackend
from controllers.SLAM.slam_map import SLAMMap


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

CNN_SEQ_LEN = 10          # temporal window fed to CNN (frames)
CNN_UPDATE_EVERY = 5      # run CNN every N IEKF steps (efficiency)
MAP_OPTIM_EVERY = 50      # run pose-graph optimisation every N keyframes
DISPLAY_SCALE = 40        # pixels per metre for trajectory display
DISPLAY_ORIGIN = (300, 300)  # pixel origin for the display (centre)

GOAL_POS = np.array([-2.0, 0.0], dtype=np.float32)  # world-frame goal XY
GOAL_RADIUS = 0.3          # metres — within this distance = success
COLLISION_DIST = 0.06      # metres — closer than this = collision

MODEL_CHECKPOINT = Path(__file__).parent / "cnn_checkpoint.pt"


# ─────────────────────────────────────────────────────────────────────────────
# Sensor helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_lidar_angles(lidar, n_points: int) -> np.ndarray:
    """Generate angle array matching the actual range image length."""
    fov = lidar.getFov()
    return np.linspace(-fov / 2.0, fov / 2.0, n_points, dtype=np.float32)


def _read_lidar(lidar) -> np.ndarray:
    """Return raw range array (NaN/Inf → 0)."""
    raw = np.array(lidar.getRangeImage(), dtype=np.float32)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    return raw


def _read_imu(inertial: Any, accel_dev: Any, gyro_dev: Any
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read raw IMU values from Webots devices.

    Returns:
        rpy   – (3,) roll/pitch/yaw from InertialUnit (radians)
        accel – (3,) accelerometer reading (m/s²)
        gyro  – (3,) gyroscope reading (rad/s)
    """
    rpy = np.array(inertial.getRollPitchYaw(), dtype=np.float32)
    accel = np.array(accel_dev.getValues(), dtype=np.float32)
    gyro = np.array(gyro_dev.getValues(), dtype=np.float32)
    return rpy, accel, gyro


def _rpy_to_quat(rpy: np.ndarray) -> np.ndarray:
    """Convert roll-pitch-yaw Euler angles to quaternion [w, x, y, z]."""
    r, p, y = rpy
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Main SLAM controller
# ─────────────────────────────────────────────────────────────────────────────

class CNNLidarSLAMController:
    """
    Webots Robot controller implementing CNN-LiDAR-SLAM.

    Instantiate once and call `run()` to start the control loop.
    """

    def __init__(self) -> None:
        self.robot: Any = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep())
        self.dt = self.timestep / 1000.0  # seconds

        print("[CNN-LiDAR-SLAM] Initialising …", flush=True)

        # ── Sensors ──────────────────────────────────────────────────────────
        self.lidar: Any = self.robot.getDevice("lidar")
        self.lidar.enable(self.timestep)
        # enablePointCloud() is only valid for 3-D LiDARs; skip for 2-D
        self.lidar_max_range = self.lidar.getMaxRange()
        # Number of points per horizontal layer (e.g. 512 for a 4-layer LiDAR
        # that returns 2048 total).  We use only the first layer so that floor
        # returns from angled layers do not falsely trigger obstacle avoidance.
        self._lidar_h_res = self.lidar.getHorizontalResolution()
        print(
            f"[CNN-LiDAR-SLAM] LiDAR: h_res={self._lidar_h_res} "
            f"layers={self.lidar.getNumberOfLayers()} "
            f"max_range={self.lidar_max_range:.1f}m "
            f"fov={math.degrees(self.lidar.getFov()):.0f}deg",
            flush=True,
        )
        self.lidar_angles = None  # built after first step (length = _lidar_h_res)

        self.inertial: Any = self.robot.getDevice("inertial unit")
        self.inertial.enable(self.timestep)

        self.accel_dev: Any = self.robot.getDevice("accelerometer")
        self.accel_dev.enable(self.timestep)

        self.gyro_dev: Any = self.robot.getDevice("gyro")
        self.gyro_dev.enable(self.timestep)

        # Optional GPS for ground-truth comparison / ATE logging
        self._gps: Optional[Any] = None
        gps_dev = self.robot.getDevice("gps")
        if gps_dev is not None:
            self._gps = gps_dev
            gps_dev.enable(self.timestep)

        # Optional display for trajectory visualisation (not required)
        self._display: Optional[Any] = None
        disp_dev = self.robot.getDevice("display")
        if disp_dev is not None:
            self._display = disp_dev

        # ── Motors ───────────────────────────────────────────────────────────
        self._left_steer: Any = self.robot.getDevice("left_steer")
        self._right_steer: Any = self.robot.getDevice("right_steer")
        self._left_steer.setPosition(0.0)
        self._right_steer.setPosition(0.0)
        self._left_steer.setVelocity(1.0)
        self._right_steer.setVelocity(1.0)

        self._wheels: List[Any] = [
            self.robot.getDevice("left_front_wheel"),
            self.robot.getDevice("right_front_wheel"),
            self.robot.getDevice("left_rear_wheel"),
            self.robot.getDevice("right_rear_wheel"),
        ]
        for w in self._wheels:
            w.setPosition(float("inf"))
            w.setVelocity(0.0)

        self._MAX_SPEED = 5.0        # rad/s wheel speed
        self._WHEEL_RADIUS = 0.033   # metres — ALTINO wheel radius
        self._SAFE_DIST = 0.40       # metres — obstacle avoidance threshold
        self._steer_state = 0.0      # current steering angle (smoothed)
        self._stuck_counter = 0      # steps spent blocked
        self._cmd_speed = 0.0        # last commanded forward speed (m/s)

        # ── SLAM modules ─────────────────────────────────────────────────────
        self.lidar_prep = LiDARPreprocessor(
            curvature_threshold=0.05,
            neighbor_half_width=5,
            voxel_size=0.05,
            max_range=self.lidar_max_range,
        )
        self.imu_proc = IMUProcessor(dt=self.dt)
        self.iekf = IEKFBackend(sigma_lidar=0.02, sigma_landmark=0.08)
        self.slam_map = SLAMMap(map_resolution=0.05)

        # ── CNN model ────────────────────────────────────────────────────────
        self.device = torch.device("cpu")  # embedded platform → CPU
        self.cnn: CNNObjectDetector = build_model()
        self.cnn.to(self.device)
        self._load_checkpoint()

        # Temporal feature window for CNN
        self.feature_window: Deque[np.ndarray] = deque(maxlen=CNN_SEQ_LEN)
        self._cnn_landmarks: List[Tuple[float, float, float]] = []
        self._step_count = 0
        self._keyframe_count = 0

        # Trajectory log for metrics
        self.trajectory: List[np.ndarray] = []
        self.gps_trajectory: List[np.ndarray] = []

        # Cache LiDAR geometry for 3-D point reconstruction
        self._lidar_n_layers = self.lidar.getNumberOfLayers()
        self._lidar_v_fov = self.lidar.getVerticalFov()
        self._lidar_h_angles: Optional[np.ndarray] = None  # built on first step
        self._lidar_v_angles: Optional[np.ndarray] = None  # built on first step

    # ── Initialisation helpers ───────────────────────────────────────────────

    def _load_checkpoint(self) -> None:
        """Load CNN weights if a checkpoint file exists."""
        if MODEL_CHECKPOINT.exists():
            try:
                ckpt = torch.load(MODEL_CHECKPOINT, map_location=self.device)
                self.cnn.load_state_dict(ckpt)
                print(f"[CNN-LiDAR-SLAM] Loaded CNN checkpoint from {MODEL_CHECKPOINT}", flush=True)
            except Exception as e:
                print(f"[CNN-LiDAR-SLAM] Warning: could not load checkpoint: {e}", flush=True)
        else:
            print("[CNN-LiDAR-SLAM] No checkpoint found; CNN running with random weights.", flush=True)

    # ── Main control loop ────────────────────────────────────────────────────

    def run(self) -> None:
        """Main loop – called once; steps until simulation ends."""
        print("[CNN-LiDAR-SLAM] Starting control loop …", flush=True)
        while self.robot.step(self.timestep) != -1:
            self._tick()

    def _tick(self) -> None:
        """Process one simulation step."""
        self._step_count += 1

        # ── 1. Read sensors ──────────────────────────────────────────────────
        raw_ranges = _read_lidar(self.lidar)

        # Use only the first horizontal layer to avoid floor/ceiling returns
        # from angled layers (which would otherwise keep front_min < SAFE_DIST
        # and prevent the robot from driving forward).
        h = self._lidar_h_res if self._lidar_h_res > 0 else len(raw_ranges)
        ranges = raw_ranges[:h]

        # First-step diagnostics — log raw values to diagnose floor returns
        if self._step_count == 1:
            valid = raw_ranges[raw_ranges > 0]
            print(
                f"[CNN-LiDAR-SLAM] DIAG raw_len={len(raw_ranges)} h={h} "
                f"layer0_min={ranges[ranges>0].min() if (ranges>0).any() else 'N/A':.3f} "
                f"raw_min={valid.min() if len(valid)>0 else 'N/A':.3f} "
                f"layer0_sample={ranges[h//2-2:h//2+3].tolist()}",
                flush=True,
            )

        if self.lidar_angles is None:
            self.lidar_angles = _build_lidar_angles(self.lidar, h)

        rpy, accel_raw, gyro_raw = _read_imu(
            self.inertial, self.accel_dev, self.gyro_dev
        )

        # ── 1a. Collision detection — restart simulation on impact ───────────
        valid_ranges = ranges[ranges > 0.0]
        if len(valid_ranges) > 0 and float(valid_ranges.min()) < COLLISION_DIST:
            print(
                f"[CNN-LiDAR-SLAM] COLLISION detected "
                f"(min range {valid_ranges.min():.3f} m) — restarting …",
                flush=True,
            )
            self.robot.simulationRevert()
            return

        # ── 1b. Goal detection — stop when robot reaches the target ──────────
        if self._gps is not None:
            gps_vals = self._gps.getValues()
            gps_xy = np.array([gps_vals[0], gps_vals[1]], dtype=np.float32)
            dist_to_goal = float(np.linalg.norm(gps_xy - GOAL_POS))
            if dist_to_goal < GOAL_RADIUS:
                print(
                    f"[CNN-LiDAR-SLAM] GOAL REACHED! "
                    f"(distance {dist_to_goal:.3f} m) — stopping simulation.",
                    flush=True,
                )
                self.robot.simulationSetMode(0)  # pause simulation
                return

        # ── 2. IMU filter ────────────────────────────────────────────────────
        imu_state: IMUState = self.imu_proc.step(gyro_raw, accel_raw)

        # ── 3. LiDAR preprocessing ───────────────────────────────────────────
        lidar_feat: LiDARFeatures = self.lidar_prep.process(
            ranges, self.lidar_angles
        )

        # ── 4. Build 26-D feature vector and push to CNN window ──────────────
        feat_vec = self.lidar_prep.build_feature_vector(
            lidar_feat,
            imu_accel=imu_state.accel_body,
            imu_gyro=imu_state.gyro_body,
            imu_quat=imu_state.quaternion,
        )
        self.feature_window.append(feat_vec)

        # ── 5. CNN forward pass (every CNN_UPDATE_EVERY steps) ───────────────
        if (self._step_count % CNN_UPDATE_EVERY == 0 and
                len(self.feature_window) == CNN_SEQ_LEN):
            seq = np.stack(list(self.feature_window), axis=0)   # (T, 26)
            self._cnn_landmarks = self.cnn.get_landmarks(seq)

        # ── 6. IEKF propagate via wheel odometry ─────────────────────────────
        # Wheel odometry is far more reliable than IMU double-integration for
        # a velocity-controlled robot; use gyro only for heading rate.
        gyro_z = float(gyro_raw[2])
        self.iekf.propagate_odom(self._cmd_speed, gyro_z, self.dt)

        # ── 7. IEKF measurement update ───────────────────────────────────────
        self.iekf.update(
            edge_points=lidar_feat.edge_points,
            planar_points=lidar_feat.planar_points,
            semantic_landmarks=self._cnn_landmarks,
        )

        # ── 8. Map update ────────────────────────────────────────────────────
        state = self.iekf.state
        pos = state.position
        heading = state.heading

        # Scan points in world frame for occupancy update
        world_pts = self._scan_to_world(lidar_feat, state)

        new_kf = self.slam_map.try_add_keyframe(
            float(pos[0]), float(pos[1]), heading,
            scan_points=world_pts,
        )
        if new_kf is not None:
            self._keyframe_count += 1

            # Update semantic landmark map
            R = self._rot2_mat(heading)
            for (cx, cy, r) in self._cnn_landmarks:
                if cx == 0.0 and cy == 0.0:
                    continue
                p_world = R @ np.array([cx, cy]) + pos
                self.slam_map.update_landmark(p_world, r)

            # Periodic global optimisation
            if self._keyframe_count % MAP_OPTIM_EVERY == 0:
                print(f"[CNN-LiDAR-SLAM] Running pose-graph optimisation "
                      f"({self._keyframe_count} keyframes) …", flush=True)
                self.slam_map.optimise()

        # ── Periodic map plot (every 500 steps) ──────────────────────────────
        if self._step_count % 500 == 0 and self._step_count > 0:
            self.slam_map.save_plot(f"slam_map_step{self._step_count:06d}.png")

        # ── 9. Log trajectory ────────────────────────────────────────────────
        self.trajectory.append(np.array([pos[0], pos[1], heading]))
        if self._gps is not None:
            gps_vals = self._gps.getValues()
            self.gps_trajectory.append(np.array([gps_vals[0], gps_vals[1]]))

        # ── 10. Display ──────────────────────────────────────────────────────
        if self._display is not None and self._step_count % 5 == 0:
            self._draw_display(pos)

        # ── 11. Drive ────────────────────────────────────────────────────────
        self._drive(ranges)

        # ── Periodic console log ─────────────────────────────────────────────
        if self._step_count % 100 == 0:
            self._log_status()

    # ── Reactive drive ───────────────────────────────────────────────────────

    def _drive(self, ranges: np.ndarray) -> None:
        """
        LiDAR-reactive navigation: always turn right when blocked.

        When anything in the ±45° front arc is closer than SAFE_DIST the robot
        slows and steers right.  It keeps turning right until the full forward
        arc is clear again, then resumes driving straight.  If the robot stays
        blocked for too long (stuck counter) it reverses briefly before resuming
        the right-turn strategy.

        Angles in the range image run from -FOV/2 (index 0) to +FOV/2 (index
        n-1).  Positive angles = left of the robot's forward direction.
        Negative steer = turn right.
        """
        n = len(ranges)
        if n == 0:
            return

        valid = np.where(ranges > 0, ranges, self.lidar_max_range)

        fov = self.lidar.getFov()
        def _idx(angle_rad: float) -> int:
            frac = (angle_rad + fov / 2.0) / fov
            return int(np.clip(frac * n, 0, n - 1))

        # Sectors used for decisions
        front = valid[_idx(-math.radians(45)):_idx(math.radians(45)) + 1]
        right = valid[_idx(-math.radians(90)):_idx(-math.radians(45)) + 1]

        front_min = float(front.min()) if len(front) > 0 else self.lidar_max_range
        right_min = float(right.min()) if len(right) > 0 else self.lidar_max_range

        # Periodic drive diagnostics
        if self._step_count % 200 == 1:
            print(
                f"[CNN-LiDAR-SLAM] DRIVE n={n} fov={math.degrees(fov):.0f}deg "
                f"front_min={front_min:.3f} right_min={right_min:.3f} "
                f"safe={self._SAFE_DIST} stuck={self._stuck_counter}",
                flush=True,
            )

        speed_cmd = self._MAX_SPEED
        steer = 0.0

        if front_min < self._SAFE_DIST:
            self._stuck_counter += 1

            if self._stuck_counter > 60:
                # Been stuck a long time — reverse straight to create distance
                speed_cmd = -self._MAX_SPEED * 0.4
                steer = 0.0
                if self._stuck_counter > 90:
                    self._stuck_counter = 0   # reset and retry right-turn
            else:
                # Standard response: slow down and turn right in place
                speed_cmd = self._MAX_SPEED * 0.15
                steer = -1.0                  # hard right

        else:
            self._stuck_counter = 0
            speed_cmd = self._MAX_SPEED

            # Drift right slightly when right side is open (wall-following bias)
            if right_min > self._SAFE_DIST * 2.0:
                steer = -0.2   # gentle right drift when space available
            else:
                steer = 0.0    # drive straight when right side is close

        # Smooth steering to avoid jerking
        self._steer_state += 0.35 * (steer - self._steer_state)

        # Record speed in m/s for IEKF odometry
        self._cmd_speed = speed_cmd * self._WHEEL_RADIUS

        self._set_motors(speed_cmd, self._steer_state)

    def _set_motors(self, speed: float, steer: float) -> None:
        steer = float(np.clip(steer, -1.0, 1.0))
        speed = float(np.clip(speed, -self._MAX_SPEED, self._MAX_SPEED))
        self._left_steer.setPosition(steer)
        self._right_steer.setPosition(steer)
        for w in self._wheels:
            w.setVelocity(speed)

    # ── Utilities ────────────────────────────────────────────────────────────

    def _scan_to_world(
        self, feat: LiDARFeatures, state
    ) -> Optional[np.ndarray]:
        """Project all valid LiDAR feature points to the world frame."""
        pts = np.vstack([feat.edge_points, feat.planar_points]) \
            if (len(feat.edge_points) > 0 and len(feat.planar_points) > 0) \
            else (feat.edge_points if len(feat.edge_points) > 0
                  else feat.planar_points)
        if len(pts) == 0:
            return None
        R = self._rot2_mat(state.heading)
        return (R @ pts.T).T + state.position

    @staticmethod
    def _rot2_mat(theta: float) -> np.ndarray:
        c, s = math.cos(theta), math.sin(theta)
        return np.array([[c, -s], [s, c]])

    def _raw_to_3d_world(
        self, raw_ranges: np.ndarray, state
    ) -> np.ndarray:
        """
        Reconstruct 3-D world-frame points from the full multi-layer range image.

        Each layer l has a vertical angle v_angles[l]; each column c within a
        layer has the same horizontal angle h_angles[c].  The local-frame point
        is (r·cos(v)·cos(h),  r·cos(v)·sin(h),  r·sin(v)), then rotated into
        the world frame using the current IEKF heading.

        Returns:
            (N, 3) float32 array of [x_world, y_world, z_local] for all valid
            (range > 0) points across all layers.
        """
        h = self._lidar_h_res if self._lidar_h_res > 0 else len(raw_ranges)
        n_layers = self._lidar_n_layers if self._lidar_n_layers > 0 else 1

        # Build angle grids once
        if self._lidar_h_angles is None:
            h_fov = self.lidar.getFov()
            self._lidar_h_angles = np.linspace(
                -h_fov / 2.0, h_fov / 2.0, h, dtype=np.float32
            )
        if self._lidar_v_angles is None:
            v_fov = self._lidar_v_fov if self._lidar_v_fov > 0 else 0.0
            self._lidar_v_angles = np.linspace(
                -v_fov / 2.0, v_fov / 2.0, n_layers, dtype=np.float32
            )

        cos_head = math.cos(state.heading)
        sin_head = math.sin(state.heading)
        px, py = float(state.position[0]), float(state.position[1])

        chunks = []
        for l_idx in range(n_layers):
            layer = raw_ranges[l_idx * h : (l_idx + 1) * h]
            v_a = float(self._lidar_v_angles[l_idx])
            cos_v = math.cos(v_a)
            sin_v = math.sin(v_a)

            mask = layer > 0.01
            if not mask.any():
                continue
            r = layer[mask]
            h_a = self._lidar_h_angles[mask]

            # Robot-local frame (x = forward, y = left, z = up)
            x_l = r * cos_v * np.cos(h_a)
            y_l = r * cos_v * np.sin(h_a)
            z_l = r * sin_v

            # Rotate XY into world frame; Z stays local (robot is ground-flat)
            x_w = cos_head * x_l - sin_head * y_l + px
            y_w = sin_head * x_l + cos_head * y_l + py

            chunks.append(np.stack([x_w, y_w, z_l], axis=1))

        return np.concatenate(chunks, axis=0) if chunks else np.empty((0, 3), dtype=np.float32)

    def _draw_display(self, pos: np.ndarray) -> None:
        """Draw the current robot position on the Webots display."""
        if self._display is None:
            return
        try:
            ox, oy = DISPLAY_ORIGIN
            px = int(ox + pos[0] * DISPLAY_SCALE)
            py = int(oy - pos[1] * DISPLAY_SCALE)
            self._display.setColor(0x00FF00)
            self._display.drawPixel(px, py)
        except Exception:
            pass  # display ops are non-critical

    def _log_status(self) -> None:
        state = self.iekf.state
        pos = state.position
        head_deg = math.degrees(state.heading)
        n_kf = len(self.slam_map.nodes)
        n_lm = len(self.slam_map.landmarks)

        ate_str = ""
        if len(self.gps_trajectory) > 0 and len(self.trajectory) > 0:
            slam_xy = np.array([t[:2] for t in self.trajectory[-100:]])
            gps_xy = np.array(self.gps_trajectory[-100:])
            min_len = min(len(slam_xy), len(gps_xy))
            if min_len > 0:
                ate = float(np.mean(
                    np.linalg.norm(slam_xy[:min_len] - gps_xy[:min_len], axis=1)
                ))
                ate_str = f"  ATE≈{ate:.3f}m"

        print(
            f"[CNN-LiDAR-SLAM] step={self._step_count:5d}  "
            f"pos=({pos[0]:.2f},{pos[1]:.2f})  "
            f"heading={head_deg:.1f}deg  "
            f"keyframes={n_kf}  landmarks={n_lm}{ate_str}",
            flush=True,
        )

    # ── Offline training data collection ─────────────────────────────────────

    def save_trajectory(self, path: str = "slam_trajectory.npz") -> None:
        """Save trajectory and GPS ground truth to disk for evaluation."""
        traj = np.array(self.trajectory, dtype=np.float32)
        gps = np.array(self.gps_trajectory, dtype=np.float32) if self.gps_trajectory else np.empty((0, 2))
        np.savez(path, trajectory=traj, gps=gps)
        print(f"[CNN-LiDAR-SLAM] Trajectory saved to {path}", flush=True)
        self.slam_map.save_plot("slam_map_final.png")


# ─────────────────────────────────────────────────────────────────────────────
# Training utilities  (run offline, not in simulation loop)
# ─────────────────────────────────────────────────────────────────────────────

def collect_training_data(
    controller: CNNLidarSLAMController,
    output_path: str = "training_data.npz",
) -> None:
    """
    Collect labelled LiDAR-IMU frames for CNN training.

    Ground-truth object positions come from GPS + known object positions.
    Call this during a dedicated data-collection episode.
    """
    features_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []

    print(f"[CNN-LiDAR-SLAM] Collecting training data → {output_path}")
    while controller.robot.step(controller.timestep) != -1:
        ranges = _read_lidar(controller.lidar)
        if controller.lidar_angles is None:
            controller.lidar_angles = _build_lidar_angles(controller.lidar, len(ranges))
        rpy, accel, gyro = _read_imu(
            controller.inertial, controller.accel_dev, controller.gyro_dev
        )
        imu_state = controller.imu_proc.step(gyro, accel)
        lidar_feat = controller.lidar_prep.process(ranges, controller.lidar_angles)

        feat_vec = controller.lidar_prep.build_feature_vector(
            lidar_feat,
            imu_accel=imu_state.accel_body,
            imu_gyro=imu_state.gyro_body,
            imu_quat=imu_state.quaternion,
        )
        features_list.append(feat_vec)

        # Label: use preliminary candidates as pseudo-ground-truth
        # (replace with real annotation for production use)
        label = np.zeros(12, dtype=np.float32)
        for i, (cx, cy, r) in enumerate(lidar_feat.object_candidates[:4]):
            label[i * 3:i * 3 + 3] = [cx, cy, r]
        labels_list.append(label)

    features = np.array(features_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.float32)
    np.savez(output_path, features=features, labels=labels)
    print(f"[CNN-LiDAR-SLAM] Saved {len(features)} frames.")


def train_cnn(
    data_path: str = "training_data.npz",
    checkpoint_path: str = str(MODEL_CHECKPOINT),
    epochs: int = 100,
    seq_len: int = CNN_SEQ_LEN,
    lr: float = 1e-4,
    batch_size: int = 16,
) -> None:
    """
    Offline CNN training (Section III-F: Adam, lr=1e-4, batch=16, 100 epochs).

    Args:
        data_path:       path to .npz produced by collect_training_data().
        checkpoint_path: where to save the trained model.
        epochs:          number of training epochs.
        seq_len:         temporal sequence length for the CNN.
        lr:              Adam learning rate.
        batch_size:      mini-batch size.
    """
    import torch.utils.data as td

    data = np.load(data_path)
    features = data["features"]   # (N, 26)
    labels = data["labels"]       # (N, 12)

    # Build sliding-window sequences
    X, Y = [], []
    for i in range(seq_len, len(features)):
        X.append(features[i - seq_len:i])
        Y.append(labels[i])
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)

    n_train = int(0.70 * len(X))
    n_val = int(0.15 * len(X))
    X_train, Y_train = X[:n_train], Y[:n_train]
    X_val, Y_val = X[n_train:n_train + n_val], Y[n_train:n_train + n_val]

    train_ds = td.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
    val_ds = td.TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val))
    train_dl = td.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = td.DataLoader(val_ds, batch_size=batch_size)

    device = torch.device("cpu")
    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    criterion = torch.nn.SmoothL1Loss(beta=0.1)

    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dl)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item()
        val_loss /= max(len(val_dl), 1)

        scheduler.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), checkpoint_path)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}/{epochs}  train={train_loss:.4f}  val={val_loss:.4f}")

    print(f"[CNN-LiDAR-SLAM] Training done. Best val loss = {best_val:.4f}")
    print(f"[CNN-LiDAR-SLAM] Model saved to {checkpoint_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback
    try:
        ctrl = CNNLidarSLAMController()
        ctrl.run()
    except KeyboardInterrupt:
        pass
    except Exception:
        traceback.print_exc()
    finally:
        try:
            ctrl.save_trajectory()
        except Exception:
            pass
