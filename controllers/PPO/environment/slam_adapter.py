"""PPO-side SLAM runtime adapter for integrating CNN-LiDAR-SLAM into the environment."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Deque, List, Tuple
import sys

import numpy as np
import torch

# Make sibling controller packages importable from the PPO controller folder.
_CONTROLLERS_DIR = Path(__file__).resolve().parents[2]
if str(_CONTROLLERS_DIR) not in sys.path:
    sys.path.insert(0, str(_CONTROLLERS_DIR))

from SLAM.cnn_model import build_model
from SLAM.iekf_backend import IEKFBackend
from SLAM.imu_filter import IMUProcessor
from SLAM.lidar_preprocessing import LiDARPreprocessor
from SLAM.slam_map import SLAMMap


@dataclass
class SLAMInputFrame:
    """Raw sensor and control packet consumed by one adapter step."""

    lidar_ranges: np.ndarray
    lidar_angles: np.ndarray
    gps_xyz: np.ndarray
    heading: float
    rpy: np.ndarray
    accel: np.ndarray
    gyro: np.ndarray
    commanded_speed_mps: float = 0.0


@dataclass
class SLAMTelemetry:
    """Timing and health metrics from the latest adapter step."""

    step_index: int = 0
    imu_ms: float = 0.0
    lidar_ms: float = 0.0
    cnn_ms: float = 0.0
    iekf_ms: float = 0.0
    map_ms: float = 0.0
    total_ms: float = 0.0
    ran_cnn: bool = False
    cnn_landmark_count: int = 0
    keyframe_count: int = 0
    map_landmark_count: int = 0


@dataclass
class SLAMStateSnapshot:
    """Structured SLAM state returned to the PPO environment each step."""

    pose_xy: np.ndarray
    heading: float
    velocity_xy: np.ndarray
    covariance_diag: np.ndarray
    covariance_trace: float
    keyframe_count: int
    landmark_count: int
    cnn_landmarks: List[Tuple[float, float, float]] = field(default_factory=list)
    telemetry: SLAMTelemetry = field(default_factory=SLAMTelemetry)


class PPOSLAMAdapter:
    """Runs the full SLAM stack (IMU + LiDAR + CNN + IEKF + map) inside PPO."""

    def __init__(self, timestep_seconds: float, config: Any):
        self.config = config
        self.dt = float(max(timestep_seconds, 1e-4))

        self.enable_cnn = bool(getattr(config, "enable_slam_runtime", False))
        self.enable_pose_graph = bool(getattr(config, "enable_slam_runtime", False))
        self.cnn_update_every = max(1, int(getattr(config, "slam_cnn_update_every", 5)))
        self.pose_graph_optimize_every = max(1, int(getattr(config, "slam_pose_graph_optimize_every", 50)))
        self.telemetry_interval = max(1, int(getattr(config, "slam_telemetry_interval", 100)))
        self.cnn_seq_len = 10

        self.lidar_preprocessor = LiDARPreprocessor(max_range=12.0)
        self.imu_processor = IMUProcessor(dt=self.dt)
        self.iekf = IEKFBackend()
        self.slam_map = SLAMMap(map_resolution=0.05)

        self.device = torch.device("cpu")
        self.cnn = build_model().to(self.device)
        self.cnn.eval()

        self.feature_window: Deque[np.ndarray] = deque(maxlen=self.cnn_seq_len)
        self._cnn_landmarks: List[Tuple[float, float, float]] = []
        self._step_index = 0
        self._keyframe_count = 0
        self._last_telemetry = SLAMTelemetry()

        self._load_checkpoint(getattr(config, "slam_cnn_checkpoint_path", None))

    def _load_checkpoint(self, checkpoint_path: Any) -> None:
        """Load optional CNN checkpoint for landmark regression."""
        if not checkpoint_path:
            return

        checkpoint = Path(str(checkpoint_path))
        if not checkpoint.is_absolute():
            checkpoint = (Path(__file__).resolve().parents[1] / checkpoint).resolve()

        if not checkpoint.exists():
            print(f"[PPO][SLAM] WARNING: CNN checkpoint not found: {checkpoint}")
            return

        try:
            state = torch.load(checkpoint, map_location=self.device)
            self.cnn.load_state_dict(state)
            print(f"[PPO][SLAM] Loaded CNN checkpoint: {checkpoint}")
        except Exception as exc:
            print(f"[PPO][SLAM] WARNING: Could not load checkpoint '{checkpoint}': {exc}")

    def reset(self) -> SLAMStateSnapshot:
        """Reset full SLAM runtime state for a new episode."""
        self.imu_processor.reset()
        self.iekf = IEKFBackend()
        self.slam_map = SLAMMap(map_resolution=0.05)

        self.feature_window.clear()
        self._cnn_landmarks = []
        self._step_index = 0
        self._keyframe_count = 0
        self._last_telemetry = SLAMTelemetry(step_index=0)

        return self._build_snapshot()

    def step(self, frame: SLAMInputFrame) -> SLAMStateSnapshot:
        """Advance one SLAM step from raw sensors and commanded speed context."""
        self._step_index += 1
        t_total_start = perf_counter()

        lidar_ranges = np.asarray(frame.lidar_ranges, dtype=np.float32).reshape(-1)
        lidar_angles = np.asarray(frame.lidar_angles, dtype=np.float32).reshape(-1)
        accel = np.asarray(frame.accel, dtype=np.float32).reshape(3)
        gyro = np.asarray(frame.gyro, dtype=np.float32).reshape(3)

        if lidar_ranges.size == 0:
            telemetry = SLAMTelemetry(step_index=self._step_index)
            self._last_telemetry = telemetry
            return self._build_snapshot(telemetry)

        if lidar_angles.size != lidar_ranges.size:
            lidar_angles = np.linspace(-np.pi, np.pi, lidar_ranges.size, dtype=np.float32)

        t0 = perf_counter()
        imu_state = self.imu_processor.step(gyro, accel)
        t1 = perf_counter()

        lidar_features = self.lidar_preprocessor.process(lidar_ranges, lidar_angles)
        t2 = perf_counter()

        feature_vec = self.lidar_preprocessor.build_feature_vector(
            lidar_features,
            imu_accel=imu_state.accel_body,
            imu_gyro=imu_state.gyro_body,
            imu_quat=imu_state.quaternion,
        )
        self.feature_window.append(feature_vec)

        ran_cnn = False
        t_cnn_start = perf_counter()
        if (
            self.enable_cnn
            and self._step_index % self.cnn_update_every == 0
            and len(self.feature_window) == self.cnn_seq_len
        ):
            seq = np.stack(list(self.feature_window), axis=0)
            try:
                self._cnn_landmarks = self.cnn.get_landmarks(seq)
                ran_cnn = True
            except Exception as exc:
                print(f"[PPO][SLAM] WARNING: CNN inference failed: {exc}")
        t3 = perf_counter()

        t_iekf_start = perf_counter()
        commanded_speed = float(frame.commanded_speed_mps)
        gyro_z = float(gyro[2])
        self.iekf.propagate_odom(commanded_speed, gyro_z, self.dt)
        self.iekf.update(
            edge_points=lidar_features.edge_points,
            planar_points=lidar_features.planar_points,
            semantic_landmarks=self._cnn_landmarks,
        )
        t4 = perf_counter()

        t_map_start = perf_counter()
        state = self.iekf.state
        world_pts = self._scan_to_world(lidar_features.edge_points, lidar_features.planar_points, state)
        new_keyframe = self.slam_map.try_add_keyframe(
            float(state.position[0]),
            float(state.position[1]),
            float(state.heading),
            scan_points=world_pts,
        )

        if new_keyframe is not None:
            self._keyframe_count += 1
            if self._cnn_landmarks:
                rot = self._rot2(state.heading)
                for cx, cy, radius in self._cnn_landmarks:
                    if cx == 0.0 and cy == 0.0:
                        continue
                    p_world = rot @ np.array([cx, cy], dtype=np.float64) + state.position
                    self.slam_map.update_landmark(p_world, float(radius))

            if self.enable_pose_graph and self._keyframe_count % self.pose_graph_optimize_every == 0:
                self.slam_map.optimise()
        t5 = perf_counter()

        telemetry = SLAMTelemetry(
            step_index=self._step_index,
            imu_ms=(t1 - t0) * 1000.0,
            lidar_ms=(t2 - t1) * 1000.0,
            cnn_ms=(t3 - t_cnn_start) * 1000.0,
            iekf_ms=(t4 - t_iekf_start) * 1000.0,
            map_ms=(t5 - t_map_start) * 1000.0,
            total_ms=(perf_counter() - t_total_start) * 1000.0,
            ran_cnn=ran_cnn,
            cnn_landmark_count=len(self._cnn_landmarks),
            keyframe_count=len(self.slam_map.nodes),
            map_landmark_count=len(self.slam_map.landmarks),
        )
        self._last_telemetry = telemetry

        return self._build_snapshot(telemetry)

    def last_telemetry(self) -> SLAMTelemetry:
        """Return latest timing/health telemetry."""
        return self._last_telemetry

    def _build_snapshot(self, telemetry: SLAMTelemetry | None = None) -> SLAMStateSnapshot:
        state = self.iekf.state
        covariance_diag = np.diag(state.P).astype(np.float32)
        covariance_trace = float(np.trace(state.P[:3, :3]))
        return SLAMStateSnapshot(
            pose_xy=state.position.astype(np.float32),
            heading=float(state.heading),
            velocity_xy=state.velocity.astype(np.float32),
            covariance_diag=covariance_diag,
            covariance_trace=covariance_trace,
            keyframe_count=len(self.slam_map.nodes),
            landmark_count=len(self.slam_map.landmarks),
            cnn_landmarks=list(self._cnn_landmarks),
            telemetry=telemetry if telemetry is not None else self._last_telemetry,
        )

    @staticmethod
    def _rot2(theta: float) -> np.ndarray:
        c = float(np.cos(theta))
        s = float(np.sin(theta))
        return np.array([[c, -s], [s, c]], dtype=np.float64)

    def _scan_to_world(self, edge_points: np.ndarray, planar_points: np.ndarray, state) -> np.ndarray | None:
        """Project the current LiDAR feature set to world frame for map updates."""
        if edge_points.size == 0 and planar_points.size == 0:
            return None

        if edge_points.size > 0 and planar_points.size > 0:
            points = np.vstack([edge_points, planar_points])
        elif edge_points.size > 0:
            points = edge_points
        else:
            points = planar_points

        rot = self._rot2(state.heading)
        return (rot @ points.T).T + state.position
