"""
LiDAR preprocessing module for CNN-LiDAR-SLAM.

Implements feature extraction (edge/planar points via local curvature),
voxel-grid downsampling, and preliminary object candidate detection as
described in Section III-C and III-D of the paper.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class LiDARFeatures:
    """Container for extracted LiDAR features from one scan frame."""
    edge_points: np.ndarray    # (N_e, 2) edge points in Cartesian (x, y)
    planar_points: np.ndarray  # (N_p, 2) planar points in Cartesian (x, y)
    scan_stats: np.ndarray     # (4,) [min_range, max_range, avg_range, point_density]
    object_candidates: List[Tuple[float, float, float]]  # [(cx, cy, r), ...] up to 4


class LiDARPreprocessor:
    """
    Processes raw 2D LiDAR scans into features for SLAM.

    Equation (1) from the paper defines local curvature σ^(m,n)_k as the
    mean distance from a point to its neighbours.  Points above threshold
    σ_t are classified as edge features E_k; points below are planar S_k.
    """

    def __init__(
        self,
        curvature_threshold: float = 0.05,
        neighbor_half_width: int = 5,
        voxel_size: float = 0.05,
        min_cluster_size: int = 4,
        max_range: float = 12.0,
        min_range: float = 0.05,
    ) -> None:
        self.sigma_t = curvature_threshold
        self.k = neighbor_half_width
        self.voxel_size = voxel_size
        self.min_cluster_size = min_cluster_size
        self.max_range = max_range
        self.min_range = min_range

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, ranges: np.ndarray, angles: np.ndarray) -> LiDARFeatures:
        """
        Full preprocessing pipeline for one LiDAR scan.

        Args:
            ranges: (N,) raw range measurements in metres.
            angles: (N,) corresponding angles in radians.

        Returns:
            LiDARFeatures containing edge points, planar points, scan
            statistics, and up to 4 preliminary object candidates.
        """
        # ---- 1. Statistical outlier removal ----
        ranges = self._filter_outliers(ranges)

        # ---- 2. Valid-range mask ----
        valid = (ranges > self.min_range) & (ranges < self.max_range)
        r_valid = ranges[valid]
        a_valid = angles[valid]

        # ---- 3. Cartesian conversion ----
        x = r_valid * np.cos(a_valid)
        y = r_valid * np.sin(a_valid)
        points = np.column_stack([x, y])  # (M, 2)

        # ---- 4. Curvature-based feature extraction (Eq. 1) ----
        curvatures = self._compute_curvature(r_valid)
        edge_mask = curvatures > self.sigma_t
        planar_mask = (curvatures > 0) & ~edge_mask

        edge_points = points[edge_mask]
        planar_points = points[planar_mask]

        # ---- 5. Voxel downsampling (Eq. 14) ----
        edge_points = self._voxel_downsample(edge_points)
        planar_points = self._voxel_downsample(planar_points)

        # ---- 6. Scan statistics ----
        if len(r_valid) > 0:
            scan_stats = np.array([
                float(r_valid.min()),
                float(r_valid.max()),
                float(r_valid.mean()),
                float(len(r_valid) / len(ranges)),  # point density ratio
            ], dtype=np.float32)
        else:
            scan_stats = np.zeros(4, dtype=np.float32)

        # ---- 7. Preliminary object candidates ----
        objects = self._extract_object_candidates(r_valid, a_valid, points)

        return LiDARFeatures(
            edge_points=edge_points,
            planar_points=planar_points,
            scan_stats=scan_stats,
            object_candidates=objects,
        )

    def build_feature_vector(
        self,
        features: LiDARFeatures,
        imu_accel: np.ndarray,
        imu_gyro: np.ndarray,
        imu_quat: np.ndarray,
    ) -> np.ndarray:
        """
        Assemble the 26-dimensional input feature vector for the CNN.

        Layout (paper Section III-D-2):
          [0:4]   scan stats (min, max, avg range, density)
          [4:16]  object geometry: (cx, cy, r) × 4 objects
          [16:19] IMU linear acceleration (x, y, z)
          [19:22] IMU angular velocity (x, y, z)
          [22:26] IMU orientation quaternion (w, x, y, z)

        Args:
            features:  extracted LiDAR features for this frame.
            imu_accel: (3,) accelerometer reading in m/s².
            imu_gyro:  (3,) gyroscope reading in rad/s.
            imu_quat:  (4,) orientation quaternion [w, x, y, z].

        Returns:
            (26,) float32 feature vector.
        """
        # Object block: pad/truncate to exactly 4 objects
        obj_block = np.zeros(12, dtype=np.float32)
        for i, (cx, cy, r) in enumerate(features.object_candidates[:4]):
            obj_block[i * 3:i * 3 + 3] = [cx, cy, r]

        vec = np.concatenate([
            features.scan_stats.astype(np.float32),
            obj_block,
            imu_accel.astype(np.float32),
            imu_gyro.astype(np.float32),
            imu_quat.astype(np.float32),
        ])
        assert vec.shape == (26,), f"Expected 26-D vector, got {vec.shape}"
        return vec

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _filter_outliers(self, ranges: np.ndarray) -> np.ndarray:
        """Statistical outlier removal: discard points > 2σ from mean."""
        valid = (ranges > self.min_range) & (ranges < self.max_range)
        if valid.sum() < 3:
            return ranges.copy()
        mean = ranges[valid].mean()
        std = ranges[valid].std()
        cleaned = ranges.copy()
        outlier = valid & (np.abs(ranges - mean) > 2.0 * std)
        cleaned[outlier] = 0.0
        return cleaned

    def _compute_curvature(self, ranges: np.ndarray) -> np.ndarray:
        """
        Local curvature σ^(m,n)_k per Equation (1).

        σ^(m,n)_k = (1/|S|) Σ_{j∈S} ||p_j − p_n||

        Approximated here on the 1-D range signal for efficiency.
        """
        n = len(ranges)
        sigma = np.zeros(n, dtype=np.float32)
        for i in range(self.k, n - self.k):
            neighbourhood = ranges[i - self.k:i + self.k + 1]
            sigma[i] = float(np.mean(np.abs(neighbourhood - ranges[i])))
        return sigma

    def _voxel_downsample(self, points: np.ndarray) -> np.ndarray:
        """
        Voxel grid filter per Equation (14).

        Replaces every non-empty voxel cell with the centroid of points
        inside it.
        """
        if len(points) == 0:
            return points

        indices = np.floor(points / self.voxel_size).astype(int)
        # Use a structured array as hashable key
        unique_idx, inverse = np.unique(indices, axis=0, return_inverse=True)
        centroids = np.zeros((len(unique_idx), points.shape[1]), dtype=np.float32)
        for i in range(len(unique_idx)):
            centroids[i] = points[inverse == i].mean(axis=0)
        return centroids

    def _extract_object_candidates(
        self,
        ranges: np.ndarray,
        angles: np.ndarray,
        points: np.ndarray,
    ) -> List[Tuple[float, float, float]]:
        """
        Cluster scan returns into preliminary object candidates (cx, cy, r).

        Objects are identified by angular gaps (scan discontinuities) between
        consecutive returns that are larger than a jump threshold.  Each
        sufficiently large cluster is represented as its centroid and mean
        radius.
        """
        if len(ranges) < self.min_cluster_size:
            return []

        # Detect cluster boundaries by range discontinuities
        gap_threshold = 0.5  # metres
        diffs = np.abs(np.diff(ranges))
        boundaries = np.where(diffs > gap_threshold)[0] + 1
        splits = np.concatenate([[0], boundaries, [len(ranges)]])

        candidates: List[Tuple[float, float, float]] = []
        for start, end in zip(splits[:-1], splits[1:]):
            if end - start < self.min_cluster_size:
                continue
            cluster = points[start:end]
            cx = float(cluster[:, 0].mean())
            cy = float(cluster[:, 1].mean())
            # radius = RMS distance from centroid
            r = float(np.sqrt(((cluster - [cx, cy]) ** 2).sum(axis=1)).mean())
            candidates.append((cx, cy, r))

        # Sort by proximity to sensor origin; keep up to 4
        candidates.sort(key=lambda o: o[0] ** 2 + o[1] ** 2)
        return candidates[:4]
