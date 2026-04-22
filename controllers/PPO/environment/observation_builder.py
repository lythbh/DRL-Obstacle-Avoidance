"""Pure geometry and observation construction helpers."""

from typing import Optional, Tuple

import numpy as np


def goal_geometry(
    pos: np.ndarray,
    heading: float,
    endpoint: np.ndarray,
    heading_frame_offset: float,
) -> Tuple[float, float]:
    """Compute goal distance and heading error."""
    goal_vec = endpoint - pos
    goal_distance = float(np.linalg.norm(goal_vec))
    goal_direction = float(np.arctan2(goal_vec[1], goal_vec[0]))

    aligned_heading = heading + heading_frame_offset
    aligned_heading = float(np.arctan2(np.sin(aligned_heading), np.cos(aligned_heading)))
    goal_error = float(np.arctan2(np.sin(goal_direction - aligned_heading), np.cos(goal_direction - aligned_heading)))
    return goal_distance, goal_error


def build_observation(
    lidar: np.ndarray,
    pos: np.ndarray,
    heading: float,
    endpoint: np.ndarray,
    reference_distance: float,
    heading_frame_offset: float,
    mode: str = "baseline",
    slam_pose: Optional[np.ndarray] = None,
    slam_heading: Optional[float] = None,
    slam_cov_diag: Optional[np.ndarray] = None,
    slam_cov_trace: Optional[float] = None,
    slam_keyframe_count: int = 0,
    slam_landmark_count: int = 0,
) -> np.ndarray:
    """Build compact observation vector with goal-direction context."""
    goal_distance, goal_error = goal_geometry(pos, heading, endpoint, heading_frame_offset)
    direction_features = np.array(
        [
            np.sin(goal_error),
            np.cos(goal_error),
            goal_distance / max(float(reference_distance), 1e-6),
        ],
        dtype=np.float32,
    )

    if mode != "slam_v1":
        return np.concatenate([lidar, direction_features])

    pose_source = np.asarray(slam_pose if slam_pose is not None else pos, dtype=np.float32)
    heading_source = float(slam_heading if slam_heading is not None else heading)
    ref_dist = max(float(reference_distance), 1e-6)

    if slam_cov_diag is None:
        cov_diag = np.zeros(3, dtype=np.float32)
    else:
        cov_diag = np.asarray(slam_cov_diag, dtype=np.float32)
        if cov_diag.size < 3:
            cov_diag = np.pad(cov_diag, (0, 3 - cov_diag.size), mode="constant", constant_values=0.0)
        else:
            cov_diag = cov_diag[:3]

    cov_trace = float(slam_cov_trace if slam_cov_trace is not None else float(np.sum(cov_diag)))

    slam_features = np.array(
        [
            pose_source[0] / ref_dist,
            pose_source[1] / ref_dist,
            np.sin(heading_source),
            np.cos(heading_source),
            cov_diag[0],
            cov_diag[1],
            cov_diag[2],
            cov_trace,
            np.log1p(max(0, int(slam_keyframe_count))) / 10.0,
            np.log1p(max(0, int(slam_landmark_count))) / 10.0,
        ],
        dtype=np.float32,
    )

    return np.concatenate([lidar, direction_features, slam_features])
