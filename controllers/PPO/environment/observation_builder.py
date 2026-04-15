"""Pure geometry and observation construction helpers."""

from typing import Tuple

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

    return np.concatenate([lidar, direction_features])
