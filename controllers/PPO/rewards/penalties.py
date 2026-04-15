"""Additional reward and safety penalty helpers."""


def calculate_clearance_penalty(
    min_lidar_norm: float,
    clearance_target_norm: float,
    clearance_penalty_scale: float,
) -> float:
    """Return a negative penalty when obstacle clearance is below target."""
    if min_lidar_norm >= clearance_target_norm:
        return 0.0

    clearance_ratio = (
        (clearance_target_norm - min_lidar_norm)
        / max(clearance_target_norm, 1e-6)
    )
    return -clearance_penalty_scale * clearance_ratio
