# IEKF odometry: dead-reckons position and heading from wheel speed + gyro.
#
# State vector: [px, py, θ, vx, vy, bωz, bax, bay]
#   px, py — position in global frame
#   θ      — heading (yaw)
#   vx, vy — velocity in global frame
#   bωz    — gyroscope z-bias
#   bax, bay — accelerometer x/y bias

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


STATE_DIM = 8
STATE_DIM = 8


@dataclass
class IEKFState:
    x: np.ndarray   # (8,) mean state
    P: np.ndarray   # (8,8) covariance
    x: np.ndarray   # (8,) mean state
    P: np.ndarray   # (8,8) covariance
    timestamp: float = 0.0

    @property
    def position(self) -> np.ndarray:
        return self.x[:2].copy()

    @property
    def heading(self) -> float:
        return float(self.x[2])


def _wrap_angle(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


class IEKFBackend:
    """Dead-reckons position and heading from wheel odometry and gyro bias correction."""

    def __init__(
        self,
        init_pos: np.ndarray = np.zeros(2),
        init_heading: float = 0.0,
        sigma_pos: float = 0.05,
        sigma_heading: float = 0.02,
        sigma_vel: float = 0.1,
        sigma_gyro_bias: float = 0.005,
        sigma_accel_bias: float = 0.01,
        sigma_gyro: float = 0.01,
        sigma_accel: float = 0.05,
    ) -> None:
        x0 = np.zeros(STATE_DIM)
        x0[:2] = init_pos
        x0[:2] = init_pos
        x0[2] = init_heading
        P0 = np.diag([
            sigma_pos**2, sigma_pos**2,
            sigma_heading**2,
            sigma_vel**2, sigma_vel**2,
            sigma_gyro_bias**2,
            sigma_accel_bias**2, sigma_accel_bias**2,
            sigma_pos**2, sigma_pos**2,
            sigma_heading**2,
            sigma_vel**2, sigma_vel**2,
            sigma_gyro_bias**2,
            sigma_accel_bias**2, sigma_accel_bias**2,
        ])
        self.state = IEKFState(x=x0, P=P0)
        self._sigma_gyro = sigma_gyro
        self._sigma_accel = sigma_accel

    def propagate_odom(self, speed: float, gyro_z: float, dt: float) -> None:
        """Dead-reckon position and heading from wheel speed and gyro yaw rate."""
        x = self.state.x.copy()
        θ = x[2]
        bωz = x[5]
        ω = gyro_z - bωz

        x[0] += speed * np.cos(θ) * dt
        x[1] += speed * np.sin(θ) * dt
        x[2] = _wrap_angle(θ + ω * dt)
        x[3] = speed * np.cos(θ)
        x[4] = speed * np.sin(θ)

        x[0] += speed * np.cos(θ) * dt
        x[1] += speed * np.sin(θ) * dt
        x[2] = _wrap_angle(θ + ω * dt)
        x[3] = speed * np.cos(θ)
        x[4] = speed * np.sin(θ)

        F = np.eye(STATE_DIM)
        F[0, 2] = -speed * np.sin(θ) * dt
        F[1, 2] =  speed * np.cos(θ) * dt
        F[2, 5] = -dt
        F[2, 5] = -dt
        F[3, 2] = -speed * np.sin(θ)
        F[4, 2] =  speed * np.cos(θ)

        self.state.x = x
        self.state.P = F @ self.state.P @ F.T + self._build_process_noise(dt)

    def _build_process_noise(self, dt: float) -> np.ndarray:
        q_pos = (self._sigma_accel * dt**2) ** 2
        q_pos = (self._sigma_accel * dt**2) ** 2
        q_head = (self._sigma_gyro * dt) ** 2
        q_vel = (self._sigma_accel * dt) ** 2
        return np.diag([q_pos, q_pos, q_head, q_vel, q_vel, 1e-6, 1e-6, 1e-6])
        return np.diag([q_pos, q_pos, q_head, q_vel, q_vel, 1e-6, 1e-6, 1e-6])
