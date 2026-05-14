# IMU filtering: EKF-based orientation estimation with gyroscope bias tracking.

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class IMUState:
    quaternion: np.ndarray   # (4,)  [w, x, y, z]
    accel_body: np.ndarray   # (3,)  m/s² in body frame
    gyro_body: np.ndarray    # (3,)  rad/s in body frame
    accel_world: np.ndarray  # (3,)  gravity-subtracted acceleration in world frame


class IMUEKF:
    """
    7-state EKF: quaternion [w,x,y,z] + gyroscope bias [bx,by,bz].

    Predict step integrates gyro; update step corrects using accelerometer
    as a gravity-direction measurement.
    """

    DIM = 7

    def __init__(
        self,
        dt: float = 0.032,
        sigma_gyro: float = 0.01,
        sigma_accel: float = 0.1,
        sigma_bias: float = 0.001,
    ) -> None:
        self.dt = dt
        self.x = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.P = np.eye(self.DIM) * 0.01
        q_quat = (sigma_gyro * dt) ** 2
        q_bias = (sigma_bias * dt) ** 2
        self.Q = np.diag([q_quat] * 4 + [q_bias] * 3)
        self.R_acc = np.eye(3) * sigma_accel ** 2

    def reset(self) -> None:
        self.x = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.P = np.eye(self.DIM) * 0.01

    def predict(self, gyro_raw: np.ndarray) -> None:
        q, b = self.x[:4], self.x[4:]
        gx, gy, gz = gyro_raw - b
        # Quaternion kinematics: q_dot = 0.5 * Omega(ω) * q
        Omega = 0.5 * np.array([
            [0,   -gx, -gy, -gz],
            [gx,   0,   gz, -gy],
            [gy,  -gz,  0,   gx],
            [gz,   gy, -gx,  0 ],
        ])
        F = np.eye(self.DIM)
        F[:4, :4] += Omega * self.dt
        # Jacobian of q w.r.t. gyro bias
        q_n = q / (np.linalg.norm(q) + 1e-10)
        qw, qx, qy, qz = q_n
        F[:4, 4:] = 0.5 * self.dt * np.array([
            [ qx,  qy,  qz],
            [-qw,  qz, -qy],
            [-qz, -qw,  qx],
            [ qy, -qx, -qw],
        ])
        self.x[:4] = (np.eye(4) + Omega * self.dt) @ q
        self.x[:4] /= np.linalg.norm(self.x[:4]) + 1e-10
        self.P = F @ self.P @ F.T + self.Q

    def update(self, accel_raw: np.ndarray) -> None:
        a_norm = np.linalg.norm(accel_raw)
        if a_norm < 0.1:
            return
        a_meas = accel_raw / a_norm
        q = self.x[:4] / (np.linalg.norm(self.x[:4]) + 1e-10)
        qw, qx, qy, qz = q
        # Predicted gravity direction in body frame: R(q)^T [0,0,1]
        h = np.array([
            2.0 * (qx * qz - qw * qy),
            2.0 * (qw * qx + qy * qz),
            qw*qw - qx*qx - qy*qy + qz*qz,
        ])
        H_q = 2.0 * np.array([
            [-qy,  qz, -qw,  qx],
            [ qx,  qw,  qz,  qy],
            [ qw, -qx, -qy,  qz],
        ])
        H = np.zeros((3, self.DIM))
        H[:, :4] = H_q
        S = H @ self.P @ H.T + self.R_acc
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x += K @ (a_meas - h)
        self.x[:4] /= np.linalg.norm(self.x[:4]) + 1e-10
        self.P = (np.eye(self.DIM) - K @ H) @ self.P

    @property
    def quaternion(self) -> np.ndarray:
        return self.x[:4].copy()


class IMUProcessor:
    """EKF-based IMU pipeline: predict from gyro, correct with accelerometer."""

    GRAVITY = 9.81

    def __init__(self, dt: float = 0.032) -> None:
        self.dt = dt
        self.ekf = IMUEKF(dt=dt)

    def reset(self) -> None:
        self.ekf.reset()

    def step(self, gyro: np.ndarray, accel: Optional[np.ndarray] = None) -> IMUState:
        if accel is None:
            accel = np.zeros(3, dtype=np.float32)
        self.ekf.predict(gyro)
        self.ekf.update(accel)
        q = self.ekf.quaternion
        a_world = self._rotate_vector(q, accel)
        a_world[2] -= self.GRAVITY
        return IMUState(
            quaternion=q,
            accel_body=accel.astype(np.float32),
            gyro_body=gyro.astype(np.float32),
            accel_world=a_world.astype(np.float32),
        )

    @staticmethod
    def _rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        qw, qx, qy, qz = q
        R = np.array([
            [1 - 2*(qy*qy + qz*qz),  2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz),       1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy),       2*(qy*qz + qw*qx),     1 - 2*(qx*qx + qy*qy)],
        ])
        return R @ v
