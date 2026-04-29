"""
IMU filtering module for CNN-LiDAR-SLAM.

Implements the hybrid Madgwick + EKF pipeline described in Section III-C-2.

Madgwick filter:  low computational overhead; provides real-time orientation
                  quaternion from gyroscope, accelerometer (+ optional magnetometer).
EKF wrapper:      refines state estimate and mitigates sensor drift / noise.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IMUState:
    """Filtered IMU state at a single timestep."""
    quaternion: np.ndarray   # (4,)  [w, x, y, z]  orientation
    accel_body: np.ndarray   # (3,)  accelerometer in body frame (m/s²)
    gyro_body: np.ndarray    # (3,)  angular velocity in body frame (rad/s)
    accel_world: np.ndarray  # (3,)  gravity-subtracted acceleration in world frame


# ─────────────────────────────────────────────────────────────────────────────
# Madgwick orientation filter
# ─────────────────────────────────────────────────────────────────────────────

class MadgwickFilter:
    """
    Madgwick orientation filter [42].

    Fuses gyroscope and accelerometer (and optional magnetometer) to
    estimate the orientation quaternion q = [w, x, y, z] in real time
    with low computational cost.
    """

    def __init__(self, beta: float = 0.1, dt: float = 0.032) -> None:
        """
        Args:
            beta: Algorithm gain (trade-off between gyro integration and
                  accelerometer correction).  Typical range 0.01–0.5.
            dt:   Sensor sample period in seconds.
        """
        self.beta = beta
        self.dt = dt
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def reset(self) -> None:
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def update(
        self,
        gyro: np.ndarray,
        accel: np.ndarray,
        mag: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Propagate orientation estimate by one step.

        Args:
            gyro:  (3,) angular velocity [gx, gy, gz] in rad/s.
            accel: (3,) acceleration     [ax, ay, az] in m/s².
            mag:   (3,) magnetometer     [mx, my, mz] in µT  (optional).

        Returns:
            (4,) updated orientation quaternion [w, x, y, z].
        """
        q = self.q
        gx, gy, gz = gyro.astype(float)

        # ── Normalise accelerometer ──────────────────────────────────────────
        a_norm = np.linalg.norm(accel)
        if a_norm < 1e-6:
            # Degenerate reading – integrate gyro only
            q = q + 0.5 * self.dt * self._quat_product(q, np.array([0, gx, gy, gz]))
            self.q = q / np.linalg.norm(q)
            return self.q.copy()
        ax, ay, az = accel / a_norm

        # ── Gradient step from accelerometer ────────────────────────────────
        # Objective function: f = R(q)^T [0,0,1] - [ax,ay,az]
        qw, qx, qy, qz = q
        f1 = 2.0 * (qx * qz - qw * qy) - ax
        f2 = 2.0 * (qw * qx + qy * qz) - ay
        f3 = 1.0 - 2.0 * (qx * qx + qy * qy) - az

        # Jacobian J^T (transposed)  [4×3]
        J = np.array([
            [-2.0 * qy,  2.0 * qx,  0.0],
            [ 2.0 * qz,  2.0 * qw, -4.0 * qx],
            [-2.0 * qw,  2.0 * qz, -4.0 * qy],
            [ 2.0 * qx,  2.0 * qy,  0.0],
        ])
        step = J @ np.array([f1, f2, f3])

        # ── Add magnetometer correction if available ─────────────────────────
        if mag is not None:
            m_norm = np.linalg.norm(mag)
            if m_norm > 1e-6:
                mx, my, mz = mag / m_norm
                # Reference direction of earth's magnetic field
                h = self._quat_rotate(q, np.array([mx, my, mz]))
                bx = float(np.sqrt(h[0] ** 2 + h[1] ** 2))
                bz = float(h[2])

                f4 = 2.0 * bx * (0.5 - qy * qy - qz * qz) + 2.0 * bz * (qx * qz - qw * qy) - mx
                f5 = 2.0 * bx * (qx * qy - qw * qz) + 2.0 * bz * (qw * qx + qy * qz) - my
                f6 = 2.0 * bx * (qw * qy + qx * qz) + 2.0 * bz * (0.5 - qx * qx - qy * qy) - mz

                Jm = np.array([
                    [-2.0 * bz * qy,  2.0 * bz * qz, -4.0 * bx * qy - 2.0 * bz * qw,  -4.0 * bx * qz + 2.0 * bz * qx],
                    [-2.0 * bx * qz + 2.0 * bz * qx,  2.0 * bx * qy + 2.0 * bz * qw,  2.0 * bx * qx + 2.0 * bz * qz,  -2.0 * bx * qw + 2.0 * bz * qy],
                    [ 2.0 * bx * qy,  2.0 * bx * qz - 4.0 * bz * qx,  2.0 * bx * qw - 4.0 * bz * qy,  2.0 * bx * qx],
                ]).T
                step = step + Jm @ np.array([f4, f5, f6])

        step_norm = np.linalg.norm(step)
        if step_norm > 1e-10:
            step /= step_norm

        # ── Gyroscope quaternion derivative ─────────────────────────────────
        q_dot = (0.5 * self._quat_product(q, np.array([0.0, gx, gy, gz]))
                 - self.beta * step)

        # ── Integrate ────────────────────────────────────────────────────────
        q = q + q_dot * self.dt
        self.q = q / np.linalg.norm(q)
        return self.q.copy()

    # ── Quaternion helpers ───────────────────────────────────────────────────

    @staticmethod
    def _quat_product(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Hamilton product q1 ⊗ q2."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ])

    @staticmethod
    def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Rotate vector v by quaternion q:  v' = q ⊗ [0,v] ⊗ q*."""
        qw, qx, qy, qz = q
        # Using the formula directly avoids full matrix construction
        t = 2.0 * np.cross([qx, qy, qz], v)
        return v + qw * t + np.cross([qx, qy, qz], t)


# ─────────────────────────────────────────────────────────────────────────────
# EKF wrapper for IMU state refinement
# ─────────────────────────────────────────────────────────────────────────────

class IMUEKF:
    """
    Extended Kalman Filter that refines the Madgwick orientation and
    estimates accelerometer / gyroscope biases.

    State vector x = [qw, qx, qy, qz, b_gx, b_gy, b_gz]  (7-D).
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

        # State: quaternion + gyro bias
        self.x = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        # Covariance
        self.P = np.eye(self.DIM) * 0.01

        # Process noise
        q_quat = (sigma_gyro * dt) ** 2
        q_bias = (sigma_bias * dt) ** 2
        self.Q = np.diag([q_quat] * 4 + [q_bias] * 3)

        # Measurement noise (accelerometer)
        self.R_acc = np.eye(3) * sigma_accel ** 2

    def predict(self, gyro_raw: np.ndarray) -> None:
        """Propagate state using gyroscope measurement."""
        q = self.x[:4]
        b = self.x[4:]
        gyro_corrected = gyro_raw - b

        gx, gy, gz = gyro_corrected
        # Quaternion kinematics: q_dot = 0.5 * Omega(ω) * q
        Omega = 0.5 * np.array([
            [0,  -gx, -gy, -gz],
            [gx,  0,   gz, -gy],
            [gy, -gz,  0,   gx],
            [gz,  gy, -gx,  0 ],
        ])
        F = np.eye(self.DIM)
        F[:4, :4] += Omega * self.dt
        # Jacobian for bias terms
        q_n = q / (np.linalg.norm(q) + 1e-10)
        qw, qx, qy, qz = q_n
        dqdB = 0.5 * self.dt * np.array([
            [ qx,  qy,  qz],
            [-qw,  qz, -qy],
            [-qz, -qw,  qx],
            [ qy, -qx, -qw],
        ])
        F[:4, 4:] = dqdB

        self.x[:4] = (np.eye(4) + Omega * self.dt) @ q
        self.x[:4] /= np.linalg.norm(self.x[:4]) + 1e-10

        self.P = F @ self.P @ F.T + self.Q

    def update(self, accel_raw: np.ndarray) -> None:
        """Correct state using accelerometer (gravity direction)."""
        a_norm = np.linalg.norm(accel_raw)
        if a_norm < 0.1:
            return
        a_meas = accel_raw / a_norm  # unit gravity direction in body frame

        q = self.x[:4] / (np.linalg.norm(self.x[:4]) + 1e-10)
        qw, qx, qy, qz = q

        # Predicted gravity direction in body frame: R^T * [0,0,1]
        h = np.array([
            2.0 * (qx * qz - qw * qy),
            2.0 * (qw * qx + qy * qz),
            qw * qw - qx * qx - qy * qy + qz * qz,
        ])
        # Jacobian of h w.r.t. quaternion (3×4)
        H_q = 2.0 * np.array([
            [-qy,  qz, -qw,  qx],
            [ qx,  qw,  qz,  qy],
            [ qw, -qx, -qy,  qz],
        ])
        H = np.zeros((3, self.DIM))
        H[:, :4] = H_q

        innov = a_meas - h
        S = H @ self.P @ H.T + self.R_acc
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ innov
        self.x[:4] /= np.linalg.norm(self.x[:4]) + 1e-10
        self.P = (np.eye(self.DIM) - K @ H) @ self.P

    @property
    def quaternion(self) -> np.ndarray:
        return self.x[:4].copy()

    @property
    def gyro_bias(self) -> np.ndarray:
        return self.x[4:].copy()


# ─────────────────────────────────────────────────────────────────────────────
# Combined pipeline
# ─────────────────────────────────────────────────────────────────────────────

# Optional import for type hint only
from typing import Optional


class IMUProcessor:
    """
    Full IMU pipeline: Madgwick pre-filter → EKF refinement (Section III-C-2).

    The Madgwick filter provides low-latency orientation estimates; the EKF
    refines them and tracks gyroscope bias.  The pipeline outputs a filtered
    IMUState for each incoming sensor reading.
    """

    GRAVITY = 9.81  # m/s²

    def __init__(self, dt: float = 0.032, madgwick_beta: float = 0.1) -> None:
        self.dt = dt
        self.madgwick = MadgwickFilter(beta=madgwick_beta, dt=dt)
        self.ekf = IMUEKF(dt=dt)

    def reset(self) -> None:
        self.madgwick.reset()
        self.ekf = IMUEKF(dt=self.dt)

    def step(
        self,
        gyro: np.ndarray,
        accel: np.ndarray,
        mag: Optional[np.ndarray] = None,
    ) -> IMUState:
        """
        Process one IMU sample.

        Args:
            gyro:  (3,) angular velocity in rad/s.
            accel: (3,) accelerometer reading in m/s².
            mag:   (3,) magnetometer reading (optional).

        Returns:
            Filtered IMUState.
        """
        # ── Madgwick orientation ─────────────────────────────────────────────
        q_madgwick = self.madgwick.update(gyro, accel, mag)

        # ── EKF predict + update ─────────────────────────────────────────────
        self.ekf.predict(gyro)
        self.ekf.update(accel)

        # Blend: use EKF quaternion as the refined estimate
        q = self.ekf.quaternion

        # ── Gravity compensation in world frame ──────────────────────────────
        # Rotate accelerometer reading to world frame and subtract gravity
        a_world = self._rotate_vector(q, accel)
        a_world[2] -= self.GRAVITY  # subtract gravity (assumed along +z)

        return IMUState(
            quaternion=q,
            accel_body=accel.astype(np.float32),
            gyro_body=gyro.astype(np.float32),
            accel_world=a_world.astype(np.float32),
        )

    @staticmethod
    def _rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Rotate vector v from body to world frame using quaternion q."""
        qw, qx, qy, qz = q
        # Rotation matrix from quaternion
        R = np.array([
            [1 - 2*(qy*qy + qz*qz),   2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz),        1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy),        2*(qy*qz + qw*qx),     1 - 2*(qx*qx + qy*qy)],
        ])
        return R @ v
