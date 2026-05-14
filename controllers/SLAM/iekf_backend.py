# Iterated Extended Kalman Filter for 2-D LiDAR-IMU SLAM.
#
# State vector: x = [px, py, θ, vx, vy, bωz, bax, bay]  (8-D)
#   px, py  — position in global frame
#   θ       — heading (yaw)
#   vx, vy  — velocity in global frame
#   bωz     — gyroscope z-bias
#   bax,bay — accelerometer x/y bias

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


STATE_DIM = 8


@dataclass
class IEKFState:
    x: np.ndarray   # (8,) mean state
    P: np.ndarray   # (8,8) covariance
    timestamp: float = 0.0

    @property
    def position(self) -> np.ndarray:
        return self.x[:2].copy()

    @property
    def heading(self) -> float:
        return float(self.x[2])


def _rot2(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _wrap_angle(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


class IEKFBackend:
    """
    2-D LiDAR-IMU IEKF.

    Per-timestep usage:
        1. iekf.propagate_odom(speed, gyro_z, dt)   — dead-reckon position/heading
        2. iekf.update(edge_pts, planar_pts, landmarks)  — correct with scan+CNN
        3. pose = iekf.state.position, iekf.state.heading
    """

    MAX_ITER = 5
    CONVERGE_THR = 1e-4

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
        sigma_lidar: float = 0.02,
        sigma_landmark: float = 0.08,
    ) -> None:
        x0 = np.zeros(STATE_DIM)
        x0[:2] = init_pos
        x0[2] = init_heading
        P0 = np.diag([
            sigma_pos**2, sigma_pos**2,
            sigma_heading**2,
            sigma_vel**2, sigma_vel**2,
            sigma_gyro_bias**2,
            sigma_accel_bias**2, sigma_accel_bias**2,
        ])
        self.state = IEKFState(x=x0, P=P0)
        self._sigma_gyro = sigma_gyro
        self._sigma_accel = sigma_accel
        self._R_lidar = sigma_lidar ** 2
        self._R_landmark = np.eye(2) * sigma_landmark ** 2
        self._line_map: List[dict] = []

    # ── Propagation ───────────────────────────────────────────────────────────

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

        F = np.eye(STATE_DIM)
        F[0, 2] = -speed * np.sin(θ) * dt
        F[1, 2] =  speed * np.cos(θ) * dt
        F[2, 5] = -dt
        F[3, 2] = -speed * np.sin(θ)
        F[4, 2] =  speed * np.cos(θ)

        self.state.x = x
        self.state.P = F @ self.state.P @ F.T + self._build_process_noise(dt)

    # ── Measurement update ────────────────────────────────────────────────────

    def update(
        self,
        edge_points: np.ndarray,
        planar_points: np.ndarray,
        semantic_landmarks: List[Tuple[float, float, float]],
    ) -> None:
        """IEKF iterative update fusing LiDAR geometry + CNN semantic residuals."""
        if len(self._line_map) == 0:
            self._init_map_from_scan(edge_points, planar_points)
            return

        x_hat = self.state.x.copy()
        P_hat = self.state.P.copy()
        K = None

        for _ in range(self.MAX_ITER):
            z_geo, H_geo, R_geo = self._geometric_residuals(x_hat, edge_points, planar_points)
            z_sem, H_sem, R_sem = self._semantic_residuals(x_hat, semantic_landmarks)

            if z_geo is None and z_sem is None:
                break

            z_list, H_list, R_list = [], [], []
            if z_geo is not None:
                z_list.append(z_geo); H_list.append(H_geo); R_list.append(R_geo)
            if z_sem is not None:
                z_list.append(z_sem); H_list.append(H_sem); R_list.append(R_sem)

            z = np.concatenate(z_list)
            H = np.vstack(H_list)
            R = np.block([[R_list[i] if i == j else np.zeros((R_list[i].shape[0], R_list[j].shape[0]))
                           for j in range(len(R_list))]
                          for i in range(len(R_list))])

            S = H @ P_hat @ H.T + R
            K = P_hat @ H.T @ np.linalg.solve(S.T, np.eye(S.shape[0])).T
            x_error = x_hat - self.state.x
            delta = -K @ (z + H @ x_error)
            x_hat = x_hat + delta
            x_hat[2] = _wrap_angle(x_hat[2])
            if np.linalg.norm(delta) < self.CONVERGE_THR:
                break

        if K is not None:
            self.state.P = (np.eye(STATE_DIM) - K @ H) @ P_hat
        self.state.x = x_hat
        self._update_map(x_hat, edge_points, planar_points)

    # ── Geometric residuals ───────────────────────────────────────────────────

    def _geometric_residuals(
        self,
        x_hat: np.ndarray,
        edge_pts: np.ndarray,
        planar_pts: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        all_pts = (np.vstack([edge_pts, planar_pts])
                   if len(edge_pts) > 0 and len(planar_pts) > 0
                   else (edge_pts if len(edge_pts) > 0 else planar_pts))
        if len(all_pts) == 0 or len(self._line_map) == 0:
            return None, None, None

        θ = x_hat[2]
        R = _rot2(θ)
        t = x_hat[:2]
        pts_w = (R @ all_pts.T).T + t

        z_list, H_list = [], []
        for i, p_w in enumerate(pts_w):
            best_line, _ = self._nearest_line(p_w)
            if best_line is None:
                continue
            mu = best_line["normal"]
            q = best_line["p"]
            z_list.append(float(mu @ (p_w - q)))
            dR_dθ = np.array([[-np.sin(θ), -np.cos(θ)], [np.cos(θ), -np.sin(θ)]])
            H_j = np.zeros(STATE_DIM)
            H_j[0] = mu[0]
            H_j[1] = mu[1]
            H_j[2] = float(mu @ (dR_dθ @ all_pts[i]))
            H_list.append(H_j)

        if not z_list:
            return None, None, None
        return (np.array(z_list), np.array(H_list),
                np.eye(len(z_list)) * self._R_lidar)

    # ── Semantic residuals ────────────────────────────────────────────────────

    def _semantic_residuals(
        self,
        x_hat: np.ndarray,
        landmarks_lidar: List[Tuple[float, float, float]],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        if not hasattr(self, '_landmark_map'):
            self._landmark_map: List[dict] = []

        θ = x_hat[2]
        R = _rot2(θ)
        t = x_hat[:2]
        dR_dθ = np.array([[-np.sin(θ), -np.cos(θ)], [np.cos(θ), -np.sin(θ)]])

        z_list, H_list = [], []
        for cx, cy, r in landmarks_lidar:
            if cx == 0.0 and cy == 0.0:
                continue
            p_L = np.array([cx, cy])
            p_W = R @ p_L + t
            match = self._associate_landmark(p_W, r)
            if match is None:
                self._landmark_map.append({"p": p_W.copy(), "r": r, "count": 1})
                continue
            z_list.append(p_W - match["p"])
            H_sem = np.zeros((2, STATE_DIM))
            H_sem[:, 0] = [1.0, 0.0]
            H_sem[:, 1] = [0.0, 1.0]
            H_sem[:, 2] = dR_dθ @ p_L
            H_list.append(H_sem)
            n = match["count"]
            match["p"] = (match["p"] * n + p_W) / (n + 1)
            match["count"] += 1

        if not z_list:
            return None, None, None
        z = np.concatenate(z_list)
        H = np.vstack(H_list)
        R_mat = np.kron(np.eye(len(z_list)), self._R_landmark)
        return z, H, R_mat

    # ── Map management ────────────────────────────────────────────────────────

    def _init_map_from_scan(self, edge_pts: np.ndarray, planar_pts: np.ndarray) -> None:
        self._update_map(self.state.x, edge_pts, planar_pts)

    def _update_map(
        self,
        x: np.ndarray,
        edge_pts: np.ndarray,
        planar_pts: np.ndarray,
        min_dist: float = 0.3,
    ) -> None:
        all_pts = (np.vstack([edge_pts, planar_pts])
                   if len(edge_pts) > 0 and len(planar_pts) > 0
                   else (edge_pts if len(edge_pts) > 0 else planar_pts))
        if len(all_pts) < 2:
            return
        pts_w = (_rot2(x[2]) @ all_pts.T).T + x[:2]
        for i in range(len(pts_w) - 1):
            p1, p2 = pts_w[i], pts_w[i + 1]
            seg_len = np.linalg.norm(p2 - p1)
            if seg_len < 0.01:
                continue
            mid = (p1 + p2) / 2.0
            direction = (p2 - p1) / seg_len
            normal = np.array([-direction[1], direction[0]])
            if self._nearest_line(mid)[1] > min_dist:
                self._line_map.append({"p": mid.copy(), "normal": normal.copy()})

    def _nearest_line(self, point: np.ndarray) -> Tuple[Optional[dict], float]:
        best_line, best_d = None, float("inf")
        for line in self._line_map:
            d = abs(float(line["normal"] @ (point - line["p"])))
            if d < best_d:
                best_d = d
                best_line = line
        return best_line, best_d

    def _associate_landmark(
        self, p_W: np.ndarray, r: float, tau_p: float = 0.5, tau_r: float = 0.2,
    ) -> Optional[dict]:
        for lm in self._landmark_map:
            if np.linalg.norm(lm["p"] - p_W) < tau_p and abs(lm["r"] - r) < tau_r:
                return lm
        return None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_process_noise(self, dt: float) -> np.ndarray:
        q_pos = (self._sigma_accel * dt**2) ** 2
        q_head = (self._sigma_gyro * dt) ** 2
        q_vel = (self._sigma_accel * dt) ** 2
        return np.diag([q_pos, q_pos, q_head, q_vel, q_vel, 1e-6, 1e-6, 1e-6])
