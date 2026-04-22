"""
Iterated Extended Kalman Filter (IEKF) backend for CNN-LiDAR-SLAM.

Implements tightly-coupled LiDAR–IMU fusion with CNN-derived semantic
landmarks as described in Sections III-C-3 through III-E of the paper.

State vector (2-D indoor SLAM):
    x = [px, py, θ, vx, vy, bωz, bax, bay]   (8-D)
    where:
        (px, py) – position in global frame
        θ        – heading angle (yaw)
        (vx, vy) – velocity in global frame
        bωz      – gyroscope z-bias
        (bax, bay) – accelerometer x/y bias

Key equations implemented:
    (2)  State vector definition
    (3)  Linearised measurement model  z ≈ H x̃ + v
    (4)  Point projection to global frame
    (5)  Residual: z = μ^T (p̂ − q)
    (7)–(13) Iterative state update (MAP minimisation, Kalman gain)
    (20)–(23) Semantic residual fusion
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

STATE_DIM = 8   # [px, py, θ, vx, vy, bωz, bax, bay]

@dataclass
class IEKFState:
    """Full IEKF state at one timestep."""
    x: np.ndarray          # (8,)  mean state vector
    P: np.ndarray          # (8,8) covariance matrix
    timestamp: float = 0.0

    @property
    def position(self) -> np.ndarray:
        return self.x[:2].copy()

    @property
    def heading(self) -> float:
        return float(self.x[2])

    @property
    def velocity(self) -> np.ndarray:
        return self.x[3:5].copy()

    def pose_matrix(self) -> np.ndarray:
        """Return 3×3 homogeneous transform [R|t; 0 1] for 2-D."""
        θ = self.x[2]
        c, s = np.cos(θ), np.sin(θ)
        T = np.array([
            [c, -s, self.x[0]],
            [s,  c, self.x[1]],
            [0,  0,       1.0],
        ])
        return T


# ─────────────────────────────────────────────────────────────────────────────
# Helper: 2-D rotation matrix
# ─────────────────────────────────────────────────────────────────────────────

def _rot2(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _wrap_angle(a: float) -> float:
    """Wrap angle to (−π, π]."""
    return float((a + np.pi) % (2 * np.pi) - np.pi)


# ─────────────────────────────────────────────────────────────────────────────
# IEKF core
# ─────────────────────────────────────────────────────────────────────────────

class IEKFBackend:
    """
    Tightly coupled 2-D LiDAR–IMU IEKF following FAST-LIO2 conventions,
    augmented with CNN semantic landmark residuals (paper Sections III-C-3,
    III-E).

    Usage per timestep:
        1. iekf.propagate(gyro_z, accel_xy, dt)   – IMU forward propagation
        2. iekf.update(edge_pts, planar_pts, landmarks)  – measurement update
        3. pose = iekf.state.pose_matrix()
    """

    MAX_ITER = 5          # IEKF inner iterations
    CONVERGE_THR = 1e-4   # |Δx| convergence threshold

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
        # ── Initial state ────────────────────────────────────────────────────
        x0 = np.zeros(STATE_DIM)
        x0[0:2] = init_pos
        x0[2] = init_heading

        P0 = np.diag([
            sigma_pos ** 2, sigma_pos ** 2,
            sigma_heading ** 2,
            sigma_vel ** 2, sigma_vel ** 2,
            sigma_gyro_bias ** 2,
            sigma_accel_bias ** 2, sigma_accel_bias ** 2,
        ])

        self.state = IEKFState(x=x0, P=P0)

        # ── Process noise ────────────────────────────────────────────────────
        self._sigma_gyro = sigma_gyro
        self._sigma_accel = sigma_accel

        # ── Measurement noise ────────────────────────────────────────────────
        self._R_lidar = sigma_lidar ** 2
        self._R_landmark = np.eye(2) * sigma_landmark ** 2

        # ── Global line map  (list of (p, normal) line primitives) ───────────
        # Each entry: {"p": (2,), "normal": (2,)}
        self._line_map: List[dict] = []

    # ─────────────────────────────────────────────────────────────────────────
    # 1.  IMU forward propagation
    # ─────────────────────────────────────────────────────────────────────────

    def propagate(
        self,
        gyro_z: float,
        accel_xy: np.ndarray,
        dt: float,
    ) -> None:
        """
        Forward propagation using IMU measurements (Section III-C-3,
        propagation stage).

        Integrates the state using a constant-velocity + heading model with
        IMU-derived acceleration.
        """
        x = self.state.x.copy()
        θ = x[2]
        vx, vy = x[3], x[4]
        bωz = x[5]
        ba = x[6:8]

        # Corrected measurements
        ω = gyro_z - bωz
        a_body = accel_xy - ba
        a_world = _rot2(θ) @ a_body

        # State prediction
        x_new = x.copy()
        x_new[0] += vx * dt + 0.5 * a_world[0] * dt ** 2
        x_new[1] += vy * dt + 0.5 * a_world[1] * dt ** 2
        x_new[2] = _wrap_angle(θ + ω * dt)
        x_new[3] += a_world[0] * dt
        x_new[4] += a_world[1] * dt
        # biases modelled as random walk → unchanged

        # Jacobian F = ∂f/∂x  (8×8)
        F = np.eye(STATE_DIM)
        # ∂pos / ∂vel
        F[0, 3] = dt
        F[1, 4] = dt
        # ∂pos / ∂heading (through acceleration rotation)
        dR_dθ = np.array([[-np.sin(θ), -np.cos(θ)],
                          [ np.cos(θ), -np.sin(θ)]])
        da_dθ = dR_dθ @ a_body
        F[0, 2] = 0.5 * da_dθ[0] * dt ** 2
        F[1, 2] = 0.5 * da_dθ[1] * dt ** 2
        F[3, 2] = da_dθ[0] * dt
        F[4, 2] = da_dθ[1] * dt
        # ∂pos / ∂acc_bias  (via a_world)
        F[0, 6] = -0.5 * np.cos(θ) * dt ** 2
        F[0, 7] = 0.5 * np.sin(θ) * dt ** 2
        F[1, 6] = -0.5 * np.sin(θ) * dt ** 2
        F[1, 7] = -0.5 * np.cos(θ) * dt ** 2
        F[3, 6] = -np.cos(θ) * dt
        F[3, 7] = np.sin(θ) * dt
        F[4, 6] = -np.sin(θ) * dt
        F[4, 7] = -np.cos(θ) * dt
        # ∂heading / ∂gyro_bias
        F[2, 5] = -dt

        # Process noise Q
        Q = self._build_process_noise(dt)

        self.state.x = x_new
        self.state.P = F @ self.state.P @ F.T + Q

    def propagate_odom(
        self,
        speed: float,
        gyro_z: float,
        dt: float,
    ) -> None:
        """
        Wheel-odometry propagation — far more reliable than IMU integration
        for a velocity-controlled wheeled robot.

        Uses the commanded wheel speed (rad/s → m/s via effective radius) and
        the gyroscope yaw rate to dead-reckon position and heading.

        Args:
            speed:  forward speed in m/s (positive = forward).
            gyro_z: yaw rate in rad/s from gyroscope.
            dt:     timestep in seconds.
        """
        x = self.state.x.copy()
        θ = x[2]
        bωz = x[5]

        ω = gyro_z - bωz  # bias-corrected yaw rate

        x_new = x.copy()
        x_new[0] += speed * np.cos(θ) * dt
        x_new[1] += speed * np.sin(θ) * dt
        x_new[2] = _wrap_angle(θ + ω * dt)
        x_new[3] = speed * np.cos(θ)   # velocity in world frame
        x_new[4] = speed * np.sin(θ)

        # Simple Jacobian for odometry model
        F = np.eye(STATE_DIM)
        F[0, 2] = -speed * np.sin(θ) * dt
        F[1, 2] =  speed * np.cos(θ) * dt
        F[2, 5] = -dt   # heading / gyro_bias
        F[3, 2] = -speed * np.sin(θ)
        F[4, 2] =  speed * np.cos(θ)

        Q = self._build_process_noise(dt)
        self.state.x = x_new
        self.state.P = F @ self.state.P @ F.T + Q

    # ─────────────────────────────────────────────────────────────────────────
    # 2.  Measurement update (IEKF)
    # ─────────────────────────────────────────────────────────────────────────

    def update(
        self,
        edge_points: np.ndarray,
        planar_points: np.ndarray,
        semantic_landmarks: List[Tuple[float, float, float]],  # (cx,cy,r) in LiDAR frame
    ) -> None:
        """
        IEKF iterative measurement update fusing geometric + semantic
        residuals (Equations 7–13, 22–23).

        Args:
            edge_points:        (N_e, 2) edge feature points in LiDAR frame.
            planar_points:      (N_p, 2) planar feature points in LiDAR frame.
            semantic_landmarks: up to 4 CNN-detected (cx, cy, r) in LiDAR frame.
        """
        if len(self._line_map) == 0:
            # Initialise map from first scan
            self._init_map_from_scan(edge_points, planar_points)
            return

        x_hat = self.state.x.copy()
        P_hat = self.state.P.copy()

        for _ in range(self.MAX_ITER):
            # ── Build residuals and Jacobians ────────────────────────────────
            z_geo, H_geo, R_geo = self._geometric_residuals(x_hat, edge_points, planar_points)
            z_sem, H_sem, R_sem = self._semantic_residuals(x_hat, semantic_landmarks)

            # Stack geometric + semantic (Eq. 22)
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

            # ── Kalman gain and update  (Eq. 11) ────────────────────────────
            S = H @ P_hat @ H.T + R
            K = P_hat @ H.T @ np.linalg.solve(S.T, np.eye(S.shape[0])).T

            # Innovation: z_k + H x̃_k  (MAP objective, Eq. 10)
            x_error = x_hat - self.state.x      # error state x̃
            delta = -K @ (z + H @ x_error)

            x_new = x_hat + delta
            x_new[2] = _wrap_angle(x_new[2])

            # Convergence check
            if np.linalg.norm(delta) < self.CONVERGE_THR:
                x_hat = x_new
                break
            x_hat = x_new

        # ── Final covariance update  (Eq. 13) ────────────────────────────────
        if 'K' in dir() and K is not None:
            self.state.P = (np.eye(STATE_DIM) - K @ H) @ P_hat

        self.state.x = x_hat

        # ── Update global map with current scan ──────────────────────────────
        self._update_map(x_hat, edge_points, planar_points)

    # ─────────────────────────────────────────────────────────────────────────
    # 3.  Geometric residuals from LiDAR scan-to-map
    # ─────────────────────────────────────────────────────────────────────────

    def _geometric_residuals(
        self,
        x_hat: np.ndarray,
        edge_pts: np.ndarray,
        planar_pts: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Compute perpendicular-distance residuals for LiDAR feature points
        (Equations 4–6).

        Each feature point is projected to the world frame (Eq. 4), then its
        residual is the signed distance to the nearest map line (Eq. 5).
        """
        all_pts = np.vstack([edge_pts, planar_pts]) if (len(edge_pts) > 0 and len(planar_pts) > 0) \
            else (edge_pts if len(edge_pts) > 0 else planar_pts)

        if len(all_pts) == 0 or len(self._line_map) == 0:
            return None, None, None

        θ = x_hat[2]
        R = _rot2(θ)
        t = x_hat[:2]

        # Eq. (4): project LiDAR points to world frame
        pts_w = (R @ all_pts.T).T + t  # (M, 2)

        z_list, H_list = [], []
        for p_w in pts_w:
            # Find nearest line in map
            best_line, best_d = self._nearest_line(p_w)
            if best_line is None:
                continue
            mu = best_line["normal"]  # unit normal of the line
            q = best_line["p"]        # a point on the line

            # Residual (Eq. 5): z = μ^T (p̂_w − q)
            z_j = float(mu @ (p_w - q))
            z_list.append(z_j)

            # Jacobian of z_j w.r.t. error state x̃ (Eq. 6):
            # ∂z/∂px = μ_x,  ∂z/∂py = μ_y
            # ∂z/∂θ  = μ^T * ∂(R p_L)/∂θ  where p_L is the LiDAR-frame point
            p_L = all_pts[len(z_list) - 1]  # original LiDAR-frame point
            dR_dθ = np.array([[-np.sin(θ), -np.cos(θ)],
                              [ np.cos(θ), -np.sin(θ)]])
            dz_dθ = float(mu @ (dR_dθ @ p_L))

            H_j = np.zeros(STATE_DIM)
            H_j[0] = mu[0]   # ∂/∂px
            H_j[1] = mu[1]   # ∂/∂py
            H_j[2] = dz_dθ   # ∂/∂θ
            H_list.append(H_j)

        if not z_list:
            return None, None, None

        z = np.array(z_list, dtype=np.float64)
        H = np.array(H_list, dtype=np.float64)
        R_mat = np.eye(len(z)) * self._R_lidar
        return z, H, R_mat

    # ─────────────────────────────────────────────────────────────────────────
    # 4.  Semantic residuals from CNN landmarks
    # ─────────────────────────────────────────────────────────────────────────

    def _semantic_residuals(
        self,
        x_hat: np.ndarray,
        landmarks_lidar: List[Tuple[float, float, float]],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Semantic residuals from CNN-detected object centroids (Eq. 20–21).

        Each detected centroid is projected to the world frame (Eq. 18).
        The residual measures displacement vs. the stored global landmark
        under the current motion hypothesis.
        """
        if not hasattr(self, '_landmark_map'):
            self._landmark_map: List[dict] = []

        θ = x_hat[2]
        R = _rot2(θ)
        t = x_hat[:2]

        z_list, H_list = [], []

        for (cx, cy, r) in landmarks_lidar:
            if cx == 0.0 and cy == 0.0:
                continue
            # Project to world frame (Eq. 18)
            p_L = np.array([cx, cy])
            p_W = R @ p_L + t

            # Data association (Eq. 19): find nearest stored landmark
            match = self._associate_landmark(p_W, r)
            if match is None:
                # New landmark – add to map
                self._landmark_map.append({"p": p_W.copy(), "r": r, "count": 1})
                continue

            p_prev = match["p"]

            # Semantic residual (Eq. 20): z_sem = p_W - p_prev
            z_sem = p_W - p_prev           # (2,)

            # Jacobian w.r.t. state error (Eq. 21): H_sem
            dR_dθ = np.array([[-np.sin(θ), -np.cos(θ)],
                              [ np.cos(θ), -np.sin(θ)]])
            H_sem = np.zeros((2, STATE_DIM))
            H_sem[:, 0] = [1.0, 0.0]          # ∂/∂px
            H_sem[:, 1] = [0.0, 1.0]          # ∂/∂py
            H_sem[:, 2] = dR_dθ @ p_L         # ∂/∂θ

            z_list.append(z_sem)
            H_list.append(H_sem)

            # Update stored position with running average
            n = match["count"]
            match["p"] = (match["p"] * n + p_W) / (n + 1)
            match["count"] += 1

        if not z_list:
            return None, None, None

        z = np.concatenate(z_list)
        H = np.vstack(H_list)
        R_mat = np.kron(np.eye(len(z_list)), self._R_landmark)
        return z, H, R_mat

    # ─────────────────────────────────────────────────────────────────────────
    # Map management
    # ─────────────────────────────────────────────────────────────────────────

    def _init_map_from_scan(self, edge_pts: np.ndarray, planar_pts: np.ndarray) -> None:
        """Seed the global line map from the first LiDAR scan."""
        self._update_map(self.state.x, edge_pts, planar_pts)

    def _update_map(
        self,
        x: np.ndarray,
        edge_pts: np.ndarray,
        planar_pts: np.ndarray,
        map_update_distance: float = 0.3,
    ) -> None:
        """
        Add new line primitives to the global map from the current scan.

        Consecutive planar points are converted to line segments; new segments
        distant from existing map lines are added.
        """
        θ = x[2]
        R = _rot2(θ)
        t = x[:2]

        all_pts = np.vstack([edge_pts, planar_pts]) if (len(edge_pts) > 0 and len(planar_pts) > 0) \
            else (edge_pts if len(edge_pts) > 0 else planar_pts)
        if len(all_pts) < 2:
            return

        # Project to world frame
        pts_w = (R @ all_pts.T).T + t

        # Extract simple line primitives from consecutive point pairs
        for i in range(len(pts_w) - 1):
            p1, p2 = pts_w[i], pts_w[i + 1]
            seg_len = np.linalg.norm(p2 - p1)
            if seg_len < 0.01:
                continue
            mid = (p1 + p2) / 2.0
            direction = (p2 - p1) / seg_len
            normal = np.array([-direction[1], direction[0]])

            # Only add if far from existing lines
            if self._nearest_line(mid)[1] > map_update_distance:
                self._line_map.append({"p": mid.copy(), "normal": normal.copy()})

    def _nearest_line(
        self, point: np.ndarray
    ) -> Tuple[Optional[dict], float]:
        """Find the nearest map line to `point` and its perpendicular distance."""
        best_line, best_d = None, float("inf")
        for line in self._line_map:
            d = abs(float(line["normal"] @ (point - line["p"])))
            if d < best_d:
                best_d = d
                best_line = line
        return best_line, best_d

    def _associate_landmark(
        self,
        p_W: np.ndarray,
        r: float,
        tau_p: float = 0.5,
        tau_r: float = 0.2,
    ) -> Optional[dict]:
        """
        Nearest-neighbour landmark association (Eq. 19).

        A stored landmark matches if it is within tau_p metres AND its
        radius is within tau_r of the detected radius.
        """
        for lm in self._landmark_map:
            if (np.linalg.norm(lm["p"] - p_W) < tau_p and
                    abs(lm["r"] - r) < tau_r):
                return lm
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _build_process_noise(self, dt: float) -> np.ndarray:
        """Diagonal process noise matrix scaled by dt."""
        q_pos = (self._sigma_accel * dt ** 2) ** 2
        q_head = (self._sigma_gyro * dt) ** 2
        q_vel = (self._sigma_accel * dt) ** 2
        q_bias = 1e-6  # slow random walk for biases
        return np.diag([q_pos, q_pos, q_head, q_vel, q_vel, q_bias, q_bias, q_bias])
