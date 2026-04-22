"""
Global map management and pose-graph optimisation for CNN-LiDAR-SLAM.

Implements the global mapping optimisation described in Section III-G:

    e_{ij} = z_{ij} − h(x_i, x_j)                      (Eq. 24)
    min_x  Σ_{(i,j)∈C} ‖e_{ij}‖²                       (Eq. 25)
    T = argmin_T Σ_{(i,j)∈C} ‖z_{ij} − h(x_i,x_j)‖²   (Eq. 26)
    min_T  Σ_i (d_ε(p_i(T)) + d_S(p_j(T)))             (Eq. 27)

Solved iteratively with the Levenberg-Marquardt algorithm [46].
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PoseNode:
    """A single pose in the pose graph."""
    id: int
    x: float       # world-frame x position
    y: float       # world-frame y position
    theta: float   # heading angle (yaw)

    @property
    def pose(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta])

    def transform_matrix(self) -> np.ndarray:
        """3×3 homogeneous transform."""
        c, s = np.cos(self.theta), np.sin(self.theta)
        return np.array([
            [c, -s, self.x],
            [s,  c, self.y],
            [0,  0,    1.0],
        ])


@dataclass
class PoseEdge:
    """A relative pose constraint between two nodes."""
    i: int          # source node id
    j: int          # target node id
    z: np.ndarray   # (3,) observed relative pose [Δx, Δy, Δθ]
    Omega: np.ndarray = field(default_factory=lambda: np.eye(3))  # information matrix

    def error(self, xi: PoseNode, xj: PoseNode) -> np.ndarray:
        """e_{ij} = z_{ij} − h(x_i, x_j)  (Eq. 24)."""
        return self.z - _relative_pose(xi, xj)


@dataclass
class MapLandmark:
    """A semantic object landmark in the global map."""
    id: int
    pos: np.ndarray   # (2,) position in world frame
    radius: float
    obs_count: int = 1
    confidence: float = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _relative_pose(xi: PoseNode, xj: PoseNode) -> np.ndarray:
    """
    Compute the relative pose h(x_i, x_j) = T_i^{-1} ∘ T_j.

    Returns (3,) = [Δx, Δy, Δθ] in x_i's frame.
    """
    dθ = xj.theta - xi.theta
    dx = xj.x - xi.x
    dy = xj.y - xi.y
    ci, si = np.cos(xi.theta), np.sin(xi.theta)
    # Rotate (dx, dy) into x_i's local frame
    local_dx =  ci * dx + si * dy
    local_dy = -si * dx + ci * dy
    return np.array([local_dx, local_dy, _wrap_angle(dθ)])


def _wrap_angle(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def _rot2(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


# ─────────────────────────────────────────────────────────────────────────────
# Pose-graph optimiser (Levenberg-Marquardt)
# ─────────────────────────────────────────────────────────────────────────────

class PoseGraphOptimizer:
    """
    Sparse pose-graph optimisation using Levenberg-Marquardt.

    Minimises Σ_{(i,j)} ‖e_{ij}‖²_{Ω_{ij}}  (Eq. 25).

    This implements the standard linearise-and-solve loop:
        1. Build linear system  H Δx = b
        2. Augment diagonal: H ← H + λ I   (Marquardt damping)
        3. Solve and update poses
        4. Adjust λ based on cost reduction
    """

    def __init__(
        self,
        max_iter: int = 50,
        lm_lambda_init: float = 1e-3,
        lm_lambda_factor: float = 10.0,
        converge_thr: float = 1e-4,
    ) -> None:
        self.max_iter = max_iter
        self.lm_lambda = lm_lambda_init
        self.lm_factor = lm_lambda_factor
        self.converge_thr = converge_thr

    def optimise(
        self,
        nodes: List[PoseNode],
        edges: List[PoseEdge],
    ) -> List[PoseNode]:
        """
        Run Levenberg-Marquardt pose-graph optimisation.

        Args:
            nodes: List of PoseNode (will be modified in-place copy).
            edges: List of PoseEdge constraints.

        Returns:
            Refined list of PoseNode objects.
        """
        if len(nodes) < 2 or len(edges) == 0:
            return nodes

        # Work on copies
        nodes = [PoseNode(n.id, n.x, n.y, n.theta) for n in nodes]
        node_idx = {n.id: i for i, n in enumerate(nodes)}

        prev_cost = self._total_cost(nodes, edges, node_idx)

        for iteration in range(self.max_iter):
            n_poses = len(nodes)
            dim = 3 * n_poses  # each pose has 3 DOF

            H_mat = np.zeros((dim, dim))
            b_vec = np.zeros(dim)

            # ── Linearise all edges ──────────────────────────────────────────
            for edge in edges:
                if edge.i not in node_idx or edge.j not in node_idx:
                    continue
                ii = node_idx[edge.i]
                jj = node_idx[edge.j]
                xi = nodes[ii]
                xj = nodes[jj]

                e = edge.error(xi, xj)            # (3,)
                Ji, Jj = self._edge_jacobians(xi, xj)  # (3,3), (3,3)
                Omega = edge.Omega

                # Accumulate into global system
                si, sj = ii * 3, jj * 3
                H_mat[si:si+3, si:si+3] += Ji.T @ Omega @ Ji
                H_mat[sj:sj+3, sj:sj+3] += Jj.T @ Omega @ Jj
                H_mat[si:si+3, sj:sj+3] += Ji.T @ Omega @ Jj
                H_mat[sj:sj+3, si:si+3] += Jj.T @ Omega @ Ji
                b_vec[si:si+3] += Ji.T @ Omega @ e
                b_vec[sj:sj+3] += Jj.T @ Omega @ e

            # ── Fix first pose (gauge freedom) ──────────────────────────────
            H_mat[:3, :] = 0.0
            H_mat[:, :3] = 0.0
            H_mat[0, 0] = 1.0
            H_mat[1, 1] = 1.0
            H_mat[2, 2] = 1.0
            b_vec[:3] = 0.0

            # ── Levenberg-Marquardt damping ──────────────────────────────────
            H_damp = H_mat + self.lm_lambda * np.eye(dim)

            # ── Solve linear system ──────────────────────────────────────────
            try:
                delta = np.linalg.solve(H_damp, -b_vec)
            except np.linalg.LinAlgError:
                break

            # ── Apply update ─────────────────────────────────────────────────
            new_nodes = [PoseNode(n.id, n.x, n.y, n.theta) for n in nodes]
            for i, node in enumerate(new_nodes):
                node.x += delta[i * 3]
                node.y += delta[i * 3 + 1]
                node.theta = _wrap_angle(node.theta + delta[i * 3 + 2])

            new_cost = self._total_cost(new_nodes, edges, node_idx)

            # ── Adjust damping ───────────────────────────────────────────────
            if new_cost < prev_cost:
                nodes = new_nodes
                self.lm_lambda /= self.lm_factor
                prev_cost = new_cost
            else:
                self.lm_lambda *= self.lm_factor

            if np.linalg.norm(delta) < self.converge_thr:
                break

        return nodes

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _total_cost(
        nodes: List[PoseNode],
        edges: List[PoseEdge],
        node_idx: Dict[int, int],
    ) -> float:
        cost = 0.0
        for edge in edges:
            if edge.i not in node_idx or edge.j not in node_idx:
                continue
            e = edge.error(nodes[node_idx[edge.i]], nodes[node_idx[edge.j]])
            cost += float(e @ edge.Omega @ e)
        return cost

    @staticmethod
    def _edge_jacobians(
        xi: PoseNode, xj: PoseNode
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Analytical Jacobians of e_{ij} = z − h(xi, xj) w.r.t. xi and xj.

        ∂h/∂xi and ∂h/∂xj for 2-D relative pose measurement.
        """
        θi = xi.theta
        ci, si = np.cos(θi), np.sin(θi)
        dx = xj.x - xi.x
        dy = xj.y - xi.y

        # ∂h/∂xi  (3×3)
        Ji = np.array([
            [-ci, -si, -si * dx + ci * dy],
            [ si, -ci, -ci * dx - si * dy],
            [ 0,   0,  -1.0             ],
        ])

        # ∂h/∂xj  (3×3)
        Jj = np.array([
            [ci, si, 0.0],
            [-si, ci, 0.0],
            [0.0, 0.0, 1.0],
        ])

        return Ji, Jj


# ─────────────────────────────────────────────────────────────────────────────
# Global occupancy map
# ─────────────────────────────────────────────────────────────────────────────

class OccupancyMap:
    """
    Simple 2-D occupancy grid for building the global map.

    Each cell stores the probability of being occupied; log-odds updates
    provide numerically stable fusion over many observations.
    """

    FREE_LOG_ODDS = -0.5
    OCC_LOG_ODDS = 1.5
    MIN_LOG_ODDS = -5.0
    MAX_LOG_ODDS = 5.0

    def __init__(
        self,
        resolution: float = 0.05,
        width_m: float = 40.0,
        height_m: float = 40.0,
        origin: Tuple[float, float] = (-20.0, -20.0),
    ) -> None:
        self.resolution = resolution
        self.origin = np.array(origin)
        self.nx = int(width_m / resolution)
        self.ny = int(height_m / resolution)
        self.log_odds = np.zeros((self.ny, self.nx), dtype=np.float32)

    def world_to_cell(self, xy: np.ndarray) -> Tuple[int, int]:
        """Convert world coordinates to grid cell indices."""
        col = int((xy[0] - self.origin[0]) / self.resolution)
        row = int((xy[1] - self.origin[1]) / self.resolution)
        return row, col

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.ny and 0 <= col < self.nx

    def update(
        self,
        robot_pos: np.ndarray,
        scan_points: np.ndarray,
        max_range: float = 10.0,
    ) -> None:
        """
        Update occupancy map with a new LiDAR scan.

        Cells along the ray to each scan endpoint are marked free;
        the endpoint cell is marked occupied.
        """
        for p in scan_points:
            dist = np.linalg.norm(p - robot_pos)
            if dist > max_range or dist < 0.01:
                continue
            # Mark endpoint occupied
            er, ec = self.world_to_cell(p)
            if self.in_bounds(er, ec):
                self.log_odds[er, ec] = np.clip(
                    self.log_odds[er, ec] + self.OCC_LOG_ODDS,
                    self.MIN_LOG_ODDS, self.MAX_LOG_ODDS
                )
            # Ray-cast: mark cells along ray as free (Bresenham)
            rr, rc = self.world_to_cell(robot_pos)
            for row, col in self._bresenham(rr, rc, er, ec):
                if self.in_bounds(row, col):
                    self.log_odds[row, col] = np.clip(
                        self.log_odds[row, col] + self.FREE_LOG_ODDS,
                        self.MIN_LOG_ODDS, self.MAX_LOG_ODDS
                    )

    @property
    def probability(self) -> np.ndarray:
        """Convert log-odds to probabilities."""
        return 1.0 / (1.0 + np.exp(-self.log_odds))

    @staticmethod
    def _bresenham(r0: int, c0: int, r1: int, c1: int) -> List[Tuple[int, int]]:
        """Bresenham's line rasterisation."""
        cells = []
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        r, c = r0, c0
        sr = 1 if r1 > r0 else -1
        sc = 1 if c1 > c0 else -1
        if dc > dr:
            err = dc // 2
            while c != c1:
                cells.append((r, c))
                err -= dr
                if err < 0:
                    r += sr
                    err += dc
                c += sc
        else:
            err = dr // 2
            while r != r1:
                cells.append((r, c))
                err -= dc
                if err < 0:
                    c += sc
                    err += dr
                r += sr
        return cells


# ─────────────────────────────────────────────────────────────────────────────
# Top-level SLAM map manager
# ─────────────────────────────────────────────────────────────────────────────

class SLAMMap:
    """
    Manages the full global map: pose graph + occupancy grid + landmarks.

    Provides a single interface for the SLAM controller to:
      • Record pose keyframes and relative constraints
      • Insert / update semantic landmarks
      • Trigger offline global pose-graph optimisation
      • Query the occupancy map for display / navigation
    """

    KEYFRAME_DIST = 0.3    # add keyframe every ~30 cm
    KEYFRAME_ANGLE = 0.15  # or ~8.5° rotation

    def __init__(self, map_resolution: float = 0.05) -> None:
        self.nodes: List[PoseNode] = []
        self.edges: List[PoseEdge] = []
        self.landmarks: List[MapLandmark] = []
        self.occ_map = OccupancyMap(resolution=map_resolution)
        self.optimizer = PoseGraphOptimizer()
        self._next_id = 0
        self._last_kf: Optional[PoseNode] = None

    # ── Keyframe management ──────────────────────────────────────────────────

    def try_add_keyframe(
        self,
        x: float, y: float, theta: float,
        scan_points: Optional[np.ndarray] = None,
    ) -> Optional[PoseNode]:
        """
        Add a new pose keyframe if the robot has moved enough.

        Returns the new node if added, else None.
        """
        new_node = PoseNode(self._next_id, x, y, theta)
        if self._last_kf is None:
            self._commit_keyframe(new_node, scan_points)
            return new_node

        dist = np.linalg.norm([x - self._last_kf.x, y - self._last_kf.y])
        dangle = abs(_wrap_angle(theta - self._last_kf.theta))

        if dist >= self.KEYFRAME_DIST or dangle >= self.KEYFRAME_ANGLE:
            # Add sequential edge (odometry constraint)
            z = _relative_pose(self._last_kf, new_node)
            edge = PoseEdge(
                i=self._last_kf.id,
                j=new_node.id,
                z=z,
                Omega=np.diag([100.0, 100.0, 50.0]),
            )
            self.edges.append(edge)
            self._commit_keyframe(new_node, scan_points)
            return new_node

        return None

    def _commit_keyframe(
        self,
        node: PoseNode,
        scan_points: Optional[np.ndarray],
    ) -> None:
        self.nodes.append(node)
        self._last_kf = node
        self._next_id += 1
        if scan_points is not None:
            self.occ_map.update(
                np.array([node.x, node.y]),
                scan_points,
            )

    # ── Landmark management ──────────────────────────────────────────────────

    def update_landmark(
        self,
        pos_world: np.ndarray,
        radius: float,
        tau_p: float = 0.5,
        tau_r: float = 0.25,
    ) -> MapLandmark:
        """Insert or update a semantic landmark."""
        for lm in self.landmarks:
            if (np.linalg.norm(lm.pos - pos_world) < tau_p and
                    abs(lm.radius - radius) < tau_r):
                n = lm.obs_count
                lm.pos = (lm.pos * n + pos_world) / (n + 1)
                lm.obs_count += 1
                return lm
        new_lm = MapLandmark(
            id=len(self.landmarks),
            pos=pos_world.copy(),
            radius=radius,
        )
        self.landmarks.append(new_lm)
        return new_lm

    # ── Global optimisation ──────────────────────────────────────────────────

    def optimise(self) -> None:
        """Run offline pose-graph optimisation (Section III-G)."""
        if len(self.nodes) < 2:
            return
        refined = self.optimizer.optimise(self.nodes, self.edges)
        self.nodes = refined

    # ── Query ────────────────────────────────────────────────────────────────

    def latest_pose(self) -> Optional[np.ndarray]:
        """Return the most recently added pose as (x, y, θ)."""
        if not self.nodes:
            return None
        n = self.nodes[-1]
        return np.array([n.x, n.y, n.theta])

    def all_positions(self) -> np.ndarray:
        """Return (N, 2) array of all keyframe positions."""
        if not self.nodes:
            return np.empty((0, 2))
        return np.array([[n.x, n.y] for n in self.nodes])

    # ── Visualisation ─────────────────────────────────────────────────────────

    def save_plot(self, output_path: str = "slam_map.png",
                  goal: Optional[Tuple[float, float]] = None) -> None:
        """Save 2-D occupancy map with trajectory, landmarks, and goal.

        Uses the non-interactive Agg backend so it works inside Webots
        without a display.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 8))

            prob = self.occ_map.probability
            ox, oy = self.occ_map.origin
            res = self.occ_map.resolution
            extent = [ox, ox + self.occ_map.nx * res,
                      oy, oy + self.occ_map.ny * res]
            ax.imshow(prob, origin="lower", extent=extent,
                      cmap="gray_r", vmin=0.0, vmax=1.0,
                      interpolation="nearest")

            if self.nodes:
                pos = self.all_positions()
                ax.plot(pos[:, 0], pos[:, 1], "r-", linewidth=1.0, label="trajectory")
                ax.scatter([pos[0, 0]], [pos[0, 1]],
                           c="lime", s=60, zorder=5, label="start")
                ax.scatter([pos[-1, 0]], [pos[-1, 1]],
                           c="red", s=60, marker="*", zorder=5, label="end")

            if goal is not None:
                ax.scatter([goal[0]], [goal[1]],
                           c="yellow", s=120, marker="*", zorder=6, label="goal")
                circle = plt.Circle(goal, 0.1, color="yellow", fill=False,
                                    linewidth=1.5, linestyle="--")
                ax.add_patch(circle)

            for lm in self.landmarks:
                circle = plt.Circle(lm.pos, max(lm.radius, 0.05),
                                    color="cyan", fill=False, linewidth=1.5)
                ax.add_patch(circle)
                ax.text(lm.pos[0], lm.pos[1] + lm.radius + 0.05,
                        f"lm{lm.id}", fontsize=6, ha="center", color="cyan")

            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.set_title(
                f"Occupancy Map  ({len(self.nodes)} keyframes, {len(self.landmarks)} landmarks)"
            )
            ax.legend(loc="upper right", fontsize=8)
            ax.set_aspect("equal")

            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[SLAMMap] Plot saved → {output_path}", flush=True)

        except Exception as exc:
            print(f"[SLAMMap] save_plot failed: {exc}", flush=True)
