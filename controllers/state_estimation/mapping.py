"""
Occupancy grid and keyframe-based trajectory tracker.

LLM level: 4 - LLM wrote majority of code, implemented and tested by us.
"""

from __future__ import annotations
import numpy as np
from typing import List, Optional, Tuple
from matplotlib.patches import Circle


class OccupancyMap:
    """Log-odds occupancy grid with Bresenham ray-casting."""

    FREE_LOG_ODDS = -0.5
    OCC_LOG_ODDS  =  1.5
    MIN_LOG_ODDS  = -5.0
    MAX_LOG_ODDS  =  5.0

    def __init__(
        self,
        resolution: float = 0.05,
        width_m: float = 40.0,
        height_m: float = 40.0,
        origin: Tuple[float, float] = (-20.0, -20.0),
    ) -> None:
        """
        Initialize an occupancy grid with given resolution and dimensions.

        Parameters
        ----------
        resolution: float
            Grid cell size in meters.
        width_m: float
            Grid width in meters.
        height_m: float
            Grid height in meters.
        origin: Tuple[float, float]
            World coordinates of the grid's bottom-left corner.
        """
        self.resolution = resolution
        self.origin = np.array(origin)
        self.nx = int(width_m / resolution)
        self.ny = int(height_m / resolution)
        self.log_odds = np.zeros((self.ny, self.nx), dtype=np.float32)


    def world_to_cell(self, xy: np.ndarray) -> Tuple[int, int]:
        """
        Convert world coordinates to (row, col) grid cell indices.
        
        Parameters
        ---------
        xy: np.ndarray
            World coordinates (x, y).
        
        Returns
        -------
        Tuple[int, int]
            Grid cell indices (row, col).
        """
        col = int((xy[0] - self.origin[0]) / self.resolution)
        row = int((xy[1] - self.origin[1]) / self.resolution)
        
        return row, col

    
    def in_bounds(self, row: int, col: int) -> bool:
        """
        Check if (row, col) is within the occupancy grid.
        
        Parameters
        ---------
        row: int
            Row index.
        col: int
            Column index.

        Returns
        -------
        bool
            True if (row, col) is within the grid.
        """
        return 0 <= row < self.ny and 0 <= col < self.nx


    def update(self, robot_pos: np.ndarray, scan_points: np.ndarray, max_range: float = 10.0) -> None:
        """
        Update the occupancy grid with new LiDAR scan data.

        Parameters
        ---------
        robot_pos: np.ndarray
            Robot's world coordinates (x, y).
        scan_points: np.ndarray
            LiDAR scan points in world coordinates (N, 2).
        max_range: float
            Maximum LiDAR range.
        """
        for p in scan_points:
            dist = np.linalg.norm(p - robot_pos)
            if dist > max_range or dist < 0.01:
                continue
            
            er, ec = self.world_to_cell(p)
            if self.in_bounds(er, ec):
                self.log_odds[er, ec] = np.clip(
                    self.log_odds[er, ec] + self.OCC_LOG_ODDS,
                    self.MIN_LOG_ODDS, self.MAX_LOG_ODDS,
                )
            
            rr, rc = self.world_to_cell(robot_pos)
            for row, col in self._bresenham(rr, rc, er, ec):
                if self.in_bounds(row, col):
                    self.log_odds[row, col] = np.clip(
                        self.log_odds[row, col] + self.FREE_LOG_ODDS,
                        self.MIN_LOG_ODDS, self.MAX_LOG_ODDS,
                    )


    @property
    def probability(self) -> np.ndarray:
        """
        Convert log-odds to probability via sigmoid.
        
        Returns
        -------
        np.ndarray
            Occupancy probability grid.
        """
        return 1.0 / (1.0 + np.exp(-self.log_odds))


    @staticmethod
    def _bresenham(r0: int, c0: int, r1: int, c1: int) -> List[Tuple[int, int]]:
        """
        Bresenham line algorithm returning all cells between (r0,c0) and (r1,c1).
        
        Parameters
        ----------
        r0, c0: int
            Start cell.
        r1, c1: int
            End cell.

        Returns
        -------
        List[Tuple[int, int]]
            List of cells between (r0,c0) and (r1,c1).
        """
        cells = []
        dr, dc = abs(r1 - r0), abs(c1 - c0)
        r, c = r0, c0
        sr = 1 if r1 > r0 else -1
        sc = 1 if c1 > c0 else -1
        if dc > dr:
            err = dc // 2
            while c != c1:
                cells.append((r, c))
                err -= dr
                if err < 0:
                    r += sr; err += dc
                
                c += sc
        else:
            err = dr // 2
            while r != r1:
                cells.append((r, c))
                err -= dc
                if err < 0:
                    c += sc; err += dr
                
                r += sr
        
        return cells


class SLAMMap:
    """
    Occupancy grid + keyframe trajectory tracker.
    """

    KEYFRAME_DIST  = 0.3    # m — add keyframe after moving this far
    KEYFRAME_ANGLE = 0.15   # rad — or rotating this much (~8.5°)

    def __init__(self, map_resolution: float = 0.05) -> None:
        """
        Initialize SLAM map with occupancy grid and trajectory storage.
        
        Parameters
        ----------
        map_resolution: float
            Resolution of the occupancy grid in meters.
        """
        self.occ_map = OccupancyMap(resolution=map_resolution)
        self._trajectory: List[Tuple[float, float]] = []   # (x, y) per keyframe
        self._last_kf: Optional[Tuple[float, float, float]] = None  # x, y, theta

    def try_add_keyframe(self, x: float, y: float, theta: float, scan_points: Optional[np.ndarray] = None) -> bool:
        """
        Add a keyframe if the robot has moved enough.
        
        Parameters
        ---------
        x, y, theta: float
            Robot pose in world frame.
        scan_points: Optional[np.ndarray]
            Scan points in robot frame. If None, no map update is performed.
        
        Returns
        -------
        bool
            True if keyframe was added.
        """
        if self._last_kf is not None:
            lx, ly, lt = self._last_kf
            dist = float(np.linalg.norm([x - lx, y - ly]))
            dangle = abs(float(((theta - lt + np.pi) % (2 * np.pi)) - np.pi))
            if dist < self.KEYFRAME_DIST and dangle < self.KEYFRAME_ANGLE:
                return False
        
        self._trajectory.append((x, y))
        self._last_kf = (x, y, theta)
        if scan_points is not None:
            self.occ_map.update(np.array([x, y]), scan_points)
        
        return True

    def save_plot(self, output_path: str = "slam_map.png", goal: Optional[Tuple[float, float]] = None) -> None:
        """
        Save a plot of the occupancy grid and trajectory to a file.

        Parameters
        ---------
        output_path: str
            Path to save the plot to.
        goal: Optional[Tuple[float, float]]
            Goal position to plot. If None, no goal is plotted.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 8))
            prob = self.occ_map.probability
            ox, oy = self.occ_map.origin
            res = self.occ_map.resolution
            extent = (ox, ox + self.occ_map.nx * res, oy, oy + self.occ_map.ny * res)
            ax.imshow(prob, origin="lower", extent=extent, cmap="gray_r",
                      vmin=0.0, vmax=1.0, interpolation="nearest")

            if self._trajectory:
                pos = np.array(self._trajectory)
                ax.plot(pos[:, 0], pos[:, 1], "r-", linewidth=1.0, label="trajectory")
                ax.scatter([pos[0, 0]], [pos[0, 1]], c="lime", s=60, zorder=5, label="start")
                ax.scatter([pos[-1, 0]], [pos[-1, 1]], c="red", s=60, marker="*", zorder=5, label="end")

            if goal is not None:
                ax.scatter([goal[0]], [goal[1]], c="yellow", s=120, marker="*", zorder=6, label="goal")
                ax.add_patch(Circle(goal, 0.1, color="yellow", fill=False,
                                    linewidth=1.5, linestyle="--"))

            ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
            ax.set_title(f"Occupancy Map  ({len(self._trajectory)} keyframes)")
            ax.legend(loc="upper right", fontsize=8)
            ax.set_aspect("equal")
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        except Exception as exc:
            print(f"[SLAMMap] save_plot failed: {exc}", flush=True)
