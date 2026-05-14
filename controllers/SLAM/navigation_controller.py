"""
GPS-guided navigation controller with occupancy map memory.

Navigation strategy: potential field combining three signals —
  1. Goal attraction  — GPS bearing to goal
  2. LiDAR repulsion  — immediate obstacle avoidance from current scan
  3. Map repulsion    — avoidance of known obstacles from the occupancy map

The occupancy map accumulates LiDAR readings over time so the robot
remembers obstacles it has seen even when they leave the sensor's view.
"""

from __future__ import annotations

import math
import sys
import traceback
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from controller import Supervisor  # pyright: ignore[reportMissingImports]
except ImportError:
    Supervisor = object  # type: ignore

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from controllers.SLAM.slam_map import OccupancyMap


# ── Configuration ─────────────────────────────────────────────────────────────

GOAL_POS       = np.array([2.0, 0.0], dtype=np.float32)
GOAL_RADIUS    = 0.30   # m — success distance
COLLISION_DIST = 0.10   # m — abort distance

MAX_SPEED   = 5.0   # rad/s wheel speed
WHEEL_RADIUS = 0.033  # m
MAX_STEER   = 1.0   # rad
SAFE_DIST   = 0.50   # m — obstacle avoidance activation range

# Map memory: sample occupancy this far ahead at N angles across the front arc
MAP_LOOKAHEAD = 1.2   # m
MAP_SAMPLES   = 11    # angular samples (-90° … +90°)
MAP_WEIGHT    = 0.5   # relative weight of map signal vs LiDAR signal


# ── Controller ────────────────────────────────────────────────────────────────

class NavigatingController:
    """Goal-seeking robot controller with occupancy-map obstacle memory."""

    def __init__(self) -> None:
        self.robot: any = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep())
        self.dt = self.timestep / 1000.0

        # Sensors
        self.lidar: any = self.robot.getDevice("lidar")
        self.lidar.enable(self.timestep)
        self.lidar_max_range: float = self.lidar.getMaxRange()
        self._h_res: int = self.lidar.getHorizontalResolution()

        self.gps: any = self.robot.getDevice("gps")
        self.gps.enable(self.timestep)

        # Heading source — InertialUnit gives yaw directly; fall back to
        # Supervisor rotation field if the device is absent.
        self._inertial: Optional[any] = None
        try:
            dev = self.robot.getDevice("inertial unit")
            if dev is not None:
                dev.enable(self.timestep)
                self._inertial = dev
        except Exception:
            pass

        self._rotation_field: Optional[any] = None
        if self._inertial is None:
            try:
                self._rotation_field = self.robot.getFromDef("ALTINO").getField("rotation")
            except Exception:
                pass

        # Motors
        self._left_steer: any  = self.robot.getDevice("left_steer")
        self._right_steer: any = self.robot.getDevice("right_steer")
        for s in (self._left_steer, self._right_steer):
            s.setPosition(0.0)
            s.setVelocity(1.0)

        self._wheels: list = [
            self.robot.getDevice("left_front_wheel"),
            self.robot.getDevice("right_front_wheel"),
            self.robot.getDevice("left_rear_wheel"),
            self.robot.getDevice("right_rear_wheel"),
        ]
        for w in self._wheels:
            w.setPosition(float("inf"))
            w.setVelocity(0.0)

        # Occupancy map: 40×40 m world, 5 cm resolution
        self.occ_map = OccupancyMap(
            resolution=0.05, width_m=40.0, height_m=40.0, origin=(-20.0, -20.0)
        )

        self._steer: float = 0.0
        self._step: int = 0
        self._lidar_angles: Optional[np.ndarray] = None

        print(f"[NAV] Ready — goal at {GOAL_POS.tolist()}", flush=True)

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        while self.robot.step(self.timestep) != -1:
            self._tick()

    def _tick(self) -> None:
        self._step += 1

        # 1. Read LiDAR (first horizontal layer only)
        raw = np.nan_to_num(
            np.array(self.lidar.getRangeImage(), dtype=np.float32),
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        h = self._h_res if self._h_res > 0 else len(raw)
        ranges = raw[:h]

        if self._lidar_angles is None:
            fov = self.lidar.getFov()
            self._lidar_angles = np.linspace(-fov / 2, fov / 2, h, dtype=np.float32)

        # 2. Read GPS and heading
        gv = self.gps.getValues()
        pos = np.array([gv[0], gv[1]], dtype=np.float32)
        heading = self._get_heading()

        # 3. Collision check
        valid = ranges[ranges > 0.0]
        if len(valid) > 0 and float(valid.min()) < COLLISION_DIST:
            print(f"[NAV] Collision at step {self._step} — reverting.", flush=True)
            self.robot.simulationRevert()
            return

        # 4. Goal check
        dist_goal = float(np.linalg.norm(pos - GOAL_POS))
        if dist_goal < GOAL_RADIUS:
            print(f"[NAV] Goal reached at step {self._step}!", flush=True)
            self.robot.simulationSetMode(0)
            return

        # 5. Update occupancy map
        scan_pts = self._scan_to_world(ranges, pos, heading)
        if len(scan_pts) > 0:
            self.occ_map.update(pos, scan_pts)

        # 6. Navigate
        self._navigate(ranges, pos, heading, dist_goal)

        if self._step % 100 == 0:
            print(
                f"[NAV] step={self._step:5d}  "
                f"pos=({pos[0]:.2f},{pos[1]:.2f})  "
                f"goal_dist={dist_goal:.2f} m",
                flush=True,
            )

    # ── Navigation (potential field) ──────────────────────────────────────────

    def _navigate(
        self,
        ranges: np.ndarray,
        pos: np.ndarray,
        heading: float,
        dist_goal: float,
    ) -> None:
        n = len(ranges)
        angles = self._lidar_angles

        # --- Goal attraction ---
        goal_vec = GOAL_POS - pos
        goal_dir = float(np.arctan2(goal_vec[1], goal_vec[0]))
        goal_error = float(
            np.arctan2(np.sin(goal_dir - heading), np.cos(goal_dir - heading))
        )

        # --- LiDAR repulsion ---
        # Each ray inside SAFE_DIST pushes laterally; sin(angle) gives left/right sign.
        eff = np.where(ranges > 0, ranges, self.lidar_max_range)
        closeness = np.clip(1.0 - eff / SAFE_DIST, 0.0, 1.0)
        lidar_lateral = float(np.clip(np.sum(-np.sin(angles) * closeness), -3.0, 3.0))

        # Minimum range in the forward ±90° arc (used for speed scaling below)
        front = eff[n // 4 : 3 * n // 4]
        front_min = float(front.min()) if len(front) > 0 else self.lidar_max_range

        # --- Map repulsion ---
        # Sample the occupancy map ahead of the robot across a 180° arc.
        # Known-occupied cells push the robot away; known-free cells are neutral.
        map_lateral = 0.0
        sample_angles = np.linspace(-math.pi / 2, math.pi / 2, MAP_SAMPLES)
        for sa in sample_angles:
            world_angle = heading + sa
            sx = pos[0] + MAP_LOOKAHEAD * math.cos(world_angle)
            sy = pos[1] + MAP_LOOKAHEAD * math.sin(world_angle)
            occ_prob = self._query_map(sx, sy)
            # occ_prob > 0.5 → occupied → push away (sign opposite to sin(sa))
            map_lateral -= math.sin(sa) * (occ_prob - 0.5) * 2.0
        map_lateral = float(np.clip(map_lateral, -3.0, 3.0))

        # --- Blend goal vs repulsion ---
        # When the robot is close to an obstacle, shift weight toward avoidance.
        proximity = float(np.clip(1.0 - front_min / SAFE_DIST, 0.0, 1.0))
        goal_weight = 1.0 - 0.75 * proximity

        raw_steer = (
            goal_weight * goal_error * 0.9
            + (1.0 - goal_weight)
            * (lidar_lateral + MAP_WEIGHT * map_lateral)
            * 0.35
        )
        raw_steer = float(np.clip(raw_steer, -MAX_STEER, MAX_STEER))

        self._steer += 0.4 * (raw_steer - self._steer)

        speed_scale = float(np.clip(front_min / SAFE_DIST, 0.25, 1.0))
        if dist_goal < 0.8:
            speed_scale *= 0.5
        speed = MAX_SPEED * speed_scale

        self._set_motors(speed, self._steer)

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _get_heading(self) -> float:
        if self._inertial is not None:
            return float(self._inertial.getRollPitchYaw()[2])
        if self._rotation_field is not None:
            rot = self._rotation_field.getSFRotation()
            if rot is not None and len(rot) == 4:
                x, y, z, angle = (float(v) for v in rot)
                norm = math.sqrt(x*x + y*y + z*z)
                if norm > 1e-8:
                    x /= norm; y /= norm; z /= norm
                c = math.cos(angle); s = math.sin(angle); oc = 1.0 - c
                yaw = math.atan2(z*s + y*x*oc, c + x*x*oc)
                return float(math.atan2(math.sin(yaw), math.cos(yaw)))
        return 0.0

    def _scan_to_world(
        self, ranges: np.ndarray, pos: np.ndarray, heading: float
    ) -> np.ndarray:
        if self._lidar_angles is None:
            return np.empty((0, 2), dtype=np.float32)
        mask = ranges > 0.01
        r = ranges[mask]
        a = self._lidar_angles[mask]
        c, s = math.cos(heading), math.sin(heading)
        xl = r * np.cos(a);  yl = r * np.sin(a)
        xw = c * xl - s * yl + pos[0]
        yw = s * xl + c * yl + pos[1]
        return np.column_stack([xw, yw]).astype(np.float32)

    def _query_map(self, x: float, y: float) -> float:
        """Occupancy probability [0,1] at world position. 0.5 = unknown."""
        row, col = self.occ_map.world_to_cell(np.array([x, y]))
        if self.occ_map.in_bounds(row, col):
            lo = float(self.occ_map.log_odds[row, col])
            return float(1.0 / (1.0 + math.exp(-lo)))
        return 0.5

    def _set_motors(self, speed: float, steer: float) -> None:
        steer = float(np.clip(steer, -MAX_STEER, MAX_STEER))
        speed = float(np.clip(speed, 0.0, MAX_SPEED))
        self._left_steer.setPosition(steer)
        self._right_steer.setPosition(steer)
        for w in self._wheels:
            w.setVelocity(speed)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        NavigatingController().run()
    except Exception:
        traceback.print_exc()
