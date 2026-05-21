"""Generate 10 static PPO training worlds with increasing obstacle density.

Worlds 1–4:  0, 2, 4, 6 obstacles — static goal at (2.0, 0.0)
Worlds 5–8:  8, 10, 12, 14 obstacles — static goal at shifted y positions
Worlds 9–10: 16, 18 obstacles — designed for per-episode goal randomization
             Set randomize_goal=True and goal_y_range=1.5 in the PPO Config
             when running these worlds.

Difficulty design:
  - Obstacles are placed ON or near the direct path, not off to the side.
  - Each pair added for the next world creates a new chokepoint or narrows
    an existing gap, so difficulty increases monotonically.
  - All pairwise Euclidean distances between obstacle centres >= 0.70 m
    → surface-to-surface gap >= 0.40 m  (> 2× Altino body width of 0.09 m)
  - A navigable path from start (-2.0, 0.0) to goal always exists.

Run from the repository root:
    python worlds/generate_training_worlds.py
"""
import re
from pathlib import Path

TEMPLATE = Path(__file__).parent / "ObstacleCourse.wbt"
OUT_DIR = Path(__file__).parent / "training"
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Primitive builders (identical API to generate_validation_worlds.py)
# ---------------------------------------------------------------------------

def _cyl(name, x, y, r=0.15, h=0.3, color="0.8 0.2 0.2"):
    return f"""\
DEF {name} Solid {{
  translation {x} {y} 0.15
  children [
    Shape {{
      appearance PBRAppearance {{ baseColor {color} roughness 0.7 }}
      geometry Cylinder {{ height {h} radius {r} }}
    }}
  ]
  boundingObject Cylinder {{ height {h} radius {r} }}
  name "{name.lower()}"
}}"""


def _box(name, x, y, sx=0.3, sy=0.3, sz=0.3, color="0.5 0.3 0.7"):
    return f"""\
DEF {name} Solid {{
  translation {x} {y} 0.15
  children [
    Shape {{
      appearance PBRAppearance {{ baseColor {color} roughness 0.8 }}
      geometry Box {{ size {sx} {sy} {sz} }}
    }}
  ]
  boundingObject Box {{ size {sx} {sy} {sz} }}
  name "{name.lower()}"
}}"""


def _goal_marker(goal_x=2.0, goal_y=0.0):
    return f"""\
DEF GOAL_MARKER Solid {{
  translation {goal_x} {goal_y:.4f} 0.001
  children [
    Shape {{
      appearance PBRAppearance {{
        baseColor 0.0 0.9 0.2
        roughness 0.8
        emissiveColor 0.0 0.5 0.1
        transparency 0.2
      }}
      geometry Cylinder {{ height 0.002 radius 0.30 }}
    }}
  ]
  name "goal_marker"
}}"""


def _barriers(goal_y=0.0, wall_x=1.5, half_span=1.55):
    return f"""\
DEF BARRIER_TOP Solid {{
  translation {wall_x} {goal_y + half_span:.4f} 0.25
  children [
    Shape {{
      appearance PBRAppearance {{ baseColor 0.5 0.5 0.5 roughness 0.8 }}
      geometry Box {{ size 0.2 2.4 0.5 }}
    }}
  ]
  boundingObject Box {{ size 0.2 2.4 0.5 }}
  name "barrier_top"
}}

DEF BARRIER_BOTTOM Solid {{
  translation {wall_x} {goal_y - half_span:.4f} 0.25
  children [
    Shape {{
      appearance PBRAppearance {{ baseColor 0.5 0.5 0.5 roughness 0.8 }}
      geometry Box {{ size 0.2 2.4 0.5 }}
    }}
  ]
  boundingObject Box {{ size 0.2 2.4 0.5 }}
  name "barrier_bottom"
}}"""


# ---------------------------------------------------------------------------
# Obstacle pool — 18 entries.
#
# Addition schedule (world → cumulative count):
#   W2: 1   W3: 3   W4: 5   W5: 7   W6: 9
#   W7: 11  W8: 13  W9: 15  W10: 18
#
# Design principles:
#   • Obs 1 is a single, easy-to-avoid blocker slightly above the path.
#   • Each subsequent pair adds one flanker that narrows a bypass and
#     one that blocks a new region, so difficulty rises smoothly.
#   • Goal y shifts gradually: 0 → 0.20 → −0.30 → 0.50 → −0.60.
#   • All pairwise centre distances ≥ 0.70 m  (verified by _verify_pool)
#     → surface gap ≥ 0.40 m  (> 2× Altino body width of 0.09 m)
#
# Coordinate system: x ∈ [−2.5, 2.5], y ∈ [−2.5, 2.5]
#   start ≈ (−2.0, 0.0),  goal = (2.0, goal_y),  barrier at x = 1.5
# ---------------------------------------------------------------------------

_POOL = [
    # ── world 1→2  (+1 obs, goal y=0.0) ─────────────────────────────
    # Single obstacle just above the direct path — wide space below.
    # Agent only needs to go slightly below y=0 to pass.
    ("cyl",  0.00,  0.20, "0.8 0.2 0.2"),   #  1  centre, slightly above path

    # ── world 2→3  (+2 obs, goal y=0.0) ─────────────────────────────
    # One left-side blocker on the path, one right-side below.
    # Creates a gentle S-curve: dip below on the left, rise above on right.
    ("cyl", -0.80,  0.00, "0.2 0.5 0.8"),   #  2  left, on path
    ("cyl",  0.90, -0.20, "0.8 0.7 0.1"),   #  3  right, slightly below path

    # ── world 3→4  (+2 obs, goal y=0.0) ─────────────────────────────
    # Flankers that start to close off the high and low bypasses.
    ("cyl", -1.30,  0.70, "0.3 0.8 0.3"),   #  4  upper-left flanker
    ("cyl",  0.40, -0.80, "0.7 0.3 0.8"),   #  5  lower-centre flanker

    # ── world 4→5  (+2 obs, goal y=+0.20) ───────────────────────────
    # Upper-left corridor and lower-right corridor both narrowed.
    # Small goal shift upward — agent must aim slightly higher.
    ("cyl", -0.50,  0.90, "0.9 0.5 0.1"),   #  6  upper left-centre
    ("cyl",  1.20,  0.50, "0.5 0.3 0.7"),   #  7  upper right near barrier

    # ── world 5→6  (+2 obs, goal y=−0.30) ───────────────────────────
    # Mirror pressure on lower side; far-left blocker on the path.
    # Goal shifts below centre for the first time.
    ("box", -1.50, -0.30, "0.9 0.2 0.5"),   #  8  far-left lower, on path
    ("cyl",  0.20,  1.40, "0.2 0.8 0.4"),   #  9  upper-centre (closes high bypass)

    # ── world 6→7  (+2 obs, goal y=+0.50) ───────────────────────────
    # Lower zone starts to fill; goal shifts back up more aggressively.
    ("cyl", -0.70, -1.00, "0.8 0.2 0.6"),   # 10  lower left-centre
    ("cyl",  1.40, -0.70, "0.8 0.4 0.1"),   # 11  lower right near barrier

    # ── world 7→8  (+2 obs, goal y=−0.60) ───────────────────────────
    # Upper-left mid and lower-centre fill; bigger negative goal shift.
    ("cyl", -1.00,  1.40, "0.4 0.2 0.8"),   # 12  upper left-mid
    ("cyl",  0.70, -1.50, "0.9 0.2 0.2"),   # 13  lower centre-right

    # ── world 8→9  (+2 obs, randomise goal) ─────────────────────────
    # Far upper-left and lower left-centre to tighten the full arena.
    ("cyl", -1.50,  2.00, "0.2 0.7 0.7"),   # 14  far upper-left
    ("cyl", -0.30, -1.60, "0.6 0.8 0.2"),   # 15  lower centre-left

    # ── world 9→10  (+3 obs, randomise goal) ────────────────────────
    # Three final obstacles complete the densest layout.
    ("cyl",  0.90,  1.40, "0.8 0.6 0.2"),   # 16  upper right
    ("cyl", -1.00, -1.70, "0.4 0.6 0.8"),   # 17  far lower-left
    ("cyl",  1.30, -1.90, "0.8 0.4 0.6"),   # 18  far lower-right
]

# ---------------------------------------------------------------------------
# Verify all pairwise distances (abort with a clear message if too close).
# ---------------------------------------------------------------------------

import math as _math

_MIN_CENTRE_DIST = 0.70   # m  → surface gap ≥ 0.40 m

def _verify_pool():
    coords = [(_POOL[i][1], _POOL[i][2]) for i in range(len(_POOL))]
    violations = []
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            d = _math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
            if d < _MIN_CENTRE_DIST:
                violations.append(
                    f"  obs {i+1} ({coords[i]}) ↔ obs {j+1} ({coords[j]}): "
                    f"dist={d:.3f} m < {_MIN_CENTRE_DIST} m"
                )
    if violations:
        raise ValueError("Obstacle spacing violations:\n" + "\n".join(violations))

_verify_pool()


def _build_obstacles(n):
    """Return Solid snippet list for the first n entries of _POOL."""
    snippets = []
    for i, (kind, x, y, color) in enumerate(_POOL[:n]):
        name = f"TR_OBS_{i + 1}"
        if kind == "cyl":
            snippets.append(_cyl(name, x, y, color=color))
        else:
            snippets.append(_box(name, x, y, color=color))
    return snippets


# ---------------------------------------------------------------------------
# World definitions
# ---------------------------------------------------------------------------

WORLDS = [
    # ── Fixed goal at (2.0, 0.0) — learn basic navigation ───────────
    {
        "filename": "train_1_empty.wbt",
        "comment": "# Training 1 — 0 obstacles, goal fixed at (2.0, 0.0)",
        "n_obs": 0,
        "goal_y": 0.0,
        "randomize_goal": False,
    },
    {
        "filename": "train_2_one_obs.wbt",
        "comment": "# Training 2 — 1 obstacle, goal fixed at (2.0, 0.0)",
        "n_obs": 1,
        "goal_y": 0.0,
        "randomize_goal": False,
    },
    {
        "filename": "train_3_three_obs.wbt",
        "comment": "# Training 3 — 3 obstacles, goal fixed at (2.0, 0.0)",
        "n_obs": 3,
        "goal_y": 0.0,
        "randomize_goal": False,
    },
    {
        "filename": "train_4_five_obs.wbt",
        "comment": "# Training 4 — 5 obstacles, goal fixed at (2.0, 0.0)",
        "n_obs": 5,
        "goal_y": 0.0,
        "randomize_goal": False,
    },
    # ── Gradually shifting goal — learn to aim off-centre ───────────
    {
        "filename": "train_5_goal_shift_pos.wbt",
        "comment": "# Training 5 — 7 obstacles, goal fixed at (2.0, +0.20)",
        "n_obs": 7,
        "goal_y": 0.20,
        "randomize_goal": False,
    },
    {
        "filename": "train_6_goal_shift_neg.wbt",
        "comment": "# Training 6 — 9 obstacles, goal fixed at (2.0, -0.30)",
        "n_obs": 9,
        "goal_y": -0.30,
        "randomize_goal": False,
    },
    {
        "filename": "train_7_goal_offset_pos.wbt",
        "comment": "# Training 7 — 11 obstacles, goal fixed at (2.0, +0.50)",
        "n_obs": 11,
        "goal_y": 0.50,
        "randomize_goal": False,
    },
    {
        "filename": "train_8_goal_offset_neg.wbt",
        "comment": "# Training 8 — 13 obstacles, goal fixed at (2.0, -0.60)",
        "n_obs": 13,
        "goal_y": -0.60,
        "randomize_goal": False,
    },
    # ── Dense layouts, static goal ───────────────────────────────────
    {
        "filename": "train_9_dense.wbt",
        "comment": "# Training 9 — 15 obstacles, goal fixed at (2.0, 0.0)",
        "n_obs": 15,
        "goal_y": 0.0,
        "randomize_goal": False,
    },
    {
        "filename": "train_10_full.wbt",
        "comment": "# Training 10 — 18 obstacles, goal fixed at (2.0, 0.0)",
        "n_obs": 18,
        "goal_y": 0.0,
        "randomize_goal": False,
    },
]

# ---------------------------------------------------------------------------
# Parse template: extract header (up to first obstacle/barrier/ALTINO) and
# the ALTINO robot block (from "DEF ALTINO Robot {" to end of file).
# ---------------------------------------------------------------------------

raw = TEMPLATE.read_text(encoding="utf-8")
lines = raw.splitlines(keepends=True)

HEADER_STOP_RE = re.compile(r"^DEF (OBS_|BARRIER_|ALTINO )")
header_end = next(i for i, ln in enumerate(lines) if HEADER_STOP_RE.match(ln))
header = "".join(lines[:header_end])

altino_start = next(i for i, ln in enumerate(lines) if ln.startswith("DEF ALTINO Robot {"))
altino = "".join(lines[altino_start:])

# ---------------------------------------------------------------------------
# Write each world file
# ---------------------------------------------------------------------------

for world in WORLDS:
    obs_snippets = _build_obstacles(world["n_obs"])
    obs_block = "\n\n".join(obs_snippets)
    marker_block = _goal_marker(goal_y=world["goal_y"])
    barrier_block = _barriers(goal_y=world["goal_y"])

    # For randomise-goal worlds, embed the signal in WorldInfo.title so the
    # controller auto-enables randomisation without any manual config change.
    if world["randomize_goal"]:
        world_header = header.replace(
            "WorldInfo {\n}", 'WorldInfo {\n  title "randomize_goal"\n}'
        )
    else:
        world_header = header

    content = (
        world_header.rstrip("\n")
        + "\n\n"
        + world["comment"]
        + "\n\n"
        + (obs_block + "\n\n" if obs_block else "")
        + marker_block
        + "\n\n"
        + barrier_block
        + "\n\n"
        + altino
    )

    out_path = OUT_DIR / world["filename"]
    out_path.write_text(content, encoding="utf-8")
    rand_note = " [randomize_goal=True]" if world["randomize_goal"] else ""
    print(f"Written: {out_path}  ({world['n_obs']} obs, goal_y={world['goal_y']}{rand_note})")

print(f"\nDone. 10 training worlds written to {OUT_DIR}/")
