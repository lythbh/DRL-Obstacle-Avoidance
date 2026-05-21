"""Generate 10 static PPO training worlds with increasing obstacle density."""

import re
from pathlib import Path

TEMPLATE = Path(__file__).parent / "ObstacleCourse.wbt"
OUT_DIR = Path(__file__).parent / "training"
OUT_DIR.mkdir(exist_ok=True)


def _cyl(name, x, y, r=0.15, h=0.3, color="0.8 0.2 0.2"):
    """Build a cylinder-shaped obstacle Solid snippet."""
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
    """Build a box-shaped obstacle Solid snippet."""
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
    """Build the green goal marker cylinder Solid snippet."""
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
    """Build the barrier wall Solid snippets that create a gap at the goal y-position."""
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


_POOL = [
    ("cyl",  0.00,  0.20, "0.8 0.2 0.2"),
    ("cyl", -0.80,  0.00, "0.2 0.5 0.8"),
    ("cyl",  0.90, -0.20, "0.8 0.7 0.1"),
    ("cyl", -1.30,  0.70, "0.3 0.8 0.3"),
    ("cyl",  0.40, -0.80, "0.7 0.3 0.8"),
    ("cyl", -0.50,  0.90, "0.9 0.5 0.1"),
    ("cyl",  1.20,  0.50, "0.5 0.3 0.7"),
    ("box", -1.50, -0.30, "0.9 0.2 0.5"),
    ("cyl",  0.20,  1.40, "0.2 0.8 0.4"),
    ("cyl", -0.70, -1.00, "0.8 0.2 0.6"),
    ("cyl",  1.40, -0.70, "0.8 0.4 0.1"),
    ("cyl", -1.00,  1.40, "0.4 0.2 0.8"),
    ("cyl",  0.70, -1.50, "0.9 0.2 0.2"),
    ("cyl", -1.50,  2.00, "0.2 0.7 0.7"),
    ("cyl", -0.30, -1.60, "0.6 0.8 0.2"),
    ("cyl",  0.90,  1.40, "0.8 0.6 0.2"),
    ("cyl", -1.00, -1.70, "0.4 0.6 0.8"),
    ("cyl",  1.30, -1.90, "0.8 0.4 0.6"),
]

import math as _math

_MIN_CENTRE_DIST = 0.70

def _verify_pool():
    """Verify all obstacle pairwise distances meet minimum spacing requirement."""
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


def _build_obstacles(n: int) -> list:
    """Return Solid snippet list for the first n entries of _POOL."""
    snippets = []
    for i, (kind, x, y, color) in enumerate(_POOL[:n]):
        name = f"TR_OBS_{i + 1}"
        if kind == "cyl":
            snippets.append(_cyl(name, x, y, color=color))
        else:
            snippets.append(_box(name, x, y, color=color))
    return snippets


WORLDS = [
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

raw = TEMPLATE.read_text(encoding="utf-8")
lines = raw.splitlines(keepends=True)

HEADER_STOP_RE = re.compile(r"^DEF (OBS_|BARRIER_|ALTINO )")
header_end = next(i for i, ln in enumerate(lines) if HEADER_STOP_RE.match(ln))

_UNUSED_PROTOS = {"WaterBottle", "WoodenChair", "BeerBottle", "OilBarrel"}
header = "".join(
    ln for ln in lines[:header_end]
    if not any(p in ln for p in _UNUSED_PROTOS)
)

header = header.replace("WorldInfo {\n}", "WorldInfo {\n  basicTimeStep 64\n}")

altino_start = next(i for i, ln in enumerate(lines) if ln.startswith("DEF ALTINO Robot {"))
altino = "".join(lines[altino_start:])

for world in WORLDS:
    obs_snippets = _build_obstacles(world["n_obs"])
    obs_block = "\n\n".join(obs_snippets)
    marker_block = _goal_marker(goal_y=world["goal_y"])
    barrier_block = _barriers(goal_y=world["goal_y"])

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
