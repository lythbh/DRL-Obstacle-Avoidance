"""Generate 5 static validation world files from the ObstacleCourse.wbt template.

Run once from the repository root:
    python worlds/generate_validation_worlds.py

Each world is written to worlds/validation/ with fixed obstacle layouts that
the PPO controller will not randomize (they are baked into the file).
The validation worlds cover three difficulty levels:
  - Stage 1 (empty): val_1_empty_center.wbt, val_2_empty_offset.wbt
  - Stage 2 (sparse): val_3_sparse_a.wbt, val_4_sparse_b.wbt
  - Stage 3 (dense):  val_5_dense.wbt
"""
import re
from pathlib import Path

TEMPLATE = Path(__file__).parent / "ObstacleCourse.wbt"
OUT_DIR   = Path(__file__).parent / "validation"
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Obstacle snippets — inline Solid cylinders and boxes, no PROTO downloads
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


# ---------------------------------------------------------------------------
# Barrier walls (gap centred on goal_y)
# ---------------------------------------------------------------------------

def _goal_marker(goal_y=0.0, goal_x=2.0):
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
# Read template: split into header (up to first DEF OBS_) and ALTINO block.
# Patch WorldInfo to include basicTimeStep 16 if not already set.
# ---------------------------------------------------------------------------

raw = TEMPLATE.read_text(encoding="utf-8")
# Ensure basicTimeStep is set for physics stability
    # No physics overrides — keep the original WorldInfo defaults (32 ms, ODE defaults)
    # which were verified to produce no physics warnings with this robot model.
lines = raw.splitlines(keepends=True)

# Header: everything up to (but not including) the first obstacle DEF or BARRIER
HEADER_STOP_RE = re.compile(r"^DEF (OBS_|BARRIER_|ALTINO )")
header_end = next(i for i, ln in enumerate(lines) if HEADER_STOP_RE.match(ln))
header = "".join(lines[:header_end])

# ALTINO block: from "DEF ALTINO Robot {" to end
altino_start = next(i for i, ln in enumerate(lines) if ln.startswith("DEF ALTINO Robot {"))
altino = "".join(lines[altino_start:])


# ---------------------------------------------------------------------------
# World definitions
# ---------------------------------------------------------------------------

WORLDS = [
    {
        "filename": "val_1_empty_center.wbt",
        "comment": "# Validation 1 — Stage 1: empty arena, goal centred at y=0",
        "goal_y": 0.0,
        "obstacles": [],
    },
    {
        "filename": "val_2_empty_offset.wbt",
        "comment": "# Validation 2 — Stage 1: empty arena, goal shifted to y=0.5",
        "goal_y": 0.5,
        "obstacles": [],
    },
    {
        # 5 obstacles verified: all surface gaps ≥ 0.45 m, all ≥ 0.9 m from start, ≥ 0.8 m from goal
        "filename": "val_3_sparse_a.wbt",
        "comment": "# Validation 3 — Stage 2A: 5 obstacles, open corridor",
        "goal_y": 0.0,
        "obstacles": [
            _cyl("VAL_CYL_1",  0.5,  0.6, color="0.8 0.2 0.2"),
            _cyl("VAL_CYL_2", -0.8, -0.5, color="0.2 0.5 0.8"),
            _cyl("VAL_CYL_3",  1.2, -0.7, color="0.8 0.7 0.1"),
            _box("VAL_BOX_1", -0.3,  0.8, color="0.5 0.3 0.7"),
            _box("VAL_BOX_2",  0.3, -0.9, color="0.9 0.4 0.1"),
        ],
    },
    {
        # 5 obstacles verified: all surface gaps ≥ 0.45 m
        "filename": "val_4_sparse_b.wbt",
        "comment": "# Validation 4 — Stage 2B: 5 obstacles, partial corridor block",
        "goal_y": 0.0,
        "obstacles": [
            _cyl("VAL_CYL_1",  0.0,  0.3, color="0.2 0.8 0.4"),
            _cyl("VAL_CYL_2",  0.8, -0.4, color="0.8 0.4 0.1"),
            _cyl("VAL_CYL_3", -1.0,  0.7, color="0.4 0.2 0.8"),
            _box("VAL_BOX_1",  0.5,  1.1, color="0.9 0.2 0.5"),
            _box("VAL_BOX_2", -0.5, -0.8, color="0.2 0.7 0.7"),
        ],
    },
    {
        # 10 obstacles verified: all surface gaps ≥ 0.45 m, all ≥ 0.9 m from start, ≥ 0.8 m from goal
        "filename": "val_5_dense.wbt",
        "comment": "# Validation 5 — Stage 3: 10 obstacles, dense layout",
        "goal_y": 0.0,
        "obstacles": [
            _cyl("VAL_CYL_1", -1.0, -1.0, color="0.8 0.2 0.2"),
            _cyl("VAL_CYL_2", -0.8,  0.8, color="0.2 0.5 0.8"),
            _cyl("VAL_CYL_3",  0.0, -0.7, color="0.8 0.7 0.1"),
            _cyl("VAL_CYL_4",  0.5,  0.7, color="0.3 0.8 0.3"),
            _cyl("VAL_CYL_5",  1.0, -0.5, color="0.7 0.3 0.8"),
            _cyl("VAL_CYL_6", -0.6,  0.0, color="0.9 0.5 0.1"),
            _cyl("VAL_CYL_7",  1.3,  0.5, color="0.5 0.3 0.7"),
            _cyl("VAL_CYL_8", -0.3, -1.6, color="0.9 0.4 0.1"),
            _cyl("VAL_CYL_9",  0.0,  1.5, color="0.2 0.8 0.4"),
            _cyl("VAL_CYL_10", 0.8, -1.5, color="0.8 0.2 0.6"),
        ],
    },
]


# ---------------------------------------------------------------------------
# Write each world
# ---------------------------------------------------------------------------

for world in WORLDS:
    obs_block = "\n\n".join(world["obstacles"])
    barrier_block = _barriers(goal_y=world["goal_y"])
    marker_block  = _goal_marker(goal_y=world["goal_y"])
    content = (
        header.rstrip("\n")
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
    print(f"Written: {out_path}")

print("Done. 5 validation worlds created in worlds/validation/")
