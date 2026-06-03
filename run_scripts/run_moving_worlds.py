"""Run the moving-world PPO curriculum stages sequentially.

The first stage resumes from the latest checkpoint in controllers/PPO/checkpoints
unless --resume-from is provided. Each following stage resumes from the newest
checkpoint produced by the previous stage.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
CHECKPOINT_ROOT = REPO_ROOT / "controllers" / "PPO" / "checkpoints"

STAGES = [
    {
        "name": "partial_moving_1",
        "world": "worlds/training/train_11_partial_moving.wbt",
        "moving_obstacle_indices": "0",
        "moving_goal": False,
    },
    {
        "name": "partial_moving_3",
        "world": "worlds/training/train_11_partial_moving.wbt",
        "moving_obstacle_indices": "0,1,2",
        "moving_goal": False,
    },
    {
        "name": "partial_moving_5",
        "world": "worlds/training/train_11_partial_moving.wbt",
        "moving_obstacle_indices": "0,1,2,3,4",
        "moving_goal": False,
    },
    {
        "name": "all_moving",
        "world": "worlds/training/train_12_all_moving.wbt",
        "moving_obstacle_indices": "all",
        "moving_goal": False,
    },
    {
        "name": "moving_goal",
        "world": "worlds/training/train_13_moving_goal.wbt",
        "moving_obstacle_indices": "all",
        "moving_goal": True,
    },
]


def _checkpoint_arch(path: Path) -> str | None:
    """Return the recurrent_cell stored in a checkpoint, or None on error."""
    try:
        import torch
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        return str(ckpt.get("recurrent_cell", "")).lower().strip() or None
    except Exception:
        return None


def _latest_checkpoint(after: float | None = None, arch: str | None = None, seed: int | None = None) -> Path:
    candidates = [path for path in CHECKPOINT_ROOT.rglob("*.pth") if after is None or path.stat().st_mtime >= after]
    if not candidates:
        detail = f" modified after {after}" if after is not None else ""
        raise FileNotFoundError(f"No PPO checkpoints found in {CHECKPOINT_ROOT}{detail}.")

    if arch is not None:
        arch_norm = {"mlp": "none", "feedforward": "none", "ff": "none"}.get(arch, arch)
        matched = [p for p in candidates if _checkpoint_arch(p) == arch_norm]
        if not matched:
            raise FileNotFoundError(
                f"No PPO checkpoints with recurrent_cell='{arch}' found in {CHECKPOINT_ROOT}."
            )
        candidates = matched

    if seed is not None:
        seed_tag = f"seed{seed:02d}"
        matched = [p for p in candidates if seed_tag in str(p)]
        if not matched:
            raise FileNotFoundError(
                f"No PPO checkpoints containing '{seed_tag}' found in {CHECKPOINT_ROOT}."
            )
        candidates = matched

    finals = [path for path in candidates if path.name.startswith("final_")]
    pool = finals or candidates
    return max(pool, key=lambda path: path.stat().st_mtime)


def _build_command(args: argparse.Namespace, stage: dict[str, object], resume_from: Path, run_group: str) -> list[str]:
    cmd = [
        sys.executable,
        "controllers/run.py",
        "worker",
        "--arch",
        args.arch,
        "--seed",
        str(args.seed),
        "--run-group",
        run_group,
        "--resume-from",
        str(resume_from),
        "--worlds",
        str(stage["world"]),
        "--moving-obstacle-indices",
        str(stage["moving_obstacle_indices"]),
    ]
    if args.episodes is not None:
        cmd.extend(["--episodes", str(args.episodes)])
    if args.max_steps is not None:
        cmd.extend(["--max-steps", str(args.max_steps)])
    if stage["moving_goal"]:
        cmd.append("--moving-goal")
    return cmd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PPO moving-world stages sequentially.")
    parser.add_argument("--arch", default="gru", help="PPO architecture to train. Default: gru.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume-from", default=None, help="Initial checkpoint. Defaults to newest PPO checkpoint.")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--run-group", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running Webots.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    arch = args.arch
    resume_from = Path(args.resume_from).resolve() if args.resume_from else _latest_checkpoint(arch=arch, seed=args.seed)
    run_group_prefix = args.run_group or datetime.now().strftime("moving_%Y%m%d_%H%M%S")

    print(f"[MOVING] arch={arch} initial checkpoint={resume_from}", flush=True)
    for index, stage in enumerate(STAGES, start=1):
        run_group = f"{run_group_prefix}_step{index:02d}_{stage['name']}"
        cmd = _build_command(args, stage, resume_from, run_group)
        print(f"[MOVING] stage={index}/{len(STAGES)} name={stage['name']}", flush=True)
        print(f"[MOVING] resume_from={resume_from}", flush=True)
        print(f"[MOVING] command={' '.join(cmd)}", flush=True)

        if args.dry_run:
            continue

        started_at = time.time()
        result = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
        if result.returncode != 0:
            print(f"[MOVING] stage failed with exit code {result.returncode}: {stage['name']}", file=sys.stderr, flush=True)
            return result.returncode

        resume_from = _latest_checkpoint(after=started_at, arch=arch, seed=args.seed)
        print(f"[MOVING] completed stage checkpoint={resume_from}", flush=True)

    print(f"[MOVING] moving-world curriculum complete final_checkpoint={resume_from}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
