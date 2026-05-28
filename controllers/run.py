"""HPC and local launcher for PPO curriculum training in Webots.

Usage examples:
  python controllers/run.py submit --sessions 10 --episodes 2500
  python controllers/run.py worker --arch gru --seed 0
  python controllers/run.py worker --arch none --worlds worlds/training/train_1_empty.wbt
  python controllers/run.py worker --arch gru --seed 0 --worlds worlds/training/train_11_partial_moving.wbt --moving-obstacle-indices "0,1,2,3,4"
  python controllers/run.py worker --arch gru --seed 0 --worlds worlds/training/train_12_all_moving.wbt --moving-obstacle-indices all
  python controllers/run.py worker --arch gru --seed 0 --worlds worlds/training/train_13_moving_goal.wbt --moving-obstacle-indices all --moving-goal
"""

from __future__ import annotations

import argparse
import gc
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
PPO_CHECKPOINT_DIR = REPO_ROOT / "controllers" / "PPO" / "checkpoints"
SLURM_SCRIPT = REPO_ROOT / "slurm.sh"
TRAINING_WORLDS_DIR = REPO_ROOT / "worlds" / "training"
DEFAULT_ARCHES = ("none", "gru", "lstm")


def _default_training_worlds() -> list[str]:
    def natural_key(path: Path) -> list[object]:
        return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)]

    worlds = sorted(TRAINING_WORLDS_DIR.glob("*.wbt"), key=natural_key)
    if not worlds:
        raise FileNotFoundError(f"No .wbt training worlds found in {TRAINING_WORLDS_DIR}")
    return [str(path.relative_to(REPO_ROOT)) for path in worlds]


def _normalize_arch(arch: str) -> str:
    arch = arch.lower().strip()
    aliases = {"mlp": "none", "feedforward": "none", "ff": "none", "rnnless": "none"}
    arch = aliases.get(arch, arch)
    if arch not in {"none", "gru", "lstm"}:
        raise ValueError(f"Unsupported PPO architecture: {arch}")
    return arch


def _resolve_worlds(worlds: Iterable[str]) -> list[Path]:
    worlds = list(worlds) if worlds else _default_training_worlds()
    resolved = []
    for world in worlds:
        path = Path(world)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"World does not exist: {path}")
        resolved.append(path)
    return resolved


def _checkpoint_path(run_id: str) -> Path:
    return PPO_CHECKPOINT_DIR / run_id / f"final_{run_id}.pth"


def run_worker(args: argparse.Namespace) -> int:
    """Run one architecture through all worlds serially, resuming between worlds."""
    arch = _normalize_arch(args.arch)
    worlds = _resolve_worlds(args.worlds)
    run_group = args.run_group or datetime.now().strftime("%Y%m%d_%H%M%S")
    previous_checkpoint: Path | None = Path(args.resume_from).resolve() if args.resume_from else None

    for stage_index, world in enumerate(worlds, start=1):
        world_name = world.stem.replace(" ", "_")
        run_id = f"{run_group}_{arch}_seed{args.seed:02d}_stage{stage_index:02d}_{world_name}"
        final_checkpoint = _checkpoint_path(run_id)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PPO_ARCH"] = arch
        env["PPO_SEED"] = str(args.seed)
        env["PPO_RUN_ID"] = run_id
        if args.episodes is not None:
            env["PPO_EPISODES"] = str(args.episodes)
        if args.max_steps is not None:
            env["PPO_MAX_STEPS"] = str(args.max_steps)
        if previous_checkpoint is not None:
            env["PPO_LOAD_MODEL"] = str(previous_checkpoint)
        if args.moving_obstacle_indices is not None:
            env["PPO_MOVING_OBSTACLE_INDICES"] = args.moving_obstacle_indices
            env["PPO_MOVING_OBSTACLE_SPEED"] = str(args.moving_obstacle_speed)
            env["PPO_MOVING_OBSTACLE_AMPLITUDE"] = str(args.moving_obstacle_amplitude)
        if args.moving_goal:
            env["PPO_MOVING_GOAL"] = "1"
            env["PPO_MOVING_GOAL_SPEED"] = str(args.moving_goal_speed)
            env["PPO_MOVING_GOAL_AMPLITUDE"] = str(args.moving_goal_amplitude)

        env.setdefault("WEBOTS_CONTROLLER_PATH", str(REPO_ROOT / "controllers"))

        cmd = [
            "webots",
            "--no-rendering",
            "--batch",
            "--minimize",
            "--stdout",
            "--stderr",
            "--mode=fast",
            str(world),
        ]
        print(f"[RUN] arch={arch} seed={args.seed} stage={stage_index}/{len(worlds)} world={world}", flush=True)
        if previous_checkpoint:
            print(f"[RUN] resuming from {previous_checkpoint}", flush=True)
        result = subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=False)
        time.sleep(5)
        gc.collect()
        if result.returncode != 0:
            print(f"[RUN] Webots failed with exit code {result.returncode}: {world}", file=sys.stderr, flush=True)
            return result.returncode
        if not final_checkpoint.exists():
            print(f"[RUN] Missing expected final checkpoint: {final_checkpoint}", file=sys.stderr, flush=True)
            return 2
        previous_checkpoint = final_checkpoint
        print(f"[RUN] completed stage checkpoint={final_checkpoint}", flush=True)

    print(f"[RUN] curriculum complete arch={arch} seed={args.seed}", flush=True)
    return 0


def _worker_command(
    *,
    arch: str,
    seed: int,
    run_group: str,
    worlds: list[str],
    episodes: int | None,
    max_steps: int | None,
) -> str:
    cmd = [
        "python",
        "controllers/run.py",
        "worker",
        "--arch",
        arch,
        "--seed",
        str(seed),
        "--run-group",
        run_group,
    ]
    if episodes is not None:
        cmd.extend(["--episodes", str(episodes)])
    if max_steps is not None:
        cmd.extend(["--max-steps", str(max_steps)])
    cmd.append("--worlds")
    cmd.extend(worlds)
    return shlex.join(cmd)


def _sbatch_command(args: argparse.Namespace, *, arch: str, seed: int, run_group: str) -> list[str]:
    job_name = f"drl_ppo_{arch}_s{seed:02d}"
    worker_command = _worker_command(
        arch=arch,
        seed=seed,
        run_group=run_group,
        worlds=args.worlds or _default_training_worlds(),
        episodes=args.episodes,
        max_steps=args.max_steps,
    )
    return [
        "sbatch",
        "--job-name",
        job_name,
        "--account",
        args.account,
        "--time",
        args.time,
        "--partition",
        args.partition,
        "--gres",
        args.gres,
        "--output",
        f"logs/{job_name}_%j.out",
        "--error",
        f"logs/{job_name}_%j.err",
        "--export",
        f"ALL,PPO_RUN_COMMAND={worker_command},PPO_FORCE_CPU=0",
        str(SLURM_SCRIPT),
    ]


def run_submit(args: argparse.Namespace) -> int:
    if not SLURM_SCRIPT.exists():
        raise FileNotFoundError(f"Missing SLURM script: {SLURM_SCRIPT}")
    arches = [_normalize_arch(arch) for arch in args.arches]
    args.worlds = args.worlds or _default_training_worlds()
    run_group = args.run_group or datetime.now().strftime("hpc_%Y%m%d_%H%M%S")

    commands = []
    for arch in arches:
        for seed in range(args.seed_offset, args.seed_offset + args.sessions):
            commands.append(_sbatch_command(args, arch=arch, seed=seed, run_group=run_group))

    print(f"[RUN] prepared {len(commands)} sbatch submissions using {SLURM_SCRIPT}", flush=True)
    if args.no_submit:
        for command in commands:
            print(shlex.join(command), flush=True)
        return 0

    for command in commands:
        print(f"[RUN] {shlex.join(command)}", flush=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch PPO Webots curriculum training.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    worker = subparsers.add_parser("worker", help="Run one architecture through all curriculum worlds.")
    worker.add_argument("--arch", required=True, help="PPO architecture: none, gru, or lstm.")
    worker.add_argument("--seed", type=int, default=0)
    worker.add_argument("--run-group", default=None)
    worker.add_argument("--episodes", type=int, default=None)
    worker.add_argument("--max-steps", type=int, default=None)
    worker.add_argument("--resume-from", default=None)
    worker.add_argument("--worlds", nargs="+", default=None)
    worker.add_argument("--moving-obstacle-indices", default=None,
                        help="Comma-separated 0-indexed obstacle indices, or 'all'.")
    worker.add_argument("--moving-obstacle-speed", type=float, default=0.3)
    worker.add_argument("--moving-obstacle-amplitude", type=float, default=0.4)
    worker.add_argument("--moving-goal", action="store_true")
    worker.add_argument("--moving-goal-speed", type=float, default=0.2)
    worker.add_argument("--moving-goal-amplitude", type=float, default=0.5)
    worker.set_defaults(func=run_worker)

    submit = subparsers.add_parser("submit", help="Generate and optionally submit SLURM jobs.")
    submit.add_argument("--sessions", type=int, default=10, help="Parallel seeds per architecture.")
    submit.add_argument("--seed-offset", type=int, default=0)
    submit.add_argument("--arches", nargs="+", default=list(DEFAULT_ARCHES))
    submit.add_argument("--worlds", nargs="+", default=None)
    submit.add_argument("--episodes", type=int, default=None)
    submit.add_argument("--max-steps", type=int, default=None)
    submit.add_argument("--run-group", default=None)
    submit.add_argument("--time", default="00:30:00")
    submit.add_argument("--account", default="ec12")
    submit.add_argument("--partition", default="accel")
    submit.add_argument("--gres", default="gpu:1")
    submit.add_argument("--no-submit", action="store_true")
    submit.set_defaults(func=run_submit)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
