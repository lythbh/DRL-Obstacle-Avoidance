"""HPC and local launcher for PPO curriculum training in Webots.

Usage examples:
  python controllers/run.py submit --sessions 10 --episodes 2500
  python controllers/run.py worker --arch gru --seed 0
  python controllers/run.py worker --arch none --worlds worlds/SimplePPO.wbt worlds/ObstacleCourse.wbt
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
PPO_CHECKPOINT_DIR = REPO_ROOT / "controllers" / "PPO" / "checkpoints"
DEFAULT_ARCHES = ("none", "gru", "lstm")
DEFAULT_WORLDS = (
    "worlds/SimplePPO.wbt",
    "worlds/Simple.wbt",
    "worlds/validation/val_1_empty_center.wbt",
    "worlds/validation/val_2_empty_offset.wbt",
    "worlds/validation/val_3_sparse_a.wbt",
    "worlds/validation/val_4_sparse_b.wbt",
    "worlds/validation/val_5_dense.wbt",
    "worlds/ObstacleCourse.wbt",
)

HPC_REPO_ROOT = "/fp/homes01/u01/ec-esbrovol/fys5429/DRL-Obstacle-Avoidance"
HPC_WEBOTS_HOME = "/fp/homes01/u01/ec-esbrovol/fys5429/webots"
HPC_TORCH_SITE = "/fp/homes01/u01/ec-esbrovol/.local/lib/python3.10/site-packages"


def _normalize_arch(arch: str) -> str:
    arch = arch.lower().strip()
    aliases = {"mlp": "none", "feedforward": "none", "ff": "none", "rnnless": "none"}
    arch = aliases.get(arch, arch)
    if arch not in {"none", "gru", "lstm"}:
        raise ValueError(f"Unsupported PPO architecture: {arch}")
    return arch


def _resolve_worlds(worlds: Iterable[str]) -> list[Path]:
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
    previous_checkpoint: Path | None = Path(args.resume_from) if args.resume_from else None

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


def _hpc_world_args(worlds: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(Path(w))) for w in worlds)


def _write_sbatch_script(
    script_path: Path,
    *,
    arch: str,
    seed: int,
    run_group: str,
    worlds: list[str],
    episodes: int | None,
    max_steps: int | None,
    time_limit: str,
    account: str,
    partition: str,
    gres: str,
) -> None:
    episodes_arg = f" --episodes {episodes}" if episodes is not None else ""
    max_steps_arg = f" --max-steps {max_steps}" if max_steps is not None else ""
    world_args = _hpc_world_args(worlds)
    job_name = f"drl_ppo_{arch}_s{seed:02d}"
    content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account={account}
#SBATCH --time={time_limit}
#SBATCH --partition={partition}
#SBATCH --gres={gres}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/{job_name}_%j.out
#SBATCH --error=logs/{job_name}_%j.err

set -e
mkdir -p logs

module load Python/3.10.8-GCCcore-12.2.0
module load GCCcore/12.2.0
module load CUDA/12.1.1

nvidia-smi
python -c "import sys; sys.path.insert(0,'{HPC_TORCH_SITE}'); import torch; print('CUDA available:', torch.cuda.is_available())" || echo "torch not found yet"

export WEBOTS_HOME={HPC_WEBOTS_HOME}
export PATH=$WEBOTS_HOME:$PATH
export LD_LIBRARY_PATH=$WEBOTS_HOME/lib/webots:/cluster/software/EL9/easybuild/software/X11/20221110-GCCcore-12.2.0/lib:$LD_LIBRARY_PATH

export WEBOTS_TMPDIR=/fp/homes01/u01/ec-esbrovol/fys5429/tmp/webots_$SLURM_JOB_ID
mkdir -p $WEBOTS_TMPDIR
mkdir -p /tmp/webots/ec-esbrovol
mkdir -p /tmp/.X11-unix

export EGL_PLATFORM=x11
export LIBGL_ALWAYS_SOFTWARE=1

DISPLAY_NUM=$((90 + SLURM_JOB_ID % 900))
Xvfb :$DISPLAY_NUM -screen 0 1024x768x24 -nolisten tcp &
XVFB_PID=$!
export DISPLAY=:$DISPLAY_NUM
sleep 3

if ! kill -0 $XVFB_PID 2>/dev/null; then
    echo "ERROR: Xvfb failed to start"
    exit 1
fi
echo "Xvfb started (PID $XVFB_PID)"

cleanup() {{
    kill $XVFB_PID 2>/dev/null || true
    rm -rf $WEBOTS_TMPDIR
}}
trap cleanup EXIT

cd {HPC_REPO_ROOT}
source {HPC_REPO_ROOT}/.venv/bin/activate
export PYTHONPATH={HPC_TORCH_SITE}:{HPC_REPO_ROOT}:$WEBOTS_HOME/lib/controller/python:$PYTHONPATH
export PYTHONUNBUFFERED=1
export PPO_FORCE_CPU=0

python controllers/run.py worker --arch {arch} --seed {seed} --run-group {shlex.quote(run_group)}{episodes_arg}{max_steps_arg} --worlds {world_args}
"""
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(content, encoding="utf-8")
    script_path.chmod(0o755)


def run_submit(args: argparse.Namespace) -> int:
    arches = [_normalize_arch(arch) for arch in args.arches]
    run_group = args.run_group or datetime.now().strftime("hpc_%Y%m%d_%H%M%S")
    script_dir = REPO_ROOT / "hpc" / run_group
    hpc_worlds = [str(Path(w)) for w in args.worlds]

    scripts = []
    for arch in arches:
        for seed in range(args.seed_offset, args.seed_offset + args.sessions):
            script_path = script_dir / f"ppo_{arch}_seed{seed:02d}.sbatch"
            _write_sbatch_script(
                script_path,
                arch=arch,
                seed=seed,
                run_group=run_group,
                worlds=hpc_worlds,
                episodes=args.episodes,
                max_steps=args.max_steps,
                time_limit=args.time,
                account=args.account,
                partition=args.partition,
                gres=args.gres,
            )
            scripts.append(script_path)

    print(f"[RUN] wrote {len(scripts)} sbatch scripts to {script_dir}", flush=True)
    if args.no_submit:
        return 0

    for script in scripts:
        print(f"[RUN] sbatch {script}", flush=True)
        subprocess.run(["sbatch", str(script)], cwd=REPO_ROOT, check=True)
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
    worker.add_argument("--worlds", nargs="+", default=list(DEFAULT_WORLDS))
    worker.set_defaults(func=run_worker)

    submit = subparsers.add_parser("submit", help="Generate and optionally submit SLURM jobs.")
    submit.add_argument("--sessions", type=int, default=10, help="Parallel seeds per architecture.")
    submit.add_argument("--seed-offset", type=int, default=0)
    submit.add_argument("--arches", nargs="+", default=list(DEFAULT_ARCHES))
    submit.add_argument("--worlds", nargs="+", default=list(DEFAULT_WORLDS))
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
