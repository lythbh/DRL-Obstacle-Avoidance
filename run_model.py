"""
Unified PPO CLI: inference, training and moving-world curriculum.

Usage:
  python run_model.py infer --episodes 10
  python run_model.py worker --arch gru --seed 0
  python run_model.py worker --arch none --worlds worlds/training/train_1_empty.wbt
  python run_model.py moving-curriculum --arch gru --seed 0
  python run_model.py submit --sessions 10 --episodes 2500

LLM level: 4 - LLM wrote the majority of the file, we did minor improvements and tested it.
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
import statistics
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional


REPO_ROOT = Path(__file__).resolve().parent
CHECKPOINT_ROOT = REPO_ROOT / "controllers" / "PPO" / "checkpoints"
SLURM_SCRIPT = REPO_ROOT / "slurm.sh"
TRAINING_WORLDS_DIR = REPO_ROOT / "worlds" / "training"
DEFAULT_ARCHES = ("none", "gru", "lstm")

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


@dataclass
class InferenceConfig:
    """
    Configuration for model inference.
    """
    model_path: Optional[str] = None
    episodes: int = 10
    show_progress: bool = True


def _default_model_path() -> str:
    """
    Find the most recently modified PPO checkpoint.
    
    Returns
    -------
    str
        Path to the most recently modified PPO checkpoint.
    """
    candidates = sorted(CHECKPOINT_ROOT.rglob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return str(candidates[0])
    
    return str(CHECKPOINT_ROOT.parent / "best_model.pth")


def _checkpoint_arch(path: Path) -> str | None:
    """
    Return the recurrent_cell stored in a checkpoint, or None on error.
    
    Parameters
    ----------
    path : Path
        Path to the checkpoint file.
    
    Returns
    -------
    str | None
        The recurrent_cell stored in the checkpoint, or None if not found or on error.
    """
    try:
        import torch
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        return str(ckpt.get("recurrent_cell", "")).lower().strip() or None
    except Exception:
        return None


def _latest_checkpoint(after: float | None = None, arch: str | None = None, seed: int | None = None) -> Path:
    """
    Locate the most recent PPO checkpoint matching the given criteria.

    Parameters
    ----------
    after : float | None, optional
        Only consider checkpoints modified after this timestamp.
    arch : str | None, optional
        Only consider checkpoints with this recurrent_cell.
    seed : int | None, optional
        Only consider checkpoints with this seed.

    Returns
    -------
    Path
        Path to the most recent matching checkpoint.
    """
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


def _default_training_worlds() -> list[str]:
    """
    Auto-discover training worlds sorted numerically.
    
    Returns
    -------
    list[str]
        List of paths to training worlds.
    """
    def natural_key(path: Path) -> list[object]:
        return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)]

    worlds = sorted(TRAINING_WORLDS_DIR.glob("*.wbt"), key=natural_key)
    if not worlds:
        raise FileNotFoundError(f"No .wbt training worlds found in {TRAINING_WORLDS_DIR}")
    
    return [str(path.relative_to(REPO_ROOT)) for path in worlds]


def _normalize_arch(arch: str) -> str:
    """
    Normalize the architecture name to a standard format.

    Parameters
    ----------
    arch : str
        The architecture name to normalize.

    Returns
    -------
    str
        The normalized architecture name.
    """
    arch = arch.lower().strip()
    aliases = {"mlp": "none", "feedforward": "none", "ff": "none", "rnnless": "none"}
    arch = aliases.get(arch, arch)
    if arch not in {"none", "gru", "lstm"}:
        raise ValueError(f"Unsupported PPO architecture: {arch}")
    
    return arch


def _resolve_worlds(worlds: Iterable[str]) -> list[Path]:
    """
    Resolve world paths to absolute paths, checking for existence.

    Parameters
    ----------
    worlds : Iterable[str]
        Iterable of world paths to resolve.

    Returns
    -------
    list[Path]
        List of resolved world paths.
    """
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
    """
    Checkpoint path for a given run ID.

    Parameters
    ----------
    run_id : str
        The run ID to get the checkpoint path for.

    Returns
    -------
    Path
        The checkpoint path.
    """
    return CHECKPOINT_ROOT / run_id / f"final_{run_id}.pth"


def run_inference(config=None) -> None:
    """
    Load trained PPO model, run episodes in Webots, and report performance metrics.
    
    Parameters
    ----------
    config : InferenceConfig, optional
        Configuration for inference. If None, uses default configuration.
    """
    if config is None:
        config = InferenceConfig()

    if config.episodes <= 0:
        print(f"[INFER] ERROR: episodes must be greater than 0, got {config.episodes}.")
        return

    model_path = config.model_path or _default_model_path()
    print(f"[INFER][PPO] model={model_path} episodes={config.episodes}", flush=True)

    try:
        import numpy as np
        import torch

        from controllers.PPO.PPO_config import Config as PPOConfig
        from controllers.PPO.PPO_agent import PPOAgent
        from controllers.Webots.webots_env import WebotsEnv, _init_supervisor
    except ImportError as e:
        print(f"[INFER] ERROR importing runtime dependencies: {e}")
        return

    _init_supervisor()

    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location="cpu")
    except FileNotFoundError:
        print(f"[INFER][PPO] ERROR: model file not found: {model_path}")
        return
    except Exception as e:
        print(f"[INFER][PPO] ERROR loading model metadata: {e}")
        return

    checkpoint_algorithm = str(checkpoint.get("algorithm", "ppo")).lower().strip()
    if checkpoint_algorithm != "ppo":
        print(f"[INFER][PPO] ERROR: checkpoint algorithm '{checkpoint_algorithm}' is not PPO.")
        return

    saved_config = checkpoint.get("config")
    if not isinstance(saved_config, dict):
        saved_config = {}

    train_config = PPOConfig(**{k: v for k, v in saved_config.items() if k in {f.name for f in fields(PPOConfig)}})
    train_config.recurrent_cell = str(checkpoint.get("recurrent_cell", "gru")).lower().strip()
    reward_computer = checkpoint.get("reward_computer")
    if reward_computer is None and isinstance(saved_config, dict):
        reward_computer = saved_config.get("reward_computer")
    if reward_computer is None:
        from controllers.PPO.PPO_rewards import PPORewardComputer
        reward_computer = PPORewardComputer(endpoint=train_config.endpoint)
    env = WebotsEnv(train_config, reward_computer=reward_computer)
    obs_size = env.observation_size
    n_actions = env.action_dim
    agent = PPOAgent(obs_size, n_actions, train_config)

    try:
        agent.load_model(model_path)
        print(f"[INFER][PPO] loaded episode={checkpoint.get('episode', 'unknown')} reward={checkpoint.get('reward', 'unknown')} goal={checkpoint.get('goal_episode', False)} cell={train_config.recurrent_cell.upper()}", flush=True)
    except Exception as e:
        print(f"[INFER][PPO] ERROR loading model: {e}")
        return

    agent.model.eval()
    agent.actor_log_std.requires_grad_(False)

    print(f"[INFER][PPO] running {config.episodes} episodes", flush=True)

    total_rewards = []
    goal_count = 0
    start_time = time.perf_counter()

    try:
        for episode in range(config.episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0.0
            steps = 0
            episode_end_reason = "max_steps"
            recurrent_state = agent.get_initial_state(batch_size=1)
            prev_done = True

            while not done:
                action, _, _, recurrent_state = agent.select_action(
                    obs, recurrent_state=recurrent_state, done=prev_done, deterministic=True,
                )

                obs_next, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                prev_done = done
                episode_reward += reward
                steps += 1

                if done:
                    if info.get("reset_reason") == "goal":
                        episode_end_reason = "goal"
                        goal_count += 1
                    elif info.get("reset_reason") == "collision":
                        episode_end_reason = "collision"
                    elif info.get("reset_reason") == "low_score":
                        episode_end_reason = "low_score"

                obs = obs_next

            total_rewards.append(episode_reward)

            if config.show_progress:
                elapsed = time.perf_counter() - start_time
                print(f"[INFER][PPO] ep={episode + 1:03d}/{config.episodes} r={episode_reward:8.2f} steps={steps:4d} min_d={env.min_episode_distance:5.2f} end={episode_end_reason} t={elapsed:7.1f}s", flush=True)

        avg_reward = float(np.mean(total_rewards))
        std_reward = float(statistics.pstdev(total_rewards))
        success_rate = goal_count / config.episodes * 100
        elapsed = time.perf_counter() - start_time

        print(f"[INFER][PPO] summary avg={avg_reward:.2f} std={std_reward:.2f} success={success_rate:.1f}% ({goal_count}/{config.episodes}) t={elapsed:7.1f}s", flush=True)
    finally:
        try:
            env.robot.stop()
        except Exception:
            pass

    print(f"[INFER][PPO] done", flush=True)


def run_worker(args: argparse.Namespace) -> int:
    """
    Run one architecture through all worlds serially, resuming between worlds.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.
    
    Returns
    -------
    int
        Exit code.
    """
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

        port = int(os.environ.get("WEBOTS_PORT", 1234))
        cmd = [
            "webots",
            f"--port={port}",
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


def _build_moving_command(args: argparse.Namespace, stage: dict[str, object], resume_from: Path, run_group: str) -> list[str]:
    """
    Build subprocess command for a moving-curriculum stage.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.
    stage : dict[str, object]
        Stage configuration.
    resume_from : Path
        Path to the checkpoint to resume from.
    run_group : str
        Run group name.
    
    Returns
    -------
    list[str]
        Command to run the stage.
    """
    cmd = [
        sys.executable,
        "run_model.py",
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


def run_moving_curriculum(args: argparse.Namespace) -> int:
    """
    Run the moving-world PPO curriculum stages sequentially.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.
    
    Returns
    -------
    int
        Exit code.
    """
    arch = args.arch
    resume_from = Path(args.resume_from).resolve() if args.resume_from else _latest_checkpoint(arch=arch, seed=args.seed)
    run_group_prefix = args.run_group or datetime.now().strftime("moving_%Y%m%d_%H%M%S")

    print(f"[MOVING] arch={arch} initial checkpoint={resume_from}", flush=True)
    for index, stage in enumerate(STAGES, start=1):
        run_group = f"{run_group_prefix}_step{index:02d}_{stage['name']}"
        cmd = _build_moving_command(args, stage, resume_from, run_group)
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


def _worker_command(
    *,
    arch: str,
    seed: int,
    run_group: str,
    worlds: list[str],
    episodes: int | None,
    max_steps: int | None,
) -> str:
    """
    Build worker CLI string for SLURM export.
    
    Parameters
    ----------
    arch : str
        Architecture name.
    seed : int
        Random seed.
    run_group : str
        Run group name.
    worlds : list[str]
        List of world names.
    episodes : int | None
        Number of episodes to run.
    max_steps : int | None
        Maximum number of steps per episode.
    
    Returns
    -------
    str
        Worker CLI string.
    """
    cmd = [
        "python",
        "run_model.py",
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
    """
    Build sbatch command for a single curriculum run.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.
    arch : str
        Architecture name.
    seed : int
        Random seed.
    run_group : str
        Run group name.
    
    Returns
    -------
    list[str]
        sbatch command.
    """
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
        "--job-name", job_name,
        "--account", args.account,
        "--time", args.time,
        "--partition", args.partition,
        "--gres", args.gres,
        "--output", f"logs/{job_name}_%j.out",
        "--error", f"logs/{job_name}_%j.err",
        "--export", f"ALL,PPO_RUN_COMMAND={worker_command},PPO_FORCE_CPU=0",
        str(SLURM_SCRIPT),
    ]


def run_submit(args: argparse.Namespace) -> int:
    """
    Submit multiple sbatch jobs for curriculum training.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.
    
    Returns
    -------
    int
        Exit code.
    """
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
    """
    Build argument parser for the CLI.
    
    Returns
    -------
    argparse.ArgumentParser
        Argument parser.
    """
    parser = argparse.ArgumentParser(description="PPO training, inference, and curriculum launcher.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    infer = subparsers.add_parser("infer", help="Run trained PPO model inference in Webots.")
    infer.add_argument("--model-path", default=None)
    infer.add_argument("--episodes", type=int, default=InferenceConfig.episodes)
    infer.add_argument("--quiet", "--no-render", dest="quiet", action="store_true")

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

    moving = subparsers.add_parser("moving-curriculum", help="Run PPO moving-world stages sequentially.")
    moving.add_argument("--arch", default="gru", help="PPO architecture to train. Default: gru.")
    moving.add_argument("--seed", type=int, default=0)
    moving.add_argument("--resume-from", default=None, help="Initial checkpoint. Defaults to newest PPO checkpoint.")
    moving.add_argument("--episodes", type=int, default=None)
    moving.add_argument("--max-steps", type=int, default=None)
    moving.add_argument("--run-group", default=None)
    moving.add_argument("--dry-run", action="store_true", help="Print commands without running Webots.")

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

    return parser


def main(argv: list[str] | None = None) -> int:
    """
    Main entry point for the CLI and the Webots controller.

    Parameters
    ----------
    argv : list[str] | None
        Command line arguments. If None, use sys.argv[1:].
    
    Returns
    -------
    int
        Exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "infer":
        run_inference(
            InferenceConfig(
                model_path=args.model_path,
                episodes=args.episodes,
                show_progress=not args.quiet,
            )
        )
        return 0
    elif args.command == "worker":
        return run_worker(args)
    elif args.command == "moving-curriculum":
        return run_moving_curriculum(args)
    elif args.command == "submit":
        return run_submit(args)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
