#!/usr/bin/env python3
"""Automate Webots validation runs for PPO static and moving architectures."""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import queue
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent
PPO_CHECKPOINT_DIR = REPO_ROOT / "controllers" / "PPO" / "checkpoints"
VALIDATION_DIR = REPO_ROOT / "worlds" / "validation"
CONTROLLERS_DIR = REPO_ROOT / "controllers"
DEFAULT_PORT = 1234

STATIC_WORLD_GLOBS = ["val_1_*.wbt", "val_2_*.wbt", "val_3_*.wbt", "val_4_*.wbt", "val_5_*.wbt"]
MOVING_WORLD_GLOBS = ["val_6_*.wbt", "val_7_*.wbt", "val_8_*.wbt", "val_9_*.wbt"]

STATIC_MODELS = {
    "gru": [
        "final_20260525_141956_gru_seed00_stage10_train_10_full.pth",
        "final_20260525_141956_gru_seed01_stage10_train_10_full.pth",
        "final_20260525_141956_gru_seed02_stage10_train_10_full.pth",
        "final_20260527_131325_gru_seed03_stage10_train_10_full.pth",
        "final_20260527_131325_gru_seed04_stage10_train_10_full.pth",
        "final_20260527_131325_gru_seed05_stage10_train_10_full.pth",
    ],
    "none": [
        "final_20260525_141956_none_seed00_stage10_train_10_full.pth",
        "final_20260525_141956_none_seed01_stage10_train_10_full.pth",
        "final_20260525_141956_none_seed02_stage10_train_10_full.pth",
        "final_20260527_131325_none_seed03_stage10_train_10_full.pth",
        "final_20260527_131325_none_seed04_stage10_train_10_full.pth",
        "final_20260527_131325_none_seed05_stage10_train_10_full.pth",
    ],
    "lstm": [
        "final_20260525_141956_lstm_seed00_stage10_train_10_full.pth",
        "final_20260525_141956_lstm_seed01_stage10_train_10_full.pth",
        "final_20260525_141956_lstm_seed02_stage10_train_10_full.pth",
        "final_20260527_131325_lstm_seed03_stage10_train_10_full.pth",
        "final_20260527_131325_lstm_seed04_stage10_train_10_full.pth",
        "final_20260527_131325_lstm_seed05_stage10_train_10_full.pth",
    ],
}

MOVING_MODELS = {
    "none": [
        "final_moving_20260603_110455_step05_moving_goal_none_seed00_stage01_train_13_moving_goal.pth",
        "final_moving_20260603_110455_step05_moving_goal_none_seed01_stage01_train_13_moving_goal.pth",
        "final_moving_20260603_110455_step05_moving_goal_none_seed02_stage01_train_13_moving_goal.pth",
        "final_moving_20260603_110455_step05_moving_goal_none_seed03_stage01_train_13_moving_goal.pth",
        "final_moving_20260603_110455_step05_moving_goal_none_seed04_stage01_train_13_moving_goal.pth",
        "final_moving_20260603_110455_step05_moving_goal_none_seed05_stage01_train_13_moving_goal.pth",
    ],
    "gru": [
        "final_moving_20260603_162611_step05_moving_goal_gru_seed00_stage01_train_13_moving_goal.pth",
        "final_moving_20260603_162611_step05_moving_goal_gru_seed01_stage01_train_13_moving_goal.pth",
        "final_moving_20260603_162611_step05_moving_goal_gru_seed02_stage01_train_13_moving_goal.pth",
        "final_moving_20260603_162611_step05_moving_goal_gru_seed03_stage01_train_13_moving_goal.pth",
        "final_moving_20260603_162611_step05_moving_goal_gru_seed04_stage01_train_13_moving_goal.pth",
        "final_moving_20260603_162611_step05_moving_goal_gru_seed05_stage01_train_13_moving_goal.pth",
    ],
    "lstm": [
        "final_moving_20260602_230908_step05_moving_goal_lstm_seed00_stage01_train_13_moving_goal.pth",
        "final_moving_20260602_230908_step05_moving_goal_lstm_seed01_stage01_train_13_moving_goal.pth",
        "final_moving_20260602_230908_step05_moving_goal_lstm_seed02_stage01_train_13_moving_goal.pth",
        "final_moving_20260602_230908_step05_moving_goal_lstm_seed03_stage01_train_13_moving_goal.pth",
        "final_moving_20260602_230908_step05_moving_goal_lstm_seed04_stage01_train_13_moving_goal.pth",
        "final_moving_20260602_230908_step05_moving_goal_lstm_seed05_stage01_train_13_moving_goal.pth",
    ],
}


def natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)]


def resolve_worlds(globs: Iterable[str]) -> list[Path]:
    worlds = []
    for glob_pattern in globs:
        worlds.extend(sorted(VALIDATION_DIR.glob(glob_pattern), key=natural_key))
    if not worlds:
        raise FileNotFoundError(f"No validation worlds found in {VALIDATION_DIR}")
    return worlds


@dataclass(frozen=True)
class ValidationRun:
    arch: str
    seed: int
    checkpoint: Path
    world: Path
    run_id: str
    output_group: str


def build_checkpoint_path(checkpoint_name: str) -> Path:
    run_id = checkpoint_name.removeprefix("final_").removesuffix(".pth")
    candidate = PPO_CHECKPOINT_DIR / run_id / checkpoint_name
    if not candidate.exists():
        raise FileNotFoundError(f"Checkpoint not found: {candidate}")
    return candidate


def make_evaluation_runs(worlds: list[Path], models: dict[str, list[str]], output_group: str) -> list[ValidationRun]:
    runs: list[ValidationRun] = []
    for arch, filenames in models.items():
        for filename in filenames:
            checkpoint = build_checkpoint_path(filename)
            seed_match = re.search(r"_seed(\d{2})_", filename)
            seed = int(seed_match.group(1)) if seed_match else 0
            for world in worlds:
                run_id = f"{output_group}/{arch}_seed{seed:02d}_{world.stem}"
                runs.append(ValidationRun(
                    arch=arch,
                    seed=seed,
                    checkpoint=checkpoint,
                    world=world,
                    run_id=run_id,
                    output_group=output_group,
                ))
    return runs


def resolve_webots_cmd(webots_cmd: str) -> str:
    if Path(webots_cmd).is_file():
        return str(Path(webots_cmd).resolve())
    found = shutil.which(webots_cmd)
    if found:
        return found
    raise FileNotFoundError(
        f"Webots executable not found: {webots_cmd}.\n"
        "Install Webots or provide the full path with --webots-cmd /path/to/webots."
    )


def patch_eval_model_path(model_path: Path) -> None:
    defaults_path = REPO_ROOT / "controllers" / "PPO" / "PPO_defaults.py"
    text = defaults_path.read_text()
    new_value = str(model_path)
    updated, count = re.subn(
        r'^(\s*eval_model_path\s*=\s*").*(")$',
        lambda m: f'{m.group(1)}{new_value}{m.group(2)}',
        text,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise RuntimeError(f"Could not patch eval_model_path in {defaults_path}")
    defaults_path.write_text(updated)
    print(f"[PATCH] wrote eval_model_path = {new_value} to {defaults_path}")


def run_webots(run: ValidationRun, port: int, webots_cmd: str, patch_defaults: bool) -> int:
    if patch_defaults:
        patch_eval_model_path(run.checkpoint)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["WEBOTS_CONTROLLER_PATH"] = str(CONTROLLERS_DIR)
    env["PPO_ARCH"] = run.arch
    env["PPO_SEED"] = str(run.seed)
    env["PPO_EVAL_MODEL_PATH"] = str(run.checkpoint)
    env["PPO_EPISODES"] = "10"
    env["PPO_RUN_ID"] = run.run_id

    cmd = [
        webots_cmd,
        f"--port={port}",
        "--no-rendering",
        "--batch",
        "--minimize",
        "--stdout",
        "--stderr",
        "--mode=fast",
        str(run.world),
    ]

    print(f"[VALIDATION] {run.output_group} arch={run.arch} seed={run.seed:02d} world={run.world.name} run_id={run.run_id}")
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
    if result.returncode != 0:
        print(f"[ERROR] Webots failed for {run.world.name} with exit code {result.returncode}", file=sys.stderr)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run validation evaluation across static and moving worlds.")
    parser.add_argument("--webots-cmd", default="webots", help="Webots executable command.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Base Webots port; each worker gets port+N.")
    parser.add_argument("--workers", type=int, default=10, help="Number of parallel Webots instances.")
    parser.add_argument("--patch-defaults", action="store_true", help="Update PPO_defaults.py eval_model_path before each run.")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned runs without launching Webots.")
    args = parser.parse_args()

    if args.patch_defaults and args.workers > 1:
        print("[ERROR] --patch-defaults is not safe with --workers > 1 (shared file race condition).", file=sys.stderr)
        return 1

    static_worlds = resolve_worlds(STATIC_WORLD_GLOBS)
    moving_worlds = resolve_worlds(MOVING_WORLD_GLOBS)
    static_runs = make_evaluation_runs(static_worlds, STATIC_MODELS, "static")
    moving_runs = make_evaluation_runs(moving_worlds, MOVING_MODELS, "moving")

    if args.dry_run:
        for run in static_runs + moving_runs:
            print(f"DRY-RUN: arch={run.arch} seed={run.seed:02d} world={run.world.name} checkpoint={run.checkpoint} run_id={run.run_id}")
        return 0

    try:
        webots_cmd = resolve_webots_cmd(args.webots_cmd)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    port_pool: queue.Queue[int] = queue.Queue()
    for i in range(args.workers):
        port_pool.put(args.port + i)

    def run_one(run: ValidationRun) -> int:
        port = port_pool.get()
        try:
            return run_webots(run, port=port, webots_cmd=webots_cmd, patch_defaults=args.patch_defaults)
        finally:
            port_pool.put(port)

    all_runs = static_runs + moving_runs
    failed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_one, run): run for run in all_runs}
        for future in concurrent.futures.as_completed(futures):
            rc = future.result()
            if rc != 0:
                failed += 1

    if failed:
        print(f"[VALIDATION] {failed}/{len(all_runs)} run(s) failed.", file=sys.stderr)
        return 1
    print(f"[VALIDATION] All {len(all_runs)} runs completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
