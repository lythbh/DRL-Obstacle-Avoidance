"""Inference script for running a trained PPO or SAC model in Webots."""

import argparse
import time
import statistics
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional, Any, cast


@dataclass
class InferenceConfig:
    model_path: Optional[str] = None
    algorithm: str = "sac"
    episodes: int = 10
    show_progress: bool = True


def _default_model_path(algorithm: str) -> str:
    """Find the most recently modified model checkpoint for the given algorithm."""
    controller_dir = "PPO" if algorithm == "ppo" else "SAC"
    checkpoint_root = Path(__file__).resolve().parent / "controllers" / controller_dir / "checkpoints"
    candidates = sorted(checkpoint_root.rglob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return str(candidates[0])
    return str(checkpoint_root.parent / "best_model.pth")


def run_inference(config=None):
    """Load trained model, run episodes in Webots, and report performance metrics."""
    if config is None:
        config = InferenceConfig()

    if config.episodes <= 0:
        print(f"[INFER] ERROR: episodes must be greater than 0, got {config.episodes}.")
        return

    algorithm = config.algorithm.lower().strip()
    if algorithm not in {"ppo", "sac"}:
        print(f"[INFER] ERROR: unsupported algorithm '{config.algorithm}'. Use 'ppo' or 'sac'.")
        return

    model_path = config.model_path or _default_model_path(algorithm)
    print(f"[INFER][{algorithm.upper()}] model={model_path} episodes={config.episodes}", flush=True)

    try:
        import numpy as np
        import torch

        from controllers.PPO.PPO import Config as PPOConfig, PPOAgent
        from controllers.SAC.SAC import Config as SACConfig, SACAgent
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
        print(f"[INFER][{algorithm.upper()}] ERROR: model file not found: {model_path}")
        return
    except Exception as e:
        print(f"[INFER][{algorithm.upper()}] ERROR loading model metadata: {e}")
        return

    checkpoint_algorithm = str(checkpoint.get("algorithm", algorithm)).lower().strip()
    if checkpoint_algorithm != algorithm:
        print(f"[INFER][{algorithm.upper()}] ERROR: checkpoint algorithm '{checkpoint_algorithm}' does not match requested '{algorithm}'.")
        return

    saved_config = checkpoint.get("config")
    if not isinstance(saved_config, dict):
        saved_config = {}

    if algorithm == "ppo":
        train_config = PPOConfig(**{k: v for k, v in saved_config.items() if k in {f.name for f in fields(PPOConfig)}})
        train_config.recurrent_cell = str(checkpoint.get("recurrent_cell", "gru")).lower().strip()
        env = WebotsEnv(train_config)
        obs_size = env.observation_size
        n_actions = env.action_dim
        agent = cast(Any, PPOAgent(obs_size, n_actions, train_config))
    else:
        train_config = SACConfig(**{k: v for k, v in saved_config.items() if k in {f.name for f in fields(SACConfig)}})
        architecture = checkpoint.get("architecture", {})
        if isinstance(architecture, dict):
            train_config.recurrent_cell = str(architecture.get("recurrent_cell", "gru")).lower().strip()
        env = WebotsEnv(train_config)
        obs_size = env.observation_size
        n_actions = env.action_dim
        agent = cast(Any, SACAgent(obs_size, n_actions, train_config))

    try:
        if algorithm == "ppo":
            agent.load_model(model_path)
        else:
            agent.load(model_path)
        print(f"[INFER][{algorithm.upper()}] loaded episode={checkpoint.get('episode', 'unknown')} reward={checkpoint.get('reward', 'unknown')} goal={checkpoint.get('goal_episode', False)} cell={train_config.recurrent_cell.upper()}", flush=True)
    except Exception as e:
        print(f"[INFER][{algorithm.upper()}] ERROR loading model: {e}")
        return

    if algorithm == "ppo":
        agent.model.eval()
        agent.actor_log_std.requires_grad_(False)
    else:
        agent.actor_enc.eval()
        agent.actor_mean.eval()
        agent.actor_log_std_head.eval()
        agent.q1_enc.eval()
        agent.q1_head.eval()
        agent.q2_enc.eval()
        agent.q2_head.eval()
        agent.target_q1_enc.eval()
        agent.target_q1_head.eval()
        agent.target_q2_enc.eval()
        agent.target_q2_head.eval()

    print(f"[INFER][{algorithm.upper()}] running {config.episodes} episodes", flush=True)

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
                if algorithm == "ppo":
                    action, _, _, recurrent_state = agent.select_action(
                        obs, recurrent_state=recurrent_state, done=prev_done, deterministic=True,
                    )
                else:
                    action, recurrent_state = agent.select_action(
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
                print(f"[INFER][{algorithm.upper()}] ep={episode + 1:03d}/{config.episodes} r={episode_reward:8.2f} steps={steps:4d} min_d={env.min_episode_distance:5.2f} end={episode_end_reason} t={elapsed:7.1f}s", flush=True)

        avg_reward = float(np.mean(total_rewards))
        std_reward = float(statistics.pstdev(total_rewards))
        success_rate = goal_count / config.episodes * 100
        elapsed = time.perf_counter() - start_time

        print(f"[INFER][{algorithm.upper()}] summary avg={avg_reward:.2f} std={std_reward:.2f} success={success_rate:.1f}% ({goal_count}/{config.episodes}) t={elapsed:7.1f}s", flush=True)
    finally:
        try:
            env.robot.stop()
        except Exception:
            pass

    print(f"[INFER][{algorithm.upper()}] done", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PPO or SAC inference in Webots.")
    parser.add_argument("--algorithm", choices=("ppo", "sac"), default=InferenceConfig.algorithm)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--episodes", type=int, default=InferenceConfig.episodes)
    parser.add_argument("--quiet", "--no-render", dest="quiet", action="store_true")
    args, _ = parser.parse_known_args()

    run_inference(
        InferenceConfig(
            model_path=args.model_path,
            algorithm=args.algorithm,
            episodes=args.episodes,
            show_progress=not args.quiet,
        )
    )
