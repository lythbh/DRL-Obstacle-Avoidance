"""Inference script for running a trained PPO or SAC model in Webots."""
import argparse
from dataclasses import dataclass, fields
from typing import Optional, Dict, Any, Type


@dataclass
class InferenceConfig:
    """Configuration for inference."""
    model_path: str = 'best_model.pth'
    algorithm: str = "sac"  # "ppo" or "sac"
    episodes: int = 10  # Number of episodes to run
    render: bool = True  # Whether to print episode info


def _checkpoint_config(config_class: Type[Any], checkpoint: Dict[str, Any]) -> Any:
    """Build a config from checkpoint metadata when available."""
    saved_config = checkpoint.get("config")
    if not isinstance(saved_config, dict):
        return config_class()

    valid_fields = {field.name for field in fields(config_class)}
    return config_class(**{key: value for key, value in saved_config.items() if key in valid_fields})


def run_inference(config: Optional[InferenceConfig] = None) -> None:
    """Run inference with the trained model.

    Args:
        config: Inference configuration
    """
    if config is None:
        config = InferenceConfig()

    if config.episodes <= 0:
        print(f"[INFERENCE] ERROR: episodes must be greater than 0, got {config.episodes}.")
        return

    algorithm = config.algorithm.lower().strip()
    if algorithm not in {"ppo", "sac"}:
        print(f"[INFERENCE] ERROR: Unsupported algorithm '{config.algorithm}'. Use 'ppo' or 'sac'.")
        return

    print(f"[INFERENCE] Using algorithm: {algorithm.upper()}", flush=True)

    try:
        import numpy as np
        import torch

        from controllers.PPO.PPO import Config as PPOConfig, PPOAgent
        from controllers.SAC.SAC import Config as SACConfig, SACAgent
        from controllers.Webots.webots_env import WebotsEnv, _init_supervisor
    except ImportError as e:
        print(f"[INFERENCE] ERROR importing runtime dependencies: {e}")
        print("[INFERENCE] Run this from the Webots controller Python environment with the project dependencies installed.")
        return

    _init_supervisor()

    try:
        checkpoint = torch.load(config.model_path, map_location="cpu")
    except FileNotFoundError:
        print(f"[INFERENCE] ERROR: Model file {config.model_path} not found!")
        return
    except Exception as e:
        print(f"[INFERENCE] ERROR loading model metadata: {e}")
        return

    if algorithm == "ppo":
        recurrent_cell = str(checkpoint.get("recurrent_cell", "gru")).lower().strip()
        train_config = _checkpoint_config(PPOConfig, checkpoint)
        train_config.recurrent_cell = recurrent_cell
    else:
        architecture = checkpoint.get("architecture")
        if not isinstance(architecture, dict):
            architecture = {}
        recurrent_cell = str(architecture.get("recurrent_cell", "gru")).lower().strip()
        train_config = _checkpoint_config(SACConfig, checkpoint)
        train_config.recurrent_cell = recurrent_cell

    env = WebotsEnv(train_config)
    env.reset()
    obs_size = env.observation_size
    n_actions = env.action_dim

    agent: Any
    if algorithm == "ppo":
        agent = PPOAgent(obs_size, n_actions, train_config)
    else:
        agent = SACAgent(obs_size, n_actions, train_config)

    try:
        if algorithm == "ppo":
            agent.load_model(config.model_path)
        else:
            agent.load(config.model_path)
        print(f"[INFERENCE] Loaded model from {config.model_path}")
        print(f"[INFERENCE] Model from episode: {checkpoint.get('episode', 'unknown')}")
        print(f"[INFERENCE] Reward: {checkpoint.get('reward', 'unknown')}")
        print(f"[INFERENCE] Goal episode: {checkpoint.get('goal_episode', False)}")
        print(f"[INFERENCE] Recurrent cell: {recurrent_cell}")
    except Exception as e:
        print(f"[INFERENCE] ERROR loading model: {e}")
        return

    if algorithm == "ppo":
        agent.model.eval()
        agent.actor_log_std.requires_grad_(False)
    else:
        agent.actor.eval()
        agent.q1.eval()
        agent.q2.eval()
        agent.target_q1.eval()
        agent.target_q2.eval()

    print(f"[INFERENCE] Running {config.episodes} episodes...")

    total_rewards = []
    goal_count = 0

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
                    obs,
                    recurrent_state=recurrent_state,
                    done=prev_done,
                    deterministic=True,
                )
            else:
                action, recurrent_state = agent.select_action(
                    obs,
                    recurrent_state=recurrent_state,
                    done=prev_done,
                    deterministic=True,
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

        if config.render:
            print(
                f"Episode {episode + 1:2d} | "
                f"Reward: {episode_reward:8.2f} | "
                f"Steps: {steps:4d} | "
                f"MinDist: {env.min_episode_distance:6.2f} | "
                f"LastDist: {env.current_distance:6.2f} | "
                f"End: {episode_end_reason}"
            )

    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    success_rate = goal_count / config.episodes * 100

    print("\n[INFERENCE] Summary:")
    print(f"  Average Reward: {avg_reward:.2f} +/- {std_reward:.2f}")
    print(f"  Goal Success Rate: {success_rate:.1f}% ({goal_count}/{config.episodes})")

    env.robot.motors.stop()
    print("[INFERENCE] Inference complete. Robot stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PPO or SAC inference in Webots.")
    parser.add_argument("--algorithm", choices=("ppo", "sac"), default=InferenceConfig.algorithm)
    parser.add_argument("--model-path", default=InferenceConfig.model_path)
    parser.add_argument("--episodes", type=int, default=InferenceConfig.episodes)
    parser.add_argument("--no-render", action="store_true")
    args, _ = parser.parse_known_args()

    run_inference(
        InferenceConfig(
            model_path=args.model_path,
            algorithm=args.algorithm,
            episodes=args.episodes,
            render=not args.no_render,
        )
    )
