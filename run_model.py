"""Inference script for running the trained PPO model in Webots."""
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
import torch

from controller import Supervisor  # pyright: ignore[reportMissingImports]

from controllers.PPO.PPO import Config, WebotsEnv, PPOAgent, _init_supervisor


@dataclass
class InferenceConfig:
    """Configuration for inference."""
    model_path: str = 'best_model.pth'
    episodes: int = 10  # Number of episodes to run
    render: bool = True  # Whether to print episode info


def run_inference(config: Optional[InferenceConfig] = None) -> None:
    """Run inference with the trained model.

    Args:
        config: Inference configuration
    """
    if config is None:
        config = InferenceConfig()

    # Initialize Webots supervisor
    _init_supervisor()

    # Create environment
    train_config = Config()  # Use default training config
    env = WebotsEnv(train_config)
    obs, _ = env.reset()
    obs_size = len(obs)
    n_actions = len(train_config.actions)

    # Create agent
    agent = PPOAgent(obs_size, n_actions, train_config)

    # Load saved model
    try:
        checkpoint = torch.load(config.model_path, map_location=agent.device)
        agent.actor.load_state_dict(checkpoint['actor'])
        agent.critic.load_state_dict(checkpoint['critic'])
        print(f"[INFERENCE] Loaded model from {config.model_path}")
        print(f"[INFERENCE] Model from episode: {checkpoint.get('episode', 'unknown')}")
        print(f"[INFERENCE] Reward: {checkpoint.get('reward', 'unknown')}")
        print(f"[INFERENCE] Goal episode: {checkpoint.get('goal_episode', False)}")
    except FileNotFoundError:
        print(f"[INFERENCE] ERROR: Model file {config.model_path} not found!")
        return
    except Exception as e:
        print(f"[INFERENCE] ERROR loading model: {e}")
        return

    # Set to evaluation mode
    agent.actor.eval()
    agent.critic.eval()

    print(f"[INFERENCE] Running {config.episodes} episodes...")

    total_rewards = []
    goal_count = 0

    for episode in range(config.episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        steps = 0
        episode_end_reason = "max_steps"

        while not done:
            # Select action (deterministic for inference, take argmax)
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
            with torch.no_grad():
                probs = agent.actor(obs_tensor)
                action = torch.argmax(probs, dim=-1).item()

            # Step environment
            obs_next, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
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

    # Summary
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    success_rate = goal_count / config.episodes * 100

    print("\n[INFERENCE] Summary:")
    print(f"  Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"  Goal Success Rate: {success_rate:.1f}% ({goal_count}/{config.episodes})")

    # Cleanup
    env.robot.motors.stop()
    print("[INFERENCE] Inference complete. Robot stopped.")


if __name__ == "__main__":
    run_inference()