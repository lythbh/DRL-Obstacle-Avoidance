"""PPO training controller for ALTINO robot in Webots obstacle avoidance task."""

from dataclasses import asdict
from pathlib import Path
from typing import List, Optional
import numpy as np
import torch

from config import Config
from controller import Supervisor  # pyright: ignore[reportMissingImports]
from environment.webots_env import WebotsEnv
from learning.agent import PPOAgent
from learning.training_utils import RollingDiagnostics, TrajectoryBuffer, resolve_episode_end_reason

# Global supervisor instance
_supervisor: Optional[Supervisor] = None

def _init_supervisor() -> None:
    """Initialize global Supervisor instance for field access."""
    global _supervisor
    _supervisor = Supervisor()


def _set_simulation_mode(config: Config) -> None:
    """Configure Webots simulation mode for throughput."""
    global _supervisor
    if not config.force_fast_mode or _supervisor is None:
        return

    try:
        _supervisor.simulationSetMode(Supervisor.SIMULATION_MODE_FAST)
        print("[PPO] Webots simulation mode set to FAST.")
    except Exception as e:
        print(f"[PPO] WARNING: Could not set FAST mode: {e}")


def _clone_state_dict_to_cpu(state_dict: dict) -> dict:
    """Clone model state dict tensors to CPU for safe checkpointing."""
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}



# ============================================================================
# TRAINING
# ============================================================================

def train(config: Optional[Config] = None) -> None:
    """Run PPO training loop.
    
    Args:
        config: Configuration object (uses defaults if None)
    """
    if config is None:
        config = Config()
    
    # Initialize Webots supervisor
    _init_supervisor()
    _set_simulation_mode(config)
    assert _supervisor is not None, "Supervisor not initialized. Call _init_supervisor() first."
    
    # Create environment and agent
    env = WebotsEnv(_supervisor, config)
    obs, _ = env.reset()
    obs_size = len(obs)
    assert config.actions is not None, "Actions not initialized"
    n_actions = len(config.actions)
    agent = PPOAgent(obs_size, n_actions, config)
    
    print(f"[TRAIN] Starting training: {config.episodes} episodes, "
          f"update every {config.update_every} episodes")
    print(f"[TRAIN] Observation size: {obs_size}, Action space: {n_actions}")
    print(
        f"[TRAIN] Device: {agent.device.type} | ActionRepeat: {max(1, int(config.action_repeat))} | "
        f"FAST mode requested: {config.force_fast_mode}"
    )
    print(
        f"[TRAIN] Goal endpoint: ({env.reward_computer.endpoint[0]:.2f}, {env.reward_computer.endpoint[1]:.2f}), "
        f"radius: {config.goal_radius:.2f}"
    )
    
    # Training buffers and rolling diagnostics.
    trajectory_buffer = TrajectoryBuffer()
    diagnostics = RollingDiagnostics(config.diagnostics_window)

    best_score = float("-inf")
    best_episode = 0
    best_actor_state = _clone_state_dict_to_cpu(agent.actor.state_dict())
    best_critic_state = _clone_state_dict_to_cpu(agent.critic.state_dict())
    ever_reached_goal = False
    collapsed_windows = 0
    rollback_cooldown_remaining = 0
    rollback_count = 0

    def _run_ppo_update(progress_episode: int) -> None:
        """Run PPO update on buffered trajectories and clear buffer."""
        if not trajectory_buffer.actions:
            return

        obs_array, actions_array, log_probs, returns, advantages = trajectory_buffer.to_arrays()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        training_progress = progress_episode / max(config.episodes, 1)
        current_entropy_coef = (
            config.entropy_coef
            + (config.entropy_final_coef - config.entropy_coef) * training_progress
        )

        agent.update(
            obs_array,
            actions_array,
            log_probs,
            returns,
            advantages,
            entropy_coef=current_entropy_coef,
        )
        trajectory_buffer.clear()
    
    # Training loop
    for episode in range(config.episodes):
        obs, _ = env.reset()
        done = False
        terminated = False
        truncated = False
        episode_observations: List[np.ndarray] = []
        episode_actions: List[int] = []
        episode_log_probs: List[float] = []
        episode_rewards: List[float] = []
        episode_end_reason = "max_steps"
        
        while not done:
            # Select action from the current policy for every transition so the
            # rollout stays on-policy for PPO updates.
            action, log_prob = agent.select_action(obs)
            
            # Step environment
            obs_next, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Track termination reason
            if done:
                episode_end_reason = resolve_episode_end_reason(info, truncated)
            
            # Accumulate
            episode_observations.append(obs)
            episode_actions.append(action)
            episode_log_probs.append(log_prob)
            episode_rewards.append(reward)
            
            obs = obs_next
        
        # Build returns and advantages per episode so reward signals do not
        # leak across episode boundaries.
        bootstrap_value = 0.0
        if truncated and not terminated:
            with torch.no_grad():
                final_obs_tensor = torch.as_tensor(
                    obs,
                    dtype=torch.float32,
                    device=agent.device,
                ).unsqueeze(0)
                bootstrap_value = float(agent.critic(final_obs_tensor).item())

        episode_returns = agent.calculate_returns(
            np.array(episode_rewards, dtype=np.float32),
            bootstrap_value=bootstrap_value,
        )
        episode_obs_array = np.array(episode_observations, dtype=np.float32)
        with torch.no_grad():
            episode_obs_tensor = torch.as_tensor(
                episode_obs_array,
                dtype=torch.float32,
                device=agent.device,
            )
            episode_values = agent.critic(episode_obs_tensor).squeeze(-1).detach().cpu().numpy()

        episode_advantages = episode_returns - episode_values

        trajectory_buffer.add_episode(
            episode_obs_array,
            episode_actions,
            episode_log_probs,
            episode_returns,
            episode_advantages,
        )
        
        # PPO update every N episodes
        if (episode + 1) % config.update_every == 0:
            _run_ppo_update(episode + 1)
        
        episode_reward_sum = sum(episode_rewards)
        diagnostics.record_episode(
            episode_end_reason,
            episode_reward_sum,
            env.min_episode_distance,
            env.current_step,
        )
        if episode_end_reason == "goal_reached":
            ever_reached_goal = True

        if diagnostics.recent_rewards:
            current_score = float(np.mean(diagnostics.recent_rewards))
            if current_score > best_score:
                best_score = current_score
                best_episode = episode + 1
                best_actor_state = _clone_state_dict_to_cpu(agent.actor.state_dict())
                best_critic_state = _clone_state_dict_to_cpu(agent.critic.state_dict())

        if diagnostics.should_emit(episode):
            summary = diagnostics.summary(env.timestep)
            print(
                f"[DIAG] Last {diagnostics.window_size} eps | "
                f"AvgReward: {summary['avg_reward']:7.2f} | "
                f"AvgMinDist: {summary['avg_min_dist']:5.2f} | "
                f"GoalRate: {summary['goal_rate'] * 100:5.1f}% | "
                f"CollisionRate: {summary['collision_rate'] * 100:5.1f}% | "
                f"StagnationRate: {summary['stagnation_rate'] * 100:5.1f}%"
            )

            if rollback_cooldown_remaining > 0:
                rollback_cooldown_remaining -= 1

            should_check_rollback = (
                config.enable_policy_rollback
                and ever_reached_goal
                and (episode + 1) >= max(1, int(config.rollback_min_episodes))
            )
            if should_check_rollback:
                if summary["goal_rate"] < float(config.rollback_goal_rate_threshold):
                    collapsed_windows += 1
                else:
                    collapsed_windows = 0

                if (
                    collapsed_windows >= max(1, int(config.rollback_patience_windows))
                    and rollback_cooldown_remaining == 0
                    and best_episode > 0
                ):
                    agent.actor.load_state_dict(best_actor_state)
                    agent.critic.load_state_dict(best_critic_state)
                    agent.optimizer.state.clear()
                    trajectory_buffer.clear()
                    rollback_count += 1
                    rollback_cooldown_remaining = max(0, int(config.rollback_cooldown_windows))
                    collapsed_windows = 0
                    print(
                        f"[ROLLBACK] Restored best policy from episode {best_episode} "
                        f"(score {best_score:.2f}) after sustained goal-rate collapse."
                    )

    # Apply one last PPO update for any leftover rollout data when episodes
    # are not divisible by update_every.
    _run_ppo_update(config.episodes)

    checkpoint_dir = Path(config.checkpoint_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = Path(__file__).resolve().parent / checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / config.checkpoint_name

    torch.save(
        {
            "actor_state_dict": best_actor_state,
            "critic_state_dict": best_critic_state,
            "best_score": best_score,
            "best_episode": best_episode,
            "config": asdict(config),
        },
        checkpoint_path,
    )
    print(
        f"[TRAIN] Saved best checkpoint from episode {best_episode} "
        f"(score {best_score:.2f}) to {checkpoint_path}"
    )
    if config.enable_policy_rollback:
        print(f"[TRAIN] Rollbacks performed: {rollback_count}")
    
    # Cleanup
    env.robot.motors.stop()
    print("[TRAIN] Training complete. Robot stopped.")


if __name__ == "__main__":
    train()