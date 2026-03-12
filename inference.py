"""
Inference script for trained Altino PPO policy

This script loads a trained policy and runs it in the environment
for evaluation and visualization of performance.
"""

import numpy as np
import torch
from ppo_altino import AltinoPPOEnv, create_actor_network, create_critic_network
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, TanhNormal


def load_trained_policy(policy_path: str = "altino_policy.pt", device="cpu"):
    """Load trained policy from saved checkpoint."""
    
    # Create network architecture (must match training)
    actor_network = create_actor_network(num_cells=256, device=device)
    
    # Load weights
    state_dict = torch.load(policy_path, map_location=device)
    actor_network.load_state_dict(state_dict)
    
    # Wrap in TensorDict module
    policy_module = TensorDictModule(
        actor_network, in_keys=['observation'], out_keys=['loc', 'scale']
    )
    
    # Create environment to get specs
    env = AltinoPPOEnv(device=device)
    
    # Wrap with ProbabilisticActor
    policy_module = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec,
        in_keys=['loc', 'scale'],
        distribution_class=TanhNormal,
        distribution_kwargs={
            'low': env.action_spec.space.low,
            'high': env.action_spec.space.high,
        },
        return_log_prob=False,  # Don't need log_prob for inference
    )
    
    policy_module.eval()  # Set to evaluation mode
    return policy_module


def run_single_episode(policy_module, max_steps: int = 500, render: bool = True):
    """
    Run a single episode with the trained policy.
    
    Args:
        policy_module: Trained policy network
        max_steps: Maximum steps per episode
        render: Whether to print episode information
    
    Returns:
        episode_reward: Total reward for the episode
        steps: Number of steps taken
        success: Whether goal was reached
    """
    
    env = AltinoPPOEnv()
    obs, _ = env.reset()
    
    episode_reward = 0.0
    success = False
    
    for step in range(max_steps):
        # Convert observation to tensor and add batch dimension
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
        
        # Get action from policy
        with torch.no_grad():
            action_tensor = policy_module(obs_tensor)
        
        # Convert to numpy
        action = action_tensor.squeeze(0).cpu().numpy()
        
        # Take step
        obs, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        
        if render:
            min_lidar = np.min(obs[:12] * 5.0)
            print(f"Step {step+1:3d} | Reward: {reward:7.4f} | "
                  f"Cumulative: {episode_reward:8.4f} | Min LiDAR: {min_lidar:5.2f}m | "
                  f"Speed: {action[1]:5.2f} | Steering: {action[0]:6.3f}")
        
        if terminated:
            success = True
            if render:
                print(f"✓ Goal reached! Total reward: {episode_reward:.4f}")
            break
        
        if truncated:
            if render:
                print(f"✗ Timeout (max steps reached). Total reward: {episode_reward:.4f}")
            break
    
    return episode_reward, step + 1, success


def evaluate_policy(policy_module, num_episodes: int = 10):
    """
    Evaluate policy over multiple episodes.
    
    Args:
        policy_module: Trained policy network
        num_episodes: Number of evaluation episodes
    
    Returns:
        results: Dictionary with evaluation metrics
    """
    
    episode_rewards = []
    episode_lengths = []
    successes = 0
    
    print(f"\n{'='*80}")
    print(f"Evaluating Policy over {num_episodes} Episodes")
    print(f"{'='*80}\n")
    
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        reward, steps, success = run_single_episode(policy_module, render=False)
        
        episode_rewards.append(reward)
        episode_lengths.append(steps)
        if success:
            successes += 1
        
        print(f"Reward: {reward:.4f} | Steps: {steps} | Success: {'✓' if success else '✗'}")
    
    # Compute statistics
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'mean_steps': np.mean(episode_lengths),
        'success_rate': successes / num_episodes,
    }
    
    # Print summary
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Mean Reward:     {results['mean_reward']:.4f} ± {results['std_reward']:.4f}")
    print(f"Max Reward:      {results['max_reward']:.4f}")
    print(f"Min Reward:      {results['min_reward']:.4f}")
    print(f"Mean Steps:      {results['mean_steps']:.1f}")
    print(f"Success Rate:    {results['success_rate']*100:.1f}%")
    print(f"{'='*80}\n")
    
    return results


def compare_policies(policy_paths: list, num_episodes: int = 5):
    """
    Compare multiple trained policies.
    
    Args:
        policy_paths: List of paths to policy checkpoints
        num_episodes: Evaluation episodes per policy
    """
    
    results_dict = {}
    
    for policy_path in policy_paths:
        print(f"\nLoading policy: {policy_path}")
        try:
            policy = load_trained_policy(policy_path)
            results = evaluate_policy(policy, num_episodes=num_episodes)
            results_dict[policy_path] = results
        except FileNotFoundError:
            print(f"✗ Policy file not found: {policy_path}")
        except Exception as e:
            print(f"✗ Error loading policy: {e}")
    
    # Print comparison
    if results_dict:
        print(f"\n{'='*80}")
        print("POLICY COMPARISON")
        print(f"{'='*80}")
        
        for policy_name, results in results_dict.items():
            print(f"\n{policy_name}:")
            print(f"  Mean Reward:   {results['mean_reward']:8.4f} ± {results['std_reward']:.4f}")
            print(f"  Success Rate:  {results['success_rate']*100:6.1f}%")
            print(f"  Mean Steps:    {results['mean_steps']:6.1f}")


def visualize_episode(policy_module, max_steps: int = 500):
    """
    Visualize an episode with detailed feedback.
    
    Args:
        policy_module: Trained policy network
        max_steps: Maximum steps
    """
    
    env = AltinoPPOEnv()
    obs, _ = env.reset()
    
    print(f"\n{'='*80}")
    print("EPISODE VISUALIZATION")
    print(f"{'='*80}\n")
    
    episode_reward = 0.0
    step_data = []
    
    for step in range(max_steps):
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
        
        with torch.no_grad():
            action_tensor = policy_module(obs_tensor)
        
        action = action_tensor.squeeze(0).cpu().numpy()
        
        # Record step data
        min_lidar = np.min(obs[:12] * 5.0)
        goal_dist = np.linalg.norm(env.goal / 5.0 - obs[12:14] * 5.0)
        
        step_data.append({
            'step': step + 1,
            'reward': 0,  # Would need to compute from obs
            'action_steering': action[0],
            'action_speed': action[1],
            'min_lidar': min_lidar,
            'obs_status': 'OK' if min_lidar > 0.5 else 'CAUTION' if min_lidar > 0.3 else 'DANGER',
        })
        
        obs, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        
        if (step + 1) % 50 == 0 or terminated or truncated:
            # Print summary every 50 steps
            print(f"Step {step+1:3d}: "
                  f"Steering={action[0]:6.3f} | Speed={action[1]:5.2f} | "
                  f"Min LiDAR={min_lidar:5.2f}m | Status: {step_data[-1]['obs_status']}")
        
        if terminated:
            print(f"\n✓ Success! Episode reward: {episode_reward:.4f}")
            break
        
        if truncated:
            print(f"\n✗ Episode timeout. Episode reward: {episode_reward:.4f}")
            break
    
    return step_data


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate trained Altino PPO policy")
    parser.add_argument('--policy', type=str, default='altino_policy.pt',
                       help='Path to trained policy')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize detailed episode')
    parser.add_argument('--compare', nargs='+',
                       help='Compare multiple policies')
    
    args = parser.parse_args()
    
    # Device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    if args.compare:
        # Compare multiple policies
        compare_policies(args.compare, num_episodes=args.episodes)
    else:
        # Load and evaluate single policy
        try:
            print(f"Loading policy from: {args.policy}")
            policy = load_trained_policy(args.policy, device=device)
            print("✓ Policy loaded successfully!\n")
            
            if args.visualize:
                # Run visualization
                visualize_episode(policy)
            else:
                # Run evaluation
                evaluate_policy(policy, num_episodes=args.episodes)
        
        except FileNotFoundError:
            print(f"✗ Error: Policy file not found: {args.policy}")
        except Exception as e:
            print(f"✗ Error: {e}")
