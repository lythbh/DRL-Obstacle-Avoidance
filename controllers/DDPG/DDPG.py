"""DDPG training controller for ALTINO robot in Webots obstacle avoidance task."""
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any, Union
import numpy as np
import torch
from torch import multiprocessing, nn

RecurrentState = Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from controllers.RNN import GRUActorCritic, LSTMActorCritic
from controllers.Webots.webots_env import WebotsEnv, _init_supervisor


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Training and environment hyperparameters."""

    # Training
    episodes: int = 500
    update_every: int = 5  # DDPG update frequency (episodes)
    batch_size: int = 64

    # DDPG Agent
    gamma: float = 0.99  # Discount factor
    tau: float = 0.005  # Soft update parameter
    learning_rate: float = 1e-4
    hidden_size: int = 128  # Network hidden layer size
    latent_size: int = 128  # Encoder output size before the recurrent core
    lstm_hidden_size: int = 128  # Hidden size of the recurrent core
    lstm_layers: int = 1
    recurrent_cell: str = "gru" # "lstm" or "gru"
    lidar_sector_dim: int = 16
    pose_goal_dim: int = 7  # [x, y] + [sin/cos heading, sin/cos goal_error, norm dist]
    imu_feature_dim: int = 10  # accel(3) + gyro(3) + quaternion(4)
    occupancy_grid_shape: Optional[Tuple[int, ...]] = None  # Optional CNN shape, e.g. (1, 16, 16)

    # Environment
    max_steps: int = 4000  # Max steps per episode
    collision_threshold: float = 0.1  # LiDAR distance threshold for collision
    low_score_threshold: float = -800.0  # Episode reset threshold
    collision_penalty: float = -20.0  # Penalty when collision happens
    progress_reward_scale: float = 3.0  # Scale for distance-progress reward
    distance_reward_scale: float = 2.0  # Dense reward for being closer to the goal than the start state
    heading_reward_scale: float = 0.5  # Bonus when facing toward the goal
    safety_reward_scale: float = 0.2  # Encourages keeping distance from obstacles
    motion_reward_scale: float = 0.05  # Bonus for moving forward to avoid stop-policy collapse
    new_best_distance_bonus: float = 1.0  # Bonus when reaching a new closest distance to goal
    step_penalty: float = -0.01  # Small per-step penalty to encourage efficiency
    endpoint: Tuple[float, float] = (2.0, 0.0)  # Goal location
    goal_threshold: float = 0.1  # Radius around goal considered reached
    goal_stop_bonus: float = 120.0  # Extra reward for stopping at the goal
    goal_hold_reward: float = 10.0  # Reward per timestep while inside goal threshold
    goal_speed_penalty: float = -60.0  # Penalty for still moving inside the goal region
    goal_overshoot_penalty: float = -50.0  # Penalty for driving past the goal region
    goal_score_threshold: float = 5000.0  # End episode when total reward reaches this threshold
    reference_distance: Optional[float] = None  # Start-to-goal distance, filled in at init

    # Robot Control
    max_steering_angle: float = 0.9
    min_speed: float = 0.0
    start_position: Optional[List[float]] = None  # [x, y, z]
    start_rotation: Optional[List[float]] = None  # [x, y, z, w]
    start_position_noise: float = 0.03  # Random position jitter at reset
    start_yaw_noise: float = 0.2  # Random yaw jitter at reset
    episode_warmup_steps: int = 12  # Random exploration steps after reset

    # Motor/Sensor Config
    max_speed: float = 10.0
    reset_settle_steps: int = 10  # Steps to wait for physics to settle after reset

    def __post_init__(self) -> None:
        """Initialize defaults for mutable fields."""
        self.recurrent_cell = self.recurrent_cell.lower().strip()
        if self.recurrent_cell not in {"lstm", "gru"}:
            raise ValueError(f"Unsupported recurrent_cell: {self.recurrent_cell}")
        if self.start_position is None:
            self.start_position = [-2.0, 0.0, 0.02]
        if self.start_rotation is None:
            self.start_rotation = [0.0, 0.0, 1.0, 0.0]
        if self.reference_distance is None:
            start_xy = np.array(self.start_position[:2], dtype=np.float32)
            endpoint_xy = np.array(self.endpoint, dtype=np.float32)
            self.reference_distance = float(np.linalg.norm(start_xy - endpoint_xy))


# ============================================================================
# DDPG AGENT
# ============================================================================

class DDPGAgent:
    """Deep Deterministic Policy Gradient agent with a recurrent actor-critic core."""

    def __init__(self, obs_size: int, action_dim: int, config: Config):
        self.config = config
        self.device = self._get_device()
        self.action_dim = action_dim
        self.obs_size = obs_size
        self._build_model(self.config.recurrent_cell)
        print(f"[DDPG] Using recurrent cell: {self.config.recurrent_cell.upper()}", flush=True)

    def _build_model(self, recurrent_cell: str) -> None:
        recurrent_cell = recurrent_cell.lower().strip()
        model_class = GRUActorCritic if recurrent_cell == "gru" else LSTMActorCritic
        self.actor = model_class(self.obs_size, self.action_dim, self.config)
        self.critic = model_class(self.obs_size, 1, self.config)  # Critic outputs value
        self.target_actor = model_class(self.obs_size, self.action_dim, self.config)
        self.target_critic = model_class(self.obs_size, 1, self.config)
        
        # Copy parameters to targets
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.learning_rate)

    def _get_device(self) -> torch.device:
        """Get appropriate device (GPU or CPU)."""
        is_fork = multiprocessing.get_start_method(allow_none=True) == 'fork'
        if torch.cuda.is_available() and not is_fork:
            return torch.device("cuda:0")
        return torch.device("cpu")

    def get_initial_state(self, batch_size: int = 1) -> RecurrentState:
        """Expose recurrent state initialization for rollouts."""
        return self.actor.get_initial_state(batch_size, device=self.device)

    def _action_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get per-dimension action bounds."""
        low = torch.tensor(
            [-self.config.max_steering_angle, self.config.min_speed],
            dtype=torch.float32,
            device=self.device,
        )
        high = torch.tensor(
            [self.config.max_steering_angle, self.config.max_speed],
            dtype=torch.float32,
            device=self.device,
        )
        return low, high

    def select_action(
        self,
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        recurrent_state: Optional[RecurrentState] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, RecurrentState]:
        """Sample a single action and advance the recurrent state."""
        with torch.no_grad():
            policy_output, _, next_state = self.actor(
                obs,
                recurrent_state=recurrent_state,
                done_mask=None,
            )
            # For DDPG, actor outputs mean action
            action = torch.tanh(policy_output.squeeze(0))
            low, high = self._action_bounds()
            center = (high + low) / 2.0
            scale = (high - low) / 2.0
            action = action * scale + center
            if not deterministic:
                # Add noise for exploration
                noise = torch.randn_like(action) * 0.1
                action = action + noise
                action = torch.clamp(action, low, high)
        return action.cpu().numpy(), next_state

    def update(self, replay_buffer: List[Dict[str, np.ndarray]]) -> None:
        """Update the DDPG policy from replay buffer."""
        if len(replay_buffer) < self.config.batch_size:
            return

        # Sample batch
        indices = np.random.choice(len(replay_buffer), self.config.batch_size, replace=False)
        batch = [replay_buffer[i] for i in indices]
        
        # Prepare tensors
        obs = torch.tensor(np.array([b['obs'] for b in batch]), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array([b['action'] for b in batch]), dtype=torch.float32, device=self.device)
        rewards = torch.tensor(np.array([b['reward'] for b in batch]), dtype=torch.float32, device=self.device)
        next_obs = torch.tensor(np.array([b['next_obs'] for b in batch]), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array([b['done'] for b in batch]), dtype=torch.float32, device=self.device)
        
        # Critic update
        with torch.no_grad():
            next_actions, _ = self.select_action(next_obs, deterministic=True)
            next_actions = torch.tensor(next_actions, dtype=torch.float32, device=self.device)
            target_q, _, _ = self.target_critic(next_obs)
            target_q = rewards + (1 - dones) * self.config.gamma * target_q.squeeze(-1)
        
        current_q, _, _ = self.critic(obs)
        critic_loss = nn.MSELoss()(current_q.squeeze(-1), target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor update
        actor_actions, _ = self.select_action(obs, deterministic=True)
        actor_actions = torch.tensor(actor_actions, dtype=torch.float32, device=self.device)
        actor_loss = -self.critic(obs, recurrent_state=None, done_mask=None)[0].mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update targets
        self._soft_update(self.target_actor, self.actor, self.config.tau)
        self._soft_update(self.target_critic, self.critic, self.config.tau)

    def _soft_update(self, target: nn.Module, source: nn.Module, tau: float) -> None:
        """Soft update target network parameters."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


# ============================================================================
# TRAINING
# ============================================================================

def train(config: Optional[Config] = None) -> None:
    """Run DDPG training loop.
    
    Args:
        config: Configuration object (uses defaults if None)
    """
    if config is None:
        config = Config()
    
    # Initialize Webots supervisor
    _init_supervisor()
    
    # Create environment and agent
    env = WebotsEnv(config)
    obs, _ = env.reset()
    obs_size = env.observation_size
    action_dim = env.action_dim
    agent = DDPGAgent(obs_size, action_dim, config)
    
    print(f"[TRAIN] Starting training: {config.episodes} episodes, "
          f"update every {config.update_every} episodes")
    print(f"[TRAIN] Observation size: {obs_size}, Action dims: {action_dim}")
    
    # Replay buffer
    replay_buffer = []
    max_replay_size = 100000
    
    # Training loop
    for episode in range(config.episodes):
        obs, _ = env.reset()
        done = False
        episode_step = 0
        episode_reward = 0.0
        episode_end_reason = "max_steps"
        recurrent_state = agent.get_initial_state(batch_size=1)
        
        while not done:
            # Select action
            action, recurrent_state = agent.select_action(obs, recurrent_state, deterministic=False)
            
            # Step environment
            obs_next, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_step += 1
            episode_reward += reward
            
            # Track termination reason
            if done:
                if info.get("reset_reason") == "low_score":
                    episode_end_reason = "low_score"
                elif info.get("reset_reason") == "collision":
                    episode_end_reason = "collision"
                elif info.get("reset_reason") == "goal":
                    episode_end_reason = "goal"
                elif truncated:
                    episode_end_reason = "max_steps"
            
            # Store transition
            transition = {
                'obs': obs,
                'action': action,
                'reward': reward,
                'next_obs': obs_next,
                'done': float(done),
            }
            replay_buffer.append(transition)
            if len(replay_buffer) > max_replay_size:
                replay_buffer.pop(0)
            
            obs = obs_next
        
        # DDPG update every N episodes
        if (episode + 1) % config.update_every == 0:
            agent.update(replay_buffer)
        
        # Logging
        print(
            f"Episode {episode + 1:2d} | "
            f"Reward: {episode_reward:8.2f} | "
            f"Steps: {env.current_step:4d} | "
            f"MinDist: {env.min_episode_distance:6.2f} | "
            f"LastDist: {env.current_distance:6.2f} | "
            f"End: {episode_end_reason}"
        )

    # Save final model
    checkpoint = {
        'actor': agent.actor.state_dict(),
        'critic': agent.critic.state_dict(),
        'target_actor': agent.target_actor.state_dict(),
        'target_critic': agent.target_critic.state_dict(),
        'episode': episode + 1,
        'reward': episode_reward,
    }
    torch.save(checkpoint, 'final_model.pth')
    print("[TRAIN] Final model saved.")

    # Cleanup
    env.robot.motors.stop()
    print("[TRAIN] Training complete. Robot stopped.")