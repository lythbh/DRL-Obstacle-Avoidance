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


class CriticNetwork(nn.Module):
    """Simple Q-network for state-action value estimation."""

    def __init__(self, obs_size: int, action_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        return self.net(x).squeeze(-1)


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
    update_every: int = 25  # DDPG update frequency (environment steps)
    batch_size: int = 64
    exploration_noise_std: float = 0.15
    exploration_noise_min: float = 0.01
    exploration_noise_decay: float = 0.995

    # DDPG Agent
    gamma: float = 0.96  # Discount factor
    tau: float = 0.06  # Soft update parameter (increased for faster convergence)
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 1e-4
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
    max_steps: int = 2000  # Max steps per episode
    collision_threshold: float = 0.1  # LiDAR distance threshold for collision
    low_score_threshold: float = -800.0  # Episode reset threshold
    collision_penalty: float = -6.0  # Penalty when collision happens
    progress_reward_scale: float = 1.5  # Scale for distance-progress reward
    distance_reward_scale: float = progress_reward_scale
    heading_reward_scale: float = 0.5  # Bonus when facing toward the goal
    safety_reward_scale: float = 0.6  # Encourages keeping distance from obstacles
    motion_reward_scale: float = 0.15  # Bonus for moving forward to avoid stop-policy collapse
    new_best_distance_bonus: float = 1.0  # Bonus when reaching a new closest distance to goal
    step_penalty: float = -0.03  # Per-step penalty (increased to penalize long episodes, discourages circling)
    stagnation_penalty: float = -2.0  # Penalty applied when no progress for N steps
    stagnation_steps: int = 40  # Steps without distance improvement before stagnation penalty kicks in
    stagnation_termination_steps: int = 100  # Force episode end if stagnant for this many steps
    endpoint: Tuple[float, float] = (2.0, 0.0)  # Goal location
    goal_threshold: float = 0.1  # Radius around goal considered reached
    goal_stop_bonus: float = 350.0  # Extra reward for stopping at the goal
    goal_hold_reward: float = 10.0  # Reward per timestep while inside goal threshold
    goal_speed_penalty: float = -60.0  # Penalty for still moving inside the goal region
    goal_overshoot_penalty: float = -50.0  # Penalty for driving past the goal region
    goal_score_threshold: float = 3500.0  # End episode when total reward reaches this threshold
    reference_distance: Optional[float] = None  # Start-to-goal distance, filled in at init

    # Robot Control
    max_steering_angle: float = 0.9
    min_speed: float = 0.0
    start_position: Optional[List[float]] = None  # [x, y, z]
    start_rotation: Optional[List[float]] = None  # [x, y, z, w]
    start_position_noise: float = 0.03  # Random position jitter at reset
    start_yaw_noise: float = 0.2  # Random yaw jitter at reset
    episode_warmup_steps: int = 60  # Random exploration steps after reset

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
        self.noise_std = self.config.exploration_noise_std
        self._build_model(self.config.recurrent_cell)
        print(f"[DDPG] Using recurrent cell: {self.config.recurrent_cell.upper()}", flush=True)
        print(f"[DDPG] Initial exploration std: {self.noise_std:.3f}", flush=True)

    def _build_model(self, recurrent_cell: str) -> None:
        recurrent_cell = recurrent_cell.lower().strip()
        model_class = GRUActorCritic if recurrent_cell == "gru" else LSTMActorCritic
        self.actor = model_class(self.obs_size, self.action_dim, self.config).to(self.device)
        self.critic = CriticNetwork(self.obs_size, self.action_dim, self.config.hidden_size).to(self.device)
        self.target_actor = model_class(self.obs_size, self.action_dim, self.config).to(self.device)
        self.target_critic = CriticNetwork(self.obs_size, self.action_dim, self.config.hidden_size).to(self.device)
        
        # Copy parameters to targets
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.critic_learning_rate)

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

    def _scale_action(self, policy_output: torch.Tensor) -> torch.Tensor:
        low, high = self._action_bounds()
        center = (high + low) / 2.0
        scale = (high - low) / 2.0
        action = torch.tanh(policy_output)
        return torch.clamp(action * scale + center, low, high)

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
            action = self._scale_action(policy_output.squeeze(0))
            if not deterministic:
                noise = torch.randn_like(action) * self.noise_std
                action = action + noise
                action = torch.clamp(action, *self._action_bounds())
        return action.cpu().numpy(), next_state

    def update(self, replay_buffer: List[Dict[str, np.ndarray]]) -> None:
        """Update the DDPG policy from replay buffer."""
        if len(replay_buffer) < self.config.batch_size:
            return

        # Ensure models are in training mode
        self.actor.train()
        self.critic.train()
        self.target_actor.eval()
        self.target_critic.eval()

        # Sample batch
        indices = np.random.choice(len(replay_buffer), self.config.batch_size, replace=False)
        batch = [replay_buffer[i] for i in indices]
        
        # Prepare tensors
        obs = torch.tensor(np.array([b['obs'] for b in batch]), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array([b['action'] for b in batch]), dtype=torch.float32, device=self.device)
        rewards = torch.tensor(np.array([b['reward'] for b in batch]), dtype=torch.float32, device=self.device)
        next_obs = torch.tensor(np.array([b['next_obs'] for b in batch]), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array([b['done'] for b in batch]), dtype=torch.float32, device=self.device)
        
        # Critic update using sampled actions from the replay buffer
        with torch.no_grad():
            # Use target networks for stability
            next_policy_output, _, _ = self.target_actor(next_obs, recurrent_state=None)
            # Ensure proper shape: (batch_size, action_dim)
            if next_policy_output.ndim == 1:
                next_policy_output = next_policy_output.unsqueeze(0)
            next_actions = self._scale_action(next_policy_output)
            target_q = self.target_critic(next_obs, next_actions)
            target_q = rewards + (1 - dones) * self.config.gamma * target_q

        current_q = self.critic(obs, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        # Check for NaN/inf
        if not torch.isfinite(critic_loss):
            print(f"[DDPG] WARNING: Non-finite critic loss: {critic_loss.item()}", flush=True)
            return

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # Actor update using critic evaluations of its predicted actions
        actor_policy_output, _, _ = self.actor(obs, recurrent_state=None)
        # Ensure proper shape: (batch_size, action_dim)
        if actor_policy_output.ndim == 1:
            actor_policy_output = actor_policy_output.unsqueeze(0)
        actor_actions = self._scale_action(actor_policy_output)
        actor_loss = -self.critic(obs, actor_actions).mean()
        
        # Check for NaN/inf
        if not torch.isfinite(actor_loss):
            print(f"[DDPG] WARNING: Non-finite actor loss: {actor_loss.item()}", flush=True)
            return

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        
        # Soft update targets
        self._soft_update(self.target_actor, self.actor, self.config.tau)
        self._soft_update(self.target_critic, self.critic, self.config.tau)

    def decay_exploration_noise(self) -> None:
        self.noise_std = max(
            self.config.exploration_noise_min,
            self.noise_std * self.config.exploration_noise_decay,
        )

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
    total_updates = 0
    
    # Training loop
    for episode in range(config.episodes):
        obs, _ = env.reset()
        done = False
        episode_step = 0
        episode_reward = 0.0
        episode_end_reason = "max_steps"
        episode_updates = 0
        best_distance = float('inf')
        steps_without_improvement = 0
        recurrent_state = agent.get_initial_state(batch_size=1)
        
        while not done:
            # Select action
            if episode_step < config.episode_warmup_steps:
                # Random exploration during warmup
                action = np.random.uniform(low=[-config.max_steering_angle, config.min_speed], 
                                         high=[config.max_steering_angle, config.max_speed], 
                                         size=(2,))
                # Advance recurrent state with random action (but since random, maybe not necessary)
                _, recurrent_state = agent.select_action(obs, recurrent_state, deterministic=True)
            else:
                action, recurrent_state = agent.select_action(obs, recurrent_state, deterministic=False)
            
            # Step environment
            obs_next, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_step += 1
            episode_reward += reward
            
            # Track stagnation (no progress toward goal)
            current_distance = env.current_distance
            if current_distance < best_distance:
                best_distance = current_distance
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1
            
            # Apply stagnation penalty if stuck, but keep the counter so termination can still occur
            if steps_without_improvement >= config.stagnation_termination_steps:
                done = True
                episode_end_reason = "stagnation"
            elif steps_without_improvement >= config.stagnation_steps and steps_without_improvement % config.stagnation_steps == 0:
                stagnation_reward = config.stagnation_penalty
                reward += stagnation_reward
                episode_reward += stagnation_reward
            
            # Track termination reason
            if done and episode_end_reason == "max_steps":  # Only override if not already set to stagnation
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
            
            # DDPG update every N environment steps (INSIDE loop)
            if episode_step > 0 and episode_step % config.update_every == 0:
                agent.update(replay_buffer)
                episode_updates += 1
                total_updates += 1
            
            obs = obs_next

        # Decay exploration after each episode
        agent.decay_exploration_noise()

        # Logging
        print(
            f"Episode {episode + 1:3d} | "
            f"Reward: {episode_reward:8.2f} | "
            f"Steps: {env.current_step:4d} | "
            f"Updates: {episode_updates:2d} | "
            f"BestDist: {best_distance:6.2f} | "
            f"Noise: {agent.noise_std:.3f} | "
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

if __name__ == "__main__":
    config = Config()
    print(
        f"[DDPG] Starting training: {config.episodes} episodes, "
        f"update every {config.update_every} episodes",
        flush=True,
    )
    print(
        f"[DDPG] Observation size: {config.lidar_sector_dim + config.pose_goal_dim + config.imu_feature_dim}, "
        f"Action dims: 2",
        flush=True,
    )
    train(config)
