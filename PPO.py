"""PPO training controller for ALTINO robot in Webots obstacle avoidance task."""

from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Any, Union
import numpy as np
import torch
from torch import multiprocessing, nn

from controller import Supervisor  # pyright: ignore[reportMissingImports]


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Training and environment hyperparameters."""
    
    # Training
    episodes: int = 50
    update_every: int = 5  # PPO update frequency (episodes)
    epochs: int = 4  # Optimization epochs per update
    batch_size: int = 64
    
    # PPO Agent
    gamma: float = 0.99  # Discount factor
    epsilon: float = 0.1  # PPO clip parameter
    learning_rate: float = 1e-4
    entropy_coef: float = 0.01  # Entropy regularization
    hidden_size: int = 128  # Network hidden layer size
    
    # Environment
    max_steps: int = 1000  # Max steps per episode
    collision_threshold: float = 0.15  # LiDAR distance threshold for collision
    low_score_threshold: float = -100.0  # Episode reset threshold
    endpoint: Tuple[float, float] = (-1.5, 1.5)  # Goal location
    
    # Robot Control
    actions: Optional[List[Tuple[float, float]]] = None  # (steering, speed) pairs
    start_position: Optional[List[float]] = None  # [x, y, z]
    start_rotation: Optional[List[float]] = None  # [x, y, z, w]
    
    # Motor/Sensor Config
    max_speed: float = 1.8
    reset_settle_steps: int = 10  # Steps to wait for physics to settle after reset
    
    def __post_init__(self) -> None:
        """Initialize defaults for mutable fields."""
        if self.actions is None:
            self.actions = [
                (-0.5, 1.2),   # left
                (0.0, 1.2),    # straight
                (0.5, 1.2),    # right
                (0.0, 0.6),    # slow
                (0.0, 0.0)     # stop
            ]
        if self.start_position is None:
            self.start_position = [0.05, -0.1, 0.02]
        if self.start_rotation is None:
            self.start_rotation = [0.0, 0.0, 1.0, 0.0]


# Global supervisor instance
_supervisor: Optional[Supervisor] = None

def _init_supervisor() -> None:
    """Initialize global Supervisor instance for field access."""
    global _supervisor
    _supervisor = Supervisor()
    print("[PPO] Supervisor instance initialized")



# ============================================================================
# HARDWARE CONTROL LAYER
# ============================================================================

class MotorController:
    """Manages steering and wheel motors."""
    
    def __init__(self, supervisor: Supervisor):
        """Initialize motor devices."""
        self.supervisor = supervisor
        
        # Steering motors
        self.left_steer = supervisor.getDevice('left_steer')
        self.right_steer = supervisor.getDevice('right_steer')
        self._init_steering()
        
        # Wheel motors
        self.wheels = [
            supervisor.getDevice('left_front_wheel'),
            supervisor.getDevice('right_front_wheel'),
            supervisor.getDevice('left_rear_wheel'),
            supervisor.getDevice('right_rear_wheel'),
        ]
        self._init_wheels()
    
    def _init_steering(self) -> None:
        """Initialize steering motors."""
        for motor in [self.left_steer, self.right_steer]:
            motor.setPosition(0.0)
            motor.setVelocity(1.0)
    
    def _init_wheels(self) -> None:
        """Initialize wheel motors."""
        for motor in self.wheels:
            motor.setPosition(float('inf'))
            motor.setVelocity(0.0)
    
    def set_steering(self, angle: float) -> None:
        """Set steering angle (radians)."""
        self.left_steer.setPosition(angle)
        self.right_steer.setPosition(angle)
    
    def set_speed(self, speed: float) -> None:
        """Set wheel velocity."""
        for motor in self.wheels:
            motor.setVelocity(speed)
    
    def stop(self) -> None:
        """Stop all motors."""
        self.set_steering(0.0)
        self.set_speed(0.0)


class SensorReader:
    """Manages LiDAR and GPS sensors."""
    
    def __init__(self, supervisor: Supervisor, timestep: int):
        """Initialize sensors."""
        self.supervisor = supervisor
        self.timestep = timestep
        
        # LiDAR
        self.lidar = supervisor.getDevice("lidar")
        self.lidar.enable(timestep)
        self.lidar_max_range = self.lidar.getMaxRange()
        
        # GPS
        self.gps = supervisor.getDevice("gps")
        self.gps.enable(timestep)
    
    def read_observation(self) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Read LiDAR and GPS data.
        
        Returns:
            lidar_data: Normalized LiDAR ranges [0, 1]
            position: GPS [x, y] in world coordinates
            collision: Boolean collision detection
        """
        # Read and normalize LiDAR
        range_array = np.array(self.lidar.getRangeImage(), dtype=np.float32)
        lidar_data = np.clip(range_array, 0, self.lidar_max_range)
        lidar_data = lidar_data / self.lidar_max_range
        
        # Read GPS
        gps_values = self.gps.getValues()
        position = np.array([gps_values[0], gps_values[1]], dtype=np.float32)
        
        # Collision detection
        collision = bool(np.any(range_array < 0.15))  # Will be parameterized
        
        return lidar_data, position, collision


class AltinoDriver:
    """High-level robot control interface."""
    
    def __init__(self, config: Config):
        """Initialize robot with config."""
        global _supervisor
        assert _supervisor is not None, "Supervisor not initialized. Call _init_supervisor() first."
        self.supervisor = _supervisor
        self.config = config
        self.timestep = int(self.supervisor.getBasicTimeStep())  # type: ignore[union-attr]
        
        # Hardware components
        self.motors = MotorController(self.supervisor)  # type: ignore[arg-type]
        self.sensors = SensorReader(self.supervisor, self.timestep)  # type: ignore[arg-type]
        
        # Position reset
        try:
            self.altino_node = self.supervisor.getFromDef('ALTINO')  # type: ignore[union-attr]
            self.translation_field = self.altino_node.getField('translation')
            self.rotation_field = self.altino_node.getField('rotation')
            print("[PPO] ALTINO node accessed for direct position reset")
        except Exception as e:
            print(f"[PPO] ERROR: Failed to get ALTINO node: {e}")
            self.altino_node = None
            self.translation_field = None
            self.rotation_field = None
    
    def set_steering(self, angle: float) -> None:
        """Set steering angle."""
        self.motors.set_steering(angle)
    
    def set_speed(self, speed: float) -> None:
        """Set velocity."""
        self.motors.set_speed(speed)
    
    def get_device(self, name: str):
        """Get a named device from supervisor."""
        return self.supervisor.getDevice(name)  # type: ignore[union-attr]
    
    def step(self, timestep: int) -> int:
        """Step simulation."""
        return self.supervisor.step(timestep)  # type: ignore[union-attr]
    
    def read_sensors(self) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Read all sensors."""
        return self.sensors.read_observation()
    
    def reset_position(self) -> None:
        """Reset robot to start position and orientation."""
        if self.translation_field is not None and self.rotation_field is not None:
            self.translation_field.setSFVec3f(self.config.start_position)  # type: ignore[arg-type]
            self.rotation_field.setSFRotation(self.config.start_rotation)  # type: ignore[arg-type]
            self.supervisor.simulationResetPhysics()  # type: ignore[union-attr]
            print(f"[PPO] Robot reset to position {self.config.start_position}")
        else:
            print("[PPO] WARNING: Cannot reset - ALTINO node not accessible!")



# ============================================================================
# ENVIRONMENT
# ============================================================================

class RewardComputer:
    """Computes rewards for the obstacle avoidance task."""
    
    def __init__(self, endpoint: np.ndarray, collision_reward: float = -100.0):
        """Initialize reward computer.
        
        Args:
            endpoint: Goal position [x, y]
            collision_reward: Penalty for collision
        """
        self.endpoint = np.array(endpoint, dtype=np.float32)
        self.collision_reward = collision_reward
        self.best_time = np.inf
    
    def compute(
        self,
        collision: bool,
        current_pos: np.ndarray,
        current_step: int,
        prev_distance: Optional[float]
    ) -> Tuple[float, Optional[float]]:
        """Compute reward for current state.
        
        Args:
            collision: Whether collision occurred
            current_pos: Current [x, y] position
            current_step: Current step in episode
            prev_distance: Distance from previous step (None on first step)
        
        Returns:
            reward: Scalar reward
            new_distance: Distance to endpoint (for next step)
        """
        if collision:
            return self.collision_reward, None
        
        distance_to_end = float(np.linalg.norm(current_pos - self.endpoint))
        
        # Goal reached
        if distance_to_end < 0.5:
            if current_step < self.best_time:
                self.best_time = current_step
                return 200.0, distance_to_end
            return 100.0, distance_to_end
        
        # Progress reward
        progress = 0.0
        if prev_distance is not None:
            progress = float(prev_distance - distance_to_end)
        
        return progress, distance_to_end


class WebotsEnv:
    """Webots simulation environment for ALTINO obstacle avoidance."""
    
    def __init__(self, config: Config):
        """Initialize environment with config.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.robot = AltinoDriver(config)
        self.timestep = self.robot.timestep
        
        # Lights
        self.headlights = self.robot.get_device("headlights")
        self.backlights = self.robot.get_device("backlights")
        
        # Reward computation
        self.reward_computer = RewardComputer(np.array(config.endpoint, dtype=np.float32))
        
        # Episode state
        self.current_step = 0
        self.episode_reward = 0.0
        self.current_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.collision = False
        self.prev_distance: Optional[float] = None
    
    def _reset_episode_state(self) -> None:
        """Reset internal episode state."""
        self.current_step = 0
        self.prev_distance = None
        self.episode_reward = 0.0
        self.collision = False
    
    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and return initial observation.
        
        Returns:
            observation: Initial LiDAR + GPS observation
            info: Info dict (empty on reset)
        """
        # Stop motors before reset
        self.robot.motors.stop()
        
        # Reset position
        print("[ENV] Resetting episode state")
        self.robot.reset_position()
        self._reset_episode_state()
        
        # Settle physics
        for _ in range(self.config.reset_settle_steps):
            self.robot.step(self.timestep)
        
        # Get observation
        lidar, pos, collision = self.robot.read_sensors()
        self.current_pos = pos
        self.collision = collision
        
        observation = np.concatenate([lidar, pos])
        print(f"[ENV] Reset complete. Collision detected: {collision}")
        return observation, {}
    
    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step environment.
        
        Args:
            action_idx: Discrete action index
        
        Returns:
            observation: Current observation
            reward: Step reward
            terminated: Whether episode ended
            truncated: Whether max steps reached
            info: Info dict with metadata
        """
        # Apply action
        assert self.config.actions is not None, "Actions not initialized"
        steering, speed = self.config.actions[action_idx]
        self.robot.set_steering(steering)
        self.robot.set_speed(speed)
        self.robot.step(self.timestep)
        self.current_step += 1
        
        # Sense
        lidar, pos, collision = self.robot.read_sensors()
        self.current_pos = pos
        self.collision = collision
        
        # Compute reward
        reward, new_distance = self.reward_computer.compute(
            collision, self.current_pos, self.current_step, self.prev_distance
        )
        self.prev_distance = new_distance
        self.episode_reward += reward
        
        # Termination conditions
        terminated = collision
        truncated = self.current_step >= self.config.max_steps
        info: Dict[str, Any] = {}
        
        # Low score reset
        if self.episode_reward <= self.config.low_score_threshold:
            terminated = True
            info["reset_reason"] = "low_score"
            print(f"[ENV] Low score detected: {self.episode_reward:.2f}, requesting reset")
            self.robot.reset_position()
        elif terminated:
            info["reset_reason"] = "collision"
        
        # Observation
        observation = np.concatenate([lidar, pos])
        
        return observation, reward, terminated, truncated, info


# ============================================================================
# PPO AGENT
# ============================================================================

class PPOAgent:
    """Proximal Policy Optimization agent."""
    
    def __init__(self, obs_size: int, n_actions: int, config: Config):
        """Initialize PPO agent.
        
        Args:
            obs_size: Observation dimension
            n_actions: Number of discrete actions
            config: Configuration object
        """
        self.config = config
        self.device = self._get_device()
        
        # Networks
        self.actor = self._build_actor(obs_size, n_actions)
        self.critic = self._build_critic(obs_size)
        
        # Optimizer
        params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.optimizer = torch.optim.Adam(params, lr=config.learning_rate)
    
    def _get_device(self) -> torch.device:
        """Get appropriate device (GPU or CPU)."""
        is_fork = multiprocessing.get_start_method(allow_none=True) == 'fork'
        if torch.cuda.is_available() and not is_fork:
            return torch.device("cuda:0")
        return torch.device("cpu")
    
    def _build_actor(self, obs_size: int, n_actions: int) -> nn.Module:
        """Build actor (policy) network."""
        return nn.Sequential(
            nn.Linear(obs_size, self.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size, n_actions),
            nn.Softmax(dim=-1)
        ).to(self.device)
    
    def _build_critic(self, obs_size: int) -> nn.Module:
        """Build critic (value) network."""
        return nn.Sequential(
            nn.Linear(obs_size, self.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size, 1)
        ).to(self.device)
    
    def select_action(self, obs: np.ndarray) -> Tuple[int, torch.Tensor]:
        """Select action using current policy.
        
        Args:
            obs: Observation array
        
        Returns:
            action: Sampled action index
            log_prob: Log probability of action
        """
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            probs = self.actor(obs_tensor)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return int(action.item()), log_prob
    
    def calculate_returns(self, rewards: np.ndarray) -> np.ndarray:
        """Calculate cumulative discounted returns.
        
        Args:
            rewards: Reward array
        
        Returns:
            returns: Cumulative returns
        """
        returns = np.zeros(len(rewards), dtype=np.float32)
        G = 0.0
        for t in reversed(range(len(rewards))):
            G = rewards[t] + self.config.gamma * G
            returns[t] = G
        return returns
    
    def update(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        log_probs_old: List[torch.Tensor],
        returns: np.ndarray,
        advantages: np.ndarray
    ) -> None:
        """Update policy using collected rollout.
        
        Args:
            observations: Trajectory observations
            actions: Trajectory actions
            log_probs_old: Old log probabilities
            returns: Cumulative returns
            advantages: Computed advantages
        """
        # Convert to tensors
        obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        act_tensor = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        logp_old_tensor = torch.stack(log_probs_old).detach().to(self.device)
        ret_tensor = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        adv_tensor = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        
        dataset_size = len(observations)
        
        # Multiple epochs of updates
        for epoch in range(self.config.epochs):
            indices = torch.randperm(dataset_size)
            
            for start in range(0, dataset_size, self.config.batch_size):
                batch_idx = indices[start:start + self.config.batch_size]
                
                obs_batch = obs_tensor[batch_idx]
                act_batch = act_tensor[batch_idx]
                logp_old_batch = logp_old_tensor[batch_idx]
                ret_batch = ret_tensor[batch_idx]
                adv_batch = adv_tensor[batch_idx]
                
                # Forward pass
                probs = self.actor(obs_batch)
                dist = torch.distributions.Categorical(probs)
                log_probs_new = dist.log_prob(act_batch)
                entropy = dist.entropy().mean()
                
                # PPO loss
                ratio = torch.exp(log_probs_new - logp_old_batch)
                surrogate1 = ratio * adv_batch
                surrogate2 = torch.clamp(ratio, 1 - self.config.epsilon, 1 + self.config.epsilon) * adv_batch
                policy_loss = -torch.min(surrogate1, surrogate2).mean()
                
                # Value loss
                values = self.critic(obs_batch).squeeze()
                value_loss = nn.MSELoss()(values, ret_batch)
                
                # Combined loss
                loss = policy_loss + 0.5 * value_loss - self.config.entropy_coef * entropy
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()



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
    
    # Create environment and agent
    env = WebotsEnv(config)
    obs, _ = env.reset()
    obs_size = len(obs)
    assert config.actions is not None, "Actions not initialized"
    n_actions = len(config.actions)
    agent = PPOAgent(obs_size, n_actions, config)
    
    print(f"[TRAIN] Starting training: {config.episodes} episodes, "
          f"update every {config.update_every} episodes")
    print(f"[TRAIN] Observation size: {obs_size}, Action space: {n_actions}")
    
    # Training buffers
    all_observations: List[np.ndarray] = []
    all_actions: List[int] = []
    all_log_probs: List[torch.Tensor] = []
    all_rewards: List[float] = []
    
    # Training loop
    for episode in range(config.episodes):
        obs, _ = env.reset()
        done = False
        episode_rewards: List[float] = []
        episode_end_reason = "max_steps"
        
        while not done:
            # Select action
            action, log_prob = agent.select_action(obs)
            
            # Step environment
            obs_next, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Track termination reason
            if done:
                if info.get("reset_reason") == "low_score":
                    episode_end_reason = "low_score"
                elif info.get("reset_reason") == "collision":
                    episode_end_reason = "collision"
                elif truncated:
                    episode_end_reason = "max_steps"
            
            # Accumulate
            all_observations.append(obs)
            all_actions.append(action)
            all_log_probs.append(log_prob)
            episode_rewards.append(reward)
            
            obs = obs_next
        
        all_rewards.extend(episode_rewards)
        
        # PPO update every N episodes
        if (episode + 1) % config.update_every == 0:
            print(f"[TRAIN] Update at episode {episode + 1}")
            
            # Compute returns
            returns = agent.calculate_returns(np.array(all_rewards, dtype=np.float32))
            
            # Compute advantages
            obs_array = np.array(all_observations, dtype=np.float32)
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32, device=agent.device)
                values = agent.critic(obs_tensor).squeeze().detach().cpu().numpy()
            
            advantages = returns - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Update policy
            agent.update(
                obs_array,
                np.array(all_actions, dtype=np.int64),
                all_log_probs,
                returns,
                advantages
            )
            
            # Clear buffers
            all_observations.clear()
            all_actions.clear()
            all_log_probs.clear()
            all_rewards.clear()
        
        # Logging
        episode_reward_sum = sum(episode_rewards)
        print(
            f"Episode {episode + 1:2d} | "
            f"Reward: {episode_reward_sum:8.2f} | "
            f"Steps: {env.current_step:4d} | "
            f"End: {episode_end_reason}"
        )
    
    # Cleanup
    env.robot.motors.stop()
    print("[TRAIN] Training complete. Robot stopped.")


if __name__ == "__main__":
    train()