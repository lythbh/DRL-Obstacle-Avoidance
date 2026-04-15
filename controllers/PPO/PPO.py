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
    episodes: int = 500
    update_every: int = 5  # PPO update frequency (episodes)
    epochs: int = 4  # Optimization epochs per update
    batch_size: int = 64
    
    # PPO Agent
    gamma: float = 0.99  # Discount factor
    epsilon: float = 0.1  # PPO clip parameter
    learning_rate: float = 3e-4
    entropy_coef: float = 0.02  # Entropy regularization
    hidden_size: int = 128  # Network hidden layer size
    
    # Environment
    max_steps: int = 2000  # Max steps per episode
    collision_threshold: float = 0.1  # LiDAR distance threshold for collision
    low_score_threshold: float = -400.0  # Episode reset threshold
    collision_penalty: float = -20.0  # Penalty when collision happens
    progress_reward_scale: float = 3.0  # Scale for distance-progress reward
    distance_reward_scale: float = 2.0  # Dense reward for being closer to the goal than the start state
    heading_reward_scale: float = 0.08  # Bonus when facing toward the goal
    safety_reward_scale: float = 0.2  # Encourages keeping distance from obstacles
    motion_reward_scale: float = 0.05  # Bonus for moving forward to avoid stop-policy collapse
    new_best_distance_bonus: float = 1.0  # Bonus when reaching a new closest distance to goal
    step_penalty: float = -0.01  # Small per-step penalty to encourage efficiency
    endpoint: Tuple[float, float] = (-2, 0)  # Goal location
    reference_distance: Optional[float] = None  # Start-to-goal distance, filled in at init
    
    # Robot Control
    actions: Optional[List[Tuple[float, float]]] = None  # (steering, speed) pairs
    start_position: Optional[List[float]] = None  # [x, y, z]
    start_rotation: Optional[List[float]] = None  # [x, y, z, w]
    start_position_noise: float = 0.03  # Random position jitter at reset
    start_yaw_noise: float = 0.4  # Random yaw jitter at reset
    
    # Motor/Sensor Config
    max_speed: float = 10.0
    reset_settle_steps: int = 10  # Steps to wait for physics to settle after reset
    
    def __post_init__(self) -> None:
        """Initialize defaults for mutable fields."""
        if self.actions is None:
            self.actions = [
                (-0.9, 0.8 * self.max_speed),   # hard left
                (-0.5, 0.5 * self.max_speed),   # medium left
                (0.0, self.max_speed),    # straight
                (0.5, 0.5 * self.max_speed),    # medium right
                (0.9, 0.8 * self.max_speed),    # hard right
                (0.0, 0.5 * self.max_speed),    # slow straight
                (0.0, 0.0)     # stop
            ]
        if self.start_position is None:
            self.start_position = [0.05, -0.1, 0.02]
        if self.start_rotation is None:
            self.start_rotation = [0.0, 0.0, 1.0, 0.0]
        if self.reference_distance is None:
            start_xy = np.array(self.start_position[:2], dtype=np.float32)
            endpoint_xy = np.array(self.endpoint, dtype=np.float32)
            self.reference_distance = float(np.linalg.norm(start_xy - endpoint_xy))


# Global supervisor instance
_supervisor: Optional[Supervisor] = None

def _init_supervisor() -> None:
    """Initialize global Supervisor instance for field access."""
    global _supervisor
    _supervisor = Supervisor()



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
    
    def __init__(self, supervisor: Supervisor, timestep: int, collision_threshold: float):
        """Initialize sensors."""
        self.supervisor = supervisor
        self.timestep = timestep
        self.collision_threshold = collision_threshold
        
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
        collision = bool(np.any(range_array < self.collision_threshold))
        
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
        self.sensors = SensorReader(
            self.supervisor,
            self.timestep,
            config.collision_threshold
        )  # type: ignore[arg-type]
        
        # Position reset
        try:
            self.altino_node = self.supervisor.getFromDef('ALTINO')  # type: ignore[union-attr]
            self.translation_field = self.altino_node.getField('translation')
            self.rotation_field = self.altino_node.getField('rotation')
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
    
    def _get_heading(self) -> float:
        """Estimate robot yaw in world frame from rotation field."""
        if self.rotation_field is None:
            return 0.0

        rotation = self.rotation_field.getSFRotation()
        if rotation is None or len(rotation) < 4:
            return 0.0

        x, y, z, angle = map(float, rotation)
        axis_norm = np.sqrt(x * x + y * y + z * z)
        if axis_norm < 1e-8:
            return 0.0
        x /= axis_norm
        y /= axis_norm
        z /= axis_norm

        # Convert axis-angle to rotation matrix and extract yaw.
        c = float(np.cos(angle))
        s = float(np.sin(angle))
        one_c = 1.0 - c
        r00 = c + x * x * one_c
        r10 = z * s + y * x * one_c
        yaw = float(np.arctan2(r10, r00))
        return float(np.arctan2(np.sin(yaw), np.cos(yaw)))

    def read_sensors(self) -> Tuple[np.ndarray, np.ndarray, float, bool]:
        """Read LiDAR, GPS, heading and collision status."""
        lidar, pos, collision = self.sensors.read_observation()
        heading = self._get_heading()
        return lidar, pos, heading, collision
    
    def reset_position(self) -> None:
        """Reset robot to start position and orientation."""
        if self.translation_field is not None and self.rotation_field is not None:
            start_position_values = self.config.start_position or [0.05, -0.1, 0.02]
            start_rotation_values = self.config.start_rotation or [0.0, 0.0, 1.0, 0.0]

            start_position = np.array(start_position_values, dtype=np.float32)
            if self.config.start_position_noise > 0.0:
                start_position[:2] += np.random.uniform(
                    -self.config.start_position_noise,
                    self.config.start_position_noise,
                    size=2,
                ).astype(np.float32)

            start_rotation = list(start_rotation_values)
            if self.config.start_yaw_noise > 0.0:
                start_rotation[3] = float(
                    start_rotation[3]
                    + np.random.uniform(-self.config.start_yaw_noise, self.config.start_yaw_noise)
                )

            self.translation_field.setSFVec3f(start_position.tolist())  # type: ignore[arg-type]
            self.rotation_field.setSFRotation(start_rotation)  # type: ignore[arg-type]
            self.supervisor.simulationResetPhysics()  # type: ignore[union-attr]
        else:
            print("[PPO] WARNING: Cannot reset - ALTINO node not accessible!")



# ============================================================================
# ENVIRONMENT
# ============================================================================

class RewardComputer:
    """Computes rewards for the obstacle avoidance task."""
    
    def __init__(
        self,
        endpoint: np.ndarray,
        reference_distance: float,
        collision_reward: float = -40.0,
        progress_scale: float = 3.0,
        distance_reward_scale: float = 2.0,
        heading_reward_scale: float = 0.08,
        safety_reward_scale: float = 0.2,
        motion_reward_scale: float = 0.05,
        new_best_distance_bonus: float = 1.0,
        step_penalty: float = -0.01,
    ):
        """Initialize reward computer.
        
        Args:
            endpoint: Goal position [x, y]
            collision_reward: Penalty for collision
        """
        self.endpoint = np.array(endpoint, dtype=np.float32)
        self.reference_distance = float(reference_distance)
        self.collision_reward = collision_reward
        self.progress_scale = progress_scale
        self.distance_reward_scale = distance_reward_scale
        self.heading_reward_scale = heading_reward_scale
        self.safety_reward_scale = safety_reward_scale
        self.motion_reward_scale = motion_reward_scale
        self.new_best_distance_bonus = new_best_distance_bonus
        self.step_penalty = step_penalty
        self.best_time = np.inf
    
    def compute(
        self,
        collision: bool,
        current_pos: np.ndarray,
        current_step: int,
        prev_distance: Optional[float],
        goal_error: float,
        min_lidar_norm: float,
        speed_norm: float,
        reached_new_best_distance: bool,
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
            delta = float(prev_distance - distance_to_end)
            progress = delta * self.progress_scale if delta >= 0.0 else delta * (0.25 * self.progress_scale)

        # Dense goal-proximity reward, normalized against the start distance.
        proximity = max(0.0, (self.reference_distance - distance_to_end) / max(self.reference_distance, 1e-6))
        proximity_reward = proximity * self.distance_reward_scale
        heading_alignment = max(0.0, float(np.cos(goal_error)))
        heading_reward = heading_alignment * self.heading_reward_scale
        safety_reward = float(np.clip(min_lidar_norm, 0.0, 1.0)) * self.safety_reward_scale
        motion_reward = float(np.clip(speed_norm, 0.0, 1.0)) * self.motion_reward_scale
        new_best_bonus = self.new_best_distance_bonus if reached_new_best_distance else 0.0

        return (
            progress
            + proximity_reward
            + heading_reward
            + safety_reward
            + motion_reward
            + new_best_bonus
            + self.step_penalty
        ), distance_to_end


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
        self.reward_computer = RewardComputer(
            np.array(config.endpoint, dtype=np.float32),
            reference_distance=float(config.reference_distance if config.reference_distance is not None else 1.0),
            collision_reward=config.collision_penalty,
            progress_scale=config.progress_reward_scale,
            distance_reward_scale=config.distance_reward_scale,
            heading_reward_scale=config.heading_reward_scale,
            safety_reward_scale=config.safety_reward_scale,
            motion_reward_scale=config.motion_reward_scale,
            new_best_distance_bonus=config.new_best_distance_bonus,
            step_penalty=config.step_penalty,
        )
        
        # Episode state
        self.current_step = 0
        self.episode_reward = 0.0
        self.current_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.current_heading = 0.0
        self.current_distance = float("inf")
        self.min_episode_distance = float("inf")
        self.collision = False
        self.prev_distance: Optional[float] = None
    
    def _reset_episode_state(self) -> None:
        """Reset internal episode state."""
        self.current_step = 0
        self.prev_distance = None
        self.episode_reward = 0.0
        self.current_heading = 0.0
        self.current_distance = float("inf")
        self.min_episode_distance = float("inf")
        self.collision = False

    def _goal_geometry(self, pos: np.ndarray, heading: float) -> Tuple[float, float]:
        """Compute goal distance and heading error."""
        goal_vec = np.array(self.config.endpoint, dtype=np.float32) - pos
        goal_distance = float(np.linalg.norm(goal_vec))
        goal_direction = float(np.arctan2(goal_vec[1], goal_vec[0]))
        goal_error = float(np.arctan2(np.sin(goal_direction - heading), np.cos(goal_direction - heading)))
        return goal_distance, goal_error

    def _build_observation(self, lidar: np.ndarray, pos: np.ndarray, heading: float) -> np.ndarray:
        """Build observation vector with heading and goal-direction context."""
        goal_distance, goal_error = self._goal_geometry(pos, heading)
        ref_dist = float(self.config.reference_distance if self.config.reference_distance is not None else 1.0)

        direction_features = np.array([
            np.sin(heading),
            np.cos(heading),
            np.sin(goal_error),
            np.cos(goal_error),
            goal_distance / max(ref_dist, 1e-6),
        ], dtype=np.float32)

        return np.concatenate([lidar, pos, direction_features])
    
    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and return initial observation.
        
        Returns:
            observation: Initial LiDAR + GPS observation
            info: Info dict (empty on reset)
        """
        # Stop motors before reset
        self.robot.motors.stop()
        
        # Reset position
        self.robot.reset_position()
        self._reset_episode_state()
        
        # Settle physics
        for _ in range(self.config.reset_settle_steps):
            self.robot.step(self.timestep)
        
        # Get observation
        lidar, pos, heading, collision = self.robot.read_sensors()
        self.current_pos = pos
        self.current_heading = heading
        self.collision = collision
        self.current_distance, _ = self._goal_geometry(pos, heading)
        self.min_episode_distance = self.current_distance
        
        observation = self._build_observation(lidar, pos, heading)
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
        lidar, pos, heading, collision = self.robot.read_sensors()
        self.current_pos = pos
        self.current_heading = heading
        self.collision = collision
        self.current_distance, goal_error = self._goal_geometry(pos, heading)
        reached_new_best_distance = self.current_distance + 1e-6 < self.min_episode_distance
        if reached_new_best_distance:
            self.min_episode_distance = self.current_distance
        min_lidar_norm = float(np.min(lidar))
        speed_norm = float(speed / max(self.config.max_speed, 1e-6))
        
        # Compute reward
        reward, new_distance = self.reward_computer.compute(
            collision,
            self.current_pos,
            self.current_step,
            self.prev_distance,
            goal_error,
            min_lidar_norm,
            speed_norm,
            reached_new_best_distance,
        )
        self.prev_distance = new_distance
        self.episode_reward += reward
        
        # Termination conditions
        terminated = collision
        truncated = self.current_step >= self.config.max_steps
        info: Dict[str, Any] = {}
        
        # Collision should take precedence over low-score bookkeeping.
        if collision:
            info["reset_reason"] = "collision"
        elif self.current_distance < 0.5:
            terminated = True
            info["reset_reason"] = "goal_reached"
        elif self.episode_reward <= self.config.low_score_threshold:
            terminated = True
            info["reset_reason"] = "low_score"

        if terminated or truncated:
            self.robot.motors.stop()
        
        # Observation
        observation = self._build_observation(lidar, pos, heading)
        
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
    all_returns: List[float] = []
    all_advantages: List[float] = []
    
    # Training loop
    for episode in range(config.episodes):
        obs, _ = env.reset()
        done = False
        episode_observations: List[np.ndarray] = []
        episode_actions: List[int] = []
        episode_log_probs: List[torch.Tensor] = []
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
                if info.get("reset_reason") == "low_score":
                    episode_end_reason = "low_score"
                elif info.get("reset_reason") == "collision":
                    episode_end_reason = "collision"
                elif info.get("reset_reason") == "goal_reached":
                    episode_end_reason = "goal_reached"
                elif truncated:
                    episode_end_reason = "max_steps"
            
            # Accumulate
            episode_observations.append(obs)
            episode_actions.append(action)
            episode_log_probs.append(log_prob)
            episode_rewards.append(reward)
            
            obs = obs_next
        
        # Build returns and advantages per episode so reward signals do not
        # leak across episode boundaries.
        episode_returns = agent.calculate_returns(np.array(episode_rewards, dtype=np.float32))
        episode_obs_array = np.array(episode_observations, dtype=np.float32)
        with torch.no_grad():
            episode_obs_tensor = torch.as_tensor(
                episode_obs_array,
                dtype=torch.float32,
                device=agent.device,
            )
            episode_values = agent.critic(episode_obs_tensor).squeeze().detach().cpu().numpy()

        episode_advantages = episode_returns - episode_values

        all_observations.extend(episode_obs_array)
        all_actions.extend(episode_actions)
        all_log_probs.extend(episode_log_probs)
        all_returns.extend(episode_returns.tolist())
        all_advantages.extend(episode_advantages.tolist())
        
        # PPO update every N episodes
        if (episode + 1) % config.update_every == 0:
            obs_array = np.array(all_observations, dtype=np.float32)
            returns = np.array(all_returns, dtype=np.float32)
            advantages = np.array(all_advantages, dtype=np.float32)
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
            all_returns.clear()
            all_advantages.clear()
        
        # Logging
        episode_reward_sum = sum(episode_rewards)
        print(
            f"Episode {episode + 1:2d} | "
            f"Reward: {episode_reward_sum:8.2f} | "
            f"Steps: {env.current_step:4d} | "
            f"MinDist: {env.min_episode_distance:6.2f} | "
            f"LastDist: {env.current_distance:6.2f} | "
            f"End: {episode_end_reason}"
        )
    
    # Cleanup
    env.robot.motors.stop()
    print("[TRAIN] Training complete. Robot stopped.")


if __name__ == "__main__":
    train()