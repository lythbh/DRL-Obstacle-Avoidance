import numpy as np
import matplotlib.pyplot as plt
# Pytorch dependensies
import torch
from torch import multiprocessing

from collections import defaultdict

from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (Compose, DoubleToFloat, ObservationNorm, StepCounter,
                          TransformedEnv)
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm

from vehicle import Driver
from controller import Camera, Lidar, GPS

EPISODES = 10
UPDATE_EVERY = 20 # episodes
EPOCHS = 4


# Creating an environment
class WebotsEnv:

    def __init__(self):

        # Setting up Altino robot
        self.altino = Driver()
        self.timestep = int(self.altino.getBasicTimeStep())
        self.headlights = self.altino.getDevice("headlights")
        self.backlights = self.altino.getDevice("backlights")

         # Constants for Altino
        self.maxSpeed = 1.8
        self.maxLeft = -3.14
        self.maxRight = 3.14
        # Minimal discrete action space: Speed, Steering
        self.actions = [
            (-0.5, 1.2),   # left
            (0.0, 1.2),    # straight
            (0.5, 1.2),    # right
            (0.0, 0.6),    # slow
            (0.0, 0.0)     # stop
        ]
        # Larger discrete action space
        # steering: [-0.6, -0.3, 0, 0.3, 0.6]
        # speed:    [0.6, 1.2]

        # Setting up sensors
        self.lidar = self.altino.getDevice("lidar")
        self.lidar.enable(self.timestep)
        self.lidarmax = self.lidar.getMaxRange()
        self.gps = self.altino.getDevice("gps")
        self.gps.enable(self.timestep)
        #self.camera = self.altino.getDevice("camera")
        #self.camera.enable(self.timestep)

        # Episode and step
        self.max_steps = 1000

        # Reward parameters
        self.endpoint = np.array([-1.5, 1.5])
        self.best_time = np.inf
        self.current_step = 0
        self.prev_distance = None
        self.current_pos = None
        self.collision = False
        self.collision_threshold = 0.15 # 0.11 is approx distance along x axis from Lidar placement to front of Altino

    def reset(self):
        # reset robot position using supervisor
        self.altino.setCustomData('reset')
        self.current_step = 0
        self.prev_distance = None

        observation = self.get_observation()
        return observation, {}

    def step(self, action):

        self.apply_action(action)
        self.altino.step(self.timestep)

        observation = self.get_observation()
        reward = self.compute_reward()

        terminated = self.check_collision()
        truncated = self.current_step >= self.max_steps

        return observation, reward, terminated, truncated, {}
    
    def get_observation(self):
        # Lidar data
        range_array = np.array(self.lidar.getRangeImage(), dtype=np.float32)
        lidar_data = np.clip(range_array, 0, self.lidarmax)
        lidar_data = lidar_data / self.lidarmax 

        # GPS data
        gps_data = self.gps.getValues()
        self.current_pos = np.array([gps_data[0], gps_data[1]], dtype=np.float32)
        obs = np.concatenate([lidar_data, self.current_pos])

        # Collision
        self.collision = bool(np.any(range_array < self.collision_threshold))
        return obs

    def apply_action(self, action_idx):
        steering, speed = self.actions[action_idx]
        self.altino.setSteeringAngle(steering)
        self.altino.setCruisingSpeed(speed)

    def compute_reward(self):
        if self.collision:
            return -100.0
        
        distance_to_end = np.linalg.norm(self.current_pos - self.endpoint)

        if distance_to_end < 0.5:
            if self.current_step < self.best_time:
                self.best_time = self.current_step
                return 200.0
            return 100.0

        progress_reward = 0.0
        if self.prev_distance is not None:
            progress_reward = self.prev_distance - distance_to_end

        self.prev_distance = distance_to_end
        self.current_step += 1
        return progress_reward
    

class PPOAgent:
    def __init__(self, obs_size, n_actions):
        # Device setup
        self.device = (
            torch.device(0)
            if torch.cuda.is_available() and not is_fork
            else torch.device("cpu")
        )
        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 0.1
        self.lr = 1e-4
        self.entropy_coef = 0.01

        # PPO networks
        self.actor = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.optimizer = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=self.lr)

    # PPO structure
    def calculate_returns(self, rewards):
        returns = np.zeros(len(rewards), dtype=np.float32)
        G = 0.0
        for t in reversed(range(len(rewards))):
            G = rewards[t] + self.gamma * G
            returns[t] = G
        return returns
    
    def select_action(self, obs):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        probs = self.actor(obs_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob
    
    def update(self, observations, actions, log_probs_old, returns, advantages, batch_size=64):
        # Converting to tensors
        observations = torch.FloatTensor(np.array(observations))
        actions = torch.LongTensor(actions)
        log_probs_old = torch.stack(log_probs_old).detach()
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)

        dataset_size = len(observations)

        for epoch in range(EPOCHS):

            indices = torch.randperm(dataset_size)

            for start in range(0, dataset_size, batch_size):
                batch_idx = indices[start:start + batch_size]
                obs_batch = observations[batch_idx]
                act_batch = actions[batch_idx]
                logp_old_batch = log_probs_old[batch_idx]
                ret_batch = returns[batch_idx]
                adv_batch = advantages[batch_idx]

                # Probabilities from current actor
                probs = self.actor(obs_batch)
                dist = torch.distributions.Categorical(probs)
                log_probs_new = dist.log_prob(act_batch)
                entropy = dist.entropy().mean()

                # PPO policy loss
                ratio = torch.exp(log_probs_new - logp_old_batch)
                surrogate1 = ratio * adv_batch
                surrogate2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * adv_batch
                policy_loss = -torch.min(surrogate1, surrogate2).mean()

                # Value loss
                values = self.critic(obs_batch).squeeze()
                value_loss = nn.MSELoss()(values, ret_batch)

                # Combined loss
                loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

def train():
    env = WebotsEnv()
    obs_size = env.lidar.getHorizontalResolution() + 2  # Lidar points + GPS x,y
    n_actions = len(env.actions)
    agent = PPOAgent(obs_size, n_actions)

    all_observations, all_actions, all_log_probs, all_rewards = [], [], [], []

    for episode in range(EPISODES):
        obs, _ = env.reset()
        done = False
        episode_rewards = []

        while not done:
            action, log_prob = agent.select_action(obs)

            obs_next, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            all_observations.append(obs)
            all_actions.append(action)
            all_log_probs.append(log_prob)
            episode_rewards.append(reward)

            obs = obs_next

        all_rewards.extend(episode_rewards)

        # Update PPO every N episodes
        if (episode + 1) % UPDATE_EVERY == 0:
            returns = agent.calculate_returns(all_rewards)

            # Advantage = returns - critic baseline
            obs_tensor = torch.FloatTensor(np.array(all_observations))
            values = agent.critic(obs_tensor).squeeze().detach().numpy()
            advantages = returns - values

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            agent.update(all_observations, all_actions, all_log_probs, returns, advantages)

            # Clear buffers
            all_observations, all_actions, all_log_probs, all_rewards = [], [], [], []

        print(f"Episode {episode + 1}, Reward: {sum(episode_rewards):.2f}")

train()

# SENSOR PROCESSING
# image = camera.getImage()
# width = camera.getWidth()
# height = camera.getHeight()
# image_array = np.frombuffer(image, np.uint8).reshape((height, width, 4))
# rgb = image_array[:, :, :3]
# camera.saveImage("image.png", 1000)


"""
NEXT TIME

- Either change Altino to Robot(), controlling its independent motors
- Or use a vehicle from the vehicle list
"""
