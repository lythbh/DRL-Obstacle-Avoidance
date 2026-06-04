"""
Feed-forward actor-critic network for PPO.

LLM level: 4 - LLM wrote the majority of the starting code, but we have since iterated on it a lot.
"""

import sys
from pathlib import Path
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from controllers.PPO.PPO_config import Config


class FeedForwardActorCritic(nn.Module):
    """
    Feed-forward actor-critic with PPO's structured observation branches.
    """

    def __init__(self, obs_size: int, action_dim: int, config: Config) -> None:
        """
        Initialize the feed-forward actor-critic network.

        Parameters
        ----------
        obs_size : int
            Size of the observation space.
        action_dim : int
            Size of the action space.
        config : Config
            Configuration object.
        """
        super().__init__()
        self.obs_size = obs_size
        self.action_dim = action_dim
        self.obstacle_dim = config.lidar_sector_dim
        self.pose_goal_dim = config.pose_goal_dim
        self.imu_dim = config.imu_feature_dim
        self.structured_obs_dim = self.obstacle_dim + self.pose_goal_dim + self.imu_dim
        self.grid_feature_dim = max(obs_size - self.structured_obs_dim, 0)
        branch_latent_dim = max(config.latent_size // 2, 32)


        def _branch(in_dim) -> torch.nn.Sequential:
            """
            Returns a branch of the network, uses ReLU activation.

            Parameters
            ----------
            in_dim : int
                Input dimension of the branch.
           
            Returns
            -------
            torch.nn.Sequential
                Branch of the network.
            """
            return nn.Sequential(
                nn.Linear(in_dim, config.hidden_size), nn.ReLU(),
                nn.Linear(config.hidden_size, branch_latent_dim), nn.ReLU(),
            )

        self.obstacle_encoder = _branch(self.obstacle_dim)
        self.pose_goal_encoder = _branch(self.pose_goal_dim)
        self.imu_encoder = _branch(self.imu_dim)
        grid_latent_dim = config.latent_size if self.grid_feature_dim > 0 else 0
        
        self.grid_encoder = None
        if self.grid_feature_dim > 0:
            self.grid_encoder = nn.Sequential(
                nn.Linear(self.grid_feature_dim, config.hidden_size), nn.ReLU(),
                nn.Linear(config.hidden_size, grid_latent_dim), nn.ReLU(),
            )
        
        self.encoder = nn.Sequential(
            nn.Linear(3 * branch_latent_dim + grid_latent_dim, config.hidden_size), nn.ReLU(),
            nn.Linear(config.hidden_size, config.latent_size), nn.ReLU(),
        )
        
        self.policy_head = nn.Linear(config.latent_size, action_dim)
        self.value_head = nn.Linear(config.latent_size, 1)


    def get_initial_state(self, batch_size: int) -> None:
        """
        Returns the initial state of the network.
        
        Parameters
        ----------
        batch_size : int
            Batch size of the input.
        """
        return None


    def forward(self, observation) -> tuple[torch.Tensor, torch.Tensor, None]:
        """
        Forward pass of the network. It uses the observation to compute the policy output and the state value.
        
        Parameters
        ----------
        observation : torch.Tensor
            Observation tensor.
        """
        obs = torch.as_tensor(observation, dtype=torch.float32, device=next(self.parameters()).device)
        single_step = obs.ndim == 1
        
        if obs.ndim == 1:
            obs = obs.view(1, 1, -1)
        elif obs.ndim == 2:
            obs = obs.view(1, obs.shape[0], -1)
        
        batch_size, seq_len = obs.shape[:2]
        flat = obs.reshape(batch_size * seq_len, -1)

        obstacle_end = self.obstacle_dim
        pose_goal_end = obstacle_end + self.pose_goal_dim
        imu_end = pose_goal_end + self.imu_dim
        encoded = [
            self.obstacle_encoder(flat[:, :obstacle_end]),
            self.pose_goal_encoder(flat[:, obstacle_end:pose_goal_end]),
            self.imu_encoder(flat[:, pose_goal_end:imu_end]),
        ]
        
        if self.grid_encoder is not None:
            encoded.append(self.grid_encoder(flat[:, imu_end:]))

        latent = self.encoder(torch.cat(encoded, dim=-1))
        policy_output = self.policy_head(latent).reshape(batch_size, seq_len, self.action_dim)
        state_value = self.value_head(latent).reshape(batch_size, seq_len)
        
        if single_step or seq_len == 1:
            policy_output = policy_output[:, 0]
        
        return policy_output, state_value, None
