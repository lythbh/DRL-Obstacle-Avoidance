"""
Shared base class for GRU/LSTM actor-critic networks.

LLM level: 3 - LLM gave the overal structure and some details, the rest implemented by us.
"""

import torch
from torch import nn
from typing import Optional, Tuple, Union
from abc import ABC, abstractmethod


RecurrentState = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


class RecurrentActorCriticBase(nn.Module, ABC):
    """
    Actor-critic network with feature encoders and a recurrent core.
    Subclasses override `get_initial_state` and `_run_recurrent`.
    """

    def __init__(self, obs_size: int, action_dim: int, config) -> None:
        """
        Initialize feature encoders and recurrent core.
        
        Parameters
        ---------
        obs_size : int
            Size of the flattened observation vector.
        action_dim : int
            Number of actions for the policy output.
        config
            Configuration object containing network and observation dimensions.
        """
        super().__init__()
        self.obs_size = obs_size
        self.action_dim = action_dim
        self.grid_shape = config.occupancy_grid_shape
        self.obstacle_dim = config.lidar_sector_dim
        self.pose_goal_dim = config.pose_goal_dim
        self.imu_dim = config.imu_feature_dim
        self.structured_obs_dim = self.obstacle_dim + self.pose_goal_dim + self.imu_dim
        if obs_size < self.structured_obs_dim:
            raise ValueError(
                f"Observation size {obs_size} is smaller than the structured feature layout \n"
                f"({self.structured_obs_dim}).\n"
            )

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

        grid_latent_dim = config.latent_size
        self.grid_encoder_cnn = None
        self.grid_encoder_mlp = None
        self.grid_feature_dim = max(obs_size - self.structured_obs_dim, 0)
        if self.grid_shape is not None:
            if len(self.grid_shape) == 2:
                in_channels = 1
            elif len(self.grid_shape) == 3:
                in_channels = self.grid_shape[0]
            else:
                raise ValueError("\noccupancy_grid_shape must be (H, W) or (C, H, W).\n")
            self.grid_encoder_cnn = nn.Sequential(
                nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(32, grid_latent_dim), nn.ReLU(),
            )
        elif self.grid_feature_dim > 0:
            self.grid_encoder_mlp = nn.Sequential(
                nn.Linear(self.grid_feature_dim, config.hidden_size), nn.ReLU(),
                nn.Linear(config.hidden_size, grid_latent_dim), nn.ReLU(),
            )
        else:
            grid_latent_dim = 0

        fusion_input_dim = 3 * branch_latent_dim + grid_latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(fusion_input_dim, config.hidden_size), nn.ReLU(),
            nn.Linear(config.hidden_size, config.latent_size), nn.ReLU(),
        )
        self.recurrent_hidden_size = config.lstm_hidden_size
        self.recurrent_layers = config.lstm_layers
        self.policy_head = nn.Linear(self.recurrent_hidden_size, action_dim)
        self.value_head = nn.Linear(self.recurrent_hidden_size, 1)


    @abstractmethod
    def get_initial_state(self, batch_size: int, device: Optional[torch.device] = None) -> RecurrentState:
        """
        Get initial hidden state for the recurrent network.

        Parameters
        ----------
        batch_size : int
            Batch size.
        device : torch.device, optional
            Device to use for the hidden state.

        Returns
        -------
        RecurrentState
            Initial hidden state for the recurrent network.
        """


    @abstractmethod
    def _run_recurrent(self, latent, recurrent_state, mask, batch_size, seq_len) -> tuple[torch.Tensor, RecurrentState]:
        """
        Execute recurrent network forward pass on latent features.
        
        Parameters
        ----------
        latent : torch.Tensor
            Latent features of shape (batch_size * seq_len, latent_dim).
        recurrent_state : RecurrentState
            Recurrent state.
        mask : torch.Tensor
            Mask of shape (batch_size, seq_len).
        batch_size : int
            Batch size.
        seq_len : int
            Sequence length.

        Returns
        -------
        torch.Tensor
            Output of the recurrent network of shape (batch_size * seq_len, recurrent_hidden_size).
        RecurrentState
            Updated recurrent state.
        """


    def _split_observation(self, observation) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, int, int]:
        """
        Parse observation tensor into obstacle, pose/goal, IMU, and optional grid components.
        
        Parameters
        ----------
        observation : torch.Tensor
            Observation tensor of shape (batch_size, seq_len, obs_size) or (obs_size,).

        Returns
        -------
        obstacle : torch.Tensor
            Obstacle features of shape (batch_size * seq_len, obstacle_dim).
        pose_goal : torch.Tensor
            Pose and goal features of shape (batch_size * seq_len, pose_goal_dim).
        imu : torch.Tensor
            IMU features of shape (batch_size * seq_len, imu_dim).
        grid : torch.Tensor or None
            Grid features of shape (batch_size * seq_len, grid_feature_dim) or None.
        batch_size : int
            Batch size.
        seq_len : int
            Sequence length.
        """
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=next(self.parameters()).device)
        if obs_tensor.ndim == 1:
            batch_size, seq_len = 1, 1
        elif obs_tensor.ndim == 2:
            batch_size, seq_len = 1, obs_tensor.shape[0]
        else:
            batch_size, seq_len = obs_tensor.shape[:2]
        obs_tensor = obs_tensor.reshape(batch_size, seq_len, -1)
        obstacle_end = self.obstacle_dim
        pose_goal_end = obstacle_end + self.pose_goal_dim
        imu_end = pose_goal_end + self.imu_dim
        obstacle = obs_tensor[..., :obstacle_end]
        pose_goal = obs_tensor[..., obstacle_end:pose_goal_end]
        imu = obs_tensor[..., pose_goal_end:imu_end]
        grid = obs_tensor[..., imu_end:] if self.grid_feature_dim > 0 else None
        return obstacle, pose_goal, imu, grid, batch_size, seq_len


    def _encode_observation(self, observation) -> tuple[torch.Tensor, int, int]:
        """
        Encode observation components through separate branches and fuse into latent representation.
        
        Parameters
        ----------
        observation : torch.Tensor
            Observation tensor of shape (batch_size, seq_len, obs_size) or (obs_size,).

        Returns
        -------
        torch.Tensor
            Latent representation of shape (batch_size, seq_len, latent_size).
        int
            Batch size.
        int
            Sequence length.
        """
        obstacle, pose_goal, imu, grid, batch_size, seq_len = self._split_observation(observation)
        n = batch_size * seq_len
        obstacle_latent = self.obstacle_encoder(obstacle.reshape(n, -1))
        pose_goal_latent = self.pose_goal_encoder(pose_goal.reshape(n, -1))
        imu_latent = self.imu_encoder(imu.reshape(n, -1))
        encoded_parts = [obstacle_latent, pose_goal_latent, imu_latent]
        if grid is not None:
            if self.grid_encoder_cnn is not None:
                grid_flat = grid.reshape(n, -1)
                grid_cnn = grid_flat.reshape(n, *self.grid_shape)
                if len(self.grid_shape) == 2:
                    grid_cnn = grid_cnn.unsqueeze(1)
                
                grid_latent = self.grid_encoder_cnn(grid_cnn)
            elif self.grid_encoder_mlp is not None:
                grid_latent = self.grid_encoder_mlp(grid.reshape(n, -1))
            else:
                raise ValueError("\nGrid observations provided but no grid encoder configured.\n")
            
            encoded_parts.append(grid_latent)
        
        latent = torch.cat(encoded_parts, dim=-1)
        latent = self.encoder(latent).reshape(batch_size, seq_len, -1)
       
        return latent, batch_size, seq_len


    def _prepare_done_mask(self, done_mask, batch_size, seq_len, device) -> torch.Tensor | None:
        """
        Convert done_mask to proper tensor shape for recurrent state reset.
        
        Parameters
        ----------
        done_mask : torch.Tensor or None
            Done mask tensor of shape (batch_size, seq_len) or (batch_size,).
        batch_size : int
            Batch size.
        seq_len : int
            Sequence length.
        device : torch.device
            Device to move the tensor to.

        Returns
        -------
        torch.Tensor or None
            Done mask tensor of shape (batch_size, seq_len) or None if done_mask is None.
        """
        if done_mask is None:
            return None
        
        mask = torch.as_tensor(done_mask, dtype=torch.float32, device=device)
        if mask.ndim == 0:
            mask = mask.view(1, 1).expand(batch_size, seq_len)
        elif mask.ndim == 1:
            mask = mask.view(batch_size, seq_len)
        
        return mask


    def encode_only(self, observation, recurrent_state=None, done_mask=None) -> Tuple[torch.Tensor, RecurrentState]:
        """
        Encode observation and run recurrent network without computing policy/value heads.
        
        Parameters
        ----------
        observation : torch.Tensor
            Observation tensor of shape (batch_size, seq_len, obs_size).
        recurrent_state : torch.Tensor or None
            Recurrent state tensor of shape (batch_size, latent_size).
        done_mask : torch.Tensor or None
            Done mask tensor of shape (batch_size, seq_len) or (batch_size,).

        Returns
        -------
        torch.Tensor
            Recurrent features tensor of shape (batch_size, seq_len, latent_size).
        RecurrentState
            Next recurrent state tensor of shape (batch_size, latent_size).
        """
        latent, batch_size, seq_len = self._encode_observation(observation)
        device = latent.device
        if recurrent_state is None:
            recurrent_state = self.get_initial_state(batch_size, device=device)
        
        mask = self._prepare_done_mask(done_mask, batch_size, seq_len, device)
        recurrent_features, next_state = self._run_recurrent(latent, recurrent_state, mask, batch_size, seq_len)
        
        return recurrent_features, next_state


    def forward(self, observation, recurrent_state=None, done_mask=None) -> Tuple[torch.Tensor, torch.Tensor, RecurrentState]:
        """
        Forward pass: encode observation, run recurrent network, compute policy and value outputs.
        
        Parameters
        ----------
        observation : torch.Tensor
            Observation tensor of shape (batch_size, seq_len, obs_size).
        recurrent_state : torch.Tensor or None
            Recurrent state tensor of shape (batch_size, latent_size).
        done_mask : torch.Tensor or None
            Done mask tensor of shape (batch_size, seq_len) or (batch_size,).

        Returns
        -------
        torch.Tensor
            Policy output tensor of shape (batch_size, seq_len, action_dim).
        torch.Tensor
            Value output tensor of shape (batch_size, seq_len).
        RecurrentState
            Next recurrent state tensor of shape (batch_size, latent_size).
        """
        latent, batch_size, seq_len = self._encode_observation(observation)
        device = latent.device
        if recurrent_state is None:
            recurrent_state = self.get_initial_state(batch_size, device=device)
        
        mask = self._prepare_done_mask(done_mask, batch_size, seq_len, device)
        recurrent_features, next_state = self._run_recurrent(latent, recurrent_state, mask, batch_size, seq_len)
        policy_output = self.policy_head(recurrent_features)
        state_value = self.value_head(recurrent_features).squeeze(-1)
        if seq_len == 1:
            policy_output = policy_output[:, 0]
            state_value = state_value[:, 0]
        
        return policy_output, state_value, next_state
