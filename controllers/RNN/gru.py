"""GRU actor-critic used by shared RL controllers."""

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn

if TYPE_CHECKING:
    from controllers.PPO.PPO import Config


RecurrentState = torch.Tensor


class GRUActorCritic(nn.Module):
    """Actor-critic network with lightweight feature branches and a GRU core."""

    def __init__(
        self,
        obs_size: int,
        action_dim: int,
        config: "Config",
        action_space: str = "continuous",
    ) -> None:
        super().__init__()
        self.obs_size = obs_size
        self.action_dim = action_dim
        self.action_space = action_space
        self.grid_shape = config.occupancy_grid_shape
        self.obstacle_dim = config.lidar_sector_dim
        self.pose_goal_dim = config.pose_goal_dim
        self.imu_dim = config.imu_feature_dim
        self.structured_obs_dim = self.obstacle_dim + self.pose_goal_dim + self.imu_dim
        if obs_size < self.structured_obs_dim:
            raise ValueError(
                f"Observation size {obs_size} is smaller than the structured feature layout "
                f"({self.structured_obs_dim})."
            )

        branch_latent_dim = max(config.latent_size // 2, 32)
        self.obstacle_encoder = nn.Sequential(
            nn.Linear(self.obstacle_dim, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, branch_latent_dim),
            nn.ReLU(),
        )
        self.pose_goal_encoder = nn.Sequential(
            nn.Linear(self.pose_goal_dim, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, branch_latent_dim),
            nn.ReLU(),
        )
        self.imu_encoder = nn.Sequential(
            nn.Linear(self.imu_dim, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, branch_latent_dim),
            nn.ReLU(),
        )

        grid_latent_dim = config.latent_size
        self.grid_encoder_cnn: Optional[nn.Module]
        self.grid_encoder_mlp: Optional[nn.Module]
        self.grid_feature_dim = max(obs_size - self.structured_obs_dim, 0)
        if self.grid_shape is not None:
            if len(self.grid_shape) == 2:
                in_channels = 1
            elif len(self.grid_shape) == 3:
                in_channels = self.grid_shape[0]
            else:
                raise ValueError(
                    "occupancy_grid_shape must be (H, W) or (C, H, W) when using CNN encoding."
                )
            self.grid_encoder_cnn = nn.Sequential(
                nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(32, grid_latent_dim),
                nn.ReLU(),
            )
            self.grid_encoder_mlp = None
        elif self.grid_feature_dim > 0:
            self.grid_encoder_cnn = None
            self.grid_encoder_mlp = nn.Sequential(
                nn.Linear(self.grid_feature_dim, config.hidden_size),
                nn.ReLU(),
                nn.Linear(config.hidden_size, grid_latent_dim),
                nn.ReLU(),
            )
        else:
            self.grid_encoder_cnn = None
            self.grid_encoder_mlp = None
            grid_latent_dim = 0

        fusion_input_dim = 3 * branch_latent_dim + grid_latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(fusion_input_dim, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.latent_size),
            nn.ReLU(),
        )
        self.recurrent_hidden_size = config.lstm_hidden_size
        self.recurrent_layers = config.lstm_layers
        self.gru = nn.GRU(
            input_size=config.latent_size,
            hidden_size=self.recurrent_hidden_size,
            num_layers=self.recurrent_layers,
            batch_first=True,
        )
        self.policy_head = nn.Linear(self.recurrent_hidden_size, action_dim)
        self.value_head = nn.Linear(self.recurrent_hidden_size, 1)

    def get_initial_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
    ) -> RecurrentState:
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros((self.recurrent_layers, batch_size, self.recurrent_hidden_size), device=device)

    def _state_batch_size(self, recurrent_state: Optional[RecurrentState]) -> Optional[int]:
        if recurrent_state is None:
            return None
        return recurrent_state.shape[1]

    def _infer_batch_time(
        self,
        tensor: torch.Tensor,
        recurrent_state: Optional[RecurrentState],
        done_mask: Optional[Union[np.ndarray, torch.Tensor]],
        feature_rank: int = 1,
    ) -> Tuple[int, int]:
        leading_dims = tensor.ndim - feature_rank
        if leading_dims <= 0:
            return 1, 1
        if leading_dims == 1:
            state_batch_size = self._state_batch_size(recurrent_state)
            if state_batch_size is not None and state_batch_size == tensor.shape[0]:
                return tensor.shape[0], 1
            if (
                recurrent_state is not None
                and done_mask is not None
                and done_mask.ndim == 1
                and done_mask.shape[0] == tensor.shape[0]
            ):
                return 1, tensor.shape[0]
            return tensor.shape[0], 1
        return tensor.shape[0], tensor.shape[1]

    def _prepare_tensor_component(
        self,
        component: Union[np.ndarray, torch.Tensor],
        recurrent_state: Optional[RecurrentState],
        done_mask: Optional[Union[np.ndarray, torch.Tensor]],
        feature_rank: int,
    ) -> torch.Tensor:
        tensor = torch.as_tensor(component, dtype=torch.float32, device=next(self.parameters()).device)
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)
        if tensor.ndim < feature_rank:
            raise ValueError(
                f"Expected at least {feature_rank} trailing dimensions, got shape {tuple(tensor.shape)}"
            )
        batch_size, seq_len = self._infer_batch_time(
            tensor,
            recurrent_state,
            done_mask,
            feature_rank=feature_rank,
        )
        return tensor.reshape(batch_size, seq_len, *tensor.shape[-feature_rank:])  # component -> [B,T,features...].

    def _grid_feature_rank(self, grid_tensor: torch.Tensor) -> int:
        if self.grid_shape is None:
            return 1
        spatial_rank = len(self.grid_shape)
        if grid_tensor.ndim >= spatial_rank and tuple(grid_tensor.shape[-spatial_rank:]) == self.grid_shape:
            return spatial_rank
        expected_flat_dim = int(np.prod(self.grid_shape))
        if grid_tensor.shape[-1] == expected_flat_dim:
            return 1
        raise ValueError(
            f"Grid observation has shape {tuple(grid_tensor.shape)} but expected trailing shape {self.grid_shape} "
            f"or a flat vector of length {expected_flat_dim}."
        )

    def _reshape_grid_for_cnn(self, grid: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        if self.grid_shape is None:
            raise RuntimeError("occupancy_grid_shape must be configured before CNN grid encoding can be used.")
        expected_flat_dim = int(np.prod(self.grid_shape))
        grid_flat = grid.reshape(batch_size * seq_len, -1)  # collapse batch/time before restoring the grid shape.
        if grid_flat.shape[-1] != expected_flat_dim:
            raise ValueError(
                f"Grid feature size {grid_flat.shape[-1]} does not match occupancy_grid_shape {self.grid_shape} "
                f"(expected {expected_flat_dim})."
            )
        grid_cnn = grid_flat.reshape(batch_size * seq_len, *self.grid_shape)
        if len(self.grid_shape) == 2:
            grid_cnn = grid_cnn.unsqueeze(1)  # [B*T,H,W] -> [B*T,1,H,W] for Conv2d.
        return grid_cnn

    def _split_observation(
        self,
        observation: Union[np.ndarray, torch.Tensor, Dict[str, Union[np.ndarray, torch.Tensor]]],
        recurrent_state: Optional[RecurrentState],
        done_mask: Optional[Union[np.ndarray, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], int, int]:
        if isinstance(observation, dict):
            obstacle_input = observation.get("obstacle_features", observation.get("lidar_sectors"))
            pose_goal_input = observation.get("pose_goal_features", observation.get("pose_goal"))
            imu_input = observation.get("imu_features", observation.get("imu"))
            if obstacle_input is None or pose_goal_input is None or imu_input is None:
                raise KeyError("Observation dict must contain obstacle/lidar, pose_goal, and imu feature groups.")
            grid_input = observation.get("grid", observation.get("occupancy_grid", observation.get("feature_map")))
            obstacle = self._prepare_tensor_component(
                obstacle_input, recurrent_state, done_mask, feature_rank=1
            )
            pose_goal = self._prepare_tensor_component(
                pose_goal_input, recurrent_state, done_mask, feature_rank=1
            )
            imu = self._prepare_tensor_component(
                imu_input, recurrent_state, done_mask, feature_rank=1
            )
            batch_size, seq_len = obstacle.shape[:2]
            grid = None
            if grid_input is not None:
                grid_tensor = torch.as_tensor(grid_input, dtype=torch.float32, device=obstacle.device)
                grid_tensor = torch.nan_to_num(grid_tensor, nan=0.0, posinf=1.0, neginf=-1.0)
                grid_feature_rank = self._grid_feature_rank(grid_tensor)
                grid = self._prepare_tensor_component(
                    grid_tensor,
                    recurrent_state,
                    done_mask,
                    feature_rank=grid_feature_rank,
                )
            return obstacle, pose_goal, imu, grid, batch_size, seq_len

        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=next(self.parameters()).device)
        obs_tensor = torch.nan_to_num(obs_tensor, nan=0.0, posinf=1.0, neginf=-1.0)
        batch_size, seq_len = self._infer_batch_time(obs_tensor, recurrent_state, done_mask)
        obs_tensor = obs_tensor.reshape(batch_size, seq_len, -1)  # flat env vector -> [B,T,F] before feature slicing.
        obstacle_end = self.obstacle_dim
        pose_goal_end = obstacle_end + self.pose_goal_dim
        imu_end = pose_goal_end + self.imu_dim
        obstacle = obs_tensor[..., :obstacle_end]
        pose_goal = obs_tensor[..., obstacle_end:pose_goal_end]
        imu = obs_tensor[..., pose_goal_end:imu_end]
        grid = obs_tensor[..., imu_end:] if self.grid_feature_dim > 0 else None
        return obstacle, pose_goal, imu, grid, batch_size, seq_len

    def _encode_observation(
        self,
        observation: Union[np.ndarray, torch.Tensor, Dict[str, Union[np.ndarray, torch.Tensor]]],
        recurrent_state: Optional[RecurrentState],
        done_mask: Optional[Union[np.ndarray, torch.Tensor]],
    ) -> Tuple[torch.Tensor, int, int]:
        obstacle, pose_goal, imu, grid, batch_size, seq_len = self._split_observation(
            observation,
            recurrent_state,
            done_mask,
        )
        obstacle_latent = self.obstacle_encoder(obstacle.reshape(batch_size * seq_len, -1))  # [B,T,D] -> [B*T,D].
        pose_goal_latent = self.pose_goal_encoder(pose_goal.reshape(batch_size * seq_len, -1))  # [B,T,D] -> [B*T,D].
        imu_latent = self.imu_encoder(imu.reshape(batch_size * seq_len, -1))  # [B,T,D] -> [B*T,D].
        encoded_parts = [obstacle_latent, pose_goal_latent, imu_latent]

        if grid is not None:
            if self.grid_encoder_cnn is not None:
                grid_latent = self.grid_encoder_cnn(self._reshape_grid_for_cnn(grid, batch_size, seq_len))
            elif self.grid_encoder_mlp is not None:
                grid_latent = self.grid_encoder_mlp(grid.reshape(batch_size * seq_len, -1))  # [B,T,G] -> [B*T,G].
            else:
                raise ValueError("Grid observations were provided but no grid encoder is configured.")
            encoded_parts.append(grid_latent)

        latent = torch.cat(encoded_parts, dim=-1)
        latent = self.encoder(latent).reshape(batch_size, seq_len, -1)  # restore recurrent layout [B,T,L].
        return latent, batch_size, seq_len

    def _prepare_done_mask(
        self,
        done_mask: Optional[Union[np.ndarray, torch.Tensor]],
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if done_mask is None:
            return None
        mask = torch.as_tensor(done_mask, dtype=torch.float32, device=device)
        if mask.ndim == 0:
            mask = mask.view(1, 1).expand(batch_size, seq_len)  # scalar reset flag -> [B,T].
        elif mask.ndim == 1:
            mask = mask.view(batch_size, seq_len)  # rollout reset vector -> [B,T].
        elif mask.ndim != 2:
            raise ValueError(f"done_mask must be scalar, [T], or [B, T], got shape {tuple(mask.shape)}")
        return mask

    def forward(
        self,
        observation: Union[np.ndarray, torch.Tensor, Dict[str, Union[np.ndarray, torch.Tensor]]],
        recurrent_state: Optional[RecurrentState] = None,
        done_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, RecurrentState]:
        latent, batch_size, seq_len = self._encode_observation(observation, recurrent_state, done_mask)
        device = latent.device
        if recurrent_state is None:
            recurrent_state = self.get_initial_state(batch_size, device=device)
        mask = self._prepare_done_mask(done_mask, batch_size, seq_len, device)

        h_t = recurrent_state
        if mask is None or not bool((mask[:, 1:] > 0).any().item()):
            if mask is not None:
                keep = (1.0 - mask[:, 0]).view(1, batch_size, 1)  # [B] -> GRU state gate [L,B,H].
                h_t = h_t * keep
            recurrent_features, h_t = self.gru(latent, h_t)
        else:
            outputs: List[torch.Tensor] = []
            for t in range(seq_len):
                if mask is not None:
                    keep = (1.0 - mask[:, t]).view(1, batch_size, 1)  # [B] -> GRU state gate [L,B,H].
                    h_t = h_t * keep
                step_output, h_t = self.gru(latent[:, t : t + 1], h_t)
                outputs.append(step_output)
            recurrent_features = torch.cat(outputs, dim=1)
        policy_output = torch.nan_to_num(self.policy_head(recurrent_features), nan=0.0, posinf=1.0, neginf=-1.0)
        state_value = torch.nan_to_num(
            self.value_head(recurrent_features).squeeze(-1),
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )

        if seq_len == 1:
            policy_output = policy_output[:, 0]
            state_value = state_value[:, 0]
        return policy_output, state_value, h_t
