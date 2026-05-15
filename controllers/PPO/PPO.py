"""PPO training controller for ALTINO robot in Webots obstacle avoidance task."""
import sys, time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn.utils.rnn import pad_sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from controllers.RNN import GRUActorCritic, LSTMActorCritic, RecurrentState
from controllers.Webots.webots_env import WebotsEnv, _init_supervisor
from controllers.common.PPO_rewards import PPORewardComputer
import controllers.common.PPO_defaults as d
from controllers.common.checkpoints import (
    run_checkpoint_dir, run_checkpoint_path, load_checkpoint,
    make_checkpoint_header as _make_checkpoint_header,
    save_checkpoint_file as _save_checkpoint_file,
)
from controllers.common.metrics_logger import MetricsLogger

_CONTROLLER_DIR = Path(__file__).resolve().parent
_CHECKPOINT_DIR = _CONTROLLER_DIR / "checkpoints"


@dataclass
class Config:
    episodes: int = d.PPODefaults.episodes
    update_every: int = d.PPODefaults.update_every
    epochs: int = d.PPODefaults.epochs
    batch_size: int = d.PPODefaults.batch_size
    save_every: int = d.PPODefaults.save_every
    gamma: float = d.PPODefaults.gamma
    gae_lambda: float = d.PPODefaults.gae_lambda
    epsilon: float = d.PPODefaults.epsilon 
    learning_rate: float = d.PPODefaults.learning_rate
    entropy_coef: float = d.PPODefaults.entropy_coef
    clip_value_loss: bool = d.PPODefaults.clip_value_loss
    hidden_size: int = d.PPODefaults.hidden_size
    latent_size: int = d.PPODefaults.latent_size
    lstm_hidden_size: int = d.PPODefaults.lstm_hidden_size
    lstm_layers: int = d.PPODefaults.lstm_layers
    recurrent_cell: str = d.PPODefaults.recurrent_cell
    sequence_length: int = d.RecurrentDefaults.sequence_length
    burn_in: int = d.RecurrentDefaults.burn_in
    sequence_stride: int = d.RecurrentDefaults.sequence_stride
    lidar_sector_dim: int = d.ENV_LIDAR_SECTOR_DIM
    pose_goal_dim: int = d.ENV_POSE_GOAL_DIM
    imu_feature_dim: int = d.ENV_IMU_FEATURE_DIM
    occupancy_grid_shape: Optional[Tuple[int, ...]] = d.ENV_OCCUPANCY_GRID_SHAPE
    max_steps: int = d.ENV_MAX_STEPS
    collision_threshold: float = d.ENV_COLLISION_THRESHOLD
    low_score_threshold: float = d.ENV_LOW_SCORE_THRESHOLD
    collision_penalty: float = d.REW_COLLISION_PENALTY
    progress_reward_scale: float = d.REW_PROGRESS_SCALE
    distance_reward_scale: float = d.REW_DISTANCE_SCALE
    heading_reward_scale: float = d.REW_HEADING_SCALE
    safety_reward_scale: float = d.REW_SAFETY_SCALE
    motion_reward_scale: float = d.REW_MOTION_SCALE
    new_best_distance_bonus: float = d.REW_NEW_BEST_DISTANCE_BONUS
    step_penalty: float = d.REW_STEP_PENALTY
    endpoint: Tuple[float, float] = d.ENV_ENDPOINT
    goal_threshold: float = d.ENV_GOAL_THRESHOLD
    goal_stop_speed_threshold: float = d.ENV_GOAL_STOP_SPEED_THRESHOLD
    goal_success_reward: float = d.REW_GOAL_SUCCESS
    goal_stop_bonus: float = d.REW_GOAL_STOP_BONUS
    goal_hold_reward: float = d.REW_GOAL_HOLD
    goal_speed_penalty: float = d.REW_GOAL_SPEED_PENALTY
    goal_overshoot_penalty: float = d.REW_GOAL_OVERSHOOT_PENALTY
    reference_distance: Optional[float] = None
    enable_slam: bool = d.SLAM_ENABLE
    profile_slam: bool = d.SLAM_PROFILE
    slam_profile_interval: int = d.SLAM_PROFILE_INTERVAL
    save_slam_plots: bool = d.SLAM_SAVE_PLOTS
    force_cpu: bool = d.SLAM_FORCE_CPU
    max_steering_angle: float = d.ENV_MAX_STEERING_ANGLE
    min_speed: float = d.ENV_MIN_SPEED

    randomize_goal: bool = False
    goal_y_range: float = 1.5
    start_position: Optional[List[float]] = None
    start_rotation: Optional[List[float]] = None
    start_position_noise: float = d.ENV_START_POSITION_NOISE
    start_yaw_noise: float = d.ENV_START_YAW_NOISE
    max_speed: float = d.ENV_MAX_SPEED
    reset_settle_steps: int = d.ENV_RESET_SETTLE_STEPS

    def __post_init__(self):
        self.recurrent_cell = self.recurrent_cell.lower().strip()
        assert self.recurrent_cell in {"lstm", "gru"}, f"Unsupported recurrent_cell: {self.recurrent_cell}"
        if self.start_position is None:
            self.start_position = list(d.ENV_START_POSITION)
        if self.start_rotation is None:
            self.start_rotation = list(d.ENV_START_ROTATION)
        if self.reference_distance is None:
            start_xy = np.array(self.start_position[:2], dtype=np.float32)
            endpoint_xy = np.array(self.endpoint, dtype=np.float32)
            self.reference_distance = float(np.linalg.norm(start_xy - endpoint_xy))


def _split_sequences(episodes, seq_len, stride):
    """Split episode trajectories into overlapping sequences for recurrent training."""
    for ep in episodes:
        total = len(ep["returns"])
        for start in range(0, total, stride):
            end = min(start + seq_len, total)
            if end > start:
                yield {k: v[start:end] for k, v in ep.items()}


class PPOAgent:
    def __init__(self, obs_size: int, action_dim: int, config: Config):
        """Initialize PPO agent with observation and action dimensions, build the neural network model."""
        self.config = config
        self.device = self._get_device()
        self.action_dim = action_dim
        self.obs_size = obs_size
        self.action_low = torch.tensor([-config.max_steering_angle, config.min_speed], dtype=torch.float32, device=self.device)
        self.action_high = torch.tensor([config.max_steering_angle, config.max_speed], dtype=torch.float32, device=self.device)
        self.action_center = (self.action_high + self.action_low) / 2.0
        self.action_scale = (self.action_high - self.action_low) / 2.0
        self._build_model(config.recurrent_cell)
        print(f"[PPO] Using recurrent cell: {config.recurrent_cell.upper()}", flush=True)

    def _build_model(self, recurrent_cell: str) -> None:
        """Build or rebuild the actor-critic model with the specified recurrent cell type (GRU or LSTM)."""
        model_class = GRUActorCritic if recurrent_cell.lower().strip() == "gru" else LSTMActorCritic
        self.model = model_class(self.obs_size, self.action_dim, self.config).to(self.device)
        self.actor = self.model.policy_head
        self.actor_log_std = nn.Parameter(torch.full((self.action_dim,), -0.5, dtype=torch.float32, device=self.device))
        params = list(self.model.parameters()) + [self.actor_log_std]
        self.optimizer = torch.optim.Adam(params, lr=self.config.learning_rate)

    def _get_device(self) -> torch.device:
        """Determine whether to use CPU or CUDA GPU for training."""
        if self.config.force_cpu or not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device("cuda")

    def get_initial_state(self, batch_size: int = 1) -> RecurrentState:
        """Get initial hidden state for the recurrent neural network."""
        return self.model.get_initial_state(batch_size, device=self.device)

    def _sample_action(self, policy_output, deterministic=False):
        """Sample action from normal distribution with tanh squashing and proper log probability computation."""
        mean = policy_output
        std = self.actor_log_std.expand_as(policy_output).exp().clamp_min(1e-3)
        dist = Normal(mean, std)
        pre_tanh = mean if deterministic else dist.rsample()
        action_tanh = torch.tanh(pre_tanh)
        action = action_tanh * self.action_scale + self.action_center
        eps = 1e-5
        action = torch.clamp(action, self.action_low + eps, self.action_high - eps)
        log_prob = dist.log_prob(pre_tanh)
        log_prob -= torch.log(self.action_scale + 1e-6)
        log_prob -= torch.log(1.0 - action_tanh.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1)

    def select_action(self, obs, recurrent_state=None, done=False, deterministic=False):
        """Select an action given observation and recurrent state, returning action, log prob, value, and next state."""
        done_mask = torch.tensor([float(done)], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            policy_output, state_value, next_state = self.model(
                obs, recurrent_state=recurrent_state, done_mask=done_mask,
            )
            action, log_prob = self._sample_action(policy_output, deterministic=deterministic)
        return (
            action.squeeze(0).cpu().numpy(),
            log_prob.squeeze(0),
            state_value.squeeze(0),
            next_state,
        )

    def calculate_gae(self, rewards, values, bootstrap_value=0.0):
        """Calculate Generalized Advantage Estimation (GAE) and returns from rewards and values."""
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        next_value = float(bootstrap_value)
        for t in reversed(range(T)):
            delta = rewards[t] + self.config.gamma * next_value - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * gae
            advantages[t] = gae
            next_value = float(values[t])
        return advantages.astype(np.float32), (advantages + values).astype(np.float32)

    def _prepare_batch(self, trajectories):
        """Pad trajectories of variable length into a batch tensor with valid masks."""
        pad_keys = ["observations", "actions", "log_probs", "returns", "advantages"]
        result = {}
        for key in pad_keys:
            result[key] = pad_sequence(
                [torch.as_tensor(t[key], dtype=torch.float32, device=self.device) for t in trajectories],
                batch_first=True,
            )
        valid_masks = [torch.ones(len(t["returns"]), dtype=torch.float32, device=self.device) for t in trajectories]
        result["valid_mask"] = pad_sequence(valid_masks, batch_first=True)
        reset_masks = []
        for t in trajectories:
            mask = torch.zeros(len(t["returns"]), dtype=torch.float32, device=self.device)
            if len(mask) > 0:
                mask[0] = 1.0
            reset_masks.append(mask)
        result["done_mask"] = pad_sequence(reset_masks, batch_first=True)
        return result

    def _sanitize_trajectories(self, trajectories):
        """Clean trajectories by removing NaNs and clamping actions to valid ranges."""
        low = np.array([-self.config.max_steering_angle, self.config.min_speed], dtype=np.float32)
        high = np.array([self.config.max_steering_angle, self.config.max_speed], dtype=np.float32)
        for t in trajectories:
            t["observations"] = np.nan_to_num(t["observations"], nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)
            t["actions"] = np.clip(np.nan_to_num(t["actions"], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32), low + 1e-5, high - 1e-5)
            t["log_probs"] = np.nan_to_num(t["log_probs"], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            t["returns"] = np.nan_to_num(t["returns"], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            t["advantages"] = np.nan_to_num(t["advantages"], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def _normalize_advantages(self, trajectories):
        """Normalize advantages across all trajectories using mean and standard deviation."""
        all_adv = np.concatenate([t["advantages"] for t in trajectories], axis=0)
        adv_mean = float(all_adv.mean())
        adv_std = float(all_adv.std() + 1e-8)
        for t in trajectories:
            t["advantages"] = ((t["advantages"] - adv_mean) / adv_std).astype(np.float32)
            # Clip normalized advantages to prevent extreme ratio outliers
            t["advantages"] = np.clip(t["advantages"], -5.0, 5.0)

    @staticmethod
    def _sequence_loss_mask(valid_mask, burn_in):
        """Create learning mask that excludes burn-in steps from gradient computation."""
        valid_lengths = valid_mask.sum(dim=1).to(dtype=torch.long)
        start_index = torch.minimum(torch.full_like(valid_lengths, burn_in), torch.clamp(valid_lengths - 1, min=0))
        return valid_mask * (torch.arange(valid_mask.shape[1], device=valid_mask.device).unsqueeze(0) >= start_index.unsqueeze(1)).to(dtype=valid_mask.dtype)

    def _update_batch(self, batch):
        """Perform one PPO gradient update on a batch, returning a dict with loss components and gradient statistics."""
        log_probs_new, values, entropy = self.evaluate_sequences(
            batch["observations"], batch["actions"], batch["done_mask"],
        )
        if not (torch.isfinite(log_probs_new).all() and torch.isfinite(values).all() and torch.isfinite(entropy).all()):
            return None
        valid_mask = batch["valid_mask"]
        learn_mask = self._sequence_loss_mask(valid_mask, self.config.burn_in)
        mask_bool = learn_mask > 0
        log_ratio = torch.nan_to_num(log_probs_new - batch["log_probs"], nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
        ratio = torch.exp(log_ratio)
        surr1 = ratio * batch["advantages"]
        surr2 = torch.clamp(ratio, 1 - self.config.epsilon, 1 + self.config.epsilon) * batch["advantages"]
        surrogate = torch.where(mask_bool, torch.min(surr1, surr2), torch.zeros_like(surr1))
        # Value loss: simple smooth_l1_loss without clipping (clip_value_loss disabled to allow critic gradient flow)
        value_error = nn.functional.smooth_l1_loss(values, batch["returns"], reduction="none")
        entropy_term = torch.where(mask_bool, entropy, torch.zeros_like(entropy))
        valid_count = learn_mask.sum().clamp_min(1.0)
        loss = (-surrogate.sum() + 0.5 * torch.where(mask_bool, value_error, torch.zeros_like(value_error)).sum() - self.config.entropy_coef * entropy_term.sum()) / valid_count
        if not torch.isfinite(loss):
            return None

        with torch.no_grad():
            actor_loss_val = float(surrogate.sum() / valid_count)
            critic_loss_val = float(value_error.sum() / valid_count)
            entropy_val = float(entropy_term.sum() / valid_count)
            value_residual_val = float(torch.abs(values - batch["returns"])[mask_bool].mean().item())
            approx_kl = float((log_probs_new - batch["log_probs"])[mask_bool].mean().item())

        self.optimizer.zero_grad()
        loss.backward()
        for p in list(self.model.parameters()) + [self.actor_log_std]:
            if p.grad is not None and not torch.isfinite(p.grad).all():
                self.optimizer.zero_grad()
                return None

        actor_params = [self.model.policy_head.weight, self.model.policy_head.bias, self.actor_log_std]
        critic_params = [self.model.value_head.weight, self.model.value_head.bias]
        all_params = list(self.model.parameters()) + [self.actor_log_std]

        grad_norm_actor = MetricsLogger.compute_grad_norm(actor_params)
        grad_norm_critic = MetricsLogger.compute_grad_norm(critic_params)

        # Per-module gradient clipping for stability:
        # - Actor params get conservative clipping (max_norm=0.5)
        # - Critic and RNN params get looser clipping (max_norm=1.0)
        actor_clip = list(self.model.policy_head.parameters()) + [self.actor_log_std]
        critic_clip = list(self.model.value_head.parameters())
        rnn_attr = "gru" if hasattr(self.model, "gru") else "lstm"
        rnn_clip = list(getattr(self.model, rnn_attr).parameters())
        encoder_clip = [p for n, p in self.model.named_parameters()
                        if "policy_head" not in n and "value_head" not in n and rnn_attr not in n]
        nn.utils.clip_grad_norm_(actor_clip, max_norm=0.5)
        nn.utils.clip_grad_norm_(critic_clip, max_norm=5.0)  # raised from 1.0 â€” critic has only 2 params, needs more room
        nn.utils.clip_grad_norm_(rnn_clip, max_norm=1.0)
        nn.utils.clip_grad_norm_(encoder_clip, max_norm=0.5)
        self.optimizer.step()
        with torch.no_grad():
            self.actor_log_std.data.copy_(torch.nan_to_num(self.actor_log_std.data, nan=-0.5, posinf=2.0, neginf=-5.0).clamp(-5.0, 2.0))

        lr = float(self.optimizer.param_groups[0]["lr"])
        return {
            "actor_loss": round(actor_loss_val, 6),
            "critic_loss": round(critic_loss_val, 6),
            "policy_entropy": round(entropy_val, 6),
            "entropy_coef": round(self.config.entropy_coef, 6),
            "value_residual": round(value_residual_val, 6),
            "approx_kl": round(approx_kl, 6),
            "grad_norm_actor": round(grad_norm_actor, 6),
            "grad_norm_critic": round(grad_norm_critic, 6),
            "lr_actor": lr,
        }

    def evaluate_sequences(self, observations, actions, done_mask):
        """Evaluate log probabilities, state values, and entropy for given observations and actions (batch)."""
        policy_output, state_values, _ = self.model(
            observations, recurrent_state=self.get_initial_state(observations.shape[0]), done_mask=done_mask,
        )
        mean = policy_output
        std = self.actor_log_std.expand_as(policy_output).exp().clamp_min(1e-3)
        dist = Normal(mean, std)
        eps = 1e-6
        safe_action = torch.clamp(actions, self.action_low + 1e-5, self.action_high - 1e-5)
        squashed = ((safe_action - self.action_center) / (self.action_scale + eps)).clamp(-1.0 + eps, 1.0 - eps)
        pre_tanh = 0.5 * (torch.log1p(squashed) - torch.log1p(-squashed))
        action_tanh = torch.tanh(pre_tanh)
        log_prob = dist.log_prob(pre_tanh)
        log_prob -= torch.log(self.action_scale + 1e-6)
        log_prob -= torch.log(1.0 - action_tanh.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        entropy_pre_tanh = dist.rsample()
        entropy_action_tanh = torch.tanh(entropy_pre_tanh)
        entropy = dist.log_prob(entropy_pre_tanh)
        entropy -= torch.log(self.action_scale + 1e-6)
        entropy -= torch.log(1.0 - entropy_action_tanh.pow(2) + 1e-6)
        entropy = -entropy.sum(dim=-1)
        return log_prob, state_values, entropy

    def update(self, trajectories):
        """Perform multiple PPO epochs of training on collected trajectories, returning per-update metrics."""
        if not trajectories:
            return []
        self._sanitize_trajectories(trajectories)
        self._normalize_advantages(trajectories)
        trajectories = list(_split_sequences(trajectories, self.config.sequence_length, self.config.sequence_stride))
        if not trajectories:
            return []
        update_metrics = []
        num = len(trajectories)
        early_stop = False
        for epoch in range(self.config.epochs):
            if early_stop:
                break
            indices = torch.randperm(num).tolist()
            for start in range(0, num, self.config.batch_size):
                batch_indices = indices[start: start + self.config.batch_size]
                batch = self._prepare_batch([trajectories[i] for i in batch_indices])
                metrics = self._update_batch(batch)
                if metrics is not None:
                    update_metrics.append(metrics)
                    # Early stopping if KL divergence is too large
                    if metrics.get("approx_kl", 0) > 0.05:
                        early_stop = True
                        break
        return update_metrics

    def load_model(self, model_path: str) -> None:
        """Load a pre-trained PPO model checkpoint, adjusting recurrent cell type if needed."""
        checkpoint = load_checkpoint(model_path, map_location=self.device)
        algo = str(checkpoint.get("algorithm", "ppo")).lower().strip()
        assert algo == "ppo", f"Checkpoint algorithm '{algo}' does not match PPO."
        assert "model" in checkpoint, "Checkpoint does not contain recurrent 'model' weights."
        for key in ("obs_size", "action_dim"):
            saved = checkpoint.get(key)
            if saved is not None and int(saved) != getattr(self, key):
                raise ValueError(f"Checkpoint {key}={saved} != current {getattr(self, key)}")
        cell = str(checkpoint.get("recurrent_cell", self.config.recurrent_cell)).lower().strip()
        assert cell in {"lstm", "gru"}, f"Unsupported recurrent_cell in checkpoint: {cell}"
        if cell != self.config.recurrent_cell:
            self.config.recurrent_cell = cell
            self._build_model(cell)
        print(f"[PPO] Loaded recurrent cell: {cell.upper()}", flush=True)
        self.model.load_state_dict(checkpoint["model"])
        if "actor_log_std" in checkpoint:
            self.actor_log_std.data.copy_(checkpoint["actor_log_std"].to(self.device))


def _save_checkpoint(agent, episode, reward, is_goal, prefix, run_id):
    """Save PPO agent checkpoint with model weights, config, and training metadata."""
    header = _make_checkpoint_header(episode, reward, is_goal, "ppo", asdict(agent.config))
    header["obs_size"] = agent.obs_size
    header["action_dim"] = agent.action_dim
    header["recurrent_cell"] = agent.config.recurrent_cell
    header["model"] = agent.model.state_dict()
    header["actor_log_std"] = agent.actor_log_std.detach().cpu()
    _save_checkpoint_file(_CHECKPOINT_DIR, run_id, prefix, header)


def train(config=None):
    """Main training loop: collect episodes, perform PPO updates, log metrics, and save checkpoints."""
    if config is None:
        config = Config()
    _init_supervisor()
    reward_computer = PPORewardComputer(
        endpoint=config.endpoint,
        collision_penalty=config.collision_penalty,
        progress_reward_scale=config.progress_reward_scale,
        distance_reward_scale=config.distance_reward_scale,
        heading_reward_scale=config.heading_reward_scale,
        safety_reward_scale=config.safety_reward_scale,
        motion_reward_scale=config.motion_reward_scale,
        new_best_distance_bonus=config.new_best_distance_bonus,
        proximity_radius=getattr(config, "proximity_radius", d.REW_PROXIMITY_RADIUS),
        proximity_reward_scale=getattr(config, "proximity_reward_scale", d.REW_PROXIMITY_SCALE),
        step_penalty=config.step_penalty,
        goal_threshold=config.goal_threshold,
        goal_stop_speed_threshold=config.goal_stop_speed_threshold,
        goal_success_reward=config.goal_success_reward,
        goal_stop_bonus=config.goal_stop_bonus,
        goal_hold_reward=config.goal_hold_reward,
        goal_speed_penalty=config.goal_speed_penalty,
        goal_overshoot_penalty=config.goal_overshoot_penalty,
    )
    env = WebotsEnv(config, reward_computer)
    env.reset()
    run_id = Path(env.run_folder).name
    obs_size = env.observation_size
    action_dim = env.action_dim
    agent = PPOAgent(obs_size, action_dim, config)
    checkpoint_dir = run_checkpoint_dir(_CHECKPOINT_DIR, run_id)
    final_model_path = run_checkpoint_path(_CHECKPOINT_DIR, run_id, "final")
    print(f"[TRAIN][PPO] rnn={config.recurrent_cell.upper()} weights_dir={checkpoint_dir} final={final_model_path}", flush=True)
    print(f"[TRAIN][PPO] episodes={config.episodes} update_every={config.update_every} obs={obs_size} act={action_dim} cell={config.recurrent_cell.upper()} seq={config.sequence_length} burn_in={config.burn_in}", flush=True)

    rollout = []
    best_reward = float("-inf")
    best_goal_reward = float("-inf")
    best_goal_episode = None
    rew_w, suc_w, gol_w, col_w, to_w = [], [], [], [], []
    total_steps = 0
    start_time = time.perf_counter()
    metrics_logger = MetricsLogger(env.run_folder, algorithm="ppo")
    metrics_logger.log_hyperparams(asdict(config), recurrent_cell=config.recurrent_cell,
                                   obs_size=obs_size, action_dim=action_dim)

    for episode in range(config.episodes):
        obs, _ = env.reset()
        done = False
        ep_step = 0
        ep_obs, ep_act, ep_lp, ep_rew = [], [], [], []
        ep_speeds = []
        ep_end_reason = "max_steps"
        ep_goal = ep_success = False
        recurrent_state = agent.get_initial_state(batch_size=1)
        prev_done = True

        while not done:
            action, log_prob, _, recurrent_state = agent.select_action(obs, recurrent_state, done=prev_done)
            obs_next, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            prev_done = done
            ep_step += 1
            total_steps += 1
            ep_goal = ep_goal or bool(info.get("goal_reached", False))
            ep_success = bool(info.get("success", False))
            ep_speeds.append(float(info.get("speed_norm", 0.0)))
            if done:
                reason = info.get("reset_reason", "")
                ep_end_reason = reason if reason else ("max_steps" if truncated else ep_end_reason)
            ep_obs.append(obs)
            ep_act.append(action)
            ep_lp.append(float(log_prob.item()))
            ep_rew.append(reward)
            obs = obs_next

        ep_obs_arr = np.array(ep_obs, dtype=np.float32)
        with torch.no_grad():
            _, ep_values, _ = agent.model(
                ep_obs_arr, recurrent_state=None,
                done_mask=np.concatenate(([1.0], np.zeros(len(ep_rew) - 1, dtype=np.float32))),
            )
            ep_val_np = ep_values.detach().cpu().numpy().reshape(-1)

        bootstrap_value = 0.0
        if ep_end_reason == "max_steps":
            with torch.no_grad():
                _, bs_val, _ = agent.model(
                    np.asarray(obs, dtype=np.float32), recurrent_state=recurrent_state,
                    done_mask=np.array([0.0], dtype=np.float32),
                )
                bootstrap_value = float(bs_val.squeeze(0).item())

        scaled_rew = np.array(ep_rew, dtype=np.float32) * d.REW_SCALE
        ep_adv, ep_ret = agent.calculate_gae(
            scaled_rew, ep_val_np, bootstrap_value=bootstrap_value,
        )
        rollout.append({"observations": ep_obs_arr, "actions": np.array(ep_act, dtype=np.float32),
                        "log_probs": np.array(ep_lp, dtype=np.float32), "returns": ep_ret, "advantages": ep_adv})

        act_stats = MetricsLogger.compute_action_stats(ep_act)
        obs_stats = MetricsLogger.compute_obs_stats(ep_obs)
        ep_val_residual = MetricsLogger.compute_value_residual(ep_val_np, ep_ret)

        # Learning rate warmup: ramp from 25% to 100% over first 25 episodes.
        # Avoids freezing the model when exploration is strongest.
        if episode < 25:
            warmup_lr = config.learning_rate * (0.25 + 0.75 * (episode + 1) / 25.0)
            for pg in agent.optimizer.param_groups:
                pg['lr'] = warmup_lr

        all_update_metrics = []
        if (episode + 1) % config.update_every == 0:
            batch_metrics = agent.update(rollout)
            rollout.clear()
            for upd in batch_metrics:
                all_update_metrics.append(upd)
                metrics_logger.log_update(
                    global_step=total_steps, episode=episode + 1,
                    recurrent_cell=config.recurrent_cell,
                    **upd,
                )

        agg_upd = MetricsLogger.aggregate_update_metrics(all_update_metrics)

        # Entropy coefficient decays linearly over training, maintaining some
        # exploration throughout to prevent premature convergence to poor policies.
        decay_frac = min(1.0, episode / max(1, config.episodes))
        agent.config.entropy_coef = d.PPODefaults.entropy_coef * (1.0 - 0.15 * decay_frac)

        ep_sum = sum(ep_rew)
        rew_w.append(ep_sum)
        suc_w.append(1.0 if ep_success else 0.0)
        gol_w.append(1.0 if ep_goal else 0.0)
        col_w.append(1.0 if ep_end_reason == "collision" else 0.0)
        to_w.append(1.0 if ep_end_reason == "max_steps" else 0.0)
        ckpt_flags = []

        if ep_end_reason == "goal" and ep_sum > best_goal_reward:
            best_goal_reward = ep_sum
            best_goal_episode = episode + 1
            env.robot.slam.save_episode(env.run_folder, episode + 1, ep_sum)
            _save_checkpoint(agent, best_goal_episode, best_goal_reward, True, "best", run_id)
            ckpt_flags.append("best_goal")
        elif best_goal_episode is None and ep_sum > best_reward:
            best_reward = ep_sum
            env.robot.slam.save_episode(env.run_folder, episode + 1, ep_sum)
            _save_checkpoint(agent, episode + 1, best_reward, False, "best", run_id)
            ckpt_flags.append("best")

        if config.save_every > 0 and (episode + 1) % config.save_every == 0:
            _save_checkpoint(agent, episode + 1, ep_sum, ep_end_reason == "goal", "checkpoint", run_id)
            ckpt_flags.append("latest")

        r10 = float(np.mean(rew_w[-10:]))
        s10 = float(np.mean(suc_w[-10:]))
        g10 = float(np.mean(gol_w[-10:]))
        c10 = float(np.mean(col_w[-10:]))
        t10 = float(np.mean(to_w[-10:]))
        avg_spd = float(np.mean(ep_speeds)) * config.max_speed if ep_speeds else 0.0
        elapsed = time.perf_counter() - start_time
        ckpt_note = f" ckpt={'+'.join(ckpt_flags)}" if ckpt_flags else ""
        print(f"[TRAIN][PPO] ep={episode + 1:03d}/{config.episodes} r={ep_sum:8.2f} avg10={r10:8.2f} steps={ep_step:4d} succ10={s10:4.2f} touch10={g10:4.2f} col10={c10:4.2f} to10={t10:4.2f} min_d={env.min_episode_distance:5.2f} avg_spd={avg_spd:4.2f}m/s end={ep_end_reason} t={elapsed:7.1f}s{ckpt_note}", flush=True)

        metrics_logger.log_episode(
            episode=episode + 1,
            global_step=total_steps,
            reward=round(ep_sum, 4),
            avg10=round(r10, 4),
            length=ep_step,
            success=int(ep_success),
            goal_touched=int(ep_goal),
            collision=int(ep_end_reason == "collision"),
            timeout=int(ep_end_reason == "max_steps"),
            min_dist=round(env.min_episode_distance, 4),
            avg_speed_ms=round(avg_spd, 3),
            end_reason=ep_end_reason,
            elapsed_s=round(elapsed, 1),
            recurrent_cell=config.recurrent_cell,
            replay_buffer_size=0,
            **act_stats,
            **obs_stats,
            **agg_upd,
        )

    if rollout:
        agent.update(rollout)
    metrics_logger.close()
    print(f"[TRAIN][PPO] metrics saved to {metrics_logger.path}", flush=True)
    print(f"[TRAIN][PPO] updates saved to {metrics_logger.update_path}", flush=True)
    print(f"[TRAIN][PPO] hyperparams saved to {metrics_logger.hyperparams_path}", flush=True)
    final_reward = best_goal_reward if best_goal_episode is not None else best_reward
    _save_checkpoint(agent, "final", final_reward, best_goal_episode is not None, "final", run_id)
    elapsed = time.perf_counter() - start_time
    print(f"[TRAIN][PPO] final reward={final_reward:.2f} t={elapsed:7.1f}s", flush=True)
    env.robot.stop()
    print("[TRAIN][PPO] done", flush=True)


if __name__ == "__main__":
    train()


