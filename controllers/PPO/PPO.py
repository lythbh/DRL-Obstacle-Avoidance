"""PPO training controller for ALTINO robot in Webots obstacle avoidance task."""
import sys, time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any, Union
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn.utils.rnn import pad_sequence

RecurrentState = Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from controllers.RNN import GRUActorCritic, LSTMActorCritic
from controllers.Webots.webots_env import WebotsEnv, _init_supervisor
from controllers.common.reward_defaults import (
    COLLISION_THRESHOLD, COLLISION_PENALTY, DISTANCE_REWARD_SCALE,
    ENABLE_SLAM, ENDPOINT, FORCE_CPU, GOAL_HOLD_REWARD,
    GOAL_STOP_SPEED_THRESHOLD, GOAL_OVERSHOOT_PENALTY, GOAL_SPEED_PENALTY,
    GOAL_STOP_BONUS, GOAL_THRESHOLD, GOAL_SUCCESS_REWARD,
    HEADING_REWARD_SCALE, IMU_FEATURE_DIM, LIDAR_SECTOR_DIM,
    LOW_SCORE_THRESHOLD, MAX_SPEED, MAX_STEERING_ANGLE, MAX_STEPS,
    MIN_SPEED, MOTION_REWARD_SCALE, NEW_BEST_DISTANCE_BONUS,
    OCCUPANCY_GRID_SHAPE, POSE_GOAL_DIM, PROGRESS_REWARD_SCALE,
    PROFILE_SLAM, RESET_SETTLE_STEPS, REWARD_SCALE, SAFETY_REWARD_SCALE,
    SAVE_SLAM_PLOTS, SLAM_PROFILE_INTERVAL, START_POSITION,
    START_POSITION_NOISE, START_ROTATION, START_YAW_NOISE, STEP_PENALTY,
)
from controllers.common.training_defaults import PPODefaults, RecurrentDefaults

_CONTROLLER_DIR = Path(__file__).resolve().parent
_CHECKPOINT_DIR = _CONTROLLER_DIR / "checkpoints"

from controllers.common.checkpoints import (
    run_checkpoint_dir, run_checkpoint_path, load_checkpoint,
    make_checkpoint_header as _make_checkpoint_header,
    save_checkpoint_file as _save_checkpoint_file,
)
from controllers.common.metrics_logger import MetricsLogger



@dataclass
class Config:
    episodes: int = PPODefaults.episodes
    update_every: int = PPODefaults.update_every
    epochs: int = PPODefaults.epochs
    batch_size: int = PPODefaults.batch_size
    save_every: int = PPODefaults.save_every
    gamma: float = 0.99
    gae_lambda: float = PPODefaults.gae_lambda
    epsilon: float = 0.2
    learning_rate: float = PPODefaults.learning_rate
    entropy_coef: float = PPODefaults.entropy_coef
    hidden_size: int = PPODefaults.hidden_size
    latent_size: int = PPODefaults.latent_size
    lstm_hidden_size: int = PPODefaults.lstm_hidden_size
    lstm_layers: int = PPODefaults.lstm_layers
    recurrent_cell: str = PPODefaults.recurrent_cell
    sequence_length: int = RecurrentDefaults.sequence_length
    burn_in: int = RecurrentDefaults.burn_in
    sequence_stride: int = RecurrentDefaults.sequence_stride
    lidar_sector_dim: int = LIDAR_SECTOR_DIM
    pose_goal_dim: int = POSE_GOAL_DIM
    imu_feature_dim: int = IMU_FEATURE_DIM
    occupancy_grid_shape: Optional[Tuple[int, ...]] = OCCUPANCY_GRID_SHAPE
    max_steps: int = MAX_STEPS
    collision_threshold: float = COLLISION_THRESHOLD
    low_score_threshold: float = LOW_SCORE_THRESHOLD
    collision_penalty: float = COLLISION_PENALTY
    progress_reward_scale: float = PROGRESS_REWARD_SCALE
    distance_reward_scale: float = DISTANCE_REWARD_SCALE
    heading_reward_scale: float = HEADING_REWARD_SCALE
    safety_reward_scale: float = SAFETY_REWARD_SCALE
    motion_reward_scale: float = MOTION_REWARD_SCALE
    new_best_distance_bonus: float = NEW_BEST_DISTANCE_BONUS
    step_penalty: float = STEP_PENALTY
    endpoint: Tuple[float, float] = ENDPOINT
    goal_threshold: float = GOAL_THRESHOLD
    goal_stop_speed_threshold: float = GOAL_STOP_SPEED_THRESHOLD
    goal_success_reward: float = GOAL_SUCCESS_REWARD
    goal_stop_bonus: float = GOAL_STOP_BONUS
    goal_hold_reward: float = GOAL_HOLD_REWARD
    goal_speed_penalty: float = GOAL_SPEED_PENALTY
    goal_overshoot_penalty: float = GOAL_OVERSHOOT_PENALTY
    reference_distance: Optional[float] = None
    enable_slam: bool = ENABLE_SLAM
    profile_slam: bool = PROFILE_SLAM
    slam_profile_interval: int = SLAM_PROFILE_INTERVAL
    save_slam_plots: bool = SAVE_SLAM_PLOTS
    force_cpu: bool = FORCE_CPU
    max_steering_angle: float = MAX_STEERING_ANGLE
    min_speed: float = MIN_SPEED
    start_position: Optional[List[float]] = None
    start_rotation: Optional[List[float]] = None
    start_position_noise: float = START_POSITION_NOISE
    start_yaw_noise: float = START_YAW_NOISE
    max_speed: float = MAX_SPEED
    reset_settle_steps: int = RESET_SETTLE_STEPS

    def __post_init__(self):
        self.recurrent_cell = self.recurrent_cell.lower().strip()
        assert self.recurrent_cell in {"lstm", "gru"}, f"Unsupported recurrent_cell: {self.recurrent_cell}"
        if self.start_position is None:
            self.start_position = list(START_POSITION)
        if self.start_rotation is None:
            self.start_rotation = list(START_ROTATION)
        if self.reference_distance is None:
            start_xy = np.array(self.start_position[:2], dtype=np.float32)
            endpoint_xy = np.array(self.endpoint, dtype=np.float32)
            self.reference_distance = float(np.linalg.norm(start_xy - endpoint_xy))


class PPOAgent:
    def __init__(self, obs_size: int, action_dim: int, config: Config):
        """Initialise PPO agent with model, action bounds, and optimizer."""
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
        """Construct recurrent actor-critic network and set up optimiser."""
        model_class = GRUActorCritic if recurrent_cell.lower().strip() == "gru" else LSTMActorCritic
        self.model = model_class(self.obs_size, self.action_dim, self.config).to(self.device)
        self.actor = self.model.policy_head
        with torch.no_grad():
            if self.actor.bias is not None and self.action_dim > 1:
                self.actor.bias[1] = -0.8
        self.actor_log_std = nn.Parameter(torch.full((self.action_dim,), -0.5, dtype=torch.float32, device=self.device))
        params = list(self.model.parameters()) + [self.actor_log_std]
        self.optimizer = torch.optim.Adam(params, lr=self.config.learning_rate)

    def _get_device(self) -> torch.device:
        """Select CUDA if available and not force_cpu, else CPU."""
        if self.config.force_cpu or not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device("cuda")

    def get_initial_state(self, batch_size: int = 1) -> RecurrentState:
        """Return zeroed recurrent state for rollouts."""
        return self.model.get_initial_state(batch_size, device=self.device)

    def _action_stats(self, policy_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract mean and standard deviation from policy output."""
        return policy_output, self.actor_log_std.expand_as(policy_output).exp().clamp_min(1e-3)

    def _tanh_log_prob(self, dist: Normal, pre_tanh: torch.Tensor) -> torch.Tensor:
        """Compute log-probability under a tanh-squashed Normal, per dimension."""
        log_prob = dist.log_prob(pre_tanh)
        log_prob -= torch.log(self.action_scale + 1e-6)
        log_prob -= torch.log(1.0 - torch.tanh(pre_tanh).pow(2) + 1e-6)
        return log_prob

    def _sample(self, mean: torch.Tensor, std: torch.Tensor,
                deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample tanh-squashed action and compute its log-probability."""
        dist = Normal(mean, std)
        pre_tanh = mean if deterministic else dist.rsample()
        action = torch.tanh(pre_tanh) * self.action_scale + self.action_center
        eps = 1e-5
        action = torch.max(torch.min(action, self.action_high - eps), self.action_low + eps)
        return action, self._tanh_log_prob(dist, pre_tanh).sum(dim=-1)

    def select_action(
        self,
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        recurrent_state: Optional[RecurrentState] = None,
        done: bool = False,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor, RecurrentState]:
        """Sample action from policy for environment interaction."""
        done_mask = torch.tensor([float(done)], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            policy_output, state_value, next_state = self.model(
                obs, recurrent_state=recurrent_state, done_mask=done_mask,
            )
            mean, std = self._action_stats(policy_output)
            action, log_prob = self._sample(mean, std, deterministic=deterministic)
        return (
            action.squeeze(0).cpu().numpy(),
            log_prob.squeeze(0),
            state_value.squeeze(0),
            next_state,
        )

    def calculate_gae(
        self, rewards: np.ndarray, values: np.ndarray, bootstrap_value: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantages and TD(lambda) returns."""
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        next_value = float(bootstrap_value)
        for t in reversed(range(T)):
            delta = rewards[t] + self.config.gamma * next_value - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * gae
            advantages[t] = gae
            next_value = float(values[t])
        returns = (advantages + values).astype(np.float32)
        return advantages.astype(np.float32), returns

    def _prepare_batch(self, trajectories: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
        """Pad variable-length trajectories into a fixed-size batch tensor."""
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

    def _split_trajectories(self, trajectories: List[Dict[str, np.ndarray]]) -> List[Dict[str, np.ndarray]]:
        """Split episodes into overlapping sequences via sliding window."""
        chunked: List[Dict[str, np.ndarray]] = []
        chunk_length = self.config.sequence_length
        chunk_stride = self.config.sequence_stride
        for trajectory in trajectories:
            total_length = len(trajectory["returns"])
            for start in range(0, total_length, chunk_stride):
                end = min(start + chunk_length, total_length)
                if end <= start:
                    continue
                chunked.append({key: value[start:end] for key, value in trajectory.items()})
                if end >= total_length:
                    break
        return chunked

    def _sanitize_trajectories(self, trajectories: List[Dict[str, np.ndarray]]) -> None:
        """Replace NaN/inf in trajectory data with safe defaults."""
        low = np.array([-self.config.max_steering_angle, self.config.min_speed], dtype=np.float32)
        high = np.array([self.config.max_steering_angle, self.config.max_speed], dtype=np.float32)
        for t in trajectories:
            t["observations"] = np.nan_to_num(t["observations"], nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)
            t["actions"] = np.clip(np.nan_to_num(t["actions"], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32), low + 1e-5, high - 1e-5)
            t["log_probs"] = np.nan_to_num(t["log_probs"], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            t["returns"] = np.nan_to_num(t["returns"], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            t["advantages"] = np.nan_to_num(t["advantages"], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def _normalize_advantages(self, trajectories: List[Dict[str, np.ndarray]]) -> None:
        """Normalise advantages to zero mean and unit variance."""
        all_adv = np.concatenate([t["advantages"] for t in trajectories], axis=0)
        adv_mean = float(all_adv.mean())
        adv_std = float(all_adv.std() + 1e-8)
        for t in trajectories:
            t["advantages"] = ((t["advantages"] - adv_mean) / adv_std).astype(np.float32)

    @staticmethod
    def _sequence_loss_mask(valid_mask: torch.Tensor, burn_in: int) -> torch.Tensor:
        """Build loss mask that ignores burn-in steps per sequence."""
        valid_lengths = valid_mask.sum(dim=1).to(dtype=torch.long)
        start_index = torch.minimum(torch.full_like(valid_lengths, burn_in), torch.clamp(valid_lengths - 1, min=0))
        return valid_mask * (torch.arange(valid_mask.shape[1], device=valid_mask.device).unsqueeze(0) >= start_index.unsqueeze(1)).to(dtype=valid_mask.dtype)

    def _update_batch(self, batch: Dict[str, torch.Tensor]) -> None:
        """Single PPO update: clipped surrogate, value loss, entropy bonus."""
        log_probs_new, values, entropy = self.evaluate_sequences(
            batch["observations"], batch["actions"], batch["done_mask"],
        )
        if not (torch.isfinite(log_probs_new).all() and torch.isfinite(values).all() and torch.isfinite(entropy).all()):
            print("[PPO] WARNING: Skipping update batch due to non-finite policy evaluation.", flush=True)
            return
        valid_mask = batch["valid_mask"]
        learn_mask = self._sequence_loss_mask(valid_mask, self.config.burn_in)
        mask_bool = learn_mask > 0
        log_ratio = torch.nan_to_num(log_probs_new - batch["log_probs"], nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
        ratio = torch.exp(log_ratio)
        surr1 = ratio * batch["advantages"]
        surr2 = torch.clamp(ratio, 1 - self.config.epsilon, 1 + self.config.epsilon) * batch["advantages"]
        surrogate = torch.where(mask_bool, torch.min(surr1, surr2), torch.zeros_like(surr1))
        value_error = nn.functional.smooth_l1_loss(values, batch["returns"], reduction="none")
        entropy_term = torch.where(mask_bool, entropy, torch.zeros_like(entropy))
        valid_count = learn_mask.sum().clamp_min(1.0)
        loss = (-surrogate.sum() + 0.5 * torch.where(mask_bool, value_error, torch.zeros_like(value_error)).sum() - self.config.entropy_coef * entropy_term.sum()) / valid_count
        if not torch.isfinite(loss):
            print("[PPO] WARNING: Skipping update batch due to non-finite loss.", flush=True)
            return
        self.optimizer.zero_grad()
        loss.backward()
        for p in list(self.model.parameters()) + [self.actor_log_std]:
            if p.grad is not None and not torch.isfinite(p.grad).all():
                print("[PPO] WARNING: Skipping update batch due to non-finite gradients.", flush=True)
                self.optimizer.zero_grad()
                return
        nn.utils.clip_grad_norm_(list(self.model.parameters()) + [self.actor_log_std], max_norm=1.0)
        self.optimizer.step()
        with torch.no_grad():
            self.actor_log_std.data.copy_(torch.nan_to_num(self.actor_log_std.data, nan=-0.5, posinf=2.0, neginf=-5.0).clamp(-5.0, 2.0))

    def evaluate_sequences(
        self, observations: torch.Tensor, actions: torch.Tensor, done_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute log-probs, state values, and entropy over sequences."""
        policy_output, state_values, _ = self.model(
            observations, recurrent_state=self.get_initial_state(observations.shape[0]), done_mask=done_mask,
        )
        mean, std = self._action_stats(policy_output)
        eps = 1e-6
        safe_action = torch.max(torch.min(actions, self.action_high - 1e-5), self.action_low + 1e-5)
        squashed = ((safe_action - self.action_center) / (self.action_scale + eps)).clamp(-1.0 + eps, 1.0 - eps)
        pre_tanh = 0.5 * (torch.log1p(squashed) - torch.log1p(-squashed))
        dist = Normal(mean, std)
        log_prob = self._tanh_log_prob(dist, pre_tanh).sum(dim=-1)
        entropy_pre_tanh = dist.rsample()
        entropy = -self._tanh_log_prob(dist, entropy_pre_tanh).sum(dim=-1)
        return log_prob, state_values, entropy

    def update(self, trajectories: List[Dict[str, np.ndarray]]) -> None:
        """Update policy from collected trajectories with multiple epochs."""
        if not trajectories:
            return
        self._sanitize_trajectories(trajectories)
        self._normalize_advantages(trajectories)
        trajectories = self._split_trajectories(trajectories)
        if not trajectories:
            return
        num = len(trajectories)
        for _ in range(self.config.epochs):
            indices = torch.randperm(num).tolist()
            for start in range(0, num, self.config.batch_size):
                batch_indices = indices[start: start + self.config.batch_size]
                self._update_batch(self._prepare_batch([trajectories[i] for i in batch_indices]))

    def load_model(self, model_path: str) -> None:
        """Load saved PPO checkpoint weights."""
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



def _save_checkpoint(agent: PPOAgent, episode, reward: float, is_goal: bool, prefix: str, run_id: str) -> None:
    """Build and persist a checkpoint file."""
    header = _make_checkpoint_header(episode, reward, is_goal, "ppo", asdict(agent.config))
    header["obs_size"] = agent.obs_size
    header["action_dim"] = agent.action_dim
    header["recurrent_cell"] = agent.config.recurrent_cell
    header["model"] = agent.model.state_dict()
    header["actor_log_std"] = agent.actor_log_std.detach().cpu()
    _save_checkpoint_file(_CHECKPOINT_DIR, run_id, prefix, header)


def train(config: Optional[Config] = None) -> None:
    """Main training loop: environment interaction and policy updates."""
    if config is None:
        config = Config()
    _init_supervisor()
    env = WebotsEnv(config)
    env.reset()
    run_id = Path(env.run_folder).name
    obs_size = env.observation_size
    action_dim = env.action_dim
    agent = PPOAgent(obs_size, action_dim, config)
    checkpoint_dir = run_checkpoint_dir(_CHECKPOINT_DIR, run_id)
    final_model_path = run_checkpoint_path(_CHECKPOINT_DIR, run_id, "final")
    print(f"[TRAIN][PPO] rnn={config.recurrent_cell.upper()} weights_dir={checkpoint_dir} final={final_model_path}", flush=True)
    print(f"[TRAIN][PPO] episodes={config.episodes} update_every={config.update_every} obs={obs_size} act={action_dim} cell={config.recurrent_cell.upper()} seq={config.sequence_length} burn_in={config.burn_in}", flush=True)

    rollout: List[Dict[str, np.ndarray]] = []
    best_reward = float("-inf")
    best_goal_reward = float("-inf")
    best_goal_episode: Optional[int] = None
    rew_w, suc_w, gol_w, col_w, to_w = [], [], [], [], []
    start_time = time.perf_counter()
    metrics_logger = MetricsLogger(env.run_folder, algorithm="ppo")

    for episode in range(config.episodes):
        obs, _ = env.reset()
        done = False
        ep_step = 0
        ep_obs, ep_act, ep_lp, ep_rew = [], [], [], []
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
            ep_goal = ep_goal or bool(info.get("goal_reached", False))
            ep_success = bool(info.get("success", False))
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
                ep_obs_arr, recurrent_state=agent.get_initial_state(batch_size=1),
                done_mask=np.concatenate(([1.0], np.zeros(len(ep_rew) - 1, dtype=np.float32))),
            )
            ep_val_np = ep_values.squeeze(0).detach().cpu().numpy()

        bootstrap_value = 0.0
        if ep_end_reason == "max_steps":
            with torch.no_grad():
                _, bs_val, _ = agent.model(
                    np.asarray(obs, dtype=np.float32), recurrent_state=recurrent_state,
                    done_mask=np.array([0.0], dtype=np.float32),
                )
                bootstrap_value = float(bs_val.squeeze(0).item())

        ep_adv, ep_ret = agent.calculate_gae(np.array(ep_rew, dtype=np.float32) * REWARD_SCALE, ep_val_np, bootstrap_value=bootstrap_value)
        rollout.append({"observations": ep_obs_arr, "actions": np.array(ep_act, dtype=np.float32),
                        "log_probs": np.array(ep_lp, dtype=np.float32), "returns": ep_ret, "advantages": ep_adv})

        if (episode + 1) % config.update_every == 0:
            agent.update(rollout)
            rollout.clear()

        ep_sum = sum(ep_rew)
        rew_w.append(ep_sum)
        suc_w.append(1.0 if ep_success else 0.0)
        gol_w.append(1.0 if ep_goal else 0.0)
        col_w.append(1.0 if ep_end_reason == "collision" else 0.0)
        to_w.append(1.0 if ep_end_reason == "max_steps" else 0.0)
        ckpt_flags: List[str] = []

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
        elapsed = time.perf_counter() - start_time
        ckpt_note = f" ckpt={'+'.join(ckpt_flags)}" if ckpt_flags else ""
        print(f"[TRAIN][PPO] ep={episode + 1:03d}/{config.episodes} r={ep_sum:8.2f} avg10={r10:8.2f} steps={ep_step:4d} succ10={s10:4.2f} touch10={g10:4.2f} col10={c10:4.2f} to10={t10:4.2f} min_d={env.min_episode_distance:5.2f} end={ep_end_reason} t={elapsed:7.1f}s{ckpt_note}", flush=True)
        metrics_logger.log(episode=episode + 1, reward=round(ep_sum, 4), avg10=round(r10, 4), steps=ep_step,
                          success=int(ep_success), goal_touched=int(ep_goal), collision=int(ep_end_reason == "collision"),
                          timeout=int(ep_end_reason == "max_steps"), min_dist=round(env.min_episode_distance, 4),
                          end_reason=ep_end_reason, elapsed_s=round(elapsed, 1))

    if rollout:
        agent.update(rollout)
    metrics_logger.close()
    print(f"[TRAIN][PPO] metrics saved to {metrics_logger.path}", flush=True)
    final_reward = best_goal_reward if best_goal_episode is not None else best_reward
    _save_checkpoint(agent, "final", final_reward, best_goal_episode is not None, "final", run_id)
    elapsed = time.perf_counter() - start_time
    print(f"[TRAIN][PPO] final reward={final_reward:.2f} t={elapsed:7.1f}s", flush=True)
    env.robot.motors.stop()
    print("[TRAIN][PPO] done", flush=True)


if __name__ == "__main__":
    train()
