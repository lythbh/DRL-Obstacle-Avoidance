"""Soft Actor-Critic controller for the ALTINO Webots task."""
from __future__ import annotations

from collections import deque
import random
import sys, time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from controllers.Webots.webots_env import WebotsEnv, _init_supervisor
from controllers.common.SAC_rewards import SACRewardComputer
from controllers.RNN import GRUActorCritic, LSTMActorCritic
from controllers.SAC.replay import SequenceReplayBuffer
import controllers.common.SAC_defaults as d
from controllers.common.checkpoints import (
    run_checkpoint_dir as _run_checkpoint_dir,
    run_checkpoint_path as _run_checkpoint_path,
    load_checkpoint as _load_checkpoint,
    save_checkpoint_file as _save_checkpoint_file,
)
from controllers.common.metrics_logger import MetricsLogger
from controllers.common.seed import set_all_seeds
from controllers.SAC.agent import SACAgent

_CONTROLLER_DIR = Path(__file__).resolve().parent
_CHECKPOINT_DIR = _CONTROLLER_DIR / "checkpoints"


@dataclass
class Config:
    episodes: int = d.SACDefaults.episodes
    seed: Optional[int] = None
    update_after_steps: int = d.SACDefaults.update_after_steps
    update_freq: int = getattr(d.SACDefaults, 'update_freq', 32)
    action_repeat: int = getattr(d.SACDefaults, 'action_repeat', 2)
    save_every: int = d.SACDefaults.save_every
    gamma: float = d.SACDefaults.gamma
    tau: float = d.SACDefaults.tau
    target_update_interval: int = d.SACDefaults.target_update_interval
    actor_lr: float = d.SACDefaults.actor_lr
    critic_lr: float = d.SACDefaults.critic_lr
    alpha_lr: float = d.SACDefaults.alpha_lr
    initial_alpha: float = d.SACDefaults.initial_alpha
    auto_entropy_tuning: bool = d.SACDefaults.auto_entropy_tuning
    target_entropy_scale: float = d.SACDefaults.target_entropy_scale
    hidden_size: int = d.SACDefaults.hidden_size
    latent_size: int = d.SACDefaults.latent_size
    recurrent_cell: str = d.SACDefaults.recurrent_cell
    recurrent_hidden_size: Optional[int] = d.SACDefaults.recurrent_hidden_size
    recurrent_layers: int = d.SACDefaults.recurrent_layers
    lstm_hidden_size: int = d.SACDefaults.lstm_hidden_size
    lstm_layers: int = d.SACDefaults.lstm_layers
    log_std_min: float = d.SACDefaults.log_std_min
    log_std_max: float = d.SACDefaults.log_std_max
    sequence_length: int = d.RecurrentDefaults.sequence_length
    burn_in: int = d.RecurrentDefaults.burn_in
    sequence_stride: int = d.RecurrentDefaults.sequence_stride
    replay_capacity: int = d.SACDefaults.replay_capacity
    replay_batch_size: int = d.SACDefaults.replay_batch_size
    min_replay_sequences: int = d.SACDefaults.min_replay_sequences
    lidar_sector_dim: int = d.ENV_LIDAR_SECTOR_DIM
    pose_goal_dim: int = d.ENV_POSE_GOAL_DIM
    imu_feature_dim: int = d.ENV_IMU_FEATURE_DIM
    occupancy_grid_shape: Optional[Tuple[int, ...]] = d.ENV_OCCUPANCY_GRID_SHAPE
    max_steps: int = d.ENV_MAX_STEPS
    collision_threshold: float = d.ENV_COLLISION_THRESHOLD
    low_score_threshold: float = d.ENV_LOW_SCORE_THRESHOLD
    collision_penalty: float = d.REW_COLLISION_PENALTY
    heading_reward_scale: float = d.REW_HEADING_SCALE
    endpoint: Tuple[float, float] = d.ENV_ENDPOINT
    goal_threshold: float = d.ENV_GOAL_THRESHOLD
    goal_stop_speed_threshold: float = d.ENV_GOAL_STOP_SPEED_THRESHOLD
    goal_success_reward: float = d.REW_GOAL_SUCCESS
    goal_overshoot_penalty: float = d.REW_GOAL_OVERSHOOT_PENALTY
    reference_distance: Optional[float] = None
    enable_slam: bool = d.SLAM_ENABLE
    profile_slam: bool = d.SLAM_PROFILE
    slam_profile_interval: int = d.SLAM_PROFILE_INTERVAL
    save_slam_plots: bool = d.SLAM_SAVE_PLOTS
    force_cpu: bool = d.SLAM_FORCE_CPU
    max_steering_angle: float = d.ENV_MAX_STEERING_ANGLE
    min_speed: float = d.ENV_MIN_SPEED
    start_position: Optional[List[float]] = None
    start_rotation: Optional[List[float]] = None
    start_position_noise: float = d.ENV_START_POSITION_NOISE
    start_yaw_noise: float = d.ENV_START_YAW_NOISE
    max_speed: float = d.ENV_MAX_SPEED
    reset_settle_steps: int = d.ENV_RESET_SETTLE_STEPS

    def __post_init__(self):
        if self.seed is None:
            self.seed = random.SystemRandom().randrange(0, 2**32)
        self.recurrent_cell = self.recurrent_cell.lower().strip()
        assert self.recurrent_cell in {"gru", "lstm"}, f"Unsupported recurrent_cell: {self.recurrent_cell}"
        if self.recurrent_hidden_size is None:
            self.recurrent_hidden_size = self.hidden_size
        if self.start_position is None:
            self.start_position = list(d.ENV_START_POSITION)
        if self.start_rotation is None:
            self.start_rotation = list(d.ENV_START_ROTATION)
        if self.reference_distance is None:
            start_xy = np.array(self.start_position[:2], dtype=np.float32)
            endpoint_xy = np.array(self.endpoint, dtype=np.float32)
            self.reference_distance = float(np.linalg.norm(start_xy - endpoint_xy))



def train(config=None):
    """Main training loop: collect episodes, sample from replay buffer, perform SAC updates, and save checkpoints."""
    if config is None:
        config = Config()

    assert config.seed is not None
    set_all_seeds(config.seed)
    _init_supervisor()
    reference_distance = float(config.reference_distance if config.reference_distance is not None else 4.0)
    reward_computer = SACRewardComputer(
        endpoint=config.endpoint,
        reference_distance=reference_distance,
        collision_penalty=config.collision_penalty,
        heading_reward_scale=config.heading_reward_scale,
        goal_threshold=config.goal_threshold,
        goal_success_reward=config.goal_success_reward,
    )
    env = WebotsEnv(config, reward_computer)
    env.reset()
    run_id = Path(env.run_folder).name
    agent = SACAgent(env.observation_size, env.action_dim, config)
    replay = SequenceReplayBuffer(env.observation_size, env.action_dim, config)
    checkpoint_dir = _run_checkpoint_dir(_CHECKPOINT_DIR, run_id)
    final_model_path = _run_checkpoint_path(_CHECKPOINT_DIR, run_id, "final")
    print(f"[TRAIN][SAC] rnn={config.recurrent_cell.upper()} weights_dir={checkpoint_dir} final={final_model_path}", flush=True)
    print(f"[TRAIN][SAC] replay=on cap={config.replay_capacity} seq={config.sequence_length} stride={config.sequence_stride} batch={config.replay_batch_size}", flush=True)

    total_steps = 0
    best_reward = float("-inf")
    best_goal_reward = float("-inf")
    best_goal_episode = None
    rew_w, suc_w, gol_w, col_w, to_w = (deque(maxlen=10) for _ in range(5))
    start_time = time.perf_counter()
    metrics_logger = MetricsLogger(env.run_folder, algorithm="sac")
    metrics_logger.log_hyperparams(asdict(config), recurrent_cell=config.recurrent_cell,
                                   obs_size=env.observation_size, action_dim=env.action_dim)

    def _snapshot_recurrent_state(state):
        if isinstance(state, tuple):
            return tuple(_snapshot_recurrent_state(part) for part in state)
        return state.detach().cpu().clone()

    for episode in range(config.episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        ep_end_reason = "max_steps"
        ep_obs, ep_act, ep_rew, ep_next, ep_done = [], [], [], [], []
        ep_states = []
        ep_goal = ep_success = False
        ep_speeds = []
        actor_state = agent.get_initial_state(batch_size=1)
        prev_done = True
        ep_step = 0
        all_update_metrics = []

        warmup_episodes = max(5, config.update_after_steps // 200)
        if episode < warmup_episodes:
            warmup_lr = config.actor_lr * (0.25 + 0.75 * (episode + 1) / warmup_episodes)
            for pg in agent.actor_optimizer.param_groups:
                pg['lr'] = warmup_lr
            warmup_lr_c = config.critic_lr * (0.25 + 0.75 * (episode + 1) / warmup_episodes)
            for pg in agent.critic_optimizer.param_groups:
                pg['lr'] = warmup_lr_c
            warmup_lr_a = config.alpha_lr * (0.25 + 0.75 * (episode + 1) / warmup_episodes)
            for pg in agent.alpha_optimizer.param_groups:
                pg['lr'] = warmup_lr_a
        else:
            for pg in agent.actor_optimizer.param_groups:
                pg['lr'] = config.actor_lr
            for pg in agent.critic_optimizer.param_groups:
                pg['lr'] = config.critic_lr
            for pg in agent.alpha_optimizer.param_groups:
                pg['lr'] = config.alpha_lr

        while not done:
            ep_states.append(_snapshot_recurrent_state(actor_state))
            action, actor_state = agent.select_action(obs, actor_state, done=prev_done, deterministic=False)

            # 1. Step the environment
            next_obs, reward, terminated, truncated, info = env.step(action)

            ep_step += 1
            transition_done = bool(terminated)
            ep_obs.append(np.asarray(obs, dtype=np.float32))
            ep_act.append(np.asarray(action, dtype=np.float32))
            ep_rew.append(float(reward))
            ep_next.append(np.asarray(next_obs, dtype=np.float32))
            ep_done.append(transition_done)
            obs = next_obs
            episode_reward += reward
            ep_speeds.append(float(info.get("speed_norm", 0.0)))
            ep_goal = ep_goal or bool(info.get("goal_reached", False))
            ep_success = bool(info.get("success", False))
            done = bool(terminated or truncated)
            prev_done = done
            total_steps += 1

            # --- NEW: DO UPDATES DURING THE SIMULATION LOOP ---
            if total_steps >= config.update_after_steps and replay.can_sample(config.replay_batch_size, config.min_replay_sequences):
                if total_steps % config.update_freq == 0:
                    upd = agent.update(replay.sample(config.replay_batch_size, agent.device))
                    if upd is not None:
                        all_update_metrics.append(upd)
                        metrics_logger.log_update(
                            global_step=total_steps, episode=episode + 1,
                            recurrent_cell=config.recurrent_cell,
                            **upd,
                        )

            if done:
                reason = info.get("reset_reason", "")
                ep_end_reason = reason if reason else ("max_steps" if truncated else ep_end_reason)

        replay.add_episode(ep_obs, ep_act, ep_rew, ep_next, ep_done, ep_states=ep_states)
        for _obs in ep_obs:
            agent.obs_rms.update(_obs)

        act_stats = MetricsLogger.compute_action_stats(ep_act)
        obs_stats = MetricsLogger.compute_obs_stats(ep_obs)
        agg_upd = MetricsLogger.aggregate_update_metrics(all_update_metrics)

        rew_w.append(episode_reward)
        suc_w.append(1.0 if ep_success else 0.0)
        gol_w.append(1.0 if ep_goal else 0.0)
        col_w.append(1.0 if ep_end_reason == "collision" else 0.0)
        to_w.append(1.0 if ep_end_reason == "max_steps" else 0.0)
        ckpt_flags = []

        if ep_end_reason == "goal" and episode_reward > best_goal_reward:
            best_goal_reward = episode_reward
            best_goal_episode = episode + 1
            env.robot.slam.save_episode(env.run_folder, episode + 1, episode_reward)
            ckpt = agent.checkpoint(best_goal_episode, best_goal_reward)
            ckpt["goal_episode"] = True
            _save_checkpoint_file(_CHECKPOINT_DIR, run_id, "best", ckpt)
            ckpt_flags.append("best_goal")
        elif best_goal_episode is None and episode_reward > best_reward:
            best_reward = episode_reward
            env.robot.slam.save_episode(env.run_folder, episode + 1, episode_reward)
            ckpt = agent.checkpoint(episode + 1, best_reward)
            ckpt["goal_episode"] = False
            _save_checkpoint_file(_CHECKPOINT_DIR, run_id, "best", ckpt)
            ckpt_flags.append("best")

        if config.save_every > 0 and (episode + 1) % config.save_every == 0:
            ckpt = agent.checkpoint(episode + 1, episode_reward)
            ckpt["goal_episode"] = ep_end_reason == "goal"
            _save_checkpoint_file(_CHECKPOINT_DIR, run_id, "checkpoint", ckpt)
            ckpt_flags.append("latest")

        r10 = float(np.mean(rew_w))
        s10 = float(np.mean(suc_w))
        g10 = float(np.mean(gol_w))
        c10 = float(np.mean(col_w))
        t10 = float(np.mean(to_w))
        avg_spd = float(np.mean(ep_speeds)) * config.max_speed if ep_speeds else 0.0
        elapsed = time.perf_counter() - start_time
        ckpt_note = f" ckpt={'+'.join(ckpt_flags)}" if ckpt_flags else ""
        print(f"[TRAIN][SAC] ep={episode + 1:03d}/{config.episodes} r={episode_reward:8.2f} avg10={r10:8.2f} steps={env.current_step:4d} succ10={s10:4.2f} touch10={g10:4.2f} col10={c10:4.2f} to10={t10:4.2f} min_d={env.min_episode_distance:5.2f} avg_spd={avg_spd:4.2f}m/s end={ep_end_reason} replay={len(replay):4d} t={elapsed:7.1f}s{ckpt_note}", flush=True)

        metrics_logger.log_episode(
            episode=episode + 1,
            global_step=total_steps,
            reward=round(episode_reward, 4),
            avg10=round(r10, 4),
            length=env.current_step,
            success=int(ep_success),
            goal_touched=int(ep_goal),
            collision=int(ep_end_reason == "collision"),
            timeout=int(ep_end_reason == "max_steps"),
            min_dist=round(env.min_episode_distance, 4),
            avg_speed_ms=round(avg_spd, 3),
            end_reason=ep_end_reason,
            elapsed_s=round(elapsed, 1),
            recurrent_cell=config.recurrent_cell,
            replay_buffer_size=len(replay),
            **act_stats,
            **obs_stats,
            **agg_upd,
        )

    metrics_logger.close()
    print(f"[TRAIN][SAC] metrics saved to {metrics_logger.path}", flush=True)
    print(f"[TRAIN][SAC] updates saved to {metrics_logger.update_path}", flush=True)
    print(f"[TRAIN][SAC] hyperparams saved to {metrics_logger.hyperparams_path}", flush=True)
    final_reward = best_goal_reward if best_goal_episode is not None else best_reward
    agent.save(final_model_path, "final", final_reward)
    elapsed = time.perf_counter() - start_time
    print(f"[TRAIN][SAC] final reward={final_reward:.2f} t={elapsed:7.1f}s", flush=True)
    env.robot.stop()
    print("[TRAIN][SAC] done", flush=True)


if __name__ == "__main__":
    train()

