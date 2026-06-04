"""
PPO training controller for ALTINO robot in Webots obstacle avoidance task.

LLM level: 4 - LLM wrote the majority of the starting code, but we have since iterated on it a lot.
"""

import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from controllers.PPO.PPO_config import Config, _apply_env_overrides
from controllers.PPO.PPO_agent import PPOAgent, _save_checkpoint, _CHECKPOINT_DIR
from controllers.Webots.webots_env import WebotsEnv, _init_supervisor
from controllers.PPO.PPO_rewards import PPORewardComputer
import controllers.PPO.PPO_defaults as d
from controllers.common.checkpoints import run_checkpoint_dir, run_checkpoint_path
from controllers.common.metrics_logger import MetricsLogger


def _make_reward_computer(config: Config) -> PPORewardComputer:
    """
    Computes reward for the PPO agent.

    Parameters
    ----------
    config : Config
        Configuration object.

    Returns
    -------
    PPORewardComputer
        Reward computer object.
    """
    return PPORewardComputer(
        endpoint=config.endpoint,
        collision_penalty=config.collision_penalty,
        progress_reward_scale=config.progress_reward_scale,
        distance_reward_scale=config.distance_reward_scale,
        heading_reward_scale=config.heading_reward_scale,
        safety_reward_scale=config.safety_reward_scale,
        motion_reward_scale=config.motion_reward_scale,
        new_best_distance_bonus=config.new_best_distance_bonus,
        step_penalty=config.step_penalty,
        goal_threshold=config.goal_threshold,
        goal_success_reward=config.goal_success_reward,
        goal_hold_reward=config.goal_hold_reward,
    )


def train(config=None) -> None:
    """
    Train the PPO agent in the Webots environment.

    Parameters
    ----------
    config : Config, optional
        Configuration object. If None, the default configuration is used.
    """
    if config is None:
        config = Config()
    config, load_model_path, run_id_override = _apply_env_overrides(config)
    
    _init_supervisor()
    reward_computer = _make_reward_computer(config)
    
    env = WebotsEnv(config, reward_computer)
    env.reset()
    run_id = run_id_override or Path(env.run_folder).name
    if run_id_override:
        env.run_folder = str(Path(env.run_folder).parent / run_id_override)
    
    os.makedirs(env.run_folder, exist_ok=True)
    obs_size = env.observation_size
    action_dim = env.action_dim
    agent = PPOAgent(obs_size, action_dim, config)
    if load_model_path:
        agent.load_model(load_model_path)
    
    checkpoint_dir = run_checkpoint_dir(_CHECKPOINT_DIR, run_id)
    final_model_path = run_checkpoint_path(_CHECKPOINT_DIR, run_id, "final")
    print(f"[TRAIN][PPO] arch={config.recurrent_cell.upper()} weights_dir={checkpoint_dir} final={final_model_path}", flush=True)
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
            reward = np.clip(reward, -100.0, 100.0)
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

        decay_frac = min(1.0, episode / max(1, config.episodes))
        #Added 1.35 to LSTM as it was unstable and suffered catastrophic forgetting
        arch_scale = {"none": 1.0, "gru": 1.0, "lstm": 1.35}.get(config.recurrent_cell, 1.0)
        base_entropy = d.PPODefaults.entropy_coef * arch_scale
        agent.config.entropy_coef = base_entropy * (1.0 - 0.30 * decay_frac)

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
            env.robot.mapping.save_episode(env.run_folder, episode + 1, ep_sum)
            _save_checkpoint(agent, best_goal_episode, best_goal_reward, True, "best", run_id)
            ckpt_flags.append("best_goal")
        elif best_goal_episode is None and ep_sum > best_reward:
            best_reward = ep_sum
            env.robot.mapping.save_episode(env.run_folder, episode + 1, ep_sum)
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
    final_reward = rew_w[-1] if rew_w else (best_goal_reward if best_goal_episode is not None else best_reward)
    _save_checkpoint(agent, "final", final_reward, best_goal_episode is not None, "final", run_id)
    elapsed = time.perf_counter() - start_time
    print(f"[TRAIN][PPO] final reward={final_reward:.2f} t={elapsed:7.1f}s", flush=True)
    env.robot.stop()
    env.robot.supervisor.simulationQuit(0)
    print("[TRAIN][PPO] done", flush=True)


def evaluate(config=None, model_path=None, episodes=10, deterministic=True) -> Dict[str, float | int]:
    """
    Evaluate the PPO agent in the Webots environment.

    Parameters
    ----------
    config : Config, optional
        Configuration object. If None, a default Config object is used.
    model_path : str, optional
        Path to the model file. If None, the model path is taken from the environment variable PPO_EVAL_MODEL.
    episodes : int, optional
        Number of episodes to evaluate. Default is 10.
    deterministic : bool, optional
        Whether to use deterministic actions. Default is True.

    Returns
    -------
    Dict[str, float | int]
        Dictionary containing the evaluation metrics.
    """
    if config is None:
        config = Config()
    
    config, load_model_path, run_id_override = _apply_env_overrides(config)
    model_path = model_path or os.getenv("PPO_EVAL_MODEL") or load_model_path
    if model_path is None:
        raise ValueError("evaluate() requires model_path or PPO_LOAD_MODEL/PPO_EVAL_MODEL.")

    _init_supervisor()
    reward_computer = _make_reward_computer(config)
    env = WebotsEnv(config, reward_computer)
    env.reset()
    if run_id_override:
        env.run_folder = str(Path(env.run_folder).parent / run_id_override)
        os.makedirs(env.run_folder, exist_ok=True)

    agent = PPOAgent(env.observation_size, env.action_dim, config)
    agent.load_model(str(model_path))
    agent.model.eval()

    rewards, successes, goal_touches, collisions, timeouts = [], [], [], [], []
    total_steps = 0
    start_time = time.perf_counter()
    print(
        f"[EVAL][PPO] model={model_path} episodes={episodes} "
        f"deterministic={deterministic} cell={agent.config.recurrent_cell.upper()}",
        flush=True,
    )

    try:
        for episode in range(episodes):
            obs, _ = env.reset()
            done = False
            ep_step = 0
            ep_reward = 0.0
            ep_goal = False
            ep_success = False
            ep_end_reason = "max_steps"
            ep_speeds = []
            recurrent_state = agent.get_initial_state(batch_size=1)
            prev_done = True

            while not done:
                action, _, _, recurrent_state = agent.select_action(
                    obs, recurrent_state, done=prev_done, deterministic=deterministic,
                )
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                prev_done = done
                ep_step += 1
                total_steps += 1
                ep_reward += float(reward)
                ep_goal = ep_goal or bool(info.get("goal_reached", False))
                ep_success = bool(info.get("success", False))
                ep_speeds.append(float(info.get("speed_norm", 0.0)))
                if done:
                    reason = info.get("reset_reason", "")
                    ep_end_reason = reason if reason else ("max_steps" if truncated else ep_end_reason)

            rewards.append(ep_reward)
            successes.append(1.0 if ep_success else 0.0)
            goal_touches.append(1.0 if ep_goal else 0.0)
            collisions.append(1.0 if ep_end_reason == "collision" else 0.0)
            timeouts.append(1.0 if ep_end_reason == "max_steps" else 0.0)
            avg_spd = float(np.mean(ep_speeds)) * config.max_speed if ep_speeds else 0.0
            print(
                f"[EVAL][PPO] ep={episode + 1:03d}/{episodes} r={ep_reward:8.2f} "
                f"steps={ep_step:4d} success={int(ep_success)} touch={int(ep_goal)} "
                f"min_d={env.min_episode_distance:5.2f} avg_spd={avg_spd:4.2f}m/s "
                f"end={ep_end_reason}",
                flush=True,
            )
    finally:
        env.robot.stop()
        env.robot.supervisor.simulationQuit(0)

    elapsed = time.perf_counter() - start_time
    summary = {
        "episodes": episodes,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "goal_touch_rate": float(np.mean(goal_touches)) if goal_touches else 0.0,
        "collision_rate": float(np.mean(collisions)) if collisions else 0.0,
        "timeout_rate": float(np.mean(timeouts)) if timeouts else 0.0,
        "total_steps": total_steps,
        "elapsed_s": elapsed,
    }
    print(
        f"[EVAL][PPO] mean_reward={summary['mean_reward']:.2f} "
        f"success={summary['success_rate']:.2f} touch={summary['goal_touch_rate']:.2f} "
        f"collision={summary['collision_rate']:.2f} timeout={summary['timeout_rate']:.2f} "
        f"steps={total_steps} t={elapsed:.1f}s",
        flush=True,
    )
    return summary

if __name__ == "__main__":
    train()
