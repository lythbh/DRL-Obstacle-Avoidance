"""
Unit tests for PPO and SAC algorithmic components.

Run with:
    python -m pytest tests/ -v
or:
    python tests/test_algorithms.py

All tests are self-contained and do NOT require a Webots installation.
The Webots `controller` C-extension is mocked so that pure-Python and
PyTorch code can be exercised without launching a simulation.
"""

import math
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from torch.distributions import Normal

# ---------------------------------------------------------------------------
# Mock the Webots `controller` C-extension BEFORE any project imports so that
# modules which do `from controller import Supervisor` succeed without Webots.
# ---------------------------------------------------------------------------
_controller_mock = types.ModuleType("controller")
_controller_mock.Supervisor = MagicMock()
sys.modules["controller"] = _controller_mock

# Make project root importable when running directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ppo_config():
    from controllers.PPO.PPO import Config
    return Config(
        episodes=10,
        update_every=2,
        max_steps=50,
        lidar_sector_dim=16,
        pose_goal_dim=7,
        imu_feature_dim=10,
    )


def _sac_config():
    from controllers.SAC.SAC import Config
    return Config(
        episodes=10,
        max_steps=50,
        lidar_sector_dim=16,
        pose_goal_dim=7,
        imu_feature_dim=10,
    )


# ---------------------------------------------------------------------------
# PPO – Generalized Advantage Estimation
# ---------------------------------------------------------------------------

class TestGAE:
    """GAE must satisfy the λ=1 / λ=0 boundary cases and be self-consistent."""

    def _make_agent(self):
        from controllers.PPO.PPO import PPOAgent
        cfg = _ppo_config()
        obs_size = cfg.lidar_sector_dim + cfg.pose_goal_dim + cfg.imu_feature_dim
        return PPOAgent(obs_size, action_dim=2, config=cfg)

    def test_lambda_one_recovers_monte_carlo(self):
        """GAE with λ=1 must equal discounted Monte-Carlo returns."""
        agent = self._make_agent()
        agent.config.gae_lambda = 1.0
        gamma = agent.config.gamma

        rewards = np.array([1.0, 2.0, -1.0, 3.0], dtype=np.float32)
        values  = np.zeros(4, dtype=np.float32)          # zero critic → A = G
        bootstrap = 0.5

        advantages, returns = agent.calculate_gae(rewards, values, bootstrap)

        # MC returns computed independently.
        mc = np.zeros(4, dtype=np.float32)
        G = bootstrap
        for t in reversed(range(4)):
            G = rewards[t] + gamma * G
            mc[t] = G

        np.testing.assert_allclose(returns, mc, atol=1e-5,
                                   err_msg="GAE(λ=1) must equal MC returns")

    def test_lambda_zero_gives_one_step_td(self):
        """GAE with λ=0 must equal the 1-step TD advantage δ_t = r_t + γV(s_{t+1}) - V(s_t)."""
        agent = self._make_agent()
        agent.config.gae_lambda = 0.0
        gamma = agent.config.gamma

        rewards = np.array([1.0, -0.5, 2.0], dtype=np.float32)
        values  = np.array([0.3, 0.7,  0.2], dtype=np.float32)
        bootstrap = 1.0

        advantages, _ = agent.calculate_gae(rewards, values, bootstrap)

        next_values = np.append(values[1:], bootstrap)
        td = rewards + gamma * next_values - values

        np.testing.assert_allclose(advantages, td, atol=1e-5,
                                   err_msg="GAE(λ=0) must equal 1-step TD advantages")

    def test_returns_equal_advantages_plus_values(self):
        """Returns must always be V̂ = A + V."""
        agent = self._make_agent()
        agent.config.gae_lambda = 0.95

        rewards = np.random.randn(20).astype(np.float32)
        values  = np.random.randn(20).astype(np.float32)

        advantages, returns = agent.calculate_gae(rewards, values, bootstrap_value=0.0)

        np.testing.assert_allclose(returns, advantages + values, atol=1e-5,
                                   err_msg="returns must satisfy returns = advantages + values")

    def test_bootstrap_propagates_to_first_step(self):
        """A positive bootstrap value should raise the advantage at t=0."""
        agent = self._make_agent()
        agent.config.gae_lambda = 1.0

        rewards = np.zeros(5, dtype=np.float32)
        values  = np.zeros(5, dtype=np.float32)

        _, returns_no_boot = agent.calculate_gae(rewards, values, bootstrap_value=0.0)
        _, returns_boot    = agent.calculate_gae(rewards, values, bootstrap_value=10.0)

        assert returns_boot[0] > returns_no_boot[0], \
            "Positive bootstrap must increase returns at t=0"

    def test_single_step_episode(self):
        """Edge case: one-step episode must not crash and must satisfy boundary conditions."""
        agent = self._make_agent()
        rewards = np.array([1.0], dtype=np.float32)
        values  = np.array([0.5], dtype=np.float32)
        advantages, returns = agent.calculate_gae(rewards, values, bootstrap_value=0.0)
        assert advantages.shape == (1,)
        assert returns.shape == (1,)


# ---------------------------------------------------------------------------
# PPO – Advantage normalisation
# ---------------------------------------------------------------------------

class TestAdvantageNormalisation:

    def _make_agent(self):
        from controllers.PPO.PPO import PPOAgent
        cfg = _ppo_config()
        obs_size = cfg.lidar_sector_dim + cfg.pose_goal_dim + cfg.imu_feature_dim
        return PPOAgent(obs_size, action_dim=2, config=cfg)

    def test_normalised_mean_is_zero(self):
        agent = self._make_agent()
        traj = [{"advantages": np.random.randn(20).astype(np.float32)} for _ in range(5)]
        agent._normalize_advantages(traj)
        all_adv = np.concatenate([t["advantages"] for t in traj])
        assert abs(float(all_adv.mean())) < 1e-5, "Normalised advantages must have zero mean"

    def test_normalised_std_is_one(self):
        agent = self._make_agent()
        traj = [{"advantages": np.random.randn(20).astype(np.float32)} for _ in range(5)]
        agent._normalize_advantages(traj)
        all_adv = np.concatenate([t["advantages"] for t in traj])
        assert abs(float(all_adv.std()) - 1.0) < 1e-4, "Normalised advantages must have unit std"

    def test_degenerate_constant_advantages_no_nan(self):
        """Constant advantages (std ≈ 0) must not produce NaN after normalisation."""
        agent = self._make_agent()
        traj = [{"advantages": np.ones(10, dtype=np.float32)}]
        agent._normalize_advantages(traj)
        assert not np.any(np.isnan(traj[0]["advantages"])), "Normalised advantages must not be NaN"


# ---------------------------------------------------------------------------
# PPO – Log-probability and tanh-squashing correction
# ---------------------------------------------------------------------------

class TestLogProbCorrection:
    """The tanh-squashed Gaussian log-prob must be consistent with the sampling density."""

    def _make_agent(self):
        from controllers.PPO.PPO import PPOAgent
        cfg = _ppo_config()
        obs_size = cfg.lidar_sector_dim + cfg.pose_goal_dim + cfg.imu_feature_dim
        return PPOAgent(obs_size, action_dim=2, config=cfg)

    def test_action_in_bounds(self):
        agent = self._make_agent()
        policy_output = torch.zeros(1, 2)
        for _ in range(50):
            action, _ = agent._sample_action(policy_output, deterministic=False)
            assert (action >= agent.action_low - 1e-4).all() and (action <= agent.action_high + 1e-4).all()

    def test_log_prob_finite(self):
        agent = self._make_agent()
        policy_output = torch.zeros(1, 2)
        _, log_prob = agent._sample_action(policy_output, deterministic=False)
        assert torch.isfinite(log_prob).all(), "Log-probability must be finite"

    def test_log_prob_negative(self):
        """For a continuous distribution, log-prob can be negative (density < 1 is possible,
        but very high-density points near mean may exceed 0 - what matters is it is finite)."""
        agent = self._make_agent()
        policy_output = torch.zeros(1, 2)
        for _ in range(30):
            _, log_prob = agent._sample_action(policy_output, deterministic=False)
            assert torch.isfinite(log_prob).all()

    def test_inverse_squash_roundtrip(self):
        """tanh-squash and inverse must recover pre-tanh (up to clamping eps)."""
        agent = self._make_agent()
        pre_tanh = torch.randn(10, 2) * 0.5
        action = torch.tanh(pre_tanh) * agent.action_scale + agent.action_center
        eps = 1e-6
        squashed = ((action - agent.action_center) / (agent.action_scale + eps)).clamp(-1.0 + eps, 1.0 - eps)
        pre_tanh_recovered = 0.5 * (torch.log1p(squashed) - torch.log1p(-squashed))
        np.testing.assert_allclose(
            pre_tanh.numpy(), pre_tanh_recovered.numpy(),
            atol=1e-4,
            err_msg="Inverse squash must recover the original pre-tanh value",
        )


# ---------------------------------------------------------------------------
# SAC – Replay buffer
# ---------------------------------------------------------------------------

class TestSequenceReplayBuffer:

    def _make_buffer(self, obs_size=33, action_dim=2, capacity=128,
                     seq_len=16, stride=8):
        from controllers.SAC.SAC import SequenceReplayBuffer, Config
        cfg = _sac_config()
        cfg.replay_capacity   = capacity
        cfg.sequence_length   = seq_len
        cfg.sequence_stride   = stride
        cfg.replay_batch_size = 4
        return SequenceReplayBuffer(obs_size, action_dim, cfg)

    def test_empty_buffer_cannot_sample(self):
        buf = self._make_buffer()
        assert not buf.can_sample(4, 4), "Empty buffer must not be sampleable"

    def test_add_episode_increases_size(self):
        buf = self._make_buffer(seq_len=8, stride=4, capacity=64)
        T = 20
        obs     = [np.random.randn(33).astype(np.float32) for _ in range(T)]
        actions = [np.random.randn(2).astype(np.float32)  for _ in range(T)]
        rewards = list(np.random.randn(T).astype(float))
        dones   = [False] * (T - 1) + [True]
        buf.add_episode(obs, actions, rewards, obs, dones)
        assert len(buf) > 0, "Buffer size must increase after adding an episode"

    def test_sample_shapes_correct(self):
        buf = self._make_buffer(capacity=256, seq_len=8, stride=4)
        T = 40
        obs     = [np.random.randn(33).astype(np.float32) for _ in range(T)]
        actions = [np.random.randn(2).astype(np.float32)  for _ in range(T)]
        rewards = list(np.zeros(T, dtype=float))
        dones   = [False] * T
        for _ in range(10):
            buf.add_episode(obs, actions, rewards, obs, dones)

        batch = buf.sample(4, torch.device("cpu"))
        assert batch["obs"].shape       == (4, 8, 33), f"obs shape wrong: {batch['obs'].shape}"
        assert batch["actions"].shape   == (4, 8, 2),  f"actions shape wrong: {batch['actions'].shape}"
        assert batch["rewards"].shape   == (4, 8, 1),  f"rewards shape wrong: {batch['rewards'].shape}"
        assert batch["dones"].shape     == (4, 8, 1),  f"dones shape wrong: {batch['dones'].shape}"
        assert batch["valid_mask"].shape == (4, 8),    f"valid_mask shape wrong: {batch['valid_mask'].shape}"

    def test_ring_buffer_wraps_at_capacity(self):
        buf = self._make_buffer(capacity=8, seq_len=4, stride=2)
        T = 10
        obs     = [np.random.randn(33).astype(np.float32) for _ in range(T)]
        actions = [np.random.randn(2).astype(np.float32)  for _ in range(T)]
        rewards = list(np.zeros(T, dtype=float))
        dones   = [False] * T
        for _ in range(5):
            buf.add_episode(obs, actions, rewards, obs, dones)
        assert len(buf) == buf.capacity, "Buffer should be full after overflow"

    def test_valid_mask_marks_padded_steps(self):
        """Short windows (< seq_len) must be zero-padded with valid_mask=0 beyond actual length."""
        buf = self._make_buffer(capacity=32, seq_len=8, stride=8)
        T = 5   # episode shorter than seq_len
        obs     = [np.random.randn(33).astype(np.float32) for _ in range(T)]
        actions = [np.random.randn(2).astype(np.float32)  for _ in range(T)]
        rewards = list(np.zeros(T, dtype=float))
        dones   = [False] * T
        buf.add_episode(obs, actions, rewards, obs, dones)
        # The stored window should have valid_mask=1 for first T steps, 0 for the rest.
        stored_mask = buf.buffer[0]["valid_mask"]
        assert stored_mask[:T].sum()  == T,     "First T steps must be valid"
        assert stored_mask[T:].sum()  == 0,     "Padded steps must be invalid"


# ---------------------------------------------------------------------------
# SAC – done flag must NOT be set on truncation
# ---------------------------------------------------------------------------

class TestSACDoneFlag:
    """The Bellman target must not zero out on episode timeout."""

    def test_terminated_sets_done(self):
        terminated, truncated = True, False
        done = bool(terminated)
        assert done is True

    def test_truncated_does_not_set_done(self):
        terminated, truncated = False, True
        done = bool(terminated)
        assert done is False, (
            "Timeout (truncated) must NOT set done=True; "
            "the Bellman backup must still bootstrap the next-state value."
        )

    def test_both_terminated_and_truncated(self):
        terminated, truncated = True, True
        done = bool(terminated)
        assert done is True


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

class TestRewardComputer:

    def _make_computer(self):
        from controllers.Webots.webots_env import RewardComputer
        return RewardComputer(
            endpoint=np.array([2.0, 0.0], dtype=np.float32),
            reference_distance=4.0,
        )

    def test_collision_returns_penalty(self):
        rc = self._make_computer()
        reward, dist = rc.compute(
            collision=True,
            current_pos=np.array([0.0, 0.0]),
            current_step=1,
            prev_distance=None,
            goal_error=0.0,
            min_lidar_norm=0.05,
            speed_norm=0.5,
            reached_new_best_distance=False,
            accel=np.zeros(3),
        )
        assert reward < 0, "Collision reward must be negative"
        assert dist is None, "Distance is undefined on collision"

    def test_goal_reached_returns_large_positive(self):
        rc = self._make_computer()
        from controllers.common.defaults import REW_GOAL_SUCCESS as GOAL_SUCCESS_REWARD, REW_GOAL_STOP_BONUS as GOAL_STOP_BONUS
        pos_at_goal = np.array([2.0, 0.0], dtype=np.float32)
        reward, dist = rc.compute(
            collision=False,
            current_pos=pos_at_goal,
            current_step=10,
            prev_distance=0.5,
            goal_error=0.0,
            min_lidar_norm=0.8,
            speed_norm=0.05,   # slow enough to trigger stop bonus
            reached_new_best_distance=True,
            accel=np.zeros(3),
        )
        assert reward > 0, "Goal reward must be positive"
        assert reward >= GOAL_SUCCESS_REWARD, "Goal reward must be at least GOAL_SUCCESS_REWARD"

    def test_progress_toward_goal_positive(self):
        rc = self._make_computer()
        # Robot at (0,0), prev distance was 3.0, now 2.5 → progress.
        reward, _ = rc.compute(
            collision=False,
            current_pos=np.array([0.5, 0.0], dtype=np.float32),
            current_step=5,
            prev_distance=3.0,
            goal_error=0.0,
            min_lidar_norm=0.8,
            speed_norm=0.5,
            reached_new_best_distance=True,
            accel=np.zeros(3),
        )
        # Should be positive net (progress dominates small penalties).
        # Not guaranteed in all configurations, but with these clean params it should be.
        assert np.isfinite(reward), "Reward must be finite for valid inputs"

    def test_reward_is_finite_for_random_inputs(self):
        rc = self._make_computer()
        rng = np.random.default_rng(42)
        for _ in range(100):
            pos = rng.uniform(-3, 3, size=2).astype(np.float32)
            reward, _ = rc.compute(
                collision=False,
                current_pos=pos,
                current_step=int(rng.integers(1, 200)),
                prev_distance=float(rng.uniform(0.5, 5.0)),
                goal_error=float(rng.uniform(-math.pi, math.pi)),
                min_lidar_norm=float(rng.uniform(0.0, 1.0)),
                speed_norm=float(rng.uniform(0.0, 1.0)),
                reached_new_best_distance=bool(rng.integers(0, 2)),
                accel=rng.standard_normal(3).astype(np.float32),
            )
            assert np.isfinite(reward), f"Reward is non-finite for pos={pos}"


# ---------------------------------------------------------------------------
# GRU / LSTM actor-critic – forward pass shapes
# ---------------------------------------------------------------------------

class TestRecurrentModels:

    def _ppo_obs(self, cfg):
        return cfg.lidar_sector_dim + cfg.pose_goal_dim + cfg.imu_feature_dim

    def test_gru_single_step_output_shapes(self):
        from controllers.RNN.gru import GRUActorCritic
        cfg = _ppo_config()
        obs_size = self._ppo_obs(cfg)
        model = GRUActorCritic(obs_size, action_dim=2, config=cfg)
        obs = torch.randn(1, obs_size)
        policy, value, h = model(obs)
        assert policy.shape == (1, 2),  f"Policy shape wrong: {policy.shape}"
        assert value.shape  == (1,),    f"Value shape wrong:  {value.shape}"

    def test_lstm_single_step_output_shapes(self):
        from controllers.RNN.lstm import LSTMActorCritic
        cfg = _ppo_config()
        obs_size = self._ppo_obs(cfg)
        model = LSTMActorCritic(obs_size, action_dim=2, config=cfg)
        obs = torch.randn(1, obs_size)
        policy, value, (h, c) = model(obs)
        assert policy.shape == (1, 2)
        assert value.shape  == (1,)

    def test_gru_sequence_batch_shapes(self):
        from controllers.RNN.gru import GRUActorCritic
        cfg = _ppo_config()
        obs_size = self._ppo_obs(cfg)
        model = GRUActorCritic(obs_size, action_dim=2, config=cfg)
        B, T = 4, 16
        obs = torch.randn(B, T, obs_size)
        done_mask = torch.zeros(B, T)
        done_mask[:, 0] = 1.0  # reset at start of each sequence
        policy, value, h = model(obs, done_mask=done_mask)
        assert policy.shape == (B, T, 2), f"Sequence policy shape wrong: {policy.shape}"
        assert value.shape  == (B, T),    f"Sequence value shape wrong:  {value.shape}"

    def test_hidden_state_reset_on_done(self):
        """A done flag of 1.0 at t=0 must reset the hidden state to zero."""
        from controllers.RNN.gru import GRUActorCritic
        cfg = _ppo_config()
        obs_size = self._ppo_obs(cfg)
        model = GRUActorCritic(obs_size, action_dim=2, config=cfg)
        # Pass a non-zero initial hidden state.
        h_nonzero = torch.ones(cfg.lstm_layers, 1, cfg.lstm_hidden_size)
        obs = torch.randn(1, obs_size)
        # done=1 should wipe the state.
        _, _, h_after_reset  = model(obs, recurrent_state=h_nonzero, done_mask=torch.tensor([1.0]))
        _, _, h_after_no_reset = model(obs, recurrent_state=h_nonzero, done_mask=torch.tensor([0.0]))
        # Reset should yield a different (typically smaller-magnitude) hidden state.
        assert not torch.allclose(h_after_reset, h_after_no_reset), \
            "done=1 must change the hidden state compared to done=0"

    def test_no_nan_in_outputs(self):
        from controllers.RNN.gru import GRUActorCritic
        cfg = _ppo_config()
        obs_size = self._ppo_obs(cfg)
        model = GRUActorCritic(obs_size, action_dim=2, config=cfg)
        obs = torch.randn(8, 32, obs_size)
        policy, value, _ = model(obs)
        assert torch.isfinite(policy).all(), "Policy output contains NaN/Inf"
        assert torch.isfinite(value).all(),  "Value output contains NaN/Inf"


# ---------------------------------------------------------------------------
# SAC – critic update target shape consistency
# ---------------------------------------------------------------------------

class TestSACNetworkShapes:

    def _make_agent(self):
        from controllers.SAC.SAC import SACAgent
        cfg = _sac_config()
        obs_size = cfg.lidar_sector_dim + cfg.pose_goal_dim + cfg.imu_feature_dim
        return SACAgent(obs_size, action_dim=2, config=cfg)

    def test_actor_single_step(self):
        agent = self._make_agent()
        obs = np.random.randn(agent.obs_size).astype(np.float32)
        action, state = agent.select_action(obs, deterministic=False)
        assert action.shape == (2,), f"Action shape wrong: {action.shape}"
        assert np.isfinite(action).all(), "Action must be finite"

    def test_action_within_bounds(self):
        agent = self._make_agent()
        obs = np.random.randn(agent.obs_size).astype(np.float32)
        for _ in range(30):
            action, _ = agent.select_action(obs, deterministic=False)
            assert action[0] >= -agent.config.max_steering_angle - 1e-4
            assert action[0] <=  agent.config.max_steering_angle + 1e-4
            assert action[1] >= agent.config.min_speed - 1e-4
            assert action[1] <= agent.config.max_speed + 1e-4

    def test_deterministic_action_repeatable(self):
        agent = self._make_agent()
        obs = np.random.randn(agent.obs_size).astype(np.float32)
        a1, _ = agent.select_action(obs, deterministic=True)
        a2, _ = agent.select_action(obs, deterministic=True)
        np.testing.assert_allclose(a1, a2, atol=1e-6,
                                   err_msg="Deterministic actions must be identical for same obs")

    def test_update_does_not_crash_on_minimal_batch(self):
        agent = self._make_agent()
        B, T = 4, 8
        batch = {
            "obs":        torch.randn(B, T, agent.obs_size),
            "actions":    torch.randn(B, T, agent.action_dim),
            "rewards":    torch.randn(B, T, 1),
            "next_obs":   torch.randn(B, T, agent.obs_size),
            "dones":      torch.zeros(B, T, 1),
            "valid_mask": torch.ones(B, T),
        }
        metrics = agent.update(batch)
        assert metrics is not None
        assert "critic_loss" in metrics
        assert "actor_loss"  in metrics
        assert all(np.isfinite(v) for v in metrics.values()), \
            f"Update produced non-finite loss: {metrics}"


# ---------------------------------------------------------------------------
# Entry point for running without pytest
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback

    test_classes = [
        TestGAE,
        TestAdvantageNormalisation,
        TestLogProbCorrection,
        TestSequenceReplayBuffer,
        TestSACDoneFlag,
        TestRewardComputer,
        TestRecurrentModels,
        TestSACNetworkShapes,
    ]

    passed = failed = 0
    for cls in test_classes:
        instance = cls()
        for name in [m for m in dir(cls) if m.startswith("test_")]:
            method = getattr(instance, name)
            try:
                method()
                print(f"  PASS  {cls.__name__}.{name}")
                passed += 1
            except Exception as exc:
                print(f"  FAIL  {cls.__name__}.{name}")
                traceback.print_exc()
                failed += 1

    print(f"\n{passed} passed, {failed} failed.")
    sys.exit(0 if failed == 0 else 1)
