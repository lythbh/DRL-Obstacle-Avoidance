"""Reward modules for PPO controller."""

from .base import RewardComputer
from .penalties import calculate_clearance_penalty

__all__ = ["RewardComputer", "calculate_clearance_penalty"]
