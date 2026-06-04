"""
Entry point for the PPO controller in the Webots world. Its act as a wrapper for the PPO algorithm
as Webots needs the controller to be at the same folder hierarchy as the worlds.

LLM level: 0 - Written independently
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_project_root))

from controllers.PPO.PPO import train

if __name__ == "__main__":
    train()
