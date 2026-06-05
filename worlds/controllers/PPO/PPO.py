import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_project_root))

from controllers.PPO.PPO import evaluate

evaluate()
