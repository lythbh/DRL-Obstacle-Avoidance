# Environment Setup
1. Create and activate a Python `3.10.19` environment.
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

# Repository Layout
- `run_model.py` runs inference with the trained PPO agent.
- `controllers/PPO/PPO.py` contains the PPO controller, Webots environment wrapper, and training loop.
- `controllers/RNN/lstm.py` contains the shared LSTM actor-critic used by PPO and future agents.
- `controllers/SLAM/` contains LiDAR, IMU, SLAM, and map preprocessing utilities.
- `worlds/Simple.wbt` is the Webots world used for training and evaluation.
- `plots/` and `controllers/PPO/slam_runs/` are generated outputs.

# Webots Setup
In Webots, open **Preferences** and point **Python command** to the Python executable in your environment.

# Controller Setup
Use the ALTINO robot in Webots with the PPO controller selected from the project tree. The current controller stack is split into:
- PPO logic in `controllers/PPO/`
- Shared recurrent policy code in `controllers/RNN/`
- Sensor processing in `controllers/SLAM/`

# World Setup
The repository currently uses `worlds/Simple.wbt`.

# Run
- Start the simulation from Webots.

