# Environment Setup
1. Create and activate a Python `3.10.19` environment.
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

# Repository Layout
- `run_model.py` runs inference with the trained PPO or SAC agent.
- `controllers/PPO/PPO.py` contains the PPO controller and training loop.
- `controllers/SAC/SAC.py` contains the SAC controller and training loop.
- `controllers/RNN/lstm.py` contains the shared recurrent actor-critic code used by PPO and SAC.
- `controllers/SLAM/` contains LiDAR, IMU, SLAM, and map preprocessing utilities.
- `worlds/Simple.wbt` is the SAC world.
- `worlds/ObstacleCourse.wbt` is the PPO world.
- `plots/` and the controller folders store generated outputs and checkpoints.

# Webots Setup
In Webots, open **Preferences** and point **Python command** to the Python executable in your environment.

# Controller Setup
Use the ALTINO robot in Webots with the controller selected in the world file. The current controller stack is split into:
- PPO logic in `controllers/PPO/`
- SAC logic in `controllers/SAC/`
- Shared recurrent policy code in `controllers/RNN/`
- Sensor processing in `controllers/SLAM/`

# World Setup
`worlds/Simple.wbt` uses the SAC controller.
`worlds/ObstacleCourse.wbt` uses the PPO controller.

# Run
- Start the simulation from Webots.

