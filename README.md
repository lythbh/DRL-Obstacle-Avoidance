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

# Pipeline Summary
This repository uses the following Webots training and inference flow:

- `worlds/ObstacleCourse.wbt` runs the PPO controller in `controllers/PPO/PPO.py`.
- `worlds/Simple.wbt` runs the SAC controller in `controllers/SAC/SAC.py`.
- Shared recurrent models live in `controllers/RNN/`.
- Webots wrapper, sensors, reward, and SLAM preprocessing live in `controllers/Webots/` and `controllers/SLAM/`.
- `run_model.py` loads the matching checkpoint from the relevant controller folder and runs deterministic inference.

The main runtime behavior is:

1. Webots starts the selected controller folder, either `PPO` or `SAC`.
2. The controller initializes a global Webots `Supervisor`.
3. `WebotsEnv` builds the ALTINO driver, motor controller, sensors, SLAM processor, reward computer, and observation layout.
4. Observations default to 33 features: 16 LiDAR sector minima, 7 pose and goal-direction features, and 10 IMU features.
5. PPO trains on complete recurrent episode trajectories and writes `best_model.pth` and `final_model.pth`.
6. SAC trains with replay-buffer updates and writes `best_model.pth` and `final_model.pth`.
7. `run_model.py` loads the selected checkpoint from the matching controller folder and runs deterministic inference.

Recent verified changes:

- PPO checkpoints are saved to `controllers/PPO/` regardless of the process working directory.
- SAC checkpoints are saved to `controllers/SAC/` regardless of the process working directory.
- Checkpoint loading uses `weights_only=False` explicitly so locally generated checkpoints remain loadable across PyTorch versions.

Verified locally with the `FYS5429` environment:

- Parsed the project Python files without writing `.pyc` files.
- Imported the project modules with a mocked Webots `controller` module.
- Loaded existing checkpoints in `controllers/PPO/` and `controllers/SAC/`.
- Ran PPO and SAC smoke tests covering reset/step loops, recurrent action selection, PPO trajectory updates, SAC replay-buffer updates, checkpoint serialization, and cleanup.

Environment notes:

- The working local environment is `FYS5429` with Python `3.10.19` and Torch installed.
- The default shell Python is `3.13.10` and does not have Torch installed.
- In Webots, point the Python command to `C:\Users\ErikH\miniconda3\envs\FYS5429\python.exe`.

Remaining risk before a long run:

- A real Webots physics smoke test still needs to be run from the Webots GUI because the local shell does not provide the actual Webots `controller` runtime.

