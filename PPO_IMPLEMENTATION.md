# PPO Implementation for Altino Obstacle Avoidance

## Overview

This implementation provides a complete **Proximal Policy Optimization (PPO)** training framework for the Altino car robot in Webots. The agent learns to navigate to a target coordinate while avoiding obstacles detected by LiDAR.

## Architecture

### 1. **Environment (`AltinoPPOEnv`)**

Bridges Webots simulation with PyTorch RL training.

#### Sensors:
- **LiDAR**: 12 rays for obstacle detection (normalized to [0, 1])
- **GPS**: Provides current position (x, y)

#### Actuators:
- **Left/Right Motors**: Velocity control for differential steering

#### State (Observation - 19 dimensions):
```
[lidar_0...lidar_11, gps_x, gps_y, goal_x, goal_y, prev_steering, prev_speed]
```

#### Actions (2 continuous values):
```
[steering_angle ∈ [-π/2, π/2], speed ∈ [0, 1.8 m/s]]
```

### 2. **Reward Function**

Carefully balanced to encourage:
- ✅ Goal reaching: +1.0
- ✅ Progress towards goal: +0.05 per unit closer
- ✅ Active movement: +0.01 when speed > 0
- ❌ Close obstacles: -0.1 to -0.5 (distance-dependent)
- ❌ Collisions: Automatic termination

### 3. **Neural Networks**

#### Policy Network (Actor)
```
Input(19) → Linear(256) → Tanh → Linear(256) → Tanh → Linear(256) → Tanh → Linear(4) → NormalParamExtractor
```
Outputs mean (μ) and standard deviation (σ) for each action dimension.

#### Value Network (Critic)
```
Input(19) → Linear(256) → Tanh → Linear(256) → Tanh → Linear(256) → Tanh → Linear(1)
```
Estimates state values for advantage computation.

## Training Process

### Key Parameters:

| Parameter | Value | Role |
|-----------|-------|------|
| `total_frames` | 50,000 | Total environment steps |
| `frames_per_batch` | 2,048 | Rollout length before update |
| `num_epochs` | 20 | Training passes |
| `lr` | 3e-4 | Learning rate |
| `clip_epsilon` | 0.2 | PPO clipping range ±20% |
| `gamma` | 0.99 | Discount factor (0.99 ≈ long-term focus) |
| `lmbda` | 0.95 | GAE smoothing (0.95 is typical) |

### Algorithm Flow:

1. **Collect Data**: Rollout policy in environment for `frames_per_batch` steps
2. **Compute Advantages**: GAE (Generalized Advantage Estimation)
3. **Update Networks**: 
   - Actor loss (PPO clip + entropy bonus)
   - Critic loss (value function)
   - Gradient clipping (max norm = 1.0)

## Usage

### Running Training in Webots

1. **Setup**:
   ```bash
   # Install/verify requirements
   pip install -r requirements.txt
   ```

2. **Start Training**:
   - Open your Webots world with the Altino robot
   - In Webots, create a **Robot Controller** pointing to `ppo_altino.py`
   - Run the simulation

3. **Monitoring**:
   - Training progress shown via tqdm progress bar
   - Models saved as `altino_policy.pt` and `altino_value.pt` after training

### Custom Configuration

Modify the hyperparameters in `ppo_altino.py` `__main__` section:

```python
# Training hyperparameters
TOTAL_FRAMES = 50_000        # More frames = longer training
FRAMES_PER_BATCH = 2048      # Larger = more stable but slower updates
NUM_EPOCHS = 20               # More epochs = more training per batch
NUM_CELLS = 256               # Network size (256-512 typical)
LR = 3e-4                     # Lower = more stable, slower convergence
CLIP_EPSILON = 0.2            # 0.1-0.3 typical range
GAMMA = 0.99                  # 0.95-0.99 for continuous control
LMBDA = 0.95                  # GAE parameter
ENTROPY_COEF = 0.01           # Higher = more exploration

# Goal position (modify based on your Webots world)
GOAL = np.array([2.5, -2.5])
```

## Key Implementation Details

### Motor Control (Differential Steering)

```python
left_speed  = speed * (1 - steering_angle / MAX_STEERING)
right_speed = speed * (1 + steering_angle / MAX_STEERING)
```
- Steering left: reduce left wheel, increase right wheel
- Steering right: increase left wheel, reduce right wheel

### Observation Normalization

- **LiDAR**: Normalized by max range (5m) → [0, 1]
- **GPS**: Normalized by workspace size (±5m) → [-1, 1]
- **Goal**: Normalized same as GPS
- All passed through `ObservationNorm` during training for stability

### Safety Features

- **LiDAR collision threshold**: 0.15m (immediate termination)
- **Close obstacle penalty**: 0.3-0.5m (rewards halved)
- **Motor velocity clamping**: All values kept in valid ranges
- **Gradient clipping**: Max norm 1.0 (prevents exploding gradients)

## Troubleshooting

### Issue: Training doesn't start
- ✓ Verify Webots robot has LiDAR and GPS devices named correctly
- ✓ Check `Robot()` can be imported in Webots controller context
- ✓ Ensure motor names match ("left wheel motor", "right wheel motor")

### Issue: Robot doesn't learn (rewards stay low)
- Increase `entropy_coef` (0.01 → 0.05) for more exploration
- Reduce `clip_epsilon` (0.2 → 0.1) for smoother learning
- Decrease `lr` (3e-4 → 1e-4) for stability
- Increase `total_frames` for more training data

### Issue: Robot crashes too often
- Increase obstacle penalty (currently -0.1 to -0.5)
- Modify `collision_threshold` to be more conservative
- Add "safety" reward for maintaining distance even away from obstacles

### Issue: Training is too slow
- Increase `frames_per_batch` (more data per update)
- Reduce `num_epochs` (fewer training passes)
- Reduce `total_frames` for testing (50K → 10K)

## Performance Metrics to Track

1. **Episode Reward**: Should increase over time
2. **Success Rate**: % of episodes reaching goal without collision
3. **Average Steps to Goal**: Should decrease (faster navigation)
4. **Min LiDAR Distance**: Should stay > 0.3m when avoiding obstacles

## Advanced Customization

### Change Action Space
Modify `_make_specs()` in `AltinoPPOEnv`:
```python
# Example: 3D action [steering, speed, brake]
action_spec = BoundedTensorSpec(
    shape=(3,),
    low=torch.tensor([-π/2, 0.0, 0.0]),
    high=torch.tensor([π/2, 1.8, 1.0]),
)
# Update network output to 6 values (3*2 for mean+std)
```

### Change Reward Function
Modify `_get_reward()` method:
```python
# Add smoothness penalty
turning_diff = abs(action[0] - self.prev_action[0])
reward -= 0.01 * turning_diff  # Smoother trajectories
```

### Custom Goals
```python
# Train for multiple goals
goals = [
    np.array([2.5, -2.5]),
    np.array([-2.5, 2.5]),
    np.array([3.0, 3.0]),
]
for goal in goals:
    train_ppo(goal=goal)
```

## Model Persistence

After training:
```python
# Load trained models for inference
policy = torch.load("altino_policy.pt")
value = torch.load("altino_value.pt")

# Run evaluation
episodes_rewards = run_inference(policy, num_episodes=10)
```

## References

- **PPO Paper**: Schulman et al. (2017) - Proximal Policy Optimization Algorithms
- **GAE**: Schulman et al. (2016) - High-Dimensional Continuous Control Using Generalized Advantage Estimation  
- **TorchRL**: Meta's reinforcement learning library documentation
- **Altino Robot**: Webots Altino model documentation

## Contact & Issues

For questions or modifications needed for your specific Webots setup, ensure:
- [ ] Webots simulation is running with Altino robot
- [ ] All sensors (LiDAR, GPS) are present and named correctly
- [ ] Controller script has proper file path configuration
- [ ] Python environment has all required packages installed

