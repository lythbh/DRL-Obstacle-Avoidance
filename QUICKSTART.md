# Quick Start Guide - Altino PPO Training

## 📋 Prerequisites

### System Requirements
- **Webots Simulation**: Must be installed and running
- **Python 3.8+**: Ensure you have Python 3.8 or later
- **GPU** (optional but recommended): CUDA-capable GPU will speed up training

### Environment Setup

```bash
# 1. Navigate to project directory
cd /Users/lyth/Documents/Robotikk/Master/2.\ Semester/FYS5429\ Avansert\ maskinlæring\ og\ dataanalyse\ for\ fysiske\ fag/DRL-Obstacle-Avoidance

# 2. Install/verify dependencies
pip install -r requirements.txt

# 3. Verify PyTorch installation (should show CUDA availability)
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 🚀 Running Training

### Method 1: Direct Controller in Webots (Recommended)

1. **Open Webots**
   - Create or load your world with the Altino robot
   - Ensure the robot has:
     - **LiDAR sensor** (named "lidar")
     - **GPS sensor** (named "gps")
     - **Left wheel motor** (named "left wheel motor")
     - **Right wheel motor** (named "right wheel motor")

2. **Set Controller Script**
   - Right-click on Altino robot → Properties
   - Set "controller" to `ppo_altino`
   - Check "Enable physics simulation" in world settings

3. **Start Simulation**
   - Click ▶️ (Play) in Webots
   - Training will automatically begin
   - Monitor progress in console output

4. **Monitor Output**
   ```
   Training on device: cuda/cpu
   Creating environment...
   Initializing observation normalization...
   Creating networks...
   Starting training...
   Epoch 1/20  [████████--] 40%
   ```

### Method 2: Command Line (Testing/Standalone)

```bash
# Test without Webots (dummy environment)
python ppo_altino.py

# This will attempt to connect to Webots or fail gracefully
# Useful for checking configuration before running full training
```

## 📊 Monitoring Training

During training, you'll see:

```
Starting training...
Epoch 1/20: Loss: 0.2345
Epoch 2/20: Loss: 0.1876
...
Training complete!
Models saved!
```

### Training Artifacts Generated
- `altino_policy.pt` - Trained policy network weights
- `altino_value.pt` - Trained value network weights

## 🎯 Key Configuration Points

### Before Training - Modify These in `ppo_altino.py`

```python
# Line ~750: Hyperparameters
TOTAL_FRAMES = 50_000      # Total training steps (start with 50k for testing)
FRAMES_PER_BATCH = 2048    # Batch size
NUM_EPOCHS = 20             # Training epochs
NUM_CELLS = 256             # Network hidden size

# Line ~760: Goal position (IMPORTANT - match your Webots world!)
GOAL = np.array([2.5, -2.5])  # Modify based on your target location
```

### During Training - Monitor In Webots

- Watch robot's behavior in simulation
- Check that LiDAR readings are non-zero
- Verify robot attempts to reach goal
- Ensure no immediate crashes

## ✅ Verification Checklist

- [ ] Webots world is open with Altino robot
- [ ] Robot has LiDAR sensor (provides distance readings)
- [ ] Robot has GPS sensor (provides position)
- [ ] Motors are configured for velocity control
- [ ] `ppo_altino.py` is set as the robot controller
- [ ] Goal coordinates match your Webots world layout
- [ ] Python requirements installed (`pip install -r requirements.txt`)

## 🔍 Troubleshooting Quick

| Issue | Solution |
|-------|----------|
| **"Warning: Webots controller not available"** | Make sure you're running inside Webots as a controller |
| **"Could not initialize all devices"** | Check sensor names: "lidar", "gps", "left wheel motor", "right wheel motor" |
| **Robot doesn't move** | Verify motor setup - should accept velocity commands (not position) |
| **Training crashes immediately** | Check observation/action specs match your sensor/motor setup |
| **GPU not detected** | CUDA might not be properly installed - falls back to CPU (slower but works) |

## 📈 Expected Training Progression

### Early Training (Epoch 1-5)
- Robot behavior random
- Reward negative (hitting obstacles)
- Gradually learns to avoid immediate dangers

### Mid Training (Epoch 6-15)
- Robot learns obstacle avoidance reasonably
- Starting to navigate towards goal
- Reward becomes positive and stable

### Late Training (Epoch 16-20)
- Efficient goal navigation
- Smooth trajectories
- Consistent success on repeated trials

## 🧪 After Training - Testing Your Model

### Quick Evaluation
```bash
python inference.py --policy altino_policy.pt --episodes 5
```

### Output
```
======================================================================
EVALUATION SUMMARY
======================================================================
Mean Reward:     3.2456 ± 1.2345
Success Rate:    80.0%
Mean Steps:      150.3
======================================================================
```

### Detailed Visualization
```bash
python inference.py --policy altino_policy.pt --visualize
```

## 💡 Tips for Success

1. **Start Small**: Train for just 10K frames first to verify setup works
2. **Monitor LiDAR**: Add manual debug prints if robot hits obstacles repeatedly
3. **Goal Placement**: Place goal far enough from obstacles to be learnable
4. **Iterate**: First training often reveals configuration issues - iterate quickly
5. **Save Progress**: Keep good models by renaming:
   ```bash
   cp altino_policy.pt altino_policy_v1.pt
   cp altino_value.pt altino_value_v1.pt
   ```

## 🔧 Common Customizations

### Change Robot Maximum Speed
In `ppo_altino.py`, modify:
```python
self.MAX_SPEED = 1.8  # Change from 1.8 to your preferred max
```

### Modify Goal
```python
GOAL = np.array([X, Y])  # Set to your target coordinates
```

### Adjust Difficulty
- **Easier**: Reduce number of obstacles in Webots
- **Harder**: Add more obstacles, increase max speed, reduce LiDAR range

### Fine-tune Learning Rate
```python
LR = 3e-4  # Lower (1e-4) = slower, more stable
           # Higher (1e-3) = faster, but less stable
```

## 📚 Learning Resources

### Understanding the Code
1. Read `PPO_IMPLEMENTATION.md` for detailed architecture
2. Check comments in `ppo_altino.py` for algorithm details
3. Review `ppo_tutorial.py` for similar PPO implementation

### PPO Algorithm
- Original paper: "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
- Key concept: Trust region optimization with clipping

### TorchRL Documentation
- Main docs: https://pytorch.org/rl/
- API Reference: https://pytorch.org/rl/stable/reference.html

## ❓ Need Help?

1. **Check error messages**: Python/CUDA errors usually indicate exact problems
2. **Verify Webots setup**: Ensure sensors/motors are properly configured
3. **Test incrementally**: Run inference after just a few epochs to catch issues early
4. **Review PPO_IMPLEMENTATION.md**: Contains troubleshooting section

## Next Steps After Training

1. **Evaluate model**: Use `inference.py` to test quality
2. **Adjust parameters**: If success rate is low, modify hyperparameters
3. **Train longer**: Initial 20 epochs might be just beginning - try 50+ for production
4. **Test in variants**: Deploy to different goal positions or obstacle layouts

---

**Happy Training! 🤖** 

You're now ready to train a PPO agent for Altino obstacle avoidance. Start with the default configuration and iterate from there.
