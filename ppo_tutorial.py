import warnings
warnings.filterwarnings("ignore")
from torch import multiprocessing


from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (Compose, DoubleToFloat, ObservationNorm, StepCounter,
                          TransformedEnv)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm




"""
PPO is an on-policy method, so data is expensive
"""

"""
This setup uses a TensorDict which is a way for torchrl classes to communicate.
Basically values are updated inside a dictionary, and torchrl are fetching these for 
different operations. 
For example: Extracting mean and std that are viable for training is done by 
simply running NormalParamExtractor() which update loc and scale values in the TensorDict.
"""

# Hyperparameters
is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)
num_cells = 256  # number of cells in each layer i.e. output dim.
lr = 3e-4
max_grad_norm = 1.0

# Data collection parameters
frames_per_batch = 1000
# Complete training should have 1M frames
total_frames = 50_000

# PPO parameters
sub_batch_size = 64  # Mini batches for stable gradients, better generalization and multiple passes (training) over same data
num_epochs = 10  # Loops of the collected batch
clip_epsilon = 0.2 # Limit how much the new policy can differ from the old one (20% change)
gamma = 0.99 # Discount factor - how much the agent cares about future rewards. 0.99 for most continuous control problems
lmbda = 0.95 # GAE (Generalized Advantage Estimation) - balancing between high variance (MC) and bias (TD)
entropy_eps = 1e-2 # The larger the more exploration (0.01 is common for continuous actions)


# Environment and transforms that prepare the data for the policy
base_env = GymEnv("InvertedDoublePendulum-v5")

env = TransformedEnv(
    base_env,
    Compose(
        ObservationNorm(in_keys=["observation"]),
        DoubleToFloat(),
        StepCounter(),
    ),
)

# reduce_dim: along which dimension should mean/std be computed
# cat_dim: along which dimensions are samples concatenated
env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)

# rollout = env.rollout(3)
# print("rollout of three steps:", rollout)
# print("Shape of the rollout TensorDict:", rollout.batch_size)


# Policy
actor_network = nn.Sequential(
    nn.LazyLinear(num_cells, device=device), # Hidden layer
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device), # Hidden layer
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device), # Hidden layer
    nn.Tanh(),
    # 2 * env.action_spec.shape[-1] because each action dimension requires a mean 𝝁 and standard deviation 𝜎
    # So it returns a 2 x n vector for a n-dimensional action space
    nn.LazyLinear(2 * env.action_spec.shape[-1], device=device), # Output layer
    # Returns locaction (loc, mean 𝝁) and scale (standard deviation 𝜎) from actions of the last layer
    NormalParamExtractor(), # scale is positive (via softplus or similar)
    # Probability distribution:
    # X = loc + scale * Z,
    # where loc (mean) shifts left or right,
    # scale streches or shrinks it
    # Z is a standardized random variable
)

# Wrap policy in tensordict
policy_module = TensorDictModule(
    actor_network, in_keys=['observation'], out_keys=['loc', 'scale']
)

policy_module = ProbabilisticActor(
    module=policy_module,
    spec=env.action_spec,
    in_keys=['loc', 'scale'],
    distribution_class=TanhNormal,
    # Pass upper and lower bounds to the prob dist.
    # so training isn't destabilized and avoids incorrect gradients
    distribution_kwargs={
        'low': env.action_spec.space.low,
        'high': env.action_spec.space.high,
    },
    # log_prob is needed for weights
    return_log_prob=True,
)

# Value Network
critic_network = nn.Sequential(
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(1, device=device),
)

value_module = ValueOperator(
    module=critic_network,
    in_keys=['observation'],
)

policy_module(env.reset())
value_module(env.reset())

# Data collector
# Classes that execute three operations:
# 1. Reset an environment
# 2. Compute an action given the latest observation (state)
# 3. Execute a step in the environment, and repeat the last two steps until the 
# environment signals a stop, or reaches a done state
collector = SyncDataCollector(
    env,
    policy_module,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    split_trajs=False,
    device=device,
)

# Replay buffer
# Common in off-policy RLs, and in on-policy RLs they are refilled every time a batch
# of data is collected.
# NOT mandatory for PPO, but makes it easier to write the inner training loop in a 
# reproducible way
replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(max_size=frames_per_batch),
    sampler=SamplerWithoutReplacement(),
)

# Loss function
# TorchRL provides the loss function ClipPPOLoss for PPO loss, which requires some
# advantage estimation. Advantage is a value that reflects an expectancy over the return 
# value while dealing with the bias / variance tradeoff. 
advantage_module = GAE( # Generalized Advantage Estimtion
    gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True, device=device,
)

loss_module = ClipPPOLoss(
    actor_network=policy_module,
    critic_network=value_module,
    clip_epsilon=clip_epsilon,
    entropy_bonus=bool(entropy_eps),
    entropy_coef=entropy_eps,
    # these keys match by default but we set this for completeness
    critic_coef=1.0,
    loss_critic_type="smooth_l1",
)

optim = torch.optim.Adam(loss_module.parameters(), lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optim, total_frames // frames_per_batch, 0.0
)

# Training loop

# Collect data

#   Compute advantage

#       Loop over the collected to compute loss values

#       Back propagate

#       Optimize

#       Repeat

#   Repeat

# Repeat

logs = defaultdict(list) # Metrics: reward, lr
pbar = tqdm(total=total_frames) # Progress bar of collected frames
eval_str = "" # Evaluation string for display

# We iterate over the collector until it reaches the total number of frames it was
# designed to collect. Fetch batch --> Train on it --> Repeat
for i, tensordict_data in enumerate(collector): 
    # The TensorDict contains: Observation, action, reward, done, next observation, log_prob
    # we now have a batch of data to work with. Let's learn something from it.
    for _ in range(num_epochs):
        # We'll need an "advantage" signal to make PPO work.
        # We re-compute it at each epoch as its value depends on the value
        # network which is updated in the inner loop.
        advantage_module(tensordict_data)
        data_view = tensordict_data.reshape(-1) # Flatten from (batch_size, time_steps) to (batch_size * time_steps)
        replay_buffer.extend(data_view.cpu()) # Store in replay buffer

        # Mini-batch loop for stable gradient updates
        for _ in range(frames_per_batch // sub_batch_size): # Divide and round: 1000 // 64 = 15
            subdata = replay_buffer.sample(sub_batch_size)
            loss_vals = loss_module(subdata.to(device))
            loss_value = ( # L = L_ppo + L_value + L_entropy
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )

            # Optimization: 
            # Compute gradients
            loss_value.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
            # Update step
            optim.step()
            optim.zero_grad()

    # Logging and progress bar
    logs["reward"].append(tensordict_data["next", "reward"].mean().item())
    pbar.update(tensordict_data.numel())
    cum_reward_str = (
        f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
    )
    logs["step_count"].append(tensordict_data["step_count"].max().item())
    stepcount_str = f"step count (max): {logs['step_count'][-1]}"
    logs["lr"].append(optim.param_groups[0]["lr"])
    lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"
    if i % 10 == 0:
        # We evaluate the policy once every 10 batches of data.
        # Evaluation is rather simple: execute the policy without exploration
        # (take the expected value of the action distribution) for a given
        # number of steps (1000, which is our ``env`` horizon).
        # The ``rollout`` method of the ``env`` can take a policy as argument:
        # it will then execute this policy at each step.
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            # execute a rollout with the trained policy
            eval_rollout = env.rollout(1000, policy_module)
            logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
            logs["eval reward (sum)"].append(
                eval_rollout["next", "reward"].sum().item()
            )
            logs["eval step_count"].append(eval_rollout["step_count"].max().item())
            eval_str = (
                f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                f"eval step-count: {logs['eval step_count'][-1]}"
            )
            del eval_rollout
    pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))

    # Gradually decreases learning rate to stabilize late training, and helps convergence
    # Not mandatory, but useful
    scheduler.step()

plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.plot(logs["reward"])
plt.title("training rewards (average)")
plt.subplot(2, 2, 2)
plt.plot(logs["step_count"])
plt.title("Max step count (training)")
plt.subplot(2, 2, 3)
plt.plot(logs["eval reward (sum)"])
plt.title("Return (test)")
plt.subplot(2, 2, 4)
plt.plot(logs["eval step_count"])
plt.title("Max step count (test)")
plt.show()