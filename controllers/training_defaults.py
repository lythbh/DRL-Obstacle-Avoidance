"""Shared training defaults used by PPO and SAC."""


class RecurrentDefaults:
    sequence_length = 16
    burn_in = 4
    sequence_stride = 8


class PPODefaults:
    episodes = 500
    update_every = 5
    epochs = 4
    batch_size = 64
    save_every = 100
    learning_rate = 1e-4
    entropy_coef = 0.02
    hidden_size = 128
    latent_size = 128
    lstm_hidden_size = 128
    lstm_layers = 1
    recurrent_cell = "gru"


class SACDefaults:
    episodes = 10000
    update_after_steps = 500
    updates_per_step = 2
    save_every = 100
    gamma = 0.99
    tau = 0.005
    actor_lr = 3e-4
    critic_lr = 3e-4
    alpha_lr = 3e-4
    initial_alpha = 0.05
    auto_entropy_tuning = True
    target_entropy_scale = 0.5
    hidden_size = 128
    recurrent_cell = "gru"
    recurrent_hidden_size = None
    recurrent_layers = 1
    log_std_min = -5.0
    log_std_max = 2.0
    replay_capacity = 1024
    replay_batch_size = 32
    min_replay_sequences = 64