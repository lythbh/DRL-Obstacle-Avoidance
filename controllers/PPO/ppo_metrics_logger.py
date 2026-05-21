"""PPO-specific structured CSV metrics logger."""

from controllers.common.metrics_logger import MetricsLogger


class PPOMetricsLogger(MetricsLogger):
    """Structured logger specialized for PPO training runs."""

    _HYPERPARAM_FIELDNAMES = MetricsLogger._HYPERPARAM_FIELDNAMES + ["value_loss_coef"]

    def __init__(self, run_folder: str) -> None:
        super().__init__(run_folder, algorithm="ppo")