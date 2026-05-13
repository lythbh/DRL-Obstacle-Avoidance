"""CSV metrics logger shared by PPO, SAC, and any future controllers.

Each training run writes one CSV file to the run folder so that learning
curves can be plotted and compared without re-running the simulation.

Usage::

    logger = MetricsLogger(run_folder, algorithm="ppo")
    logger.log(episode=1, reward=-12.3, end_reason="collision", ...)
    logger.close()
"""

import csv
import os
from typing import Any, Dict, Optional


class MetricsLogger:
    """Appends one row per episode to a CSV file in *run_folder*."""

    def __init__(self, run_folder: str, algorithm: str) -> None:
        self._path = os.path.join(run_folder, f"{algorithm}_metrics.csv")
        self._file = open(self._path, "w", newline="", buffering=1)
        self._writer: Optional[csv.DictWriter] = None
        self._fieldnames: Optional[list] = None

    def log(self, **kwargs: Any) -> None:
        """Write one row.  Columns are inferred from the first call."""
        if self._writer is None:
            self._fieldnames = list(kwargs.keys())
            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames,
                                          extrasaction="ignore")
            self._writer.writeheader()
        self._writer.writerow(kwargs)

    def close(self) -> None:
        self._file.flush()
        self._file.close()

    @property
    def path(self) -> str:
        return self._path
