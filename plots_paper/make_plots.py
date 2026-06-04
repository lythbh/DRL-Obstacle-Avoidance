"""Generate paper plots for the static PPO curriculum runs.

The script ignores moving-world runs, keeps only complete 10-stage static
curricula, and compares feedforward PPO, GRU-PPO, and LSTM-PPO on shared axes.
"""

from __future__ import annotations

import csv
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/drl-obstacle-matplotlib-cache")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


PLOTS_DIR = Path(__file__).resolve().parents[1] / "plots"
OUT_DIR = Path(__file__).resolve().parent
N_STAGES = 10
EPS_PER_STAGE = 500
TOTAL_EPISODES = N_STAGES * EPS_PER_STAGE
ROLLING_WINDOW = 50

MODEL_LABELS = {
    "none": "Feedforward PPO",
    "gru": "GRU-PPO",
    "lstm": "LSTM-PPO",
}

MODEL_COLORS = {
    "none": "#4C78A8",
    "gru": "#F58518",
    "lstm": "#54A24B",
}

RUN_RE = re.compile(
    r"^(?P<timestamp>\d{8}_\d{6})_"
    r"(?P<model>gru|lstm|none)_"
    r"seed(?P<seed>\d+)_"
    r"stage(?P<stage>\d{2})_"
)


def _float(value: str | None) -> float:
    if value is None or value == "":
        return np.nan
    try:
        return float(value)
    except ValueError:
        return np.nan


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.astype(float)
    out = np.full(values.shape, np.nan, dtype=float)
    half = window // 2
    for idx in range(len(values)):
        lo = max(0, idx - half)
        hi = min(len(values), idx + half + 1)
        chunk = values[lo:hi]
        if np.isfinite(chunk).any():
            out[idx] = np.nanmean(chunk)
    return out


def _mean_std(curves: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.vstack([curve.astype(float) for curve in curves])
    return np.nanmean(matrix, axis=0), np.nanstd(matrix, axis=0)


def _discover_complete_runs(plots_dir: Path) -> dict[str, list[dict[str, object]]]:
    grouped: dict[tuple[str, str, str], dict[int, Path]] = defaultdict(dict)
    for folder in sorted(plots_dir.iterdir()):
        if not folder.is_dir() or "moving" in folder.name:
            continue
        match = RUN_RE.match(folder.name)
        if not match:
            continue
        model = match.group("model")
        timestamp = match.group("timestamp")
        seed = match.group("seed")
        stage = int(match.group("stage"))
        grouped[(model, timestamp, seed)][stage] = folder

    complete: dict[str, list[dict[str, object]]] = defaultdict(list)
    required = set(range(1, N_STAGES + 1))
    for (model, timestamp, seed), stages in grouped.items():
        if set(stages) != required:
            continue
        if not all((stages[stage] / "ppo_episodes.csv").exists() for stage in required):
            continue
        complete[model].append({"timestamp": timestamp, "seed": seed, "stages": stages})

    for model in complete:
        complete[model].sort(key=lambda run: int(str(run["seed"])))
    return complete


def _episode_curves(run: dict[str, object]) -> dict[str, np.ndarray]:
    success = np.full(TOTAL_EPISODES, np.nan)
    reward = np.full(TOTAL_EPISODES, np.nan)
    min_dist = np.full(TOTAL_EPISODES, np.nan)

    stages: dict[int, Path] = run["stages"]  # type: ignore[assignment]
    for stage in range(1, N_STAGES + 1):
        rows = _read_csv(stages[stage] / "ppo_episodes.csv")[:EPS_PER_STAGE]
        offset = (stage - 1) * EPS_PER_STAGE
        for idx, row in enumerate(rows):
            pos = offset + idx
            success[pos] = _float(row.get("success"))
            reward[pos] = _float(row.get("reward"))
            min_dist[pos] = _float(row.get("min_dist"))

    return {"success": success, "reward": reward, "min_dist": min_dist}


def _loss_curve(run: dict[str, object], column: str = "critic_loss") -> np.ndarray:
    """Return a dense episode-indexed loss curve from sparse PPO update logs."""
    x_points: list[int] = []
    y_points: list[float] = []
    stages: dict[int, Path] = run["stages"]  # type: ignore[assignment]

    for stage in range(1, N_STAGES + 1):
        path = stages[stage] / "ppo_updates.csv"
        if not path.exists():
            continue
        by_episode: dict[int, list[float]] = defaultdict(list)
        for row in _read_csv(path):
            episode = int(_float(row.get("episode")))
            value = _float(row.get(column))
            if 1 <= episode <= EPS_PER_STAGE and np.isfinite(value):
                by_episode[episode].append(value)

        offset = (stage - 1) * EPS_PER_STAGE
        for episode in sorted(by_episode):
            x_points.append(offset + episode)
            y_points.append(float(np.mean(by_episode[episode])))

    dense_x = np.arange(1, TOTAL_EPISODES + 1)
    if len(x_points) < 2:
        return np.full(TOTAL_EPISODES, np.nan)
    y = _rolling_mean(np.asarray(y_points, dtype=float), ROLLING_WINDOW)
    return np.interp(dense_x, np.asarray(x_points, dtype=float), y)


def _draw_stage_markers(ax: plt.Axes) -> None:
    for stage in range(2, N_STAGES + 1):
        ax.axvline((stage - 1) * EPS_PER_STAGE, color="#555555", linestyle="--", linewidth=0.7, alpha=0.35)


def _style_axis(ax: plt.Axes, ylabel: str) -> None:
    ax.set_xlim(1, TOTAL_EPISODES)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    _draw_stage_markers(ax)


def plot_success_rate(runs_by_model: dict[str, list[dict[str, object]]]) -> None:
    fig, ax = plt.subplots(figsize=(12.8, 6.2))
    x = np.arange(1, TOTAL_EPISODES + 1)

    for model in ("none", "gru", "lstm"):
        runs = runs_by_model.get(model, [])
        curves = [_rolling_mean(_episode_curves(run)["success"], ROLLING_WINDOW) for run in runs]
        mean, std = _mean_std(curves)
        color = MODEL_COLORS[model]
        label = f"{MODEL_LABELS[model]} (n={len(runs)})"
        ax.plot(x, mean, color=color, linewidth=2.0, label=label)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.16, linewidth=0)

    _style_axis(ax, f"Rolling success rate ({ROLLING_WINDOW} episodes)")
    ax.set_ylim(-0.03, 1.03)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    fig.suptitle("Static Curriculum Training Performance: Goal Success Rate", y=0.98)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.84))
    fig.savefig(OUT_DIR / "static_success_rate_comparison.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT_DIR / "static_success_rate_comparison.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "static_success_rate_comparison.svg", bbox_inches="tight")
    plt.close(fig)


def plot_value_loss(runs_by_model: dict[str, list[dict[str, object]]]) -> None:
    fig, ax = plt.subplots(figsize=(12.8, 6.2))
    x = np.arange(1, TOTAL_EPISODES + 1)

    for model in ("none", "gru", "lstm"):
        runs = runs_by_model.get(model, [])
        curves = [_loss_curve(run, "critic_loss") for run in runs]
        mean, std = _mean_std(curves)
        color = MODEL_COLORS[model]
        label = f"{MODEL_LABELS[model]} (n={len(runs)})"
        ax.plot(x, mean, color=color, linewidth=2.0, label=label)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.16, linewidth=0)

    _style_axis(ax, "Critic loss")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
    ax.yaxis.set_minor_formatter(ticker.LogFormatter(base=10.0, labelOnlyBase=False))
    ax.tick_params(axis="y", which="minor", labelsize=8)
    fig.suptitle("Static Curriculum Training Loss", y=0.98)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.84))
    fig.savefig(OUT_DIR / "static_critic_loss_comparison.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT_DIR / "static_critic_loss_comparison.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "static_critic_loss_comparison.svg", bbox_inches="tight")
    plt.close(fig)


def write_summary(runs_by_model: dict[str, list[dict[str, object]]]) -> None:
    out_path = OUT_DIR / "static_run_summary.csv"
    fieldnames = [
        "model",
        "seed",
        "timestamp",
        "final_stage_success_rate",
        "final_stage_mean_reward",
        "final_stage_mean_min_dist",
        "overall_success_rate",
    ]
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for model in ("none", "gru", "lstm"):
            for run in runs_by_model.get(model, []):
                curves = _episode_curves(run)
                final_slice = slice((N_STAGES - 1) * EPS_PER_STAGE, TOTAL_EPISODES)
                writer.writerow({
                    "model": MODEL_LABELS[model],
                    "seed": run["seed"],
                    "timestamp": run["timestamp"],
                    "final_stage_success_rate": f"{np.nanmean(curves['success'][final_slice]):.4f}",
                    "final_stage_mean_reward": f"{np.nanmean(curves['reward'][final_slice]):.4f}",
                    "final_stage_mean_min_dist": f"{np.nanmean(curves['min_dist'][final_slice]):.4f}",
                    "overall_success_rate": f"{np.nanmean(curves['success']):.4f}",
                })


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs_by_model = _discover_complete_runs(PLOTS_DIR)
    for model in ("none", "gru", "lstm"):
        runs = runs_by_model.get(model, [])
        seeds = ", ".join(str(run["seed"]) for run in runs)
        print(f"{MODEL_LABELS[model]}: {len(runs)} complete static run(s), seeds=[{seeds}]")

    plot_success_rate(runs_by_model)
    plot_value_loss(runs_by_model)
    write_summary(runs_by_model)
    print(f"Saved plots and summary to {OUT_DIR}")


if __name__ == "__main__":
    main()