"""Generate plots for the moving PPO curriculum runs.

Considers only folders with "moving" in their name and compares
Feedforward PPO, GRU-PPO and LSTM-PPO.
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
N_STAGES = 5
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

MOVING_RUN_RE = re.compile(
    r"moving_(?P<timestamp>\d{8}_\d{6})_step(?P<step>\d{2})_.*_(?P<model>gru|lstm|none)_seed(?P<seed>\d+)_stage(?P<stage>\d{2})_"
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
    # Group folders by (model, seed) then by timestamp -> step mapping.
    # For each (model, seed) pick the latest timestamp that has a complete set
    # of steps 1..N_STAGES.
    grouped: dict[tuple[str, str], dict[str, dict[int, Path]]] = defaultdict(lambda: defaultdict(dict))
    for folder in sorted(plots_dir.iterdir()):
        if not folder.is_dir() or "moving" not in folder.name:
            continue
        m = MOVING_RUN_RE.search(folder.name)
        if not m:
            continue
        model = m.group("model")
        timestamp = m.group("timestamp")
        seed = m.group("seed")
        step = int(m.group("step"))
        grouped[(model, seed)][timestamp][step] = folder

    complete: dict[str, list[dict[str, object]]] = defaultdict(list)
    required = set(range(1, N_STAGES + 1))
    for (model, seed), timestamps in grouped.items():
        # prefer latest timestamp that has all steps
        valid_ts = [ts for ts, steps in timestamps.items() if set(steps) == required and all((steps[s] / "ppo_episodes.csv").exists() for s in required)]
        if not valid_ts:
            continue
        chosen = max(valid_ts)
        steps = timestamps[chosen]
        complete[model].append({"timestamp": chosen, "seed": seed, "steps": steps})

    # ensure seeds sorted numerically and limit to seeds 00..05 if present
    for model in complete:
        complete[model].sort(key=lambda run: int(str(run["seed"])))
        # filter to unique seeds 0-5 if more copies exist (take first occurrence per seed)
        seen = set()
        filtered = []
        for run in complete[model]:
            s = run["seed"]
            if s in seen:
                continue
            seen.add(s)
            filtered.append(run)
        complete[model] = filtered
    return complete


def _episode_curves(run: dict[str, object]) -> dict[str, np.ndarray]:
    success = np.full(TOTAL_EPISODES, np.nan)
    reward = np.full(TOTAL_EPISODES, np.nan)
    min_dist = np.full(TOTAL_EPISODES, np.nan)

    steps: dict[int, Path] = run["steps"]  # type: ignore[assignment]
    for step in range(1, N_STAGES + 1):
        rows = _read_csv(steps[step] / "ppo_episodes.csv")[:EPS_PER_STAGE]
        offset = (step - 1) * EPS_PER_STAGE
        for idx, row in enumerate(rows):
            pos = offset + idx
            success[pos] = _float(row.get("success"))
            reward[pos] = _float(row.get("reward"))
            min_dist[pos] = _float(row.get("min_dist"))

    return {"success": success, "reward": reward, "min_dist": min_dist}


def _loss_curve(run: dict[str, object], column: str = "critic_loss") -> np.ndarray:
    """Return a dense loss curve for the concatenated N_STAGES run."""
    x_points: list[int] = []
    y_points: list[float] = []
    steps: dict[int, Path] = run["steps"]  # type: ignore[assignment]

    for step in range(1, N_STAGES + 1):
        path = steps[step] / "ppo_updates.csv"
        if not path.exists():
            continue
        by_episode: dict[int, list[float]] = defaultdict(list)
        for row in _read_csv(path):
            episode = int(_float(row.get("episode")))
            value = _float(row.get(column))
            if 1 <= episode <= EPS_PER_STAGE and np.isfinite(value):
                by_episode[episode].append(value)

        offset = (step - 1) * EPS_PER_STAGE
        for episode in sorted(by_episode):
            x_points.append(offset + episode)
            y_points.append(float(np.mean(by_episode[episode])))

    dense_x = np.arange(1, TOTAL_EPISODES + 1)
    if len(x_points) < 2:
        return np.full(TOTAL_EPISODES, np.nan)
    y = _rolling_mean(np.asarray(y_points, dtype=float), ROLLING_WINDOW)
    return np.interp(dense_x, np.asarray(x_points, dtype=float), y)


def _style_axis(ax: plt.Axes, ylabel: str, total_episodes: int) -> None:
    ax.set_xlim(1, total_episodes)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)


def plot_success_rate(runs_by_model: dict[str, list[dict[str, object]]]) -> None:
    # build episode curves and determine max length
    curves_by_model: dict[str, list[np.ndarray]] = {}
    max_len = 0
    for model in ("none", "gru", "lstm"):
        runs = runs_by_model.get(model, [])
        curves = [ _episode_curves(run)["success"] for run in runs ]
        if curves:
            max_len = max(max_len, max(len(c) for c in curves))
        curves_by_model[model] = curves

    # use fixed total episodes across the 5 moving worlds
    max_len = TOTAL_EPISODES
    fig, ax = plt.subplots(figsize=(12.8, 6.2))
    x = np.arange(1, max_len + 1)

    for model in ("none", "gru", "lstm"):
        curves = curves_by_model.get(model, [])
        # pad to max_len and compute rolling mean
        padded = [ np.pad(_rolling_mean(c, ROLLING_WINDOW), (0, max_len - len(c)), constant_values=np.nan) for c in curves ]
        if not padded:
            mean = np.full(max_len, np.nan)
            std = np.full(max_len, np.nan)
        else:
            mean, std = _mean_std(padded)

        color = MODEL_COLORS[model]
        label = f"{MODEL_LABELS[model]} (n={len(padded)})"
        ax.plot(x, mean, color=color, linewidth=2.0, label=label)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.16, linewidth=0)

    _style_axis(ax, f"Rolling success rate ({ROLLING_WINDOW} episodes)", max_len)
    ax.set_ylim(-0.03, 1.03)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    fig.suptitle("Moving Curriculum Training Performance: Goal Success Rate", y=0.98)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=3, frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.84))
    fig.savefig(OUT_DIR / "moving_success_rate_comparison.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT_DIR / "moving_success_rate_comparison.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "moving_success_rate_comparison.svg", bbox_inches="tight")
    plt.close(fig)


def plot_value_loss(runs_by_model: dict[str, list[dict[str, object]]]) -> None:
    max_len = TOTAL_EPISODES
    fig, ax = plt.subplots(figsize=(12.8, 6.2))
    x = np.arange(1, max_len + 1)

    for model in ("none", "gru", "lstm"):
        runs = runs_by_model.get(model, [])
        curves = [ _loss_curve(run, "critic_loss") for run in runs ]
        if not curves:
            mean = np.full(max_len, np.nan)
            std = np.full(max_len, np.nan)
        else:
            mean, std = _mean_std(curves)

        color = MODEL_COLORS[model]
        label = f"{MODEL_LABELS[model]} (n={len(runs)})"
        ax.plot(x, mean, color=color, linewidth=2.0, label=label)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.16, linewidth=0)

    _style_axis(ax, "Critic loss", max_len)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
    ax.yaxis.set_minor_formatter(ticker.LogFormatter(base=10.0, labelOnlyBase=False))
    ax.tick_params(axis="y", which="minor", labelsize=8)
    fig.suptitle("Moving Curriculum Training Loss", y=0.98)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=3, frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.84))
    fig.savefig(OUT_DIR / "moving_critic_loss_comparison.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT_DIR / "moving_critic_loss_comparison.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "moving_critic_loss_comparison.svg", bbox_inches="tight")
    plt.close(fig)


def write_summary(runs_by_model: dict[str, list[dict[str, object]]]) -> None:
    out_path = OUT_DIR / "moving_run_summary.csv"
    fieldnames = [
        "model",
        "seed",
        "timestamp",
        "final_success_rate",
        "final_mean_reward",
        "final_mean_min_dist",
        "overall_success_rate",
    ]
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for model in ("none", "gru", "lstm"):
            for run in runs_by_model.get(model, []):
                curves = _episode_curves(run)
                n = len(curves["success"])
                final_slice = slice(max(0, n - EPS_PER_STAGE), n)
                writer.writerow({
                    "model": MODEL_LABELS[model],
                    "seed": run["seed"],
                    "timestamp": run["timestamp"],
                    "final_success_rate": f"{np.nanmean(curves['success'][final_slice]):.4f}",
                    "final_mean_reward": f"{np.nanmean(curves['reward'][final_slice]):.4f}",
                    "final_mean_min_dist": f"{np.nanmean(curves['min_dist'][final_slice]):.4f}",
                    "overall_success_rate": f"{np.nanmean(curves['success']):.4f}",
                })


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs_by_model = _discover_complete_runs(PLOTS_DIR)
    for model in ("none", "gru", "lstm"):
        runs = runs_by_model.get(model, [])
        seeds = ", ".join(str(run["seed"]) for run in runs)
        print(f"{MODEL_LABELS[model]}: {len(runs)} moving run(s), seeds=[{seeds}]")

    plot_success_rate(runs_by_model)
    plot_value_loss(runs_by_model)
    write_summary(runs_by_model)
    print(f"Saved moving plots and summary to {OUT_DIR}")


if __name__ == "__main__":
    main()
