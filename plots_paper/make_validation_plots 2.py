"""Generate validation comparison plots for static and moving evaluation runs."""

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

ROOT_DIR = Path(__file__).resolve().parents[1]
PLOTS_DIR = ROOT_DIR / "plots"
OUT_DIR = Path(__file__).resolve().parent

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

STATIC_PATTERN = "_val_"
MOVING_PATTERN = "_val_"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value: str | None) -> float:
    if value is None or value == "":
        return np.nan
    try:
        return float(value)
    except ValueError:
        return np.nan


def _mean_std(curves: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.vstack([curve.astype(float) for curve in curves])
    return np.nanmean(matrix, axis=0), np.nanstd(matrix, axis=0)


def _collect_validation_data(base_dir: Path) -> dict[str, dict[int, dict[str, dict[str, float]]]]:
    """Collect evaluation values grouped by dataset, model, world, and seed."""
    data: dict[str, dict[int, dict[str, dict[str, float]]]] = {
        "static": defaultdict(lambda: defaultdict(dict)),
        "moving": defaultdict(lambda: defaultdict(dict)),
    }

    run_re = re.compile(r"^(?P<model>gru|lstm|none)_seed(?P<seed>\d+)_val_(?P<world>\d+)_")
    for dataset in ("static", "moving"):
        dataset_dir = base_dir / dataset
        if not dataset_dir.exists():
            continue

        for folder in sorted(dataset_dir.iterdir()):
            if not folder.is_dir():
                continue
            match = run_re.match(folder.name)
            if not match:
                continue
            model = match.group("model")
            seed = match.group("seed")
            world = int(match.group("world"))

            episode_csv = folder / "ppo_episodes.csv"
            if not episode_csv.exists():
                continue

            rows = _read_csv(episode_csv)
            if not rows:
                continue
            row = rows[0]
            success = _float(row.get("success"))
            critic_loss = _float(row.get("critic_loss"))
            data[dataset][world][model][seed] = {
                "success": success,
                "critic_loss": critic_loss,
            }

    return data


def _build_summary(data: dict[int, dict[str, dict[str, float]]]) -> tuple[list[int], dict[str, np.ndarray], dict[str, np.ndarray]]:
    worlds = sorted(data)
    x = np.arange(1, len(worlds) + 1)
    success_curves: dict[str, list[float]] = {model: [] for model in MODEL_LABELS}
    loss_curves: dict[str, list[float]] = {model: [] for model in MODEL_LABELS}

    for world in worlds:
        models = data[world]
        for model in MODEL_LABELS:
            seeds = models.get(model, {})
            if not seeds:
                success_curves[model].append(np.nan)
                loss_curves[model].append(np.nan)
            else:
                values = np.array([metrics["success"] for metrics in seeds.values()], dtype=float)
                losses = np.array([metrics["critic_loss"] for metrics in seeds.values()], dtype=float)
                success_curves[model].append(np.nanmean(values) if np.isfinite(values).any() else np.nan)
                loss_curves[model].append(np.nanmean(losses) if np.isfinite(losses).any() else np.nan)

    return x, {model: np.array(success_curves[model], dtype=float) for model in MODEL_LABELS}, {model: np.array(loss_curves[model], dtype=float) for model in MODEL_LABELS}


def _style_plot(ax: plt.Axes, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)


def _save_plot(fig: plt.Figure, basename: str) -> None:
    fig.savefig(OUT_DIR / f"{basename}.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{basename}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{basename}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_validation_graphs(dataset: str, worlds_data: dict[int, dict[str, dict[str, float]]]) -> None:
    x, success_curves, loss_curves = _build_summary(worlds_data)
    labels = [f"Episode {i}" for i in x]
    x_labels = x

    # Success rate
    fig, ax = plt.subplots(figsize=(10, 6))
    for model in MODEL_LABELS:
        y = success_curves[model]
        ax.plot(x, y, marker="o", linewidth=2.0, color=MODEL_COLORS[model], label=MODEL_LABELS[model])
    ax.set_xticks(x_labels)
    ax.set_xticklabels(labels)
    _style_plot(ax, "Validation episode / world", "Mean success rate")
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.set_title(f"Validation Success Rate: {dataset.capitalize()} Worlds")
    ax.legend(frameon=False)
    _save_plot(fig, f"val_{dataset}_success_rate_comparison")

    # Critic loss
    loss_values = np.concatenate([loss_curves[model] for model in MODEL_LABELS])
    if not np.isfinite(loss_values).any():
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(
            0.5,
            0.5,
            "No critic loss values are recorded in the validation logs.\n" +
            "Validation episodes only contain success/reward outcomes, not training loss.",
            ha="center",
            va="center",
            fontsize=14,
            wrap=True,
        )
        ax.axis("off")
        _save_plot(fig, f"val_{dataset}_critic_loss_comparison")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for model in MODEL_LABELS:
        y = loss_curves[model]
        ax.plot(x, y, marker="o", linewidth=2.0, color=MODEL_COLORS[model], label=MODEL_LABELS[model])
    ax.set_xticks(x_labels)
    ax.set_xticklabels(labels)
    _style_plot(ax, "Validation episode / world", "Mean critic loss")
    ax.set_title(f"Validation Critic Loss: {dataset.capitalize()} Worlds")

    max_values = np.array([
        np.nanmax(loss_curves[model]) if np.isfinite(loss_curves[model]).any() else np.nan
        for model in MODEL_LABELS
    ], dtype=float)
    max_loss = np.nanmax(max_values) if np.isfinite(max_values).any() else np.nan
    min_values = np.array([
        np.nanmin(loss_curves[model]) if np.isfinite(loss_curves[model]).any() else np.nan
        for model in MODEL_LABELS
    ], dtype=float)
    min_loss = np.nanmin(min_values) if np.isfinite(min_values).any() else np.nan
    if np.isfinite(max_loss) and max_loss > 0:
        ax.set_yscale("log")
    if np.isfinite(min_loss) and min_loss >= 0 and np.isfinite(max_loss) and max_loss <= 1:
        ax.set_ylim(0, 1)

    ax.legend(frameon=False)
    _save_plot(fig, f"val_{dataset}_critic_loss_comparison")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = _collect_validation_data(PLOTS_DIR)
    for dataset in ("static", "moving"):
        if not data[dataset]:
            print(f"No validation data found for {dataset}.")
            continue
        worlds = sorted(data[dataset])
        print(f"{dataset.capitalize()} worlds found: {worlds}")
        plot_validation_graphs(dataset, data[dataset])
    print(f"Saved validation plots to {OUT_DIR}")


if __name__ == "__main__":
    main()
