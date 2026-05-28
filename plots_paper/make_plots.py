"""
Generate paper-quality training curves: one plot per RNN type (gru, lstm, none).
Each plot shows reward over 5000 episodes (10 stages × 500 each), one line per seed.
Only includes runs where all 10 stages completed.

Also generates outcome-rate plots (goal vs collision) in the same style.
"""

from __future__ import annotations

import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

PLOTS_DIR   = Path(__file__).resolve().parents[1] / "plots"
OUT_DIR     = Path(__file__).resolve().parent
N_STAGES    = 10
EPS_PER_STAGE = 500

STAGE_LABELS = {
    1:  "Empty",
    2:  "1 obs",
    3:  "3 obs",
    4:  "5 obs",
    5:  "Goal shift +",
    6:  "Goal shift −",
    7:  "Goal offset +",
    8:  "Goal offset −",
    9:  "Dense",
    10: "Full",
}

SEED_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c",
    "#d62728", "#9467bd", "#8c564b",
]

SMOOTH_WINDOW = 20  # rolling mean half-width for readability


def parse_folders(plots_dir: Path):
    """Return dict: (model, run_key) -> {stage_num: Path}."""
    pattern = re.compile(
        r"^(\d{8}_\d{6})_"      # timestamp
        r"(gru|lstm|none)_"      # model
        r"(seed\d+)_"            # seed
        r"stage(\d{2})_"         # stage
    )
    runs: dict = defaultdict(dict)  # (model, timestamp, seed) -> {stage: path}
    for folder in sorted(plots_dir.iterdir()):
        if not folder.is_dir():
            continue
        m = pattern.match(folder.name)
        if not m:
            continue
        timestamp, model, seed, stage = m.group(1), m.group(2), m.group(3), int(m.group(4))
        key = (model, timestamp, seed)
        runs[key][stage] = folder
    return runs


def load_complete_runs(plots_dir: Path):
    """Return dict: model -> list of (seed_label, DataFrame with 5000 rows)."""
    raw = parse_folders(plots_dir)
    complete: dict[str, list] = defaultdict(list)

    for (model, timestamp, seed), stages in raw.items():
        if len(stages) < N_STAGES or not all(s in stages for s in range(1, N_STAGES + 1)):
            continue

        frames = []
        for stage_num in range(1, N_STAGES + 1):
            csv = stages[stage_num] / "ppo_episodes.csv"
            if not csv.exists():
                break
            df = pd.read_csv(csv)
            df = df.head(EPS_PER_STAGE)
            df["stage"] = stage_num
            df["global_episode"] = (stage_num - 1) * EPS_PER_STAGE + np.arange(1, len(df) + 1)
            frames.append(df)
        else:
            combined = pd.concat(frames, ignore_index=True)
            complete[model].append((seed, combined))

    # Sort each model's runs by seed label for consistent ordering
    for model in complete:
        complete[model].sort(key=lambda x: x[0])

    return complete


def smooth(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, center=True, min_periods=1).mean()


def _band(runs: list, column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (x, mean, std) across seeds after smoothing."""
    smoothed = np.stack([
        smooth(df[column].astype(float), SMOOTH_WINDOW).values
        for _, df in runs
    ])
    x = runs[0][1]["global_episode"].values
    return x, smoothed.mean(axis=0), smoothed.std(axis=0)


def make_plot(model: str, runs: list, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))

    x, mean, std = _band(runs, "reward")
    ax.fill_between(x, mean - std, mean + std, alpha=0.25, color="#1f77b4", label="±1 std")
    ax.plot(x, mean, color="#1f77b4", linewidth=1.8, label="Mean")

    _draw_stage_markers(ax)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Reward", fontsize=12)
    ax.set_title(f"Training curve — {model.upper()} ({len(runs)} seeds)", fontsize=13, fontweight="bold")
    ax.set_xlim(1, N_STAGES * EPS_PER_STAGE)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out_path = out_dir / f"training_curve_{model}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def _draw_stage_markers(ax: plt.Axes) -> None:
    """Draw vertical stage-boundary lines and stage-name labels."""
    for stage_num in range(2, N_STAGES + 1):
        ax.axvline(
            x=(stage_num - 1) * EPS_PER_STAGE,
            color="black", linewidth=0.8, linestyle="--", alpha=0.4,
        )
    for stage_num in range(1, N_STAGES + 1):
        x_mid = (stage_num - 1) * EPS_PER_STAGE + EPS_PER_STAGE / 2
        ax.text(
            x_mid, 0.01,
            STAGE_LABELS[stage_num],
            transform=ax.get_xaxis_transform(),
            ha="center", va="bottom",
            fontsize=7.5, color="#333333",
        )


def make_outcome_plot(model: str, runs: list, out_dir: Path) -> None:
    """Plot rolling goal-hit rate: mean across seeds with ±1 std band."""
    fig, ax = plt.subplots(figsize=(12, 5))

    x, mean, std = _band(runs, "success")
    ax.fill_between(x, mean - std, mean + std, alpha=0.25, color="#2ca02c", label="±1 std")
    ax.plot(x, mean, color="#2ca02c", linewidth=1.8, label="Mean")

    _draw_stage_markers(ax)
    ax.set_ylabel("Goal reached rate", fontsize=12)
    ax.set_ylim(-0.02, 1.02)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.set_xlim(1, N_STAGES * EPS_PER_STAGE)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_title(f"Goal reached rate — {model.upper()} ({len(runs)} seeds)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out_path = out_dir / f"outcome_rates_{model}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    complete = load_complete_runs(PLOTS_DIR)
    for model in ("gru", "lstm", "none"):
        runs = complete.get(model, [])
        if not runs:
            print(f"No complete runs found for model={model}, skipping.")
            continue
        print(f"{model.upper()}: {len(runs)} seed(s) — {[s for s, _ in runs]}")
        make_plot(model, runs, OUT_DIR)
        make_outcome_plot(model, runs, OUT_DIR)


if __name__ == "__main__":
    main()
