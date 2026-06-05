"""Generate comprehensive test-result plots.

Plots produced (all prefixed test_):
  test_static_success_bar.pdf      -- grouped bar: success rate per world × arch
  test_static_reward_bar.pdf       -- grouped bar: mean reward per world × arch (successes only)
  test_static_steps_bar.pdf        -- grouped bar: mean steps per world × arch (successes only)
  test_static_heatmap.pdf          -- heatmap: success rate, seed × world, one panel per arch
  test_moving_success_bar.pdf      -- grouped bar: moving world success per arch
  test_moving_per_seed.pdf         -- strip + mean: per-seed success by world, shows bimodal
  test_moving_heatmap.pdf          -- heatmap: success rate, seed × world, one panel per arch
  test_moving_min_dist_bar.pdf     -- mean min obstacle distance per moving world × arch
  test_summary_bar.pdf             -- side-by-side static vs moving grouped bar

All "validation" labels renamed to "test".
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import os
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/drl-obstacle-matplotlib-cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
OUT  = Path(__file__).resolve().parent

MODEL_LABELS = {"none": "Feedforward", "gru": "GRU", "lstm": "LSTM"}
MODEL_COLORS = {"none": "#4C78A8", "gru": "#F58518", "lstm": "#54A24B"}

STATIC_WORLDS = {
    1: "Empty\n(centre)", 2: "Empty\n(offset)",
    3: "Sparse A",        4: "Sparse B",        5: "Dense",
}
MOVING_WORLDS = {
    6: "One moving\nobstacle", 7: "All moving\nobstacles",
    8: "Moving\ngoal",         9: "Offset\nmoving goal",
}

EVAL_RE = re.compile(
    r"\[EVAL\]\[PPO\] ep=(\d+)/\d+ r=\s*(-?[\d.]+) steps=\s*(\d+) "
    r"success=(\d) touch=(\d) min_d=\s*([\d.]+) avg_spd=([\d.]+)m/s end=(\w+)"
)
VAL_HDR = re.compile(
    r"\[VALIDATION\] (?P<phase>static|moving).*world=val_(?P<w>\d+)_"
)

N_SEEDS = 6


# ── parse all logs ────────────────────────────────────────────────────────────

def parse_all_logs():
    """
    Returns nested dict:
      phase -> arch -> seed(int) -> world(int) -> list[dict]
    Each dict has keys: success, reward, steps, min_dist, avg_spd, end
    """
    data = {
        "static": defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
        "moving": defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
    }
    for arch in ("gru", "lstm", "none"):
        for seed in range(N_SEEDS):
            path = LOGS / f"validation_{arch}_seed{seed}.log"
            if not path.exists():
                continue
            current_phase, current_w = None, None
            for line in path.read_text().splitlines():
                hm = VAL_HDR.search(line)
                if hm:
                    current_phase = hm["phase"]
                    current_w     = int(hm["w"])
                    continue
                em = EVAL_RE.search(line)
                if em and current_w is not None:
                    data[current_phase][arch][seed][current_w].append({
                        "success":  int(em.group(4)),
                        "reward":   float(em.group(2)),
                        "steps":    int(em.group(3)),
                        "min_dist": float(em.group(6)),
                        "avg_spd":  float(em.group(7)),
                        "end":      em.group(8),
                    })
    return data


# ── aggregation helpers ───────────────────────────────────────────────────────

def seed_world_metric(data_phase, arch, world, key, filter_success=False):
    """Return list of per-seed means for a given metric/world/arch."""
    out = []
    for seed in range(N_SEEDS):
        eps = data_phase[arch][seed].get(world, [])
        if filter_success:
            eps = [e for e in eps if e["success"] == 1]
        if not eps:
            out.append(np.nan)
        else:
            out.append(np.mean([e[key] for e in eps]))
    return np.array(out, dtype=float)


def world_mean_std(data_phase, arch, world, key, filter_success=False):
    vals = seed_world_metric(data_phase, arch, world, key, filter_success)
    finite = vals[np.isfinite(vals)]
    if len(finite) == 0:
        return np.nan, np.nan
    return float(np.mean(finite)), float(np.std(finite))


# ── shared style ──────────────────────────────────────────────────────────────

def _grouped_bars(ax, world_ids, world_lbls, means_dict, stds_dict,
                  models=("none", "gru", "lstm")):
    n_w, n_m = len(world_ids), len(models)
    width, gap = 0.22, 0.04
    gw = n_m * width + (n_m - 1) * gap
    positions = np.arange(n_w)
    for mi, model in enumerate(models):
        offsets = positions + mi * (width + gap) - gw / 2 + width / 2
        means = [means_dict[model][wi] for wi in world_ids]
        stds  = [stds_dict[model][wi]  for wi in world_ids]
        ax.bar(offsets, means, width=width, color=MODEL_COLORS[model],
               alpha=0.88, label=MODEL_LABELS[model],
               yerr=stds, capsize=3.5, error_kw={"lw": 1.1})
    ax.set_xticks(positions)
    ax.set_xticklabels(world_lbls, fontsize=9.5)
    ax.legend(frameon=False, fontsize=10)
    ax.grid(axis="y", alpha=0.25, lw=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


# ── static world plots ────────────────────────────────────────────────────────

def plot_static_success_bar(data):
    world_ids  = sorted(STATIC_WORLDS)
    world_lbls = [STATIC_WORLDS[w] for w in world_ids]
    means_d, stds_d = {}, {}
    for model in ("none", "gru", "lstm"):
        means_d[model] = {w: world_mean_std(data["static"], model, w, "success")[0]
                          for w in world_ids}
        stds_d[model]  = {w: world_mean_std(data["static"], model, w, "success")[1]
                          for w in world_ids}
    fig, ax = plt.subplots(figsize=(9, 4.8))
    _grouped_bars(ax, world_ids, world_lbls, means_d, stds_d)
    ax.set_ylim(0, 1.22)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.set_ylabel("Success rate  (9 episodes × 6 seeds)", fontsize=11)
    ax.set_xlabel("Test world", fontsize=11)
    # annotate bars
    n_m = 3
    width, gap = 0.22, 0.04
    gw = n_m * width + (n_m - 1) * gap
    for mi, model in enumerate(("none", "gru", "lstm")):
        for wi, w in enumerate(world_ids):
            m = means_d[model][w]
            if np.isfinite(m):
                x = wi + mi * (width + gap) - gw / 2 + width / 2
                ax.text(x, m + 0.03, f"{m:.0%}", ha="center", va="bottom",
                        fontsize=7, fontweight="bold")
    fig.tight_layout()
    _save(fig, "test_static_success_bar")


def plot_static_reward_bar(data):
    world_ids  = sorted(STATIC_WORLDS)
    world_lbls = [STATIC_WORLDS[w] for w in world_ids]
    means_d, stds_d = {}, {}
    for model in ("none", "gru", "lstm"):
        means_d[model] = {w: world_mean_std(data["static"], model, w, "reward", filter_success=True)[0]
                          for w in world_ids}
        stds_d[model]  = {w: world_mean_std(data["static"], model, w, "reward", filter_success=True)[1]
                          for w in world_ids}
    fig, ax = plt.subplots(figsize=(9, 4.8))
    _grouped_bars(ax, world_ids, world_lbls, means_d, stds_d)
    ax.set_ylabel("Mean episode reward (successful episodes only)", fontsize=11)
    ax.set_xlabel("Test world", fontsize=11)
    fig.tight_layout()
    _save(fig, "test_static_reward_bar")


def plot_static_steps_bar(data):
    world_ids  = sorted(STATIC_WORLDS)
    world_lbls = [STATIC_WORLDS[w] for w in world_ids]
    means_d, stds_d = {}, {}
    for model in ("none", "gru", "lstm"):
        means_d[model] = {w: world_mean_std(data["static"], model, w, "steps", filter_success=True)[0]
                          for w in world_ids}
        stds_d[model]  = {w: world_mean_std(data["static"], model, w, "steps", filter_success=True)[1]
                          for w in world_ids}
    fig, ax = plt.subplots(figsize=(9, 4.8))
    _grouped_bars(ax, world_ids, world_lbls, means_d, stds_d)
    ax.set_ylabel("Mean episode length  (steps, successful episodes)", fontsize=11)
    ax.set_xlabel("Test world", fontsize=11)
    fig.tight_layout()
    _save(fig, "test_static_steps_bar")


def plot_static_heatmap(data):
    world_ids  = sorted(STATIC_WORLDS)
    world_lbls = [STATIC_WORLDS[w].replace("\n", " ") for w in world_ids]
    models = ("none", "gru", "lstm")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), sharey=True)
    for ax, model in zip(axes, models):
        mat = np.zeros((N_SEEDS, len(world_ids)))
        for si in range(N_SEEDS):
            for wi, w in enumerate(world_ids):
                eps = data["static"][model][si].get(w, [])
                mat[si, wi] = np.mean([e["success"] for e in eps]) if eps else np.nan
        im = ax.imshow(mat, aspect="auto", vmin=0, vmax=1,
                       cmap="RdYlGn", interpolation="nearest")
        ax.set_xticks(range(len(world_ids)))
        ax.set_xticklabels(world_lbls, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(N_SEEDS))
        ax.set_yticklabels([f"Seed {s}" for s in range(N_SEEDS)], fontsize=8)
        ax.set_title(MODEL_LABELS[model], fontsize=11)
        for si in range(N_SEEDS):
            for wi in range(len(world_ids)):
                v = mat[si, wi]
                txt = f"{v:.0%}" if np.isfinite(v) else "—"
                ax.text(wi, si, txt, ha="center", va="center",
                        fontsize=7, color="black" if 0.3 < v < 0.7 else "white" if v < 0.3 else "black")
    cbar = fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.02, pad=0.02)
    cbar.set_label("Success rate", fontsize=9)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["0%", "50%", "100%"])
    fig.suptitle("Static test: per-seed success rate", fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, "test_static_heatmap")


# ── moving world plots ────────────────────────────────────────────────────────

def plot_moving_success_bar(data):
    world_ids  = sorted(MOVING_WORLDS)
    world_lbls = [MOVING_WORLDS[w] for w in world_ids]
    means_d, stds_d = {}, {}
    for model in ("none", "gru", "lstm"):
        means_d[model] = {w: world_mean_std(data["moving"], model, w, "success")[0]
                          for w in world_ids}
        stds_d[model]  = {w: world_mean_std(data["moving"], model, w, "success")[1]
                          for w in world_ids}
    fig, ax = plt.subplots(figsize=(9, 4.8))
    _grouped_bars(ax, world_ids, world_lbls, means_d, stds_d)
    ax.set_ylim(0, 1.22)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.set_ylabel("Success rate  (9 episodes × 6 seeds)", fontsize=11)
    ax.set_xlabel("Test world", fontsize=11)
    fig.tight_layout()
    _save(fig, "test_moving_success_bar")


def plot_moving_per_seed(data):
    """Strip plot showing per-seed success per world per arch. Reveals bimodal distribution."""
    world_ids  = sorted(MOVING_WORLDS)
    world_lbls = [MOVING_WORLDS[w].replace("\n", " ") for w in world_ids]
    models = ("none", "gru", "lstm")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

    rng = np.random.default_rng(42)
    for ax, model in zip(axes, models):
        for wi, w in enumerate(world_ids):
            seed_vals = seed_world_metric(data["moving"], model, w, "success")
            # mean bar (faint)
            finite = seed_vals[np.isfinite(seed_vals)]
            mean_val = float(np.nanmean(seed_vals)) if len(finite) else 0.0
            ax.barh(wi, mean_val, height=0.55, color=MODEL_COLORS[model],
                    alpha=0.25, zorder=1)
            # individual seed jitter
            jitter = rng.uniform(-0.18, 0.18, size=N_SEEDS)
            for si, (v, j) in enumerate(zip(seed_vals, jitter)):
                if np.isfinite(v):
                    ax.scatter(v, wi + j, color=MODEL_COLORS[model],
                               s=55, zorder=3, edgecolors="white", linewidths=0.5,
                               alpha=0.9)
            # mean marker
            ax.scatter(mean_val, wi, marker="|", color="black",
                       s=200, zorder=4, linewidths=2.0)

        ax.set_yticks(range(len(world_ids)))
        ax.set_yticklabels(world_lbls, fontsize=9)
        ax.set_xlim(-0.05, 1.10)
        ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.25))
        ax.set_xlabel("Success rate", fontsize=10)
        ax.set_title(MODEL_LABELS[model], fontsize=11)
        ax.grid(axis="x", alpha=0.25, lw=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # add note about mean marker
        ax.text(0.99, -0.5, "— mean", ha="right", va="center",
                fontsize=7.5, color="black",
                transform=ax.get_yaxis_transform())

    fig.suptitle(
        "Moving-world test: per-seed success rate (dots = individual seeds, bar = mean)",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    _save(fig, "test_moving_per_seed")


def plot_moving_heatmap(data):
    world_ids  = sorted(MOVING_WORLDS)
    world_lbls = [MOVING_WORLDS[w].replace("\n", " ") for w in world_ids]
    models = ("none", "gru", "lstm")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), sharey=True)
    for ax, model in zip(axes, models):
        mat = np.zeros((N_SEEDS, len(world_ids)))
        for si in range(N_SEEDS):
            for wi, w in enumerate(world_ids):
                eps = data["moving"][model][si].get(w, [])
                mat[si, wi] = np.mean([e["success"] for e in eps]) if eps else np.nan
        im = ax.imshow(mat, aspect="auto", vmin=0, vmax=1,
                       cmap="RdYlGn", interpolation="nearest")
        ax.set_xticks(range(len(world_ids)))
        ax.set_xticklabels(world_lbls, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(N_SEEDS))
        ax.set_yticklabels([f"Seed {s}" for s in range(N_SEEDS)], fontsize=8)
        ax.set_title(MODEL_LABELS[model], fontsize=11)
        for si in range(N_SEEDS):
            for wi in range(len(world_ids)):
                v = mat[si, wi]
                txt = f"{v:.0%}" if np.isfinite(v) else "—"
                col = "white" if v < 0.35 else "black"
                ax.text(wi, si, txt, ha="center", va="center",
                        fontsize=7.5, color=col)
    cbar = fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.02, pad=0.02)
    cbar.set_label("Success rate", fontsize=9)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["0%", "50%", "100%"])
    fig.suptitle("Moving-world test: per-seed success rate", fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, "test_moving_heatmap")


def plot_moving_min_dist_bar(data):
    """Mean minimum distance to goal on moving worlds (all episodes, not just successes)."""
    world_ids  = sorted(MOVING_WORLDS)
    world_lbls = [MOVING_WORLDS[w] for w in world_ids]
    means_d, stds_d = {}, {}
    for model in ("none", "gru", "lstm"):
        means_d[model] = {w: world_mean_std(data["moving"], model, w, "min_dist")[0]
                          for w in world_ids}
        stds_d[model]  = {w: world_mean_std(data["moving"], model, w, "min_dist")[1]
                          for w in world_ids}
    fig, ax = plt.subplots(figsize=(9, 4.8))
    _grouped_bars(ax, world_ids, world_lbls, means_d, stds_d)
    ax.set_ylabel("Mean closest approach to goal  (m)", fontsize=11)
    ax.set_xlabel("Test world", fontsize=11)
    ax.set_ylim(0, None)
    fig.tight_layout()
    _save(fig, "test_moving_min_dist_bar")


# ── combined summary ──────────────────────────────────────────────────────────

def plot_summary(data):
    all_worlds = list(STATIC_WORLDS.items()) + list(MOVING_WORLDS.items())
    world_ids  = [w for w, _ in all_worlds]
    world_lbls = [lbl.replace("\n", " ") for _, lbl in all_worlds]
    is_static  = [True] * len(STATIC_WORLDS) + [False] * len(MOVING_WORLDS)

    fig, ax = plt.subplots(figsize=(16, 5))
    n_m = 3
    width, gap = 0.20, 0.03
    gw = n_m * width + (n_m - 1) * gap
    positions = np.arange(len(world_ids))

    for mi, model in enumerate(("none", "gru", "lstm")):
        means, stds = [], []
        for w, static in zip(world_ids, is_static):
            phase = "static" if static else "moving"
            m, s = world_mean_std(data[phase], model, w, "success")
            means.append(m); stds.append(s)
        offsets = positions + mi * (width + gap) - gw / 2 + width / 2
        ax.bar(offsets, means, width=width, color=MODEL_COLORS[model],
               alpha=0.88, label=MODEL_LABELS[model],
               yerr=stds, capsize=3.0, error_kw={"lw": 1.0})

    ax.set_xticks(positions)
    ax.set_xticklabels(world_lbls, rotation=25, ha="right", fontsize=9)
    ax.set_ylim(0, 1.22)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.set_ylabel("Success rate", fontsize=11)

    # shade static vs moving regions
    n_s = len(STATIC_WORLDS)
    ax.axvspan(-0.5, n_s - 0.5, alpha=0.04, color="steelblue", zorder=0)
    ax.axvspan(n_s - 0.5, len(world_ids) - 0.5, alpha=0.04, color="darkorange", zorder=0)
    ax.text(n_s / 2 - 0.5, 1.14, "Static test worlds", ha="center",
            fontsize=9.5, color="steelblue")
    ax.text(n_s + len(MOVING_WORLDS) / 2 - 0.5, 1.14, "Moving test worlds (bimodal)",
            ha="center", fontsize=9.5, color="darkorange")

    ax.axvline(n_s - 0.5, color="#888", lw=1.0, ls="--")
    ax.legend(frameon=False, fontsize=10)
    ax.grid(axis="y", alpha=0.25, lw=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, "test_summary_bar")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("Parsing test logs...")
    data = parse_all_logs()

    # report counts
    for phase in ("static", "moving"):
        for arch in ("none", "gru", "lstm"):
            for seed in range(N_SEEDS):
                worlds = sorted(data[phase][arch][seed])
                ep_counts = [len(data[phase][arch][seed][w]) for w in worlds]
                print(f"  [{phase}] {arch} seed{seed}: worlds={worlds} eps={ep_counts}")

    print("\nGenerating static test plots...")
    plot_static_success_bar(data)
    plot_static_reward_bar(data)
    plot_static_steps_bar(data)
    plot_static_heatmap(data)

    print("Generating moving test plots...")
    plot_moving_success_bar(data)
    plot_moving_per_seed(data)
    plot_moving_heatmap(data)
    plot_moving_min_dist_bar(data)

    print("Generating combined summary...")
    plot_summary(data)

    print(f"\nSaved all test plots to {OUT}")


if __name__ == "__main__":
    main()
