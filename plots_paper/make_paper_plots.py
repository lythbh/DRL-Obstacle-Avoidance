"""Comprehensive paper-quality plots from all static and moving curriculum runs.

Generates:
  - static_training_success.pdf   -- success rate over 5000 training episodes (static curriculum)
  - static_training_reward.pdf    -- mean episode reward over static training
  - static_training_loss.pdf      -- critic loss over static training (log scale)
  - moving_training_success.pdf   -- success rate over 2500 moving curriculum episodes
  - moving_training_reward.pdf    -- mean episode reward over moving training
  - validation_static_bar.pdf     -- grouped bar: static-world success per arch
  - validation_moving_bar.pdf     -- grouped bar: moving-world success per arch (static model)
  - validation_summary_bar.pdf    -- combined side-by-side static vs moving summary

Also prints LaTeX table code for the results section.
"""

from __future__ import annotations

import csv
import os
import re
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/drl-obstacle-matplotlib-cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np

ROOT    = Path(__file__).resolve().parents[1]
PLOTS   = ROOT / "plots"
LOGS    = ROOT / "logs"
OUT     = Path(__file__).resolve().parent

STATIC_STAGES  = 10
MOVING_STAGES  = 5
EPS_PER_STAGE  = 500
STATIC_EPS     = STATIC_STAGES * EPS_PER_STAGE   # 5000
MOVING_EPS     = MOVING_STAGES * EPS_PER_STAGE   # 2500
WINDOW         = 50

MODEL_LABELS = {"none": "Feedforward", "gru": "GRU", "lstm": "LSTM"}
MODEL_COLORS = {"none": "#4C78A8", "gru": "#F58518", "lstm": "#54A24B"}
HATCHES      = {"none": "///", "gru": "...", "lstm": "xxx"}

STATIC_WORLDS = {
    1: "Empty\n(centre)", 2: "Empty\n(offset)",
    3: "Sparse A",        4: "Sparse B",
    5: "Dense",
}
MOVING_WORLDS = {
    6: "One moving\nobstacle", 7: "All moving\nobstacles",
    8: "Moving\ngoal",        9: "Offset\nmoving goal",
}

# ── helpers ──────────────────────────────────────────────────────────────────

def _float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan

def _read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))

def _roll(arr, w):
    out = np.full(len(arr), np.nan)
    h = w // 2
    for i in range(len(arr)):
        chunk = arr[max(0, i-h): min(len(arr), i+h+1)]
        v = chunk[np.isfinite(chunk)]
        if len(v):
            out[i] = v.mean()
    return out

def _mean_std(*curves):
    mat = np.vstack([c.astype(float) for c in curves])
    return np.nanmean(mat, 0), np.nanstd(mat, 0)

def _save(fig, stem):
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)

STATIC_RUN_RE = re.compile(
    r"^(?P<ts>\d{8}_\d{6})_(?P<model>gru|lstm|none)_seed(?P<seed>\d+)"
    r"_stage(?P<stage>\d{2})_"
)
MOVING_RUN_RE = re.compile(
    r"moving_(?P<ts>\d{8}_\d{6})_step(?P<step>\d{2})_.*_"
    r"(?P<model>gru|lstm|none)_seed(?P<seed>\d+)_stage(?P<stage>\d{2})_"
)

# ── static training data ──────────────────────────────────────────────────────

def load_static_runs():
    grouped = defaultdict(lambda: defaultdict(dict))  # (model,ts,seed) -> stage -> Path
    for d in sorted(PLOTS.iterdir()):
        if not d.is_dir() or "moving" in d.name:
            continue
        m = STATIC_RUN_RE.match(d.name)
        if not m:
            continue
        key = (m["model"], m["ts"], m["seed"])
        grouped[key][int(m["stage"])] = d

    complete = defaultdict(list)
    required = set(range(1, STATIC_STAGES+1))
    for (model, ts, seed), stages in grouped.items():
        if set(stages) != required:
            continue
        if not all((stages[s] / "ppo_episodes.csv").exists() for s in required):
            continue
        complete[model].append({"ts": ts, "seed": seed, "stages": stages})

    for model in complete:
        complete[model].sort(key=lambda r: int(r["seed"]))
        seen, deduped = set(), []
        for r in complete[model]:
            if r["seed"] not in seen:
                seen.add(r["seed"])
                deduped.append(r)
        complete[model] = deduped
    return complete

def static_episode_curves(run):
    success  = np.full(STATIC_EPS, np.nan)
    reward   = np.full(STATIC_EPS, np.nan)
    min_dist = np.full(STATIC_EPS, np.nan)
    for stage in range(1, STATIC_STAGES+1):
        rows = _read_csv(run["stages"][stage] / "ppo_episodes.csv")[:EPS_PER_STAGE]
        off  = (stage-1)*EPS_PER_STAGE
        for i, row in enumerate(rows):
            success [off+i] = _float(row.get("success"))
            reward  [off+i] = _float(row.get("reward"))
            min_dist[off+i] = _float(row.get("min_dist"))
    return success, reward, min_dist

def static_loss_curve(run, col="critic_loss"):
    xs, ys = [], []
    for stage in range(1, STATIC_STAGES+1):
        p = run["stages"][stage] / "ppo_updates.csv"
        if not p.exists():
            continue
        by_ep = defaultdict(list)
        for row in _read_csv(p):
            ep  = int(_float(row.get("episode")))
            val = _float(row.get(col))
            if 1 <= ep <= EPS_PER_STAGE and np.isfinite(val):
                by_ep[ep].append(val)
        off = (stage-1)*EPS_PER_STAGE
        for ep in sorted(by_ep):
            xs.append(off+ep)
            ys.append(np.mean(by_ep[ep]))
    if len(xs) < 2:
        return np.full(STATIC_EPS, np.nan)
    dense = np.arange(1, STATIC_EPS+1)
    y = _roll(np.array(ys, float), WINDOW)
    return np.interp(dense, np.array(xs, float), y)

# ── moving training data ──────────────────────────────────────────────────────

def load_moving_runs():
    grouped = defaultdict(lambda: defaultdict(dict))  # (model,seed) -> ts -> step -> Path
    for d in sorted(PLOTS.iterdir()):
        if not d.is_dir() or "moving" not in d.name:
            continue
        m = MOVING_RUN_RE.search(d.name)
        if not m:
            continue
        key = (m["model"], m["seed"])
        grouped[key][m["ts"]][int(m["step"])] = d

    required = set(range(1, MOVING_STAGES+1))
    complete = defaultdict(list)
    for (model, seed), tss in grouped.items():
        valid = [ts for ts, steps in tss.items()
                 if set(steps) == required
                 and all((steps[s] / "ppo_episodes.csv").exists() for s in required)]
        if not valid:
            continue
        ts    = max(valid)
        steps = tss[ts]
        complete[model].append({"ts": ts, "seed": seed, "steps": steps})

    for model in complete:
        complete[model].sort(key=lambda r: int(r["seed"]))
        seen, deduped = set(), []
        for r in complete[model]:
            if r["seed"] not in seen:
                seen.add(r["seed"])
                deduped.append(r)
        complete[model] = deduped
    return complete

def moving_episode_curves(run):
    success  = np.full(MOVING_EPS, np.nan)
    reward   = np.full(MOVING_EPS, np.nan)
    for step in range(1, MOVING_STAGES+1):
        rows = _read_csv(run["steps"][step] / "ppo_episodes.csv")[:EPS_PER_STAGE]
        off  = (step-1)*EPS_PER_STAGE
        for i, row in enumerate(rows):
            success[off+i] = _float(row.get("success"))
            reward [off+i] = _float(row.get("reward"))
    return success, reward

def moving_loss_curve(run, col="critic_loss"):
    xs, ys = [], []
    for step in range(1, MOVING_STAGES+1):
        p = run["steps"][step] / "ppo_updates.csv"
        if not p.exists():
            continue
        by_ep = defaultdict(list)
        for row in _read_csv(p):
            ep  = int(_float(row.get("episode")))
            val = _float(row.get(col))
            if 1 <= ep <= EPS_PER_STAGE and np.isfinite(val):
                by_ep[ep].append(val)
        off = (step-1)*EPS_PER_STAGE
        for ep in sorted(by_ep):
            xs.append(off+ep)
            ys.append(np.mean(by_ep[ep]))
    if len(xs) < 2:
        return np.full(MOVING_EPS, np.nan)
    dense = np.arange(1, MOVING_EPS+1)
    y = _roll(np.array(ys, float), WINDOW)
    return np.interp(dense, np.array(xs, float), y)

# ── validation log parsing ────────────────────────────────────────────────────

EVAL_RE = re.compile(
    r"\[EVAL\]\[PPO\] ep=(\d+)/(\d+) r=\s*(-?[\d.]+) steps=\s*(\d+) "
    r"success=(\d) touch=(\d) min_d=\s*([\d.]+) avg_spd=([\d.]+)m/s end=(\w+)"
)
VAL_HDR_RE = re.compile(
    r"\[VALIDATION\] (?P<phase>static|moving) arch=(?P<arch>\w+) seed=(?P<seed>\d+) "
    r"world=val_(?P<world>\d+)_"
)

N_TEST_EPS = 9   # deterministic evaluation episodes per world per seed

def parse_validation_logs():
    """
    Returns dict: phase -> arch -> seed_int -> world_int -> list[dict]
    Each dict has keys: success, reward, min_dist, steps, end_reason
    Only includes blocks where the static-phase model is used
    (confirmed via stage10 checkpoint in the log).
    """
    data = {
        "static": defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
        "moving": defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
    }
    for log_path in sorted(LOGS.glob("validation_*.log")):
        text = log_path.read_text()
        lines = text.splitlines()

        current_phase  = None
        current_arch   = None
        current_seed   = None
        current_world  = None

        for line in lines:
            hm = VAL_HDR_RE.search(line)
            if hm:
                current_phase = hm["phase"]
                current_arch  = hm["arch"]
                current_seed  = int(hm["seed"])
                current_world = int(hm["world"])
                continue
            em = EVAL_RE.search(line)
            if em and current_world is not None:
                ep_num = int(em.group(1))
                if ep_num == 10:
                    continue  # ep 10 sometimes missing due to termination
                ep = {
                    "success":   int(em.group(5)),
                    "reward":    float(em.group(3)),
                    "min_dist":  float(em.group(7)),
                    "steps":     int(em.group(4)),
                    "end":       em.group(9),
                }
                data[current_phase][current_arch][current_seed][current_world].append(ep)
    return data


def world_success(episodes):
    if not episodes:
        return np.nan
    return np.mean([e["success"] for e in episodes])

def world_min_dist(episodes):
    if not episodes:
        return np.nan
    return np.mean([e["min_dist"] for e in episodes])

def world_reward(episodes):
    if not episodes:
        return np.nan
    return np.mean([e["reward"] for e in episodes])

# ── plot helpers ──────────────────────────────────────────────────────────────

def _style(ax, ylabel, xlim=None):
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlabel("Episode", fontsize=11)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def _stage_lines(ax, n_stages, eps_per_stage, color="#666666"):
    for s in range(2, n_stages+1):
        ax.axvline((s-1)*eps_per_stage, color=color, lw=0.7, ls="--", alpha=0.40)

def _legend(fig, ax, loc="upper center", ncol=3):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc=loc, bbox_to_anchor=(0.5, 0.96),
               ncol=ncol, frameon=False, fontsize=10)

# ── training curve plots ──────────────────────────────────────────────────────

def plot_static_training(runs):
    x = np.arange(1, STATIC_EPS+1)

    # success rate
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for model in ("none", "gru", "lstm"):
        rlist = runs.get(model, [])
        if not rlist:
            continue
        curves = [_roll(static_episode_curves(r)[0], WINDOW) for r in rlist]
        mn, sd = _mean_std(*curves)
        c = MODEL_COLORS[model]
        ax.plot(x, mn, color=c, lw=1.8, label=f"{MODEL_LABELS[model]} (n={len(rlist)})")
        ax.fill_between(x, mn-sd, mn+sd, color=c, alpha=0.15, lw=0)
    _stage_lines(ax, STATIC_STAGES, EPS_PER_STAGE)
    ax.set_ylim(-0.03, 1.03)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
    _style(ax, f"Success rate (rolling {WINDOW}-ep mean)", xlim=(1, STATIC_EPS))
    _legend(fig, ax)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save(fig, "static_training_success")

    # mean episode reward
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for model in ("none", "gru", "lstm"):
        rlist = runs.get(model, [])
        if not rlist:
            continue
        curves = [_roll(static_episode_curves(r)[1], WINDOW) for r in rlist]
        mn, sd = _mean_std(*curves)
        c = MODEL_COLORS[model]
        ax.plot(x, mn, color=c, lw=1.8, label=MODEL_LABELS[model])
        ax.fill_between(x, mn-sd, mn+sd, color=c, alpha=0.15, lw=0)
    _stage_lines(ax, STATIC_STAGES, EPS_PER_STAGE)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
    _style(ax, f"Mean episode reward (rolling {WINDOW}-ep mean)", xlim=(1, STATIC_EPS))
    _legend(fig, ax)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save(fig, "static_training_reward")

    # critic loss
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for model in ("none", "gru", "lstm"):
        rlist = runs.get(model, [])
        if not rlist:
            continue
        curves = [static_loss_curve(r) for r in rlist]
        mn, sd = _mean_std(*curves)
        c = MODEL_COLORS[model]
        ax.plot(x, mn, color=c, lw=1.8, label=MODEL_LABELS[model])
        ax.fill_between(x, np.maximum(mn-sd, 1e-6), mn+sd, color=c, alpha=0.15, lw=0)
    _stage_lines(ax, STATIC_STAGES, EPS_PER_STAGE)
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
    _style(ax, "Critic (value) loss", xlim=(1, STATIC_EPS))
    _legend(fig, ax)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save(fig, "static_training_loss")


def plot_moving_training(runs):
    x = np.arange(1, MOVING_EPS+1)

    stage_labels = [
        "Partial 1", "Partial 3", "Partial 5", "All moving", "Moving goal"
    ]

    # success rate
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for model in ("none", "gru", "lstm"):
        rlist = runs.get(model, [])
        if not rlist:
            continue
        curves = [_roll(moving_episode_curves(r)[0], WINDOW) for r in rlist]
        mn, sd = _mean_std(*curves)
        c = MODEL_COLORS[model]
        ax.plot(x, mn, color=c, lw=1.8, label=f"{MODEL_LABELS[model]} (n={len(rlist)})")
        ax.fill_between(x, mn-sd, mn+sd, color=c, alpha=0.15, lw=0)
    _stage_lines(ax, MOVING_STAGES, EPS_PER_STAGE)
    # stage labels
    for i, lbl in enumerate(stage_labels):
        xc = (i + 0.5) * EPS_PER_STAGE
        ax.text(xc, 1.01, lbl, ha="center", va="bottom", fontsize=7.5,
                color="#555555", transform=ax.get_xaxis_transform())
    ax.set_ylim(-0.03, 1.08)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
    _style(ax, f"Success rate (rolling {WINDOW}-ep mean)", xlim=(1, MOVING_EPS))
    _legend(fig, ax)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save(fig, "moving_training_success")

    # reward
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for model in ("none", "gru", "lstm"):
        rlist = runs.get(model, [])
        if not rlist:
            continue
        curves = [_roll(moving_episode_curves(r)[1], WINDOW) for r in rlist]
        mn, sd = _mean_std(*curves)
        c = MODEL_COLORS[model]
        ax.plot(x, mn, color=c, lw=1.8, label=MODEL_LABELS[model])
        ax.fill_between(x, mn-sd, mn+sd, color=c, alpha=0.15, lw=0)
    _stage_lines(ax, MOVING_STAGES, EPS_PER_STAGE)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
    _style(ax, f"Mean episode reward (rolling {WINDOW}-ep mean)", xlim=(1, MOVING_EPS))
    _legend(fig, ax)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save(fig, "moving_training_reward")

    # critic loss
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for model in ("none", "gru", "lstm"):
        rlist = runs.get(model, [])
        if not rlist:
            continue
        curves = [moving_loss_curve(r) for r in rlist]
        mn, sd = _mean_std(*curves)
        c = MODEL_COLORS[model]
        ax.plot(x, mn, color=c, lw=1.8, label=MODEL_LABELS[model])
        ax.fill_between(x, np.maximum(mn-sd, 1e-6), mn+sd, color=c, alpha=0.15, lw=0)
    _stage_lines(ax, MOVING_STAGES, EPS_PER_STAGE)
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
    _style(ax, "Critic (value) loss", xlim=(1, MOVING_EPS))
    _legend(fig, ax)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save(fig, "moving_training_loss")


# ── validation bar plots ──────────────────────────────────────────────────────

def _bar_data(val_data, phase, worlds):
    """
    Returns dict: world -> model -> (mean_success, std_success)
    Aggregated over all seeds.
    """
    phase_data = val_data[phase]
    result = {}
    for w in worlds:
        result[w] = {}
        for model in ("none", "gru", "lstm"):
            seed_successes = []
            for seed, world_dict in phase_data[model].items():
                eps = world_dict.get(w, [])
                s = world_success(eps)
                if np.isfinite(s):
                    seed_successes.append(s)
            if seed_successes:
                result[w][model] = (np.mean(seed_successes), np.std(seed_successes))
            else:
                result[w][model] = (np.nan, np.nan)
    return result


def plot_validation_bar(val_data, phase, worlds, title_suffix, stem):
    bar_data = _bar_data(val_data, phase, worlds)
    world_ids  = sorted(worlds.keys())
    world_lbls = [worlds[w] for w in world_ids]
    n_worlds   = len(world_ids)
    models     = ("none", "gru", "lstm")
    n_models   = len(models)

    width = 0.22
    gap   = 0.04
    group_width = n_models * width + (n_models-1) * gap
    positions   = np.arange(n_worlds)

    fig, ax = plt.subplots(figsize=(max(8, n_worlds * 2.4), 4.8))
    for mi, model in enumerate(models):
        offsets = positions + mi * (width + gap) - group_width/2 + width/2
        means = [bar_data[w][model][0] for w in world_ids]
        stds  = [bar_data[w][model][1] for w in world_ids]
        ax.bar(offsets, means, width=width,
               color=MODEL_COLORS[model], alpha=0.88,
               label=MODEL_LABELS[model],
               yerr=stds, capsize=4, error_kw={"lw": 1.2})
        for ox, m in zip(offsets, means):
            if np.isfinite(m):
                ax.text(ox, m + 0.03, f"{m:.0%}",
                        ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax.set_xticks(positions)
    ax.set_xticklabels(world_lbls, fontsize=10)
    ax.set_ylim(0, 1.18)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.set_ylabel("Mean success rate (9 episodes per seed)", fontsize=11)
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.25, lw=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, stem)
    return bar_data


def plot_validation_summary(val_data):
    """Side-by-side static vs moving summary per architecture."""
    static_bar = _bar_data(val_data, "static", STATIC_WORLDS)
    moving_bar = _bar_data(val_data, "moving", MOVING_WORLDS)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, bar_data, worlds, phase_label in [
        (axes[0], static_bar, STATIC_WORLDS, "Static evaluation worlds"),
        (axes[1], moving_bar, MOVING_WORLDS, "Moving evaluation worlds"),
    ]:
        world_ids = sorted(worlds.keys())
        world_lbls = [worlds[w] for w in world_ids]
        n_worlds = len(world_ids)
        models   = ("none", "gru", "lstm")
        width, gap = 0.22, 0.04
        group_width = len(models)*width + (len(models)-1)*gap
        positions = np.arange(n_worlds)

        for mi, model in enumerate(models):
            offsets = positions + mi*(width+gap) - group_width/2 + width/2
            means = [bar_data[w][model][0] for w in world_ids]
            stds  = [bar_data[w][model][1] for w in world_ids]
            ax.bar(offsets, means, width=width, color=MODEL_COLORS[model], alpha=0.88,
                   label=MODEL_LABELS[model], yerr=stds, capsize=4,
                   error_kw={"lw": 1.2})

        ax.set_xticks(positions)
        ax.set_xticklabels(world_lbls, fontsize=9)
        ax.set_title(phase_label, fontsize=12, pad=8)
        ax.set_ylim(0, 1.22)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
        ax.grid(axis="y", alpha=0.25, lw=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Mean success rate", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=3, frameon=False, fontsize=11)
    fig.tight_layout()
    _save(fig, "validation_summary_bar")


# ── summary statistics ────────────────────────────────────────────────────────

def compute_summary(val_data, static_runs, moving_runs):
    lines = []

    # ---- static training summary
    lines.append("\n===== Static curriculum training =====")
    for model in ("none", "gru", "lstm"):
        rlist = static_runs.get(model, [])
        n = len(rlist)
        if not n:
            lines.append(f"  {MODEL_LABELS[model]}: no complete runs")
            continue
        final_sr, final_rew, overall_sr = [], [], []
        for r in rlist:
            s, rw, _ = static_episode_curves(r)
            final_slice  = slice((STATIC_STAGES-1)*EPS_PER_STAGE, STATIC_EPS)
            final_sr.append(np.nanmean(s[final_slice]))
            final_rew.append(np.nanmean(rw[final_slice]))
            overall_sr.append(np.nanmean(s))
        lines.append(
            f"  {MODEL_LABELS[model]:12s} n={n}  "
            f"final_SR={np.mean(final_sr):.3f}±{np.std(final_sr):.3f}  "
            f"overall_SR={np.mean(overall_sr):.3f}±{np.std(overall_sr):.3f}  "
            f"final_rew={np.mean(final_rew):.1f}±{np.std(final_rew):.1f}"
        )

    # ---- moving curriculum summary
    lines.append("\n===== Moving curriculum training =====")
    for model in ("none", "gru", "lstm"):
        rlist = moving_runs.get(model, [])
        n = len(rlist)
        if not n:
            lines.append(f"  {MODEL_LABELS[model]}: no complete runs")
            continue
        final_sr, final_rew, overall_sr = [], [], []
        for r in rlist:
            s, rw = moving_episode_curves(r)
            final_slice  = slice((MOVING_STAGES-1)*EPS_PER_STAGE, MOVING_EPS)
            final_sr.append(np.nanmean(s[final_slice]))
            final_rew.append(np.nanmean(rw[final_slice]))
            overall_sr.append(np.nanmean(s))
        lines.append(
            f"  {MODEL_LABELS[model]:12s} n={n}  "
            f"final_SR={np.mean(final_sr):.3f}±{np.std(final_sr):.3f}  "
            f"overall_SR={np.mean(overall_sr):.3f}±{np.std(overall_sr):.3f}  "
            f"final_rew={np.mean(final_rew):.1f}±{np.std(final_rew):.1f}"
        )

    # ---- validation summary
    lines.append("\n===== Static validation (worlds 1-5) =====")
    for model in ("none", "gru", "lstm"):
        all_seed_world_sr = []
        for seed, world_dict in val_data["static"][model].items():
            for w in STATIC_WORLDS:
                s = world_success(world_dict.get(w, []))
                if np.isfinite(s):
                    all_seed_world_sr.append(s)
        if all_seed_world_sr:
            lines.append(
                f"  {MODEL_LABELS[model]:12s}  "
                f"mean_SR={np.mean(all_seed_world_sr):.3f}±{np.std(all_seed_world_sr):.3f}  "
                f"n_worlds×seeds={len(all_seed_world_sr)}"
            )

    lines.append("\n===== Moving validation (worlds 6-9) =====")
    for model in ("none", "gru", "lstm"):
        all_seed_world_sr = []
        for seed, world_dict in val_data["moving"][model].items():
            for w in MOVING_WORLDS:
                s = world_success(world_dict.get(w, []))
                if np.isfinite(s):
                    all_seed_world_sr.append(s)
        if all_seed_world_sr:
            lines.append(
                f"  {MODEL_LABELS[model]:12s}  "
                f"mean_SR={np.mean(all_seed_world_sr):.3f}±{np.std(all_seed_world_sr):.3f}  "
                f"n_worlds×seeds={len(all_seed_world_sr)}"
            )

    # per-world moving validation detail
    lines.append("\n  Per-world moving validation breakdown:")
    bar_moving = _bar_data(val_data, "moving", MOVING_WORLDS)
    for w, lbl in MOVING_WORLDS.items():
        row_parts = [f"    World {w} ({lbl.replace(chr(10), ' ')}):"]
        for model in ("none", "gru", "lstm"):
            m, s = bar_moving[w].get(model, (np.nan, np.nan))
            row_parts.append(f"{MODEL_LABELS[model]}={m:.0%}±{s:.0%}")
        lines.append("  ".join(row_parts))

    return "\n".join(lines)


def latex_validation_table(val_data):
    static_bar = _bar_data(val_data, "static", STATIC_WORLDS)
    moving_bar = _bar_data(val_data, "moving", MOVING_WORLDS)

    rows = []
    rows.append(r"\begin{table}[htbp]")
    rows.append(r"\centering")
    rows.append(r"\caption{Validation success rates (\%) for each evaluation scenario, "
                r"averaged over all seeds (9 episodes per seed). "
                r"Mean\,$\pm$\,std across seeds. "
                r"Static worlds use the stage-10 static model; "
                r"moving worlds use the final moving-curriculum model.}")
    rows.append(r"\label{tab:validation}")

    # build header
    n_sw = len(STATIC_WORLDS)
    n_mw = len(MOVING_WORLDS)
    total_cols = 1 + n_sw + n_mw
    col_spec = "l" + "c"*n_sw + "c"*n_mw
    rows.append(r"\begin{tabular}{" + col_spec + r"}")
    rows.append(r"\toprule")

    # world-number row
    sw_heads = " & ".join(f"W{w}" for w in sorted(STATIC_WORLDS))
    mw_heads = " & ".join(f"W{w}" for w in sorted(MOVING_WORLDS))
    rows.append(r" & \multicolumn{" + str(n_sw) + r"}{c}{Static worlds} "
                r"& \multicolumn{" + str(n_mw) + r"}{c}{Moving worlds} \\")
    rows.append(r"\cmidrule(lr){2-" + str(1+n_sw) + r"}"
                r"\cmidrule(lr){" + str(2+n_sw) + r"-" + str(total_cols) + r"}")
    rows.append("Architecture & " + sw_heads + " & " + mw_heads + r" \\")
    rows.append(r"\midrule")

    for model in ("none", "gru", "lstm"):
        cells = [MODEL_LABELS[model]]
        for w in sorted(STATIC_WORLDS):
            m, s = static_bar[w].get(model, (np.nan, np.nan))
            cells.append(f"{m*100:.0f}\\,\\% $\\pm$ {s*100:.0f}\\,\\%" if np.isfinite(m) else "--")
        for w in sorted(MOVING_WORLDS):
            m, s = moving_bar[w].get(model, (np.nan, np.nan))
            cells.append(f"{m*100:.0f}\\,\\% $\\pm$ {s*100:.0f}\\,\\%" if np.isfinite(m) else "--")
        rows.append(" & ".join(cells) + r" \\")

    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")
    rows.append(r"\end{table}")
    return "\n".join(rows)


# ── per-episode test success rate ────────────────────────────────────────────

def parse_test_logs_per_episode():
    """
    Returns: phase -> arch -> seed(int) -> world(int) -> list[int] (success per episode, ordered)
    """
    LOGS = ROOT / "logs"
    data = {
        "static": defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
        "moving": defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
    }
    for arch in ("gru", "lstm", "none"):
        for seed in range(6):
            path = LOGS / f"validation_{arch}_seed{seed}.log"
            if not path.exists():
                continue
            current_phase, current_world = None, None
            for line in path.read_text().splitlines():
                hm = VAL_HDR_RE.search(line)
                if hm:
                    current_phase = hm["phase"]
                    current_world = int(hm["world"])
                    continue
                em = EVAL_RE.search(line)
                if em and current_world is not None:
                    ep_idx = int(em.group(1))
                    if ep_idx <= N_TEST_EPS:
                        data[current_phase][arch][seed][current_world].append(int(em.group(5)))
    return data


def plot_test_success_per_episode(test_data):
    """
    For each world, for each episode index 1..9, compute the mean success
    across all 6 seeds. Plot episode (x) vs success % (y) with one line per
    world, one subplot per architecture.
    """
    all_worlds = {**STATIC_WORLDS, **MOVING_WORLDS}
    # line styles: static worlds solid, moving worlds dashed
    static_wids = sorted(STATIC_WORLDS)
    moving_wids = sorted(MOVING_WORLDS)

    # colour palette: 5 static blues, 4 moving oranges/reds
    static_colors = ["#2166ac", "#4393c3", "#74add1", "#abd9e9", "#e0f3f8"]
    moving_colors = ["#d73027", "#f46d43", "#fdae61", "#fee090"]

    episodes = list(range(1, N_TEST_EPS + 1))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)

    for ax, arch in zip(axes, ("none", "gru", "lstm")):
        # static worlds
        for wi, (wid, col) in enumerate(zip(static_wids, static_colors)):
            lbl = STATIC_WORLDS[wid].replace("\n", " ")
            ys = []
            for ep in episodes:
                vals = []
                for seed in range(6):
                    eps_list = test_data["static"][arch][seed].get(wid, [])
                    if ep - 1 < len(eps_list):
                        vals.append(eps_list[ep - 1])
                ys.append(np.mean(vals) * 100 if vals else np.nan)
            ax.plot(episodes, ys, color=col, lw=2.0, marker="o", ms=5,
                    label=f"W{wid} {lbl}", zorder=3)

        # moving worlds
        for wi, (wid, col) in enumerate(zip(moving_wids, moving_colors)):
            lbl = MOVING_WORLDS[wid].replace("\n", " ")
            ys = []
            for ep in episodes:
                vals = []
                for seed in range(6):
                    eps_list = test_data["moving"][arch][seed].get(wid, [])
                    if ep - 1 < len(eps_list):
                        vals.append(eps_list[ep - 1])
                ys.append(np.mean(vals) * 100 if vals else np.nan)
            ax.plot(episodes, ys, color=col, lw=2.0, marker="s", ms=5,
                    linestyle="--", label=f"W{wid} {lbl}", zorder=3)

        ax.set_title(MODEL_LABELS[arch], fontsize=12)
        ax.set_xlabel("Test episode", fontsize=11)
        ax.set_xlim(0.6, N_TEST_EPS + 0.4)
        ax.set_xticks(episodes)
        ax.set_ylim(-3, 105)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(25))
        ax.grid(axis="y", alpha=0.25, lw=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Success rate (mean across 6 seeds)", fontsize=11)

    # shared legend below the subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=5, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()
    fig.savefig(OUT / "test_success_per_episode.pdf", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "test_success_per_episode.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_moving_test_success_per_episode(test_data):
    """Same as above but moving worlds only, so the y-axis is not crushed by 100% static lines."""
    moving_colors = ["#d73027", "#f46d43", "#fdae61", "#fee090"]
    episodes = list(range(1, N_TEST_EPS + 1))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)

    for ax, arch in zip(axes, ("none", "gru", "lstm")):
        for wid, col in zip(sorted(MOVING_WORLDS), moving_colors):
            lbl = MOVING_WORLDS[wid].replace("\n", " ")
            ys = []
            for ep in episodes:
                vals = []
                for seed in range(6):
                    eps_list = test_data["moving"][arch][seed].get(wid, [])
                    if ep - 1 < len(eps_list):
                        vals.append(eps_list[ep - 1])
                ys.append(np.mean(vals) * 100 if vals else np.nan)
            ax.plot(episodes, ys, color=col, lw=2.2, marker="o", ms=6,
                    label=f"W{wid} {lbl}", zorder=3)

        ax.set_title(MODEL_LABELS[arch], fontsize=12)
        ax.set_xlabel("Test episode", fontsize=11)
        ax.set_xlim(0.6, N_TEST_EPS + 0.4)
        ax.set_xticks(episodes)
        ax.set_ylim(-3, 105)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(25))
        ax.grid(axis="y", alpha=0.25, lw=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Success rate (mean across 6 seeds)", fontsize=11)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=4, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, -0.10))
    fig.tight_layout()
    fig.savefig(OUT / "test_moving_success_per_episode.pdf", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "test_moving_success_per_episode.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print("Loading static training runs...")
    static_runs = load_static_runs()
    for model in ("none", "gru", "lstm"):
        r = static_runs.get(model, [])
        seeds = [x["seed"] for x in r]
        print(f"  {MODEL_LABELS[model]:12s}: {len(r)} complete run(s), seeds={seeds}")

    print("\nLoading moving training runs...")
    moving_runs = load_moving_runs()
    for model in ("none", "gru", "lstm"):
        r = moving_runs.get(model, [])
        seeds = [x["seed"] for x in r]
        print(f"  {MODEL_LABELS[model]:12s}: {len(r)} complete run(s), seeds={seeds}")

    print("\nParsing validation logs...")
    val_data = parse_validation_logs()
    for phase in ("static", "moving"):
        for model in ("none", "gru", "lstm"):
            seeds = sorted(val_data[phase][model].keys())
            worlds_found = {}
            for s in seeds:
                for w in val_data[phase][model][s]:
                    worlds_found[w] = worlds_found.get(w, 0) + 1
            print(f"  [{phase}] {MODEL_LABELS[model]}: seeds={seeds}, worlds={sorted(worlds_found)}")

    print("\nGenerating static training plots...")
    plot_static_training(static_runs)

    print("Generating moving training plots...")
    plot_moving_training(moving_runs)

    print("Generating validation bar plots...")
    plot_validation_bar(val_data, "static", STATIC_WORLDS,
                        "Static Worlds", "validation_static_bar")
    plot_validation_bar(val_data, "moving", MOVING_WORLDS,
                        "Moving Worlds", "validation_moving_bar")
    plot_validation_summary(val_data)

    print("Generating per-episode test success plots...")
    test_ep_data = parse_test_logs_per_episode()
    plot_test_success_per_episode(test_ep_data)
    plot_moving_test_success_per_episode(test_ep_data)

    summary = compute_summary(val_data, static_runs, moving_runs)
    print(summary)
    (OUT / "summary_stats.txt").write_text(summary)

    latex_tbl = latex_validation_table(val_data)
    print("\n\n===== LaTeX validation table =====")
    print(latex_tbl)
    (OUT / "validation_table.tex").write_text(latex_tbl)

    print(f"\nAll plots saved to {OUT}")


if __name__ == "__main__":
    main()
