"""Generate publication-quality figures from experiment results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_BASE = Path("outputs/experiments")
FIGURE_DIR = Path("paper/figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Consistent styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

COLORS = {
    "uniform_r4": "#1f77b4",
    "baseline_r8": "#ff7f0e",
    "uniform_r16": "#2ca02c",
    "blockwise_no_ccd": "#d62728",
    "blockwise_ccd": "#9467bd",
}

LABELS = {
    "uniform_r4": "Uniform r=4",
    "baseline_r8": "Uniform r=8",
    "uniform_r16": "Uniform r=16",
    "blockwise_no_ccd": "Blockwise",
    "blockwise_ccd": "Blockwise+CCD",
}


def load_results():
    """Load all experiment results."""
    results = {}
    for name in ["uniform_r4", "baseline_r8", "uniform_r16", "blockwise_no_ccd", "blockwise_ccd"]:
        path = OUTPUT_BASE / name / "results.json"
        if path.exists():
            with open(path) as f:
                results[name] = json.load(f)
    return results


def fig_metrics_bar(results: dict):
    """Bar chart comparing all metrics across configurations."""
    metrics = ["dino_score", "clip_t_score", "clip_i_score", "lpips_diversity"]
    metric_labels = ["DINO", "CLIP-T", "CLIP-I", "LPIPS"]

    order = ["uniform_r4", "baseline_r8", "uniform_r16", "blockwise_no_ccd", "blockwise_ccd"]
    configs = [c for c in order if c in results]

    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.0), sharey=False)

    x = np.arange(len(configs))
    width = 0.6

    for ax, metric, label in zip(axes, metrics, metric_labels):
        vals = [results[c]["metrics"].get(metric, 0) for c in configs]
        colors = [COLORS[c] for c in configs]
        bars = ax.bar(x, vals, width, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_title(label)
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[c] for c in configs], rotation=45, ha="right", fontsize=6)
        ax.set_ylim(bottom=min(vals) * 0.9, top=max(vals) * 1.05)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Add value labels on bars
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.3f}", ha="center", va="bottom", fontsize=5.5)

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "metrics_comparison.pdf")
    fig.savefig(FIGURE_DIR / "metrics_comparison.png")
    plt.close(fig)
    print(f"Saved metrics_comparison")


def fig_param_efficiency(results: dict):
    """Scatter plot: parameter count vs DINO score."""
    order = ["uniform_r4", "baseline_r8", "uniform_r16", "blockwise_no_ccd", "blockwise_ccd"]
    configs = [c for c in order if c in results]

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    for c in configs:
        r = results[c]
        params_k = r["lora_params"] / 1000
        dino = r["metrics"].get("dino_score", 0)
        ax.scatter(params_k, dino, c=COLORS[c], s=60, zorder=5, edgecolors="black", linewidths=0.5)
        ax.annotate(LABELS[c], (params_k, dino), textcoords="offset points",
                    xytext=(5, 5), fontsize=6, ha="left")

    ax.set_xlabel("LoRA Parameters (K)")
    ax.set_ylabel("DINO Score")
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "param_efficiency.pdf")
    fig.savefig(FIGURE_DIR / "param_efficiency.png")
    plt.close(fig)
    print(f"Saved param_efficiency")


def fig_ccd_comparison(results: dict):
    """Side-by-side comparison of blockwise with and without CCD."""
    if "blockwise_no_ccd" not in results or "blockwise_ccd" not in results:
        print("Skipping CCD comparison: missing results")
        return

    metrics = ["dino_score", "clip_t_score", "clip_i_score", "cae", "lpips_diversity"]
    labels = ["DINO", "CLIP-T", "CLIP-I", "CAE", "LPIPS"]

    no_ccd = results["blockwise_no_ccd"]["metrics"]
    with_ccd = results["blockwise_ccd"]["metrics"]

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    x = np.arange(len(metrics))
    width = 0.35

    vals_no_ccd = [no_ccd.get(m, 0) for m in metrics]
    vals_ccd = [with_ccd.get(m, 0) for m in metrics]

    bars1 = ax.bar(x - width / 2, vals_no_ccd, width, label="Blockwise", color=COLORS["blockwise_no_ccd"],
                   edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, vals_ccd, width, label="Blockwise+CCD", color=COLORS["blockwise_ccd"],
                   edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.legend(fontsize=7, framealpha=0.8)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=5)

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "ccd_comparison.pdf")
    fig.savefig(FIGURE_DIR / "ccd_comparison.png")
    plt.close(fig)
    print(f"Saved ccd_comparison")


def fig_loss_bar(results: dict):
    """Final loss comparison across configs."""
    order = ["uniform_r4", "baseline_r8", "uniform_r16", "blockwise_no_ccd", "blockwise_ccd"]
    configs = [c for c in order if c in results]

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    x = np.arange(len(configs))
    width = 0.6

    losses = [results[c].get("final_loss_total", 0) for c in configs]
    colors = [COLORS[c] for c in configs]

    bars = ax.bar(x, losses, width, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in configs], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Final Loss")
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, val in zip(bars, losses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.4f}", ha="center", va="bottom", fontsize=6)

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "loss_comparison.pdf")
    fig.savefig(FIGURE_DIR / "loss_comparison.png")
    plt.close(fig)
    print(f"Saved loss_comparison")


def fig_radar(results: dict):
    """Radar chart comparing all 5 configs on normalized metrics."""
    order = ["uniform_r4", "baseline_r8", "uniform_r16", "blockwise_no_ccd", "blockwise_ccd"]
    configs = [c for c in order if c in results]
    if len(configs) < 3:
        print("Skipping radar: too few configs")
        return

    metrics = ["dino_score", "clip_t_score", "clip_i_score", "lpips_diversity"]
    labels = ["DINO", "CLIP-T", "CLIP-I", "LPIPS"]

    # Collect and normalize
    raw = {}
    for m in metrics:
        vals = [results[c]["metrics"].get(m, 0) for c in configs]
        mn, mx = min(vals), max(vals)
        rng = mx - mn if mx > mn else 1.0
        raw[m] = [(v - mn) / rng for v in vals]

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(3.5, 3.5), subplot_kw=dict(polar=True))

    for i, c in enumerate(configs):
        values = [raw[m][i] for m in metrics]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=1.2, color=COLORS[c], label=LABELS[c], markersize=3)
        ax.fill(angles, values, alpha=0.05, color=COLORS[c])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_yticklabels([])
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=6.5, framealpha=0.8)

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "radar_comparison.pdf")
    fig.savefig(FIGURE_DIR / "radar_comparison.png")
    plt.close(fig)
    print(f"Saved radar_comparison")


def main():
    results = load_results()
    print(f"Loaded {len(results)} experiment results: {list(results.keys())}")

    if not results:
        print("No results found!")
        return

    fig_metrics_bar(results)
    fig_param_efficiency(results)
    fig_ccd_comparison(results)
    fig_loss_bar(results)
    fig_radar(results)
    print("All figures generated.")


if __name__ == "__main__":
    main()
