"""Generate publication-quality figures from all experiment results (v2).

Includes new figures for:
- Rank sensitivity analysis
- CCD lambda ablation
- Training duration comparison
- Updated parameter efficiency plot with all configs
"""

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
    for path in OUTPUT_BASE.glob("*/results.json"):
        name = path.parent.name
        with open(path) as f:
            results[name] = json.load(f)
    return results


def fig_metrics_bar(results: dict):
    """Bar chart comparing all metrics across base configurations."""
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

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.3f}", ha="center", va="bottom", fontsize=5.5)

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "metrics_comparison.pdf")
    fig.savefig(FIGURE_DIR / "metrics_comparison.png")
    plt.close(fig)
    print("Saved metrics_comparison")


def fig_param_efficiency(results: dict):
    """Scatter plot: parameter count vs DINO score with all configs."""
    # Include rank sensitivity experiments too
    order = ["uniform_r4", "baseline_r8", "uniform_r16", "blockwise_no_ccd", "blockwise_ccd"]
    rank_exps = ["rank_sensitivity_id8", "rank_sensitivity_id12", "rank_sensitivity_id24", "rank_sensitivity_id32"]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    # Plot base configs
    for c in order:
        if c not in results:
            continue
        r = results[c]
        params_k = r["lora_params"] / 1000
        dino = r["metrics"].get("dino_score", 0)
        ax.scatter(params_k, dino, c=COLORS[c], s=60, zorder=5, edgecolors="black", linewidths=0.5)
        offset = (5, 5) if c != "blockwise_no_ccd" else (5, -12)
        ax.annotate(LABELS[c], (params_k, dino), textcoords="offset points",
                    xytext=offset, fontsize=6, ha="left")

    # Plot rank sensitivity as smaller gray dots with connecting line
    rank_data = []
    for rexp in rank_exps:
        if rexp in results:
            r = results[rexp]
            rank_data.append((r["lora_params"] / 1000, r["metrics"].get("dino_score", 0)))

    if rank_data:
        # Add blockwise (rank 16) to the line
        if "blockwise_no_ccd" in results:
            bw = results["blockwise_no_ccd"]
            rank_data.append((bw["lora_params"] / 1000, bw["metrics"].get("dino_score", 0)))
        rank_data.sort(key=lambda x: x[0])
        rx, ry = zip(*rank_data)
        ax.plot(rx, ry, 'o--', color='gray', markersize=4, alpha=0.6, linewidth=1,
                label='Rank sensitivity', zorder=3)

    ax.set_xlabel("LoRA Parameters (K)")
    ax.set_ylabel("DINO Score")
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if rank_data:
        ax.legend(fontsize=6, loc='lower right')

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "param_efficiency.pdf")
    fig.savefig(FIGURE_DIR / "param_efficiency.png")
    plt.close(fig)
    print("Saved param_efficiency")


def fig_rank_sensitivity(results: dict):
    """Line plot showing DINO and CLIP-T vs identity rank."""
    ranks = [8, 12, 16, 24, 32]
    exp_names = {
        8: "rank_sensitivity_id8",
        12: "rank_sensitivity_id12",
        16: "blockwise_no_ccd",
        24: "rank_sensitivity_id24",
        32: "rank_sensitivity_id32",
    }

    dino_vals, clipt_vals, params_vals = [], [], []
    valid_ranks = []

    for r in ranks:
        name = exp_names[r]
        if name in results:
            m = results[name]["metrics"]
            dino_vals.append(m.get("dino_score", 0))
            clipt_vals.append(m.get("clip_t_score", 0))
            params_vals.append(results[name]["lora_params"] / 1e6)
            valid_ranks.append(r)

    if len(valid_ranks) < 3:
        print("Skipping rank_sensitivity: too few data points")
        return

    fig, ax1 = plt.subplots(figsize=(3.5, 2.5))

    color1 = "#d62728"
    color2 = "#1f77b4"

    ax1.plot(valid_ranks, dino_vals, 'o-', color=color1, linewidth=1.5, markersize=5, label='DINO')
    ax1.set_xlabel(r'Identity Block Rank ($r_{\mathrm{id}}$)')
    ax1.set_ylabel('DINO Score', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.plot(valid_ranks, clipt_vals, 's--', color=color2, linewidth=1.5, markersize=5, label='CLIP-T')
    ax2.set_ylabel('CLIP-T Score', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='lower right')

    ax1.set_xticks(valid_ranks)
    ax1.grid(alpha=0.3, linewidth=0.5)
    ax1.spines["top"].set_visible(False)

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "rank_sensitivity.pdf")
    fig.savefig(FIGURE_DIR / "rank_sensitivity.png")
    plt.close(fig)
    print("Saved rank_sensitivity")


def fig_ccd_lambda(results: dict):
    """Bar chart showing metrics across CCD lambda values."""
    lambdas = [0.0, 0.1, 0.3, 0.5, 1.0]
    exp_names = {
        0.0: "blockwise_no_ccd",
        0.1: "ccd_lambda_01",
        0.3: "blockwise_ccd",
        0.5: "ccd_lambda_05",
        1.0: "ccd_lambda_10",
    }

    valid_lambdas = []
    dino_vals, vqa_vals = [], []

    for lam in lambdas:
        name = exp_names[lam]
        if name in results:
            m = results[name]["metrics"]
            valid_lambdas.append(lam)
            dino_vals.append(m.get("dino_score", 0))
            vqa_vals.append(m.get("vqa_alignment", 0))

    if len(valid_lambdas) < 3:
        print("Skipping ccd_lambda: too few data points")
        return

    fig, ax1 = plt.subplots(figsize=(3.5, 2.5))

    x = np.arange(len(valid_lambdas))
    width = 0.35

    color1 = "#d62728"
    color2 = "#9467bd"

    bars1 = ax1.bar(x - width/2, dino_vals, width, color=color1, alpha=0.8, label='DINO')
    bars2 = ax1.bar(x + width/2, vqa_vals, width, color=color2, alpha=0.8, label='VQA')

    ax1.set_xlabel(r'CCD Loss Weight ($\lambda_2$)')
    ax1.set_ylabel('Score')
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(l) for l in valid_lambdas])
    ax1.legend(fontsize=7)
    ax1.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h,
                     f"{h:.3f}", ha="center", va="bottom", fontsize=5.5)

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "ccd_lambda.pdf")
    fig.savefig(FIGURE_DIR / "ccd_lambda.png")
    plt.close(fig)
    print("Saved ccd_lambda")


def fig_training_duration(results: dict):
    """Grouped bar chart showing 500 vs 1000 steps for different configs."""
    configs_500 = ["baseline_r8", "blockwise_no_ccd", "blockwise_ccd"]
    configs_1000 = ["uniform_r8_1000", "blockwise_1000", "blockwise_ccd_1000"]
    group_labels = ["Uniform r=8", "Blockwise", "Blockwise+CCD"]

    # Check data availability
    available = all(c in results for c in configs_500 + configs_1000)
    if not available:
        print(f"Skipping training_duration: missing some configs")
        missing = [c for c in configs_500 + configs_1000 if c not in results]
        print(f"  Missing: {missing}")
        return

    metrics = ["dino_score", "clip_t_score", "vqa_alignment"]
    metric_labels = ["DINO", "CLIP-T", "VQA"]

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))

    x = np.arange(len(group_labels))
    width = 0.35

    for ax, metric, label in zip(axes, metrics, metric_labels):
        vals_500 = [results[c]["metrics"].get(metric, 0) for c in configs_500]
        vals_1000 = [results[c]["metrics"].get(metric, 0) for c in configs_1000]

        bars1 = ax.bar(x - width/2, vals_500, width, label='500 steps', color='#1f77b4', alpha=0.8)
        bars2 = ax.bar(x + width/2, vals_1000, width, label='1000 steps', color='#ff7f0e', alpha=0.8)

        ax.set_title(label)
        ax.set_xticks(x)
        ax.set_xticklabels(group_labels, rotation=20, ha="right", fontsize=7)
        all_vals = vals_500 + vals_1000
        ax.set_ylim(bottom=min(all_vals) * 0.92, top=max(all_vals) * 1.06)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for bars in [bars1, bars2]:
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=5)

    axes[0].legend(fontsize=6, loc='upper left')

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "training_duration.pdf")
    fig.savefig(FIGURE_DIR / "training_duration.png")
    plt.close(fig)
    print("Saved training_duration")


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

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=5)

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "ccd_comparison.pdf")
    fig.savefig(FIGURE_DIR / "ccd_comparison.png")
    plt.close(fig)
    print("Saved ccd_comparison")


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
    print("Saved loss_comparison")


def fig_radar(results: dict):
    """Radar chart comparing all 5 base configs on normalized metrics."""
    order = ["uniform_r4", "baseline_r8", "uniform_r16", "blockwise_no_ccd", "blockwise_ccd"]
    configs = [c for c in order if c in results]
    if len(configs) < 3:
        print("Skipping radar: too few configs")
        return

    metrics = ["dino_score", "clip_t_score", "clip_i_score", "lpips_diversity"]
    labels = ["DINO", "CLIP-T", "CLIP-I", "LPIPS"]

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
    print("Saved radar_comparison")


def main():
    results = load_results()
    print(f"Loaded {len(results)} experiment results: {sorted(results.keys())}")

    if not results:
        print("No results found!")
        return

    # Original figures (updated)
    fig_metrics_bar(results)
    fig_param_efficiency(results)
    fig_ccd_comparison(results)
    fig_loss_bar(results)
    fig_radar(results)

    # New figures
    fig_rank_sensitivity(results)
    fig_ccd_lambda(results)
    fig_training_duration(results)

    print("All figures generated.")


if __name__ == "__main__":
    main()
