"""Generate multi-subject comparison figure for the paper."""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load results
results_path = Path("outputs/experiments/multi_subject/results.json")
with open(results_path) as f:
    data = json.load(f)

masked = data["masked"]
naive = data["naive"]

# Figure: grouped bar chart of multi-subject metrics
fig, axes = plt.subplots(1, 4, figsize=(12, 3))

metrics = [
    ("IIS", masked["iis"], naive["iis"], True),
    ("CLIP-T", masked["clip_t"], naive["clip_t"], True),
    ("Comp. Diversity", masked["composition_diversity"], naive["composition_diversity"], True),
    ("Cross-Contam.", (masked["dog_cross_sim"] + masked["cat_cross_sim"]) / 2,
     (naive["dog_cross_sim"] + naive["cat_cross_sim"]) / 2, False),
]

colors_masked = "#4C72B0"
colors_naive = "#DD8452"

for ax, (name, m_val, n_val, higher_better) in zip(axes, metrics):
    x = np.array([0, 0.6])
    vals = [n_val, m_val]
    colors = [colors_naive, colors_masked]
    labels = ["Naïve", "Masked"]

    bars = ax.bar(x, vals, width=0.4, color=colors, edgecolor="black", linewidth=0.5)

    # Annotate bars
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + max(vals) * 0.03,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title(name, fontsize=10, fontweight="bold")
    arrow = "↑" if higher_better else "↓"
    ax.set_ylabel(f"Score ({arrow})", fontsize=8)
    ax.set_ylim(0, max(vals) * 1.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()

out_dir = Path("paper/figures")
out_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(out_dir / "multi_subject_comparison.pdf", bbox_inches="tight", dpi=300)
fig.savefig(out_dir / "multi_subject_comparison.png", bbox_inches="tight", dpi=300)
plt.close(fig)

print("Multi-subject figure saved.")
