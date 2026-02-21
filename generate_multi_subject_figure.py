"""Generate multi-subject comparison figure for the paper (2x2 factorial)."""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load results
with open("outputs/experiments/multi_subject/results.json") as f:
    ccd_data = json.load(f)
with open("outputs/experiments/multi_subject_noccd/results.json") as f:
    noccd_data = json.load(f)

# 4 conditions: naive_noccd, naive_ccd, masked_noccd, masked_ccd
conditions = [
    ("Naive\n(no CCD)", noccd_data["naive"]),
    ("Naive\n(+CCD)", ccd_data["naive"]),
    ("Masked\n(no CCD)", noccd_data["masked"]),
    ("Masked\n(+CCD)", ccd_data["masked"]),
]

fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))

metric_defs = [
    ("IIS", "iis", True),
    ("CLIP-T", "clip_t", True),
    ("Comp. Diversity", "composition_diversity", True),
    ("Cross-Contamination", None, False),  # computed below
]

colors = ["#DD8452", "#E8A87C", "#4C72B0", "#7BA3CC"]

for ax, (metric_name, metric_key, higher_better) in zip(axes, metric_defs):
    x = np.arange(len(conditions))
    if metric_key is not None:
        vals = [d[metric_key] for _, d in conditions]
    else:
        vals = [(d["dog_cross_sim"] + d["cat_cross_sim"]) / 2 for _, d in conditions]

    bars = ax.bar(x, vals, width=0.65, color=colors, edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + max(vals) * 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold",
        )

    labels = [name for name, _ in conditions]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_title(metric_name, fontsize=10, fontweight="bold")
    arrow = r"$\uparrow$" if higher_better else r"$\downarrow$"
    ax.set_ylabel(f"Score {arrow}", fontsize=8)
    ax.set_ylim(0, max(vals) * 1.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()

out_dir = Path("paper/figures")
out_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(out_dir / "multi_subject_comparison.pdf", bbox_inches="tight", dpi=300)
fig.savefig(out_dir / "multi_subject_comparison.png", bbox_inches="tight", dpi=300)
plt.close(fig)

print("Multi-subject 2x2 figure saved.")
