"""Render saved comparison statistics; no analysis or model fitting.

Presentation correction: draw random-mean markers above interval bars and move
the legend outside data rows. Numerical outputs and frozen analysis stay intact.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STUDY = Path(__file__).resolve().parents[2]
OUTPUT = STUDY / "results/group_comparison"


def run():
    results = json.loads((OUTPUT / "comparison.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, rule in zip(axes, ("primary", "size_15_500")):
        rows = [row for row in results if row["size_rule"] == rule]
        for y, row in enumerate(rows):
            stat = row["random_control_comparison"]["mean_pathway_jaccard"]
            ax.plot([stat["random_q025"], stat["random_q975"]], [y, y],
                    color="#b3bbc4", linewidth=7, zorder=1)
            ax.scatter(stat["random_mean"], y, c="#526d82", marker="o", zorder=2,
                       label="Random mean" if y == 0 else None)
            ax.scatter(stat["observed"], y, c="#ae542d", marker="D", zorder=3,
                       label="Biological groups" if y == 0 else None)
        ax.set(yticks=range(len(rows)), yticklabels=[r["label"] for r in rows], xlim=(0, 1),
               ylim=(-.4, 3.4), xlabel="Top-five agreement across folds (Jaccard)",
               title="All nonempty groups" if rule == "primary" else "15–500 measured genes per group")
        ax.invert_yaxis()
    fig.suptitle("Each collection compared with its own random groups\n"
                 "Grey bars: 2.5th–97.5th percentiles, not confidence intervals", fontsize=11)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.savefig(OUTPUT / "group_stability.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    run()
