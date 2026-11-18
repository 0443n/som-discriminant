"""Figures from results/benchmark/*.csv.

contamination.png : accuracy change vs. eps=0 under training-set contamination,
                    one panel per dataset (shared y), key methods coloured,
                    remaining baselines as grey context lines.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RES = Path(__file__).resolve().parent.parent / "results" / "benchmark"

SURFACE = "#fcfcfb"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#b9b8b2", "#e6e5e0"
# reference categorical slots, fixed order
KEY = {
    "v2_joint": ("v2 joint", "#2a78d6"),
    "v1": ("v1 (2023)", "#eb6834"),
    "lda": ("LDA", "#1baf7a"),
    "svm_rbf": ("SVM-RBF", "#eda100"),
    "rf": ("Random forest", "#e87ba4"),
    "v2_joint_t20": ("v2 joint, trim 0.2", "#008300"),
    "v3_density_t20": ("v3 density, trim 0.2", "#6250d6"),
}
TITLES = {"iris": "Iris", "wine": "Wine", "breast_cancer": "Breast cancer",
          "multimodal": "Multimodal (synthetic)", "rings": "Rings (synthetic)"}


def contamination():
    df = pd.read_csv(RES / "contamination.csv")
    m = df.groupby(["dataset", "method", "eps"]).acc.mean().reset_index()
    base = m[m.eps == 0].set_index(["dataset", "method"]).acc
    m["delta"] = m.acc - base.reindex(pd.MultiIndex.from_frame(m[["dataset", "method"]])).values
    datasets = [d for d in TITLES if d in set(m.dataset)]

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
    fig, axes = plt.subplots(1, len(datasets), figsize=(2.6 * len(datasets), 3.1),
                             sharey=True, facecolor=SURFACE)
    for ax, d in zip(axes, datasets):
        ax.set_facecolor(SURFACE)
        sub = m[m.dataset == d]
        for meth, g in sub.groupby("method"):
            if meth not in KEY:
                ax.plot(g.eps, g.delta, color=MUTED, lw=0.8, alpha=0.8, zorder=1)
        for meth, (label, col) in KEY.items():
            g = sub[sub.method == meth]
            ax.plot(g.eps, g.delta, color=col, lw=2, marker="o", ms=4.5,
                    markeredgecolor=SURFACE, markeredgewidth=1, label=label, zorder=3)
        ax.axhline(0, color=INK2, lw=0.8, zorder=2)
        ax.set_title(TITLES[d], fontsize=9.5, color=INK)
        ax.set_xticks([0, 0.1, 0.2, 0.3])
        ax.set_xlabel("contaminated fraction ε")
        ax.grid(axis="y", color=GRID, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("accuracy change vs. clean")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=MUTED, lw=0.8))
    labels.append("other baselines")
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(RES / "contamination.png", dpi=150, facecolor=SURFACE)
    print("wrote", RES / "contamination.png")


if __name__ == "__main__":
    contamination()
