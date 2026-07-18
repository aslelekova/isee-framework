import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

from .data import DIMENSIONS
from . import aggregate

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral"],
    "font.size": 11, "axes.titlesize": 11.5, "axes.labelsize": 10.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#d9d9d9", "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.dpi": 200,
})

CASE_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
DIM_COLORS = {"FV": "#0072B2", "MF": "#56B4E9", "SV": "#009E73",
              "EX": "#E69F00", "LL": "#D55E00"}
CLUSTER_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9",
                  "#D55E00"]


def radar(B, labels, path):
    axes_lbl = ["FV", "MF", "SV", "EX\n(inverted)", "LL\n(inverted)"]
    ang = np.linspace(0, 2 * np.pi, 5, endpoint=False).tolist()
    fig = plt.figure(figsize=(6.6, 5.4))
    ax = fig.add_subplot(polar=True)
    for j, lbl in enumerate(labels):
        vals = B[j].tolist()
        ax.plot(ang + ang[:1], vals + vals[:1], color=CASE_COLORS[j], lw=2,
                label=lbl)
        ax.fill(ang + ang[:1], vals + vals[:1], color=CASE_COLORS[j],
                alpha=0.08)
    ax.set_xticks(ang); ax.set_xticklabels(axes_lbl)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0]); ax.set_ylim(0, 1.02)
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8.5,
                       color="#666666")
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.08), frameon=False)
    ax.set_title("Normalised dimension profiles of the four mineral systems",
                 pad=24)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def contributions(S, w, labels, path):
    isee = aggregate.additive(S, w)
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    x = np.arange(len(labels))
    bot_pos, bot_neg = np.zeros(len(labels)), np.zeros(len(labels))
    for i, d in enumerate(DIMENSIONS):
        contrib = w[i] * aggregate.SIGNS[i] * S[:, i]
        bot = bot_pos if aggregate.SIGNS[i] > 0 else bot_neg
        ax.bar(x, contrib, 0.55, bottom=bot, color=DIM_COLORS[d],
               edgecolor="white", linewidth=1.2, label=d)
        if aggregate.SIGNS[i] > 0:
            bot_pos += contrib
        else:
            bot_neg += contrib
    ax.scatter(x, isee, marker="D", s=55, color="#111111", zorder=5,
               label="Net ISEE")
    for j in range(len(labels)):
        ax.annotate(f"{isee[j]:+.3f}", (x[j] + 0.34, isee[j]), va="center",
                    fontsize=9.5)
    ax.axhline(0, color="#444444", lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlim(-0.6, len(labels))
    ax.set_ylabel("Weighted contribution to ISEE")
    ax.set_title("Dimension contributions and net ISEE score (equal weights)")
    ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.14),
              frameon=False)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def mc_violin(scores, rank1, labels, path,
              title="Monte Carlo sensitivity analysis: ISEE under 10,000 "
                    "random weightings"):
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    x = np.arange(len(labels))
    parts = ax.violinplot([scores[:, j] for j in range(len(labels))],
                          positions=x, showmedians=False, showextrema=False)
    for j, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(CASE_COLORS[j]); pc.set_alpha(0.45)
    bp = ax.boxplot([scores[:, j] for j in range(len(labels))], positions=x,
                    widths=0.16, showfliers=False, patch_artist=True,
                    medianprops=dict(color="black", lw=1.4))
    for j, b in enumerate(bp["boxes"]):
        b.set(facecolor="white", edgecolor=CASE_COLORS[j], linewidth=1.4)
    for j in range(len(labels)):
        ax.annotate(f"rank 1 in {rank1[j]:.0%}\nof draws",
                    (x[j], scores[:, j].max() + 0.03), ha="center",
                    fontsize=8.5, color="#444444")
    ax.axhline(0, color="#444444", lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("ISEE score")
    ax.set_ylim(scores.min() - 0.05, scores.max() + 0.16)
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def typology_scatter(coords, var, clusters, systems, path):
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    for c in sorted(set(clusters)):
        m = clusters == c
        ax.scatter(coords[m, 0], coords[m, 1], s=55,
                   color=CLUSTER_COLORS[c], label=f"Cluster {c + 1}",
                   edgecolor="white", linewidth=0.8, zorder=3)
    for j, sid in enumerate(systems.ids):
        star = systems.flagship[j]
        ax.annotate(sid, (coords[j, 0], coords[j, 1]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=8.5 if star else 7.5,
                    fontweight="bold" if star else "normal",
                    color="#111111" if star else "#555555")
    for j in np.where(systems.flagship)[0]:
        ax.scatter(coords[j, 0], coords[j, 1], s=150, facecolor="none",
                   edgecolor="#111111", linewidth=1.3, zorder=4)
    ax.set_xlabel(f"PC1 ({var[0]:.0%} of variance)")
    ax.set_ylabel(f"PC2 ({var[1]:.0%} of variance)")
    ax.set_title("Typology of 30 country–mineral systems in the ISEE "
                 "dimension space\n(k-means clusters; case-study systems "
                 "circled)")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def typology_dendrogram(Z, systems, clusters, path):
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    labels = [f"{sid}*" if systems.flagship[j] else sid
              for j, sid in enumerate(systems.ids)]
    dendrogram(Z, labels=labels, ax=ax, color_threshold=0,
               above_threshold_color="#666666", leaf_font_size=8)
    for tick in ax.get_xticklabels():
        sid = tick.get_text().rstrip("*")
        j = systems.ids.index(sid)
        tick.set_color(CLUSTER_COLORS[clusters[j]])
        if systems.flagship[j]:
            tick.set_fontweight("bold")
    ax.set_ylabel("Ward linkage distance")
    ax.set_title("Hierarchical clustering of the 30 mineral systems "
                 "(leaf colour = k-means cluster; * = case-study system)")
    ax.grid(False)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
