"""
Pareto plot for TLBSB paper — two-panel layout.
  Left panel:  all H→* methods (HtoS, HtoM)
  Right panel: all S→* methods (StoH, StoM)

Usage:
    python plot_pareto.py --data x1_bs_pareto_pku.json --output pareto.pdf
"""

import json
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

matplotlib.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         12,
    "axes.linewidth":    1.2,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
})

# ── color / marker definitions ────────────────────────────────────────────────
COLOR = {
    "X1":    ("#D94F00", "#D94F00"),   # (facecolor, edgecolor)
    "SACPO": ("#2A6DB5", "#2A6DB5"),
    "V6":    ("#6A4C9C", "#6A4C9C"),
    "base":  ("#777777", "#777777"),
}
MARKER = {
    "toS": "o",   # →S  (pure safety)
    "toH": "o",   # →H  (pure helpful)
    "toM": "D",   # →M  (mixed)
}
MS = 120   # marker size

# ── per-method config ─────────────────────────────────────────────────────────
# key: (family, dest, display_label, label_dx, label_dy)
META = {
    "X1_HtoS":   ("X1",    "toS", "TLBSB H→S",   0.06, -0.50),
    "X1_HtoM":   ("X1",    "toM", "TLBSB H→M",   0.06,  0.25),
    "X1_StoH":   ("X1",    "toH", "TLBSB S→H",   0.06,  0.25),
    "X1_StoM":   ("X1",    "toM", "TLBSB S→M",   0.06, -0.50),
    "SACPO_HtoS":("SACPO", "toS", "SACPO H→S",   0.06, -0.50),
    "SACPO_HtoM":("SACPO", "toM", "SACPO H→M",   0.06,  0.25),
    "SACPO_StoH":("SACPO", "toH", "SACPO S→H",   0.06,  0.25),
    "SACPO_StoM":("SACPO", "toM", "SACPO S→M",   0.06, -0.50),
    "V6_HtoS":   ("V6",    "toS", "V6 H→S",      0.06,  0.25),
    "V6_HtoM":   ("V6",    "toM", "V6 H→M",      0.06, -0.50),
    "V6_StoH":   ("V6",    "toH", "V6 S→H",      0.06,  0.25),
    "V6_StoM":   ("V6",    "toM", "V6 S→M",      0.06, -0.50),
    "Helpful_baseline": ("base", "toS", "π_r (init)", 0.06, 0.25),
    "Safety_baseline":  ("base", "toH", "π_s (init)", 0.06, 0.25),
}

# which keys go in each panel
PANELS = {
    "H→*": ["Helpful_baseline",
             "X1_HtoS", "X1_HtoM",
             "SACPO_HtoS", "SACPO_HtoM",
             "V6_HtoS", "V6_HtoM"],
    "S→*": ["Safety_baseline",
             "X1_StoH", "X1_StoM",
             "SACPO_StoH", "SACPO_StoM",
             "V6_StoH", "V6_StoM"],
}


def pareto_frontier(points):
    pts = np.array(points)
    n = len(pts)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (pts[j, 0] >= pts[i, 0] and pts[j, 1] >= pts[i, 1] and
                    (pts[j, 0] > pts[i, 0] or pts[j, 1] > pts[i, 1])):
                dominated[i] = True
                break
    idx = np.where(~dominated)[0]
    return idx[np.argsort(pts[idx, 0])]


def draw_frontier(ax, xs, ys):
    pts = list(zip(xs, ys))
    idx = pareto_frontier(pts)
    px = [pts[i][0] for i in idx]
    py = [pts[i][1] for i in idx]
    ax.plot(px, py, color="#BBBBBB", linewidth=1.4,
            linestyle="--", zorder=1)


def plot_panel(ax, data, keys, panel_title,
               xlabel="Helpful reward (mean ± SE)",
               ylabel="Safety score (mean ± SE)"):
    ax.set_facecolor("#FAFAFA")
    ax.grid(True, color="#E2E2E2", linewidth=0.8, zorder=0)
    ax.set_title(panel_title, fontsize=13, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=10.5, labelpad=5)
    ax.set_ylabel(ylabel, fontsize=10.5, labelpad=5)
    ax.tick_params(labelsize=9.5)

    present = [k for k in keys if k in data and k in META]
    baseline_keys = {"Helpful_baseline", "Safety_baseline"}
    frontier_keys = [k for k in present if k not in baseline_keys]

    # Pareto frontier
    if frontier_keys:
        xs = [data[k]["x"] for k in frontier_keys]
        ys = [data[k]["y"] for k in frontier_keys]
        draw_frontier(ax, xs, ys)

    # points
    for key in present:
        vals = data[key]
        fam, dest, label, dx, dy = META[key]
        fc, ec = COLOR[fam]
        mk = MARKER[dest]
        x, y   = vals["x"], vals["y"]
        xe, ye = vals.get("x_se", 0), vals.get("y_se", 0)

        zo = 2 if fam == "base" else 4
        alpha_err = 0.5 if fam == "base" else 0.75

        ax.errorbar(x, y, xerr=xe, yerr=ye,
                    fmt="none", ecolor=ec, elinewidth=0.9,
                    capsize=3, capthick=0.9, zorder=zo - 1, alpha=alpha_err)

        ax.scatter(x, y, s=MS, marker=mk,
                   facecolors=fc if fam != "base" else "none",
                   edgecolors=ec, linewidths=1.4, zorder=zo)

        ax.annotate(
            label, xy=(x, y), xytext=(x + dx, y + dy),
            fontsize=8.8, color="#1a1a1a", ha="left", va="center",
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
        )


def build_legend():
    """Shared legend entries."""
    handles = []

    # family colors
    handles.append(Line2D([0], [0], linestyle="none", marker="none",
                          label="$\\bf{Method}$"))
    for fam, (fc, ec) in COLOR.items():
        if fam == "base":
            continue
        name = {"X1": "TLBSB (ours)", "SACPO": "SACPO", "V6": "V6 (ablation)"}[fam]
        handles.append(Line2D([0], [0], linestyle="none",
                               marker="o", markersize=8,
                               markerfacecolor=fc, markeredgecolor=ec,
                               markeredgewidth=1.3, label=name))

    # baseline
    handles.append(Line2D([0], [0], linestyle="none",
                           marker="o", markersize=8,
                           markerfacecolor="none", markeredgecolor="#777777",
                           markeredgewidth=1.3, label="Init baseline (π_r / π_s)"))

    # marker shapes
    handles.append(Line2D([0], [0], linestyle="none", marker="none",
                          label="$\\bf{Stage\\;2\\;data}$"))
    handles.append(Line2D([0], [0], linestyle="none",
                           marker="o", markersize=8,
                           markerfacecolor="#888888", markeredgecolor="#888888",
                           label="Pure (→S / →H)"))
    handles.append(Line2D([0], [0], linestyle="none",
                           marker="D", markersize=7,
                           markerfacecolor="#888888", markeredgecolor="#888888",
                           label="Mixed (→M)"))

    # pareto line
    handles.append(Line2D([0], [0], color="#BBBBBB", linewidth=1.4,
                           linestyle="--", label="Pareto frontier"))
    return handles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default="x1_bs_pareto_pku.json")
    parser.add_argument("--output", default="pareto.pdf")
    parser.add_argument("--ylabel", default="Safety score (mean ± SE)  →  safer")
    parser.add_argument("--xlabel", default="Helpful reward (mean ± SE)  →  more helpful")
    args = parser.parse_args()

    with open(args.data) as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.subplots_adjust(wspace=0.32)

    for ax, (panel_title, keys) in zip(axes, PANELS.items()):
        plot_panel(ax, data, keys, panel_title,
                   xlabel=args.xlabel, ylabel=args.ylabel)

    legend = build_legend()
    fig.legend(handles=legend,
               loc="lower center",
               ncol=7,
               fontsize=9,
               framealpha=0.92,
               edgecolor="#CCCCCC",
               handlelength=1.4,
               handletextpad=0.5,
               columnspacing=1.0,
               bbox_to_anchor=(0.5, -0.12))

    fig.tight_layout(rect=[0, 0.08, 1, 1])

    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Saved: {args.output}")

    alt = args.output.replace(".pdf", ".png") if args.output.endswith(".pdf") \
        else args.output.replace(".png", ".pdf")
    fig.savefig(alt, dpi=300, bbox_inches="tight")
    print(f"Saved: {alt}")
    plt.close(fig)


if __name__ == "__main__":
    main()
