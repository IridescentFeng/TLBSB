"""
Pareto plot for TLBSB paper.
Usage:
    python plot_pareto.py --data x1_bs_pareto_pku.json --output pareto.pdf
    python plot_pareto.py --data x1_bs_pareto_pku.json --output pareto.png --no_baseline
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
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "pdf.fonttype":      42,   # embeds fonts in PDF
    "ps.fonttype":       42,
})

# ── per-method display config ──────────────────────────────────────────────────
# (label, face_color, edge_color, marker, marker_size, zorder, label_dx, label_dy)
STYLE = {
    "X1_HtoS":  ("TLBSB H→S",  "#D94F00", "#D94F00", "o", 130, 6,  0.08, -0.45),
    "X1_StoH":  ("TLBSB S→H",  "#D94F00", "#D94F00", "o", 130, 6,  0.08,  0.20),
    "X1_HtoM":  ("TLBSB H→M",  "#FF8C55", "#D94F00", "o", 130, 6, -0.45,  0.20),
    "X1_StoM":  ("TLBSB S→M",  "#FF8C55", "#D94F00", "o", 130, 6, -0.52, -0.45),
    "SACPO_HtoS": ("SACPO H→S","#2A6DB5", "#2A6DB5", "s", 130, 5,  0.08, -0.45),
    "SACPO_StoH": ("SACPO S→H","#2A6DB5", "#2A6DB5", "s", 130, 5,  0.08,  0.20),
    "SACPO_HtoM": ("SACPO H→M","#7CB9E8", "#2A6DB5", "s", 130, 5, -0.52,  0.20),
    "SACPO_StoM": ("SACPO S→M","#7CB9E8", "#2A6DB5", "s", 130, 5,  0.08,  0.20),
    "V6_HtoS":  ("V6 H→S",     "#6A4C9C", "#6A4C9C", "^", 120, 4,  0.08,  0.20),
    "V6_StoH":  ("V6 S→H",     "#6A4C9C", "#6A4C9C", "^", 120, 4,  0.08, -0.45),
    "Helpful_baseline": ("π_r","#555555", "#555555", "o", 110, 3,  0.08,  0.20),
    "Safety_baseline":  ("π_s","#555555", "#555555", "D", 110, 3,  0.08,  0.20),
}

# legend group order
LEGEND_GROUPS = [
    ("TLBSB (ours)", ["X1_HtoS", "X1_StoH", "X1_HtoM", "X1_StoM"]),
    ("SACPO",        ["SACPO_HtoS", "SACPO_StoH", "SACPO_HtoM", "SACPO_StoM"]),
    ("V6 (ablation)",["V6_HtoS", "V6_StoH"]),
    ("Baselines",    ["Helpful_baseline", "Safety_baseline"]),
]


def pareto_frontier(points):
    """Return indices of Pareto-optimal points (maximise both axes)."""
    pts = np.array(points)
    n = len(pts)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if pts[j, 0] >= pts[i, 0] and pts[j, 1] >= pts[i, 1] and (
                    pts[j, 0] > pts[i, 0] or pts[j, 1] > pts[i, 1]):
                dominated[i] = True
                break
    idx = np.where(~dominated)[0]
    idx = idx[np.argsort(pts[idx, 0])]  # sort by x for line drawing
    return idx


def draw_pareto_line(ax, data, methods_in_data):
    xs = [data[m]["x"] for m in methods_in_data]
    ys = [data[m]["y"] for m in methods_in_data]
    pts = list(zip(xs, ys))
    idx = pareto_frontier(pts)
    px = [pts[i][0] for i in idx]
    py = [pts[i][1] for i in idx]
    ax.plot(px, py, color="#AAAAAA", linewidth=1.4,
            linestyle="--", zorder=1, label="Pareto frontier")


def plot_pareto(data_path, output_path, no_baseline=False,
                xlabel="Helpful reward (mean ± SE)  →  more helpful",
                ylabel="Safety score (mean ± SE)  →  safer",
                title=None):

    with open(data_path) as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    ax.set_facecolor("#FAFAFA")
    ax.grid(True, color="#E0E0E0", linewidth=0.8, zorder=0)

    baseline_keys = {"Helpful_baseline", "Safety_baseline"}
    active = {k: v for k, v in data.items()
              if k in STYLE and (not no_baseline or k not in baseline_keys)}

    # Pareto frontier (exclude baselines from frontier)
    frontier_methods = [k for k in active if k not in baseline_keys]
    if frontier_methods:
        draw_pareto_line(ax, data, frontier_methods)

    # plot points + error bars
    for key, vals in active.items():
        if key not in STYLE:
            continue
        label, fc, ec, mk, ms, zo, dx, dy = STYLE[key]
        x, y   = vals["x"], vals["y"]
        xe, ye = vals.get("x_se", 0), vals.get("y_se", 0)

        ax.errorbar(x, y, xerr=xe, yerr=ye,
                    fmt="none", ecolor=ec, elinewidth=1.0,
                    capsize=3, capthick=1.0, zorder=zo - 1, alpha=0.7)

        ax.scatter(x, y, s=ms, marker=mk,
                   facecolors=fc, edgecolors=ec,
                   linewidths=1.2, zorder=zo)

        # annotation
        ax.annotate(
            label, xy=(x, y), xytext=(x + dx, y + dy),
            fontsize=9.5, color="#222222",
            ha="left", va="center",
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
        )

    # ── legend ────────────────────────────────────────────────────────────────
    legend_handles = []
    for group_name, keys in LEGEND_GROUPS:
        present = [k for k in keys if k in active]
        if not present:
            continue
        # group header (invisible spacer)
        legend_handles.append(
            Line2D([0], [0], linestyle="none", marker="none",
                   label=f"$\\bf{{{group_name}}}$")
        )
        seen_shapes = {}
        for k in present:
            lbl, fc, ec, mk, ms, *_ = STYLE[k]
            shape_key = (mk, fc)
            if shape_key in seen_shapes:
                continue
            seen_shapes[shape_key] = True
            legend_handles.append(
                Line2D([0], [0], linestyle="none",
                       marker=mk, markersize=7,
                       markerfacecolor=fc, markeredgecolor=ec,
                       markeredgewidth=1.2, label=lbl)
            )

    # pareto line entry
    legend_handles.append(
        Line2D([0], [0], color="#AAAAAA", linewidth=1.4,
               linestyle="--", label="Pareto frontier")
    )

    ax.legend(handles=legend_handles,
              loc="lower left", fontsize=8.5,
              framealpha=0.92, edgecolor="#CCCCCC",
              handlelength=1.5, handletextpad=0.6,
              borderpad=0.7, labelspacing=0.35)

    ax.set_xlabel(xlabel, fontsize=11, labelpad=6)
    ax.set_ylabel(ylabel, fontsize=11, labelpad=6)
    if title:
        ax.set_title(title, fontsize=12, pad=10)

    ax.tick_params(labelsize=10)
    fig.tight_layout(pad=1.5)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")

    # also save companion format
    if output_path.endswith(".pdf"):
        fig.savefig(output_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path.replace('.pdf', '.png')}")
    elif output_path.endswith(".png"):
        fig.savefig(output_path.replace(".png", ".pdf"), dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path.replace('.png', '.pdf')}")

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",        default="x1_bs_pareto_pku.json",
                        help="Pareto data JSON ({method: {x, x_se, y, y_se}})")
    parser.add_argument("--output",      default="pareto.pdf")
    parser.add_argument("--xlabel",      default="Helpful reward (mean ± SE)  →  more helpful")
    parser.add_argument("--ylabel",      default="Safety score (mean ± SE)  →  safer")
    parser.add_argument("--title",       default=None)
    parser.add_argument("--no_baseline", action="store_true",
                        help="Hide π_r / π_s baseline points")
    args = parser.parse_args()

    plot_pareto(args.data, args.output,
                no_baseline=args.no_baseline,
                xlabel=args.xlabel,
                ylabel=args.ylabel,
                title=args.title)


if __name__ == "__main__":
    main()
