"""
Pareto plot for TLBSB paper - two-panel layout.
  Left panel:  all H->* methods (HtoS, HtoM)
  Right panel: all S->* methods (StoH, StoM)

Each point is one method x stage-2-data configuration; absolute scores
(not normalized) from MD-Judge (safety) and Beaver-Helpful (helpful).
Dashed line: empirical Pareto frontier across all methods within panel.
All single-stage / two-stage methods (SACPO, TLBSB, V6) use beta=0.05.
Significance markers vs SACPO (paired t-test): * p<0.05, ** p<0.01,
*** p<0.001.

Usage:
    python plot_pareto.py --data pareto_mdjudge.json --output pareto.pdf
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
    "font.size":         9,
    "axes.linewidth":    0.9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
})

# ── color / marker config ─────────────────────────────────────────────────────
COLOR = {
    "X1":    ("#D94F00", "#D94F00"),
    "SACPO": ("#2A6DB5", "#2A6DB5"),
    "V6":    ("#6A4C9C", "#6A4C9C"),
    "base":  ("#777777", "#777777"),
}
MARKER = {
    "toS": "o",
    "toH": "o",
    "toM": "D",
}
MS = 55

# ── per-method metadata ───────────────────────────────────────────────────────
# (color_family, marker_dest, display_label)
# Significance: TLBSB H→S vs SACPO is *** (p<0.001) per ttest.py.
# Edit the ★/★★/★★★ on the other TLBSB labels after running ttest.py
# on the corresponding pairs.
META = {
    "X1_HtoS":          ("X1",    "toS", "TLBSB H→S ★★★"),
    "X1_HtoM":          ("X1",    "toM", "TLBSB H→M ★★"),
    "X1_StoH":          ("X1",    "toH", "TLBSB S→H ★★"),
    "X1_StoM":          ("X1",    "toM", "TLBSB S→M ★"),
    "SACPO_HtoS":       ("SACPO", "toS", "SACPO H→S"),
    "SACPO_HtoM":       ("SACPO", "toM", "SACPO H→M"),
    "SACPO_StoH":       ("SACPO", "toH", "SACPO S→H"),
    "SACPO_StoM":       ("SACPO", "toM", "SACPO S→M"),
    "V6_HtoS":          ("V6",    "toS", "V6 H→S"),
    "V6_HtoM":          ("V6",    "toM", "V6 H→M"),
    "V6_StoH":          ("V6",    "toH", "V6 S→H"),
    "V6_StoM":          ("V6",    "toM", "V6 S→M"),
    "Helpful_baseline": ("base",  "toS", "π_r (init)"),
    "Safety_baseline":  ("base",  "toH", "π_s (init)"),
}

# ── panel-specific label positions ────────────────────────────────────────────
# (dx, dy, ha, va, use_arrow)
# dx/dy: data-unit offset from the point to the text anchor
# use_arrow=True: draw a thin connector line from text to point
H_ANNO = {
    "Helpful_baseline": ( 0.08,  0.09, "left",  "center", False),
    "SACPO_HtoS":       ( 0.08,  0.00, "left",  "center", False),
    "V6_HtoM":          (-0.10,  0.22, "right", "bottom", True),
    "X1_HtoM":          ( 0.06,  0.22, "left",  "bottom", True),
    "SACPO_HtoM":       (-0.65, -0.15, "right", "top",    True),
    "V6_HtoS":          ( 0.10, -0.30, "left",  "top",    True),
    "X1_HtoS":          ( 0.08,  0.00, "left",  "center", False),
}
S_ANNO = {
    "Safety_baseline":  (-0.12,  0.00, "right", "center", False),
    "SACPO_StoH":       (-0.10,  0.00, "right", "center", False),
    "V6_StoH":          ( 0.08, -0.14, "left",  "top",    True),
    "X1_StoH":          ( 0.08,  0.08, "left",  "bottom", False),
    "V6_StoM":          (-0.10,  0.00, "right", "center", False),
    "SACPO_StoM":       (-0.10,  0.17, "right", "bottom", True),
    "X1_StoM":          (-0.20,  0.10, "right",  "bottom", True),
}

# ── panel membership ──────────────────────────────────────────────────────────
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

ANNO_CFG = {
    "H→*": H_ANNO,
    "S→*": S_ANNO,
}

# thin connector line for crowded labels (no arrowhead)
ARROW_PROPS = dict(arrowstyle="-", color="#999999", linewidth=0.8)


# ── helpers ───────────────────────────────────────────────────────────────────

def pareto_frontier(points):
    """Return indices of non-dominated points, sorted by x."""
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
    if len(idx) < 2:
        return  # single Pareto-optimal point: nothing to draw
    px = [pts[i][0] for i in idx]
    py = [pts[i][1] for i in idx]
    ax.plot(px, py, color="#BBBBBB", linewidth=1.4, linestyle="--", zorder=1)


# ── panel drawing ─────────────────────────────────────────────────────────────

def plot_panel(ax, data, panel_title, xlabel, ylabel):
    keys     = PANELS[panel_title]
    anno_cfg = ANNO_CFG[panel_title]

    ax.set_facecolor("#FAFAFA")
    ax.grid(True, color="#E2E2E2", linewidth=0.6, zorder=0)
    ax.set_title(panel_title, fontsize=10, fontweight="bold", pad=4)
    ax.set_xlabel(xlabel, fontsize=8.5, labelpad=3)
    ax.set_ylabel(ylabel, fontsize=8.5, labelpad=3)
    ax.tick_params(labelsize=7.5)

    present       = [k for k in keys if k in data and k in META]
    baseline_keys = {"Helpful_baseline", "Safety_baseline"}
    frontier_keys = [k for k in present if k not in baseline_keys]

    if frontier_keys:
        draw_frontier(ax,
                      [data[k]["x"] for k in frontier_keys],
                      [data[k]["y"] for k in frontier_keys])

    for key in present:
        vals      = data[key]
        fam, dest, label = META[key]
        fc, ec    = COLOR[fam]
        mk        = MARKER[dest]
        x, y      = vals["x"], vals["y"]
        xe, ye    = vals.get("x_se", 0), vals.get("y_se", 0)
        zo        = 2 if fam == "base" else 4

        ax.errorbar(x, y, xerr=xe, yerr=ye,
                    fmt="none", ecolor=ec, elinewidth=0.9,
                    capsize=3, capthick=0.9, zorder=zo - 1,
                    alpha=0.5 if fam == "base" else 0.75)

        ax.scatter(x, y, s=MS, marker=mk,
                   facecolors=fc if fam != "base" else "none",
                   edgecolors=ec, linewidths=1.4, zorder=zo)

        cfg = anno_cfg.get(key)
        if cfg is None:
            continue
        dx, dy, ha, va, use_arrow = cfg
        kw = dict(fontsize=7, color="#1a1a1a", ha=ha, va=va,
                  path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])
        if use_arrow:
            kw["arrowprops"] = ARROW_PROPS
        ax.annotate(label, xy=(x, y), xytext=(x + dx, y + dy), **kw)


# ── legend ────────────────────────────────────────────────────────────────────

def build_legend():
    h = []
    h.append(Line2D([0], [0], ls="none", marker="none",
                    label="$\\bf{Method}$"))
    for fam, (fc, ec) in COLOR.items():
        if fam == "base":
            continue
        name = {"X1": "TLBSB (ours)", "SACPO": "SACPO",
                "V6": "V6 (ablation)"}[fam]
        h.append(Line2D([0], [0], ls="none", marker="o", markersize=5.5,
                        markerfacecolor=fc, markeredgecolor=ec,
                        markeredgewidth=1.3, label=name))
    h.append(Line2D([0], [0], ls="none", marker="o", markersize=5.5,
                    markerfacecolor="none", markeredgecolor="#777777",
                    markeredgewidth=1.3,
                    label="Init baseline (π_r / π_s)"))
    h.append(Line2D([0], [0], ls="none", marker="none",
                    label="$\\bf{Stage\\ 2\\ data}$"))
    h.append(Line2D([0], [0], ls="none", marker="o", markersize=5.5,
                    markerfacecolor="#888888", markeredgecolor="#888888",
                    label="Pure (→S / →H)"))
    h.append(Line2D([0], [0], ls="none", marker="D", markersize=4.8,
                    markerfacecolor="#888888", markeredgecolor="#888888",
                    label="Mixed (→M)"))
    h.append(Line2D([0], [0], color="#BBBBBB", linewidth=1.4,
                    linestyle="--", label="Pareto frontier"))
    return h


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default="pareto_mdjudge.json")
    parser.add_argument("--output", default="pareto.pdf")
    parser.add_argument("--xlabel", default="Helpful reward (mean ± SE)  →  more helpful")
    parser.add_argument("--ylabel", default="Safety score (mean ± SE)  →  safer")
    args = parser.parse_args()

    with open(args.data) as f:
        data = json.load(f)

    # AAAI 2-column figure*: 7.0 in wide is the cross-column max.
    # 6.8 x 3.2 keeps fonts >= 8pt at print scale.
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.2))
    fig.subplots_adjust(wspace=0.32)

    for ax, panel_title in zip(axes, PANELS):
        plot_panel(ax, data, panel_title, args.xlabel, args.ylabel)

    # Align the two panels so points are visually comparable across H→* and S→*.
    xlims = [ax.get_xlim() for ax in axes]
    ylims = [ax.get_ylim() for ax in axes]
    xmin = min(l[0] for l in xlims); xmax = max(l[1] for l in xlims)
    ymin = min(l[0] for l in ylims); ymax = max(l[1] for l in ylims)
    for ax in axes:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    fig.legend(handles=build_legend(),
               loc="lower center", ncol=4, fontsize=7.5,
               framealpha=0.92, edgecolor="#CCCCCC",
               handlelength=1.4, handletextpad=0.5, columnspacing=1.0,
               bbox_to_anchor=(0.5, -0.18))

    fig.tight_layout(rect=[0, 0.12, 1, 1])
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Saved: {args.output}")

    alt = (args.output.replace(".pdf", ".png") if args.output.endswith(".pdf")
           else args.output.replace(".png", ".pdf"))
    fig.savefig(alt, dpi=300, bbox_inches="tight")
    print(f"Saved: {alt}")
    plt.close(fig)


if __name__ == "__main__":
    main()
