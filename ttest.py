"""
Statistical significance testing for TLBSB paper.
Runs paired t-tests comparing methods on helpful reward (x) and safety score (y).

Usage:
    python ttest.py
"""

import json
import os
import numpy as np
from scipy import stats
from itertools import combinations

# ── paths to per-sample score files ──────────────────────────────────────────
# Each entry: (method_name, helpful_scores_file, safety_scores_file)
# helpful_scores_file: contains {"reward_scores": [...]}  (129 values)
# safety_scores_file:  contains {"safety_logodds": [...]} (83 values, from MD-Judge)
#                   OR {"cost_scores": [...]}              (83 values, from Beaver)

METHODS = {
    "X1_HtoS":    "/home/zjq/FZ2026/BSB-TDPO/PKU_results/X1_HtoS/eval_results_sequential",
    "X1_StoH":    "/home/zjq/FZ2026/BSB-TDPO/PKU_results/X1_StoH/eval_results_sequential",
    "X1_HtoM":    "/home/zjq/FZ2026/BSB-TDPO/PKU_results/X1_HtoM/eval_results_sequential",
    "X1_StoM":    "/home/zjq/FZ2026/BSB-TDPO/PKU_results/X1_StoM/eval_results_sequential",
    "V6_HtoS":    "/home/zjq/FZ2026/BSB-TDPO/PKU_results/V6_HtoS/eval_results_sequential",
    "V6_StoH":    "/home/zjq/FZ2026/BSB-TDPO/PKU_results/V6_StoH/eval_results_sequential",
    "SACPO_HtoS": "/home/zjq/FZ2026/sacpo-main/output/30K_helpful_dpo_safety/eval_results",
    "SACPO_StoH": "/home/zjq/FZ2026/sacpo-main/output/30K_safety_dpo_helpful/eval_results",
    "SACPO_HtoM": "/home/zjq/FZ2026/sacpo-main/output/PKU_Baseline_Helpful_to_Mixed/eval_results",
    "SACPO_StoM": "/home/zjq/FZ2026/sacpo-main/output/PKU_Baseline_Safety_to_Mixed/eval_results",
}

# Key comparisons for the paper (method_a vs method_b)
# "better" means: a should be better than b on the stated axis
KEY_PAIRS = [
    # Core Prop 1+2 evidence: barrier on vs off, same H→S data
    ("X1_HtoS", "SACPO_HtoS", "both",    "TLBSB vs SACPO (H→S): barrier effect"),
    # Ablation: distilled probe vs joint probe
    ("X1_HtoS", "V6_HtoS",    "both",    "TLBSB vs V6 (H→S): distillation vs joint training"),
    # S→H direction
    ("X1_StoH", "SACPO_StoH", "both",    "TLBSB vs SACPO (S→H): barrier effect"),
    ("X1_StoH", "V6_StoH",    "both",    "TLBSB vs V6 (S→H): distillation vs joint training"),
    # Mixed data
    ("X1_HtoM", "SACPO_HtoM", "both",    "TLBSB vs SACPO (H→M)"),
    ("X1_StoM", "SACPO_StoM", "both",    "TLBSB vs SACPO (S→M)"),
]

SIG = {0.001: "***", 0.01: "**", 0.05: "*", 1.0: "n.s."}


def sig_stars(p):
    for threshold, label in SIG.items():
        if p < threshold:
            return label
    return "n.s."


def load_scores(directory):
    """Load per-sample helpful reward and safety scores from a method directory."""
    helpful, safety = None, None

    # helpful: Beaver reward scores (129 values)
    for fname in ["helpful_results_beaver_scores.json", "helpful_beaver_scores.json",
                  "helpful_scores.json"]:
        path = os.path.join(directory, fname)
        if os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
            helpful = np.array(d["reward_scores"])
            break

    # safety: MD-Judge log-odds preferred, fall back to Beaver cost
    for fname in ["safety_results_md_scores.json"]:
        path = os.path.join(directory, fname)
        if os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
            safety = np.array(d["safety_logodds"])
            break
    if safety is None:
        for fname in ["safety_results_beaver_scores.json", "safety_beaver_scores.json",
                      "safety_scores.json"]:
            path = os.path.join(directory, fname)
            if os.path.exists(path):
                with open(path) as f:
                    d = json.load(f)
                safety = np.array(d["cost_scores"])
                break

    return helpful, safety


def paired_ttest(a, b):
    """Two-sided paired t-test. Returns (t, p, mean_diff, se_diff)."""
    diff = a - b
    t, p = stats.ttest_rel(a, b)
    return t, p, diff.mean(), diff.std() / np.sqrt(len(diff))


def print_table(results):
    sep = "-" * 90
    print(sep)
    print(f"{'Comparison':<40} {'Axis':<8} {'Δmean':>8} {'SE':>7} {'t':>7} {'p':>9} {'sig':>5}")
    print(sep)
    for row in results:
        print(f"{row['label']:<40} {row['axis']:<8} {row['delta']:>+8.3f} "
              f"{row['se']:>7.3f} {row['t']:>7.3f} {row['p']:>9.4f} {row['sig']:>5}")
    print(sep)


def main():
    # load all scores
    scores = {}
    for name, directory in METHODS.items():
        h, s = load_scores(directory)
        if h is None:
            print(f"[warn] No helpful scores for {name}")
        if s is None:
            print(f"[warn] No safety scores for {name}")
        scores[name] = (h, s)

    print("\n=== Key pairwise t-tests ===\n")
    results = []
    for m_a, m_b, axes, label in KEY_PAIRS:
        h_a, s_a = scores.get(m_a, (None, None))
        h_b, s_b = scores.get(m_b, (None, None))

        for axis, arr_a, arr_b in [("helpful", h_a, h_b), ("safety", s_a, s_b)]:
            if axes not in (axis, "both"):
                continue
            if arr_a is None or arr_b is None:
                print(f"[skip] {label} ({axis}): missing data")
                continue
            if len(arr_a) != len(arr_b):
                print(f"[warn] {label} ({axis}): length mismatch "
                      f"{len(arr_a)} vs {len(arr_b)}, using min")
                n = min(len(arr_a), len(arr_b))
                arr_a, arr_b = arr_a[:n], arr_b[:n]

            t, p, delta, se = paired_ttest(arr_a, arr_b)
            results.append({
                "label": label[:39],
                "axis":  axis,
                "delta": delta,
                "se":    se,
                "t":     t,
                "p":     p,
                "sig":   sig_stars(p),
            })

    print_table(results)

    # save as JSON for LaTeX table generation
    out_path = "ttest_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # summary for paper
    print("\n=== Paper-ready summary (Δ = A − B, positive = A better) ===\n")
    for r in results:
        print(f"  {r['label']} [{r['axis']}]: "
              f"Δ={r['delta']:+.3f} (SE={r['se']:.3f}), "
              f"p={r['p']:.4f} {r['sig']}")


if __name__ == "__main__":
    main()
