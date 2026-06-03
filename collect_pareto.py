"""
Collect per-method pareto_md_summary.json files into one combined JSON
for use with plot_pareto.py.

Usage:
    python collect_pareto.py --output x1_pareto_mdjudge.json
"""

import json
import os
import argparse

# method_name -> directory containing pareto_md_summary.json
METHODS = {
    "X1_HtoS":        "/home/zjq/FZ2026/BSB-TDPO/PKU_results/X1_HtoS/eval_results_sequential",
    "X1_StoH":        "/home/zjq/FZ2026/BSB-TDPO/PKU_results/X1_StoH/eval_results_sequential",
    "X1_HtoM":        "/home/zjq/FZ2026/BSB-TDPO/PKU_results/X1_HtoM/eval_results_sequential",
    "X1_StoM":        "/home/zjq/FZ2026/BSB-TDPO/PKU_results/X1_StoM/eval_results_sequential",
    "V6_HtoS":        "/home/zjq/FZ2026/BSB-TDPO/PKU_results/V6_HtoS/eval_results_sequential",
    "V6_StoH":        "/home/zjq/FZ2026/BSB-TDPO/PKU_results/V6_StoH/eval_results_sequential",
    "SACPO_HtoS":     "/home/zjq/FZ2026/sacpo-main/output/30K_helpful_dpo_safety/eval_results",
    "SACPO_StoH":     "/home/zjq/FZ2026/sacpo-main/output/30K_safety_dpo_helpful/eval_results",
    "SACPO_HtoM":     "/home/zjq/FZ2026/sacpo-main/output/PKU_Baseline_Helpful_to_Mixed/eval_results",
    "SACPO_StoM":     "/home/zjq/FZ2026/sacpo-main/output/PKU_Baseline_Safety_to_Mixed/eval_results",
    "Helpful_baseline": "/home/zjq/FZ2026/sacpo-main/output/30K_helpful_dpo_safety/helpful_baseline_results",
    "Safety_baseline":  "/home/zjq/FZ2026/sacpo-main/output/30K_safety_dpo_helpful/safety_baseline_results",
    # Single-stage joint baselines. Verify the eval subdir name on the machine.
    "SafeDPO":          "/home/zjq/FZ2026/sacpo-main/output/SafeDPO_delta5/eval_results",
    "BFPO":             "/home/zjq/FZ2026/sacpo-main/output/BFPO_b1-3_alpha-0.5/eval_results",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="x1_pareto_mdjudge.json",
                        help="Output combined JSON for plot_pareto.py")
    args = parser.parse_args()

    combined = {}
    missing = []

    for name, directory in METHODS.items():
        summary_path = os.path.join(directory, "pareto_md_summary.json")
        if not os.path.exists(summary_path):
            print(f"  [missing] {name}: {summary_path}")
            missing.append(name)
            continue
        with open(summary_path) as f:
            d = json.load(f)
        combined[name] = {
            "x":    d["x"],
            "x_se": d["x_se"],
            "y":    d["y"],
            "y_se": d["y_se"],
        }
        print(f"  [ok] {name:20s}  x={d['x']:.3f}±{d['x_se']:.3f}  "
              f"y={d['y']:.3f}±{d['y_se']:.3f}  safe_rate={d.get('y_safe_rate', float('nan')):.1%}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(combined)} methods -> {args.output}")
    if missing:
        print(f"Missing ({len(missing)}): {', '.join(missing)}")
        print("Run md_judge_eval.py on the missing directories first.")


if __name__ == "__main__":
    main()
