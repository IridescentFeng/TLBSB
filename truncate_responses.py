"""
Truncate multi-turn run-on generations back to the first response turn.

Some methods (e.g. SafeDPO under aggressive hyperparams) fail to stop at EOS
and keep generating fake "### Instruction: ... ### Response: ..." turns until
hitting max_new_tokens. This inflates Beaver reward (length bias) and makes the
point incomparable. This script keeps only the first answer turn so every
method is judged on a single-turn response.

Output goes to <output_dir> with the SAME filenames, so you can then run the
Beaver reward scorer and md_judge_eval.py pointed at the new directory.

Usage:
    python truncate_responses.py \
        --input_dir  /home/zjq/FZ2026/sacpo-main/output/SafeDPO_delta5/eval_results \
        --output_dir /home/zjq/FZ2026/sacpo-main/output/SafeDPO_delta5/eval_results_truncated
"""

import argparse
import json
import os
import re

# First occurrence of a new template turn marks the start of run-on content.
CUT_PATTERN = re.compile(r"\n[ \t]*#{2,}\s*(Instruction|Response|Input|Output|Related)", re.IGNORECASE)

RESPONSE_KEYS = ["response", "answer", "sacpo_dpo_final_answer",
                 "model_answer", "output", "text"]


def find_response_key(item):
    for key in RESPONSE_KEYS:
        if key in item and isinstance(item[key], str) and item[key].strip():
            return key
    for k, v in item.items():
        if k != "prompt" and isinstance(v, str) and v.strip():
            return k
    return None


def truncate_text(text):
    m = CUT_PATTERN.search(text)
    return text[:m.start()].rstrip() if m else text


def process_file(in_path, out_path):
    if not os.path.exists(in_path):
        print(f"  [skip] {in_path} not found")
        return
    with open(in_path, encoding="utf-8") as f:
        data = json.load(f)

    n_cut = 0
    before_words = after_words = 0
    for item in data:
        key = find_response_key(item)
        if key is None:
            continue
        orig = item[key]
        new = truncate_text(orig)
        before_words += len(orig.split())
        after_words += len(new.split())
        if new != orig:
            n_cut += 1
        item[key] = new

    n = len(data)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  {os.path.basename(in_path)}: {n_cut}/{n} truncated  "
          f"mean_words {before_words/max(n,1):.1f} -> {after_words/max(n,1):.1f}  "
          f"-> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    for fname in ("helpful_results.json", "safety_results.json"):
        process_file(os.path.join(args.input_dir, fname),
                     os.path.join(args.output_dir, fname))

    print("\nNext steps on the truncated dir:")
    print("  1. Re-run the Beaver reward scorer on helpful_results.json "
          "(produces helpful_results_beaver_scores.json)")
    print(f"  2. python md_judge_eval.py --input_dir {args.output_dir}")


if __name__ == "__main__":
    main()
