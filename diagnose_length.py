"""
Diagnose generation-length bias and degeneration across methods.

Reads each method's raw response files (helpful_results.json / safety_results.json),
reports per-method length stats and a repetition score to tell apart
"verbose but fine" from "degenerate/looping".

Usage:
    python diagnose_length.py \
        --methods \
            SafeDPO:/home/zjq/FZ2026/sacpo-main/output/SafeDPO_delta5/eval_results \
            SACPO_HtoS:/home/zjq/FZ2026/sacpo-main/output/30K_helpful_dpo_safety/eval_results \
            X1_HtoS:/home/zjq/FZ2026/BSB-TDPO/PKU_results/X1_HtoS/eval_results_sequential \
        --show_samples 2
"""

import argparse
import json
import os
import re
from collections import Counter


def get_response_text(item):
    for key in ["response", "answer", "sacpo_dpo_final_answer",
                "model_answer", "output", "text"]:
        if key in item and isinstance(item[key], str) and item[key].strip():
            return item[key]
    for k, v in item.items():
        if k != "prompt" and isinstance(v, str) and v.strip():
            return v
    return ""


def repetition_score(text):
    """Fraction of 4-grams that are repeats. ~0 = no repetition, ->1 = looping."""
    words = text.split()
    if len(words) < 8:
        return 0.0
    grams = [" ".join(words[i:i + 4]) for i in range(len(words) - 3)]
    c = Counter(grams)
    repeated = sum(v - 1 for v in c.values() if v > 1)
    return repeated / len(grams)


def max_run(text):
    """Length (in words) of the longest immediately-repeated phrase block."""
    words = text.split()
    best = 0
    for span in range(1, min(20, len(words) // 2 + 1)):
        run = 1
        i = span
        while i + span <= len(words):
            if words[i:i + span] == words[i - span:i]:
                run += 1
                i += span
            else:
                run = 1
                i += 1
            best = max(best, run if run > 1 else 0)
    return best


def stats_for_file(path, show_samples=0):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    word_lens, char_lens, reps, runs = [], [], [], []
    samples = []
    for item in data:
        r = get_response_text(item)
        wl = len(r.split())
        word_lens.append(wl)
        char_lens.append(len(r))
        reps.append(repetition_score(r))
        runs.append(max_run(r))
        samples.append((wl, item.get("prompt", "")[:80], r))
    n = len(word_lens)
    if n == 0:
        return None
    word_lens.sort()
    out = {
        "n": n,
        "mean_words": sum(word_lens) / n,
        "median_words": word_lens[n // 2],
        "max_words": word_lens[-1],
        "mean_rep4": sum(reps) / n,
        "frac_degenerate": sum(1 for x in runs if x >= 3) / n,  # phrase looped >=3x
    }
    if show_samples:
        longest = sorted(samples, key=lambda s: -s[0])[:show_samples]
        out["samples"] = longest
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", required=True, nargs="+",
                    help='List of "name:eval_dir" pairs')
    ap.add_argument("--show_samples", type=int, default=0,
                    help="Print this many longest responses per method")
    args = ap.parse_args()

    print(f"{'method':<16}{'file':<10}{'n':>5}{'mean_w':>9}{'med_w':>8}"
          f"{'max_w':>8}{'rep4':>8}{'degen%':>9}")
    print("-" * 73)

    sample_dump = []
    for spec in args.methods:
        name, d = spec.split(":", 1)
        for fname, tag in (("helpful_results.json", "helpful"),
                           ("safety_results.json", "safety")):
            st = stats_for_file(os.path.join(d, fname), args.show_samples)
            if st is None:
                print(f"{name:<16}{tag:<10}{'  --- missing ---'}")
                continue
            print(f"{name:<16}{tag:<10}{st['n']:>5}{st['mean_words']:>9.1f}"
                  f"{st['median_words']:>8}{st['max_words']:>8}"
                  f"{st['mean_rep4']:>8.3f}{st['frac_degenerate']*100:>8.1f}%")
            if args.show_samples and "samples" in st:
                sample_dump.append((name, tag, st["samples"]))

    if sample_dump:
        print("\n" + "=" * 73 + "\nLONGEST SAMPLES\n" + "=" * 73)
        for name, tag, samples in sample_dump:
            for wl, prompt, resp in samples:
                print(f"\n[{name}/{tag}]  words={wl}")
                print(f"  PROMPT:   {prompt}")
                print(f"  RESPONSE: {resp[:600]}{' ...[truncated]' if len(resp) > 600 else ''}")


if __name__ == "__main__":
    main()
