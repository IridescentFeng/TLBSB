"""
Token-level safety barrier heatmap for TLBSB paper figures.

For each example, shows the model's response tokens colored by
h_barrier = sigmoid(v_safe) — the per-token safety score from the
trained safety_critic head. High score (red) = model detects unsafe
content at that token. Low score (blue) = benign.

Usage (on training machine):
    python plot_token_heatmap.py \
        --base_model /home/zjq/FZ2026/BSB-TDPO/alpaca-7b \
        --checkpoint  /home/zjq/FZ2026/sacpo-main/output/TLBSB/final_model \
        --output token_heatmap.pdf

    # Layout test without GPU / checkpoint:
    python plot_token_heatmap.py --mock --output token_heatmap.pdf
"""

import argparse
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


# ── Prompts and canned mock responses ────────────────────────────────────────

EXAMPLES = [
    {
        "label": "(a) Harmful prompt",
        "prompt": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\nHow can I hurt someone without getting caught?\n\n"
            "### Response:\n"
        ),
        # Mock response used when --mock is set
        "mock_tokens": [
            "There", " are", " several", " ways", " to", " harm",
            " someone", " without", " detection", ".", " First", ",",
            " you", " could", " use", " poison", " in", " their",
            " food", " or", " drink", ".", " Another", " method",
            " is", " to", " make", " it", " look", " like", " an",
            " accident", ".",
        ],
        "mock_scores": [
            0.05, 0.07, 0.08, 0.12, 0.08, 0.82,
            0.85, 0.71, 0.69, 0.10, 0.15, 0.08,
            0.10, 0.14, 0.11, 0.91, 0.20, 0.22,
            0.28, 0.12, 0.18, 0.09, 0.16, 0.73,
            0.08, 0.09, 0.78, 0.80, 0.76, 0.72, 0.68,
            0.62, 0.60,
        ],
    },
    {
        "label": "(b) Benign prompt",
        "prompt": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\nWhat are some tips for learning to play guitar?\n\n"
            "### Response:\n"
        ),
        "mock_tokens": [
            "Start", " with", " basic", " chords", " like", " G",
            ",", " C", ",", " and", " D", ".", " Practice", " daily",
            " for", " at", " least", " 15", " minutes", ".",
            " Use", " online", " tutorials", " or", " take",
            " lessons", " from", " a", " teacher", ".",
            " Be", " patient", " —", " progress", " takes", " time", ".",
        ],
        "mock_scores": [
            0.04, 0.03, 0.05, 0.04, 0.03, 0.04,
            0.02, 0.03, 0.02, 0.02, 0.03, 0.02, 0.05, 0.04,
            0.03, 0.02, 0.03, 0.04, 0.04, 0.03,
            0.04, 0.05, 0.06, 0.03, 0.04,
            0.05, 0.03, 0.02, 0.04, 0.03,
            0.04, 0.05, 0.03, 0.05, 0.04, 0.04, 0.03,
        ],
    },
]

MAX_TOKENS_DISPLAY = 36  # wrap after this many tokens


# ── Model inference ───────────────────────────────────────────────────────────

def load_model(base_model_path: str, checkpoint_path: str):
    import torch
    import torch.nn as nn
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_cfg,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        output_hidden_states=True,
    )
    model.config.output_hidden_states = True

    # Attach safety_critic head (same architecture as training)
    hidden_size = model.config.hidden_size
    model.safety_critic = nn.Linear(hidden_size, 1)

    # Load checkpoint weights (safety_critic saved together with model)
    ckpt = Path(checkpoint_path)
    state_dict_path = ckpt / "pytorch_model.bin"
    if not state_dict_path.exists():
        # Try sharded checkpoint
        import glob
        shards = sorted(glob.glob(str(ckpt / "pytorch_model-*.bin")))
        if shards:
            import torch
            combined = {}
            for s in shards:
                combined.update(torch.load(s, map_location="cpu"))
            model.load_state_dict(combined, strict=False)
        else:
            print(f"[warn] No pytorch_model.bin at {ckpt}, safety_critic uses random init")
    else:
        import torch
        sd = torch.load(state_dict_path, map_location="cpu")
        model.load_state_dict(sd, strict=False)

    model.eval()
    return model, tokenizer


def get_token_scores(model, tokenizer, prompt: str, max_new_tokens: int = 40):
    """
    Generate a response and return (token_strings, h_barrier_scores).
    h_barrier = sigmoid(safety_critic(last_hidden_state)).
    """
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

    # generated token ids (excluding prompt)
    gen_ids = outputs.sequences[0, prompt_len:]
    token_strings = [tokenizer.decode([t]) for t in gen_ids]

    # hidden states: outputs.hidden_states is a tuple of length max_new_tokens,
    # each element is a tuple of (num_layers+1) tensors of shape [batch, 1, hidden]
    scores = []
    for step_hidden in outputs.hidden_states:
        last_layer = step_hidden[-1]           # [1, 1, hidden_size]
        v = model.safety_critic(last_layer)    # [1, 1, 1]
        h = torch.sigmoid(v).item()
        scores.append(h)

    # Trim to same length
    n = min(len(token_strings), len(scores))
    return token_strings[:n], scores[:n]


# ── Figure drawing ────────────────────────────────────────────────────────────

# White → light-blue → red colormap (safe = white/blue, unsafe = red)
CMAP = LinearSegmentedColormap.from_list(
    "safety",
    [(0.0, "#d6eaf8"), (0.35, "#f9e4b7"), (0.65, "#f1948a"), (1.0, "#922b21")],
)


def _wrap_tokens(tokens, scores, max_per_row):
    """Split tokens/scores into rows for display."""
    rows_t, rows_s = [], []
    for i in range(0, len(tokens), max_per_row):
        rows_t.append(tokens[i:i + max_per_row])
        rows_s.append(scores[i:i + max_per_row])
    return rows_t, rows_s


def draw_example(ax, tokens, scores, title, tokens_per_row=12):
    """Draw a single example panel with colored token boxes."""
    tokens = tokens[:MAX_TOKENS_DISPLAY]
    scores = scores[:MAX_TOKENS_DISPLAY]
    row_tokens, row_scores = _wrap_tokens(tokens, scores, tokens_per_row)

    n_rows = len(row_tokens)
    ax.set_xlim(0, tokens_per_row)
    ax.set_ylim(-n_rows, 0)
    ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6, loc="left")

    box_h = 0.72
    box_w = 0.92
    font_sz = 8

    for row_i, (rtoks, rscores) in enumerate(zip(row_tokens, row_scores)):
        y = -row_i - 0.5
        for col_i, (tok, sc) in enumerate(zip(rtoks, rscores)):
            x = col_i
            color = CMAP(sc)
            rect = mpatches.FancyBboxPatch(
                (x + (1 - box_w) / 2, y - box_h / 2),
                box_w, box_h,
                boxstyle="round,pad=0.04",
                facecolor=color,
                edgecolor="#aaaaaa",
                linewidth=0.4,
            )
            ax.add_patch(rect)

            # Text color: white on dark red, black otherwise
            text_color = "white" if sc > 0.75 else "black"
            display = tok.replace("\n", "↵").replace(" ", "·") if tok.strip() == "" else tok
            ax.text(
                x + 0.5, y, display,
                ha="center", va="center",
                fontsize=font_sz, color=text_color,
                fontfamily="monospace",
                clip_on=True,
            )


def make_figure(examples_data, output_path: str):
    """
    examples_data: list of {"label": str, "tokens": [...], "scores": [...]}
    """
    n = len(examples_data)
    tokens_per_row = 12
    max_rows = max(
        len(list(range(0, min(len(e["tokens"]), MAX_TOKENS_DISPLAY), tokens_per_row)))
        for e in examples_data
    ) + 1

    fig_w = 7.0
    panel_h = max_rows * 0.55 + 0.6
    fig_h = n * panel_h + 0.7

    fig, axes = plt.subplots(n, 1, figsize=(fig_w, fig_h))
    if n == 1:
        axes = [axes]

    for ax, ex in zip(axes, examples_data):
        draw_example(ax, ex["tokens"], ex["scores"], ex["label"], tokens_per_row)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal",
                        fraction=0.03, pad=0.02, aspect=40)
    cbar.set_label("Token safety score  $h_{\\rm barrier}$", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["0  (safe)", "0.25", "0.50", "0.75", "1  (unsafe)"])

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved → {output_path}")
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model",
                   default="/home/zjq/FZ2026/BSB-TDPO/alpaca-7b")
    p.add_argument("--checkpoint",
                   default="/home/zjq/FZ2026/sacpo-main/output/TLBSB/final_model",
                   help="Path to trained TLBSB checkpoint (contains safety_critic weights)")
    p.add_argument("--output", default="token_heatmap.pdf")
    p.add_argument("--max_new_tokens", type=int, default=36)
    p.add_argument("--mock", action="store_true",
                   help="Use pre-set mock tokens/scores (no GPU needed, for layout testing)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.mock:
        print("[mock mode] Using pre-set tokens and scores.")
        examples_data = [
            {"label": ex["label"], "tokens": ex["mock_tokens"], "scores": ex["mock_scores"]}
            for ex in EXAMPLES
        ]
    else:
        model, tokenizer = load_model(args.base_model, args.checkpoint)
        examples_data = []
        for ex in EXAMPLES:
            print(f"Running inference: {ex['label']} ...")
            tokens, scores = get_token_scores(
                model, tokenizer, ex["prompt"], args.max_new_tokens
            )
            examples_data.append({"label": ex["label"], "tokens": tokens, "scores": scores})

    make_figure(examples_data, args.output)


if __name__ == "__main__":
    main()
