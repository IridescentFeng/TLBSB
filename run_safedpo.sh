#!/bin/bash
# Run SafeDPO training on Alpaca-7B-reproduced.
# Adjust MODEL_PATH and OUTPUT_BASE to match your machine.

set -e

MODEL_PATH="/home/zjq/FZ2026/BSB-TDPO/alpaca-7b-reproduced"
# Fallback: PKU-Alignment/alpaca-7b-reproduced (HF Hub)

OUTPUT_BASE="/home/zjq/FZ2026/sacpo-main/output"
SCRIPT="$(dirname "$0")/train_safedpo.py"

# ── SafeDPO with Δ=5 (recommended in paper) ──────────────────────────────────
python "$SCRIPT" \
    --model_path "$MODEL_PATH" \
    --output_dir "${OUTPUT_BASE}/SafeDPO_delta5" \
    --safety_margin 5.0 \
    --beta 0.1 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-4 \
    --max_length 512 \
    --max_prompt_length 256 \
    --lora_r 64 \
    --lora_alpha 16

echo "SafeDPO (Δ=5) done."

# ── SafeDPO with Δ=0 (no margin, pure transformation only) ───────────────────
# Uncomment to run ablation
# python "$SCRIPT" \
#     --model_path "$MODEL_PATH" \
#     --output_dir "${OUTPUT_BASE}/SafeDPO_delta0" \
#     --safety_margin 0.0 \
#     --beta 0.1 \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 4 \
#     --gradient_accumulation_steps 4 \
#     --learning_rate 5e-4
