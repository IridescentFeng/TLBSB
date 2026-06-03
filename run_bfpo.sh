#!/bin/bash
# Run BFPO training on Alpaca-7B-reproduced.
# ICLR 2025 Spotlight baseline for TLBSB paper.
# Adjust MODEL_PATH and OUTPUT_BASE to match your machine.

set -e

MODEL_PATH="/home/zjq/FZ2026/BSB-TDPO/alpaca-7b"
DATA_DIR="/home/zjq/FZ2026/sacpo-main/data"
OUTPUT_BASE="/home/zjq/FZ2026/sacpo-main/output"
SCRIPT="$(dirname "$0")/train_bfpo.py"

# ── BFPO with paper defaults (b1=3, alpha=0.5) ───────────────────────────────
CUDA_VISIBLE_DEVICES=2 python "$SCRIPT" \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --data_files pku_helpful.jsonl pku_safety.jsonl \
    --safety_files pku_safety.jsonl \
    --output_dir "${OUTPUT_BASE}/BFPO_b1-3_alpha-0.5" \
    --b1 3.0 \
    --alpha 0.5 \
    --beta 0.1 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --max_length 512 \
    --max_prompt_length 128 \
    --lora_r 64 \
    --lora_alpha 16

echo "BFPO (b1=3, alpha=0.5) done."
