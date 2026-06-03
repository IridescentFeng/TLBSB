"""
SafeDPO training script — ICLR 2026 Oral baseline for TLBSB paper.

Implements SafeDPO on PKU-SafeRLHF-30K with Alpaca-7B-reproduced,
matching the SACPO/TLBSB experimental setup.

Key changes vs standard DPO:
  1. Transformation T on preference pairs (safety-aware reordering)
  2. Optional safety margin Δ on (safe-chosen, unsafe-rejected) pairs

Usage (run from sacpo-main directory or any directory with data/ folder):
    python train_safedpo.py \
        --model_path /path/to/alpaca-7b-reproduced \
        --output_dir ./output/SafeDPO_HtoS \
        --safety_margin 5.0 \
        --beta 0.1 \
        --num_train_epochs 1

Requires: transformers, trl>=0.7, peft, bitsandbytes, datasets
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import DPOTrainer, DPOConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)


# ── SafeDPO data transformation ───────────────────────────────────────────────

def _parse_safe_flag(val) -> bool:
    """Robustly parse is_response_X_safe: handles bool, int, str."""
    if isinstance(val, bool):
        return val
    if isinstance(val, int):
        return val != 0
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "yes")
    return bool(val)


def apply_transformation_T(example):
    """
    SafeDPO transformation T (Section 3.2 of Kim et al., ICLR 2026).

    Supports two common field layouts for PKU-SafeRLHF JSONL files:
      Layout A (HF / SACPO-style):
        better_response_id, response_0, response_1,
        is_response_0_safe, is_response_1_safe, prompt
      Layout B (chosen/rejected pre-split):
        chosen, rejected, prompt  (no safety fields → treated as safe/safe)

    Returns dict with chosen/rejected/h_rejected, or None if both unsafe.
    """
    # ── Layout B: already split into chosen/rejected, no safety fields ──────
    if "chosen" in example and "better_response_id" not in example:
        prompt = ALPACA_TEMPLATE.format(instruction=example["prompt"])
        return {
            "prompt": prompt,
            "chosen": example["chosen"],
            "rejected": example["rejected"],
            "h_rejected": 0,
        }

    # ── Layout A: standard PKU-SafeRLHF fields ───────────────────────────────
    bid = int(example["better_response_id"])
    lid = 1 - bid

    y_w = example[f"response_{bid}"]
    y_l = example[f"response_{lid}"]

    h_w = 0 if _parse_safe_flag(example[f"is_response_{bid}_safe"]) else 1
    h_l = 0 if _parse_safe_flag(example[f"is_response_{lid}_safe"]) else 1

    prompt = ALPACA_TEMPLATE.format(instruction=example["prompt"])

    if h_w == 0:
        return {"prompt": prompt, "chosen": y_w, "rejected": y_l,
                "h_rejected": h_l}
    elif h_w == 1 and h_l == 0:
        # preferred unsafe, non-preferred safe → swap
        return {"prompt": prompt, "chosen": y_l, "rejected": y_w,
                "h_rejected": 1}
    else:
        # both unsafe → discard
        return None


def _load_jsonl(path: Path) -> List[dict]:
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def build_safedpo_dataset(data_dir: str, filenames: Optional[List[str]] = None):
    """
    Load preference pairs from local JSONL files and apply transformation T.

    data_dir   : directory containing the JSONL files
    filenames  : list of filenames to load (default: pku_helpful.jsonl + pku_safety.jsonl)
    """
    data_dir = Path(data_dir)
    if filenames is None:
        filenames = ["pku_helpful.jsonl", "pku_safety.jsonl"]

    raw = []
    for fname in filenames:
        fpath = data_dir / fname
        if not fpath.exists():
            logger.warning("Data file not found, skipping: %s", fpath)
            continue
        loaded = _load_jsonl(fpath)
        logger.info("Loaded %d examples from %s", len(loaded), fpath)
        raw.extend(loaded)

    if not raw:
        raise FileNotFoundError(
            f"No data loaded from {data_dir}. "
            "Check --data_dir and --data_files."
        )

    transformed = []
    n_kept = n_swapped = n_discarded = 0
    for ex in raw:
        # Pre-check for both-unsafe in Layout A
        if "better_response_id" in ex:
            bid = int(ex["better_response_id"])
            lid = 1 - bid
            h_w = 0 if _parse_safe_flag(ex[f"is_response_{bid}_safe"]) else 1
            h_l = 0 if _parse_safe_flag(ex[f"is_response_{lid}_safe"]) else 1
            if h_w == 1 and h_l == 1:
                n_discarded += 1
                continue
            is_swap = (h_w == 1 and h_l == 0)
        else:
            is_swap = False

        result = apply_transformation_T(ex)
        if result is None:
            n_discarded += 1
            continue

        if is_swap:
            n_swapped += 1
        else:
            n_kept += 1
        transformed.append(result)

    logger.info(
        "Dataset after T: kept=%d  swapped=%d  discarded=%d  total=%d",
        n_kept, n_swapped, n_discarded, len(transformed),
    )

    from datasets import Dataset
    return Dataset.from_list(transformed)


# ── SafeDPO trainer (adds safety margin Δ) ───────────────────────────────────

class SafeDPOTrainer(DPOTrainer):
    """
    Extends TRL's DPOTrainer with the safety margin term Δ.

    Loss: -log σ(β*(log_ratio_chosen - log_ratio_rejected) - h_rejected * Δ)
    where h_rejected=1 when the rejected response is unsafe.

    h_rejected is injected via tokenize_row so it survives dataset processing
    in TRL >= 0.9 (which removed DPODataCollatorWithPadding).
    """

    def __init__(self, *args, safety_margin: float = 5.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.safety_margin = safety_margin
        self._h_rejected_batch: Optional[torch.Tensor] = None

    def tokenize_row(self, feature, *args, **kwargs):
        result = super().tokenize_row(feature, *args, **kwargs)
        result["h_rejected"] = feature.get("h_rejected", 0)
        return result

    def get_batch_loss_metrics(self, model, batch, train_eval="train"):
        h = batch.pop("h_rejected", None)
        if h is not None:
            self._h_rejected_batch = h if isinstance(h, torch.Tensor) else torch.tensor(h)
        result = super().get_batch_loss_metrics(model, batch, train_eval)
        self._h_rejected_batch = None
        return result

    def dpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
        reference_free: bool = False,
    ):
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps)
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps)
        logits = chosen_rewards - rejected_rewards

        # Safety margin: push separation wider when rejected is unsafe
        if self._h_rejected_batch is not None and self.safety_margin > 0:
            h = self._h_rejected_batch.float().to(logits.device)
            if h.shape != logits.shape:
                h = h[: logits.shape[0]]
            logits = logits - h * self.safety_margin

        losses = -F.logsigmoid(logits)
        return losses, chosen_rewards.detach(), rejected_rewards.detach()


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True,
                   help="Local path to alpaca-7b (or alpaca-7b-reproduced)")
    p.add_argument("--output_dir", default="./output/SafeDPO")
    # ── Data ─────────────────────────────────────────────────────────────────
    p.add_argument("--data_dir",
                   default="/home/zjq/FZ2026/sacpo-main/data",
                   help="Directory containing the training JSONL files")
    p.add_argument("--data_files", nargs="+",
                   default=["pku_helpful.jsonl", "pku_safety.jsonl"],
                   help="JSONL filenames inside data_dir to use for training")
    # ── SafeDPO hypers ────────────────────────────────────────────────────────
    p.add_argument("--safety_margin", type=float, default=5.0,
                   help="SafeDPO Δ parameter (0 = no margin)")
    p.add_argument("--beta", type=float, default=0.1,
                   help="DPO β (KL penalty coefficient)")
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--max_prompt_length", type=int, default=128)
    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--load_in_4bit", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=500)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── tokenizer ────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # ── model (4-bit quantization + LoRA, same as SACPO) ─────────────────────
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=args.load_in_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    ) if args.load_in_4bit else None

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=bnb_cfg,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Reference model (frozen Alpaca-7B — no LoRA)
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=bnb_cfg,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # ── dataset ───────────────────────────────────────────────────────────────
    train_ds = build_safedpo_dataset(args.data_dir, args.data_files)

    # ── training args ─────────────────────────────────────────────────────────
    training_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        remove_unused_columns=False,
        seed=args.seed,
        report_to="none",
        dataloader_num_workers=4,
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
    )

    trainer = SafeDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        safety_margin=args.safety_margin,
    )

    logger.info("Starting SafeDPO training (Δ=%.1f, β=%.2f)…",
                args.safety_margin, args.beta)
    trainer.train()

    # Save final LoRA adapter
    final_path = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info("Saved to %s", final_path)


if __name__ == "__main__":
    main()
