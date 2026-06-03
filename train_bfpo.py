"""
BFPO training script — ICLR 2025 Spotlight baseline for TLBSB paper.

Implements BFPO (Barrier-Function Policy Optimization) on PKU-SafeRLHF-30K
with Alpaca-7B-reproduced, matching the SACPO/TLBSB experimental setup.

Key: squared-error loss with a safety-aware target determined by
     is_chosen_safe / is_rejected_safe labels per sample.
     Loss = (logits - safe_factor) ** 2
     where logits   = beta * (delta_logp_chosen - delta_logp_rejected)
           safe_factor = b1*b3*is_chosen_safe - b3*is_rejected_safe - alpha

Usage:
    python train_bfpo.py \
        --model_path /path/to/alpaca-7b \
        --output_dir ./output/BFPO \
        --b1 3.0 --alpha 0.5 --beta 0.1 \
        --num_train_epochs 1

Reference: Zhang et al., "BFPO: Barrier-Function Policy Optimization", ICLR 2025.
           https://github.com/wx-zhang/bfpo
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import List, Optional

import torch
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


# ── Dataset loading ───────────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> List[dict]:
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def build_bfpo_dataset(
    data_dir: str,
    filenames: Optional[List[str]] = None,
    safety_files: Optional[List[str]] = None,
):
    """
    Load preference pairs from local JSONL files and attach BFPO safety labels.

    safety_files : filenames where chosen=safe, rejected=unsafe (safety split).
                   All other files are treated as helpful (both responses safe).

    Safety label convention (matches BFPO paper):
      pku_safety.jsonl  → is_chosen_safe=1, is_rejected_safe=0
      pku_helpful.jsonl → is_chosen_safe=1, is_rejected_safe=1
    """
    data_dir = Path(data_dir)
    if filenames is None:
        filenames = ["pku_helpful.jsonl", "pku_safety.jsonl"]
    if safety_files is None:
        safety_files = ["pku_safety.jsonl"]

    safety_set = set(safety_files)
    raw = []

    for fname in filenames:
        fpath = data_dir / fname
        if not fpath.exists():
            logger.warning("Data file not found, skipping: %s", fpath)
            continue
        loaded = _load_jsonl(fpath)
        logger.info("Loaded %d examples from %s", len(loaded), fpath)

        is_safety = fname in safety_set
        for ex in loaded:
            prompt = ALPACA_TEMPLATE.format(instruction=ex["prompt"])
            raw.append({
                "prompt": prompt,
                "chosen": ex["chosen"],
                "rejected": ex["rejected"],
                "is_chosen_safe": 1,
                "is_rejected_safe": 0 if is_safety else 1,
            })

    if not raw:
        raise FileNotFoundError(
            f"No data loaded from {data_dir}. "
            "Check --data_dir and --data_files."
        )

    logger.info("BFPO dataset: %d total examples", len(raw))

    from datasets import Dataset
    return Dataset.from_list(raw)


# ── BFPO trainer ──────────────────────────────────────────────────────────────

class BFPOTrainer(DPOTrainer):
    """
    Extends TRL's DPOTrainer with the BFPO squared-error loss.

    Loss = (logits - safe_factor) ** 2
    logits      = beta * (delta_logp_chosen - delta_logp_rejected)
    safe_factor = b1*b3 * is_chosen_safe - b3 * is_rejected_safe - alpha
    b3 = 1 / (b1 - 1)

    Safety labels are injected via tokenize_row so they survive dataset
    processing in TRL >= 0.9 (which removed DPODataCollatorWithPadding).
    """

    def __init__(self, *args, b1: float = 3.0, alpha: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.b1 = b1
        self.alpha = alpha
        self.b3 = 1.0 / (b1 - 1.0)
        self._safe_chosen_batch: Optional[torch.Tensor] = None
        self._safe_rejected_batch: Optional[torch.Tensor] = None

    def tokenize_row(self, feature, *args, **kwargs):
        result = super().tokenize_row(feature, *args, **kwargs)
        result["is_chosen_safe"] = feature.get("is_chosen_safe", 1)
        result["is_rejected_safe"] = feature.get("is_rejected_safe", 1)
        return result

    def get_batch_loss_metrics(self, model, batch, train_eval="train"):
        sc = batch.pop("is_chosen_safe", None)
        sr = batch.pop("is_rejected_safe", None)
        if sc is not None:
            self._safe_chosen_batch = sc if isinstance(sc, torch.Tensor) else torch.tensor(sc)
        if sr is not None:
            self._safe_rejected_batch = sr if isinstance(sr, torch.Tensor) else torch.tensor(sr)
        result = super().get_batch_loss_metrics(model, batch, train_eval)
        self._safe_chosen_batch = None
        self._safe_rejected_batch = None
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

        # BFPO safe_factor from safety labels
        if self._safe_chosen_batch is not None and self._safe_rejected_batch is not None:
            sc = self._safe_chosen_batch.float().to(logits.device)
            sr = self._safe_rejected_batch.float().to(logits.device)
            if sc.shape != logits.shape:
                sc = sc[: logits.shape[0]]
                sr = sr[: logits.shape[0]]
            bfpo_factor = self.b1 * self.b3 * sc - self.b3 * sr
            safe_factor = bfpo_factor - self.alpha
        else:
            safe_factor = -self.alpha  # fallback: treat as unknown-safe pair

        # target lives in the same β·Δlogp space as logits;
        # dividing by β would require Δlogp ≈ safe_factor/β² ≈ 50–100 nats (unachievable)
        target = safe_factor
        losses = (logits - target) ** 2

        return losses, chosen_rewards.detach(), rejected_rewards.detach()


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True,
                   help="Local path to alpaca-7b (alpaca-7b-reproduced)")
    p.add_argument("--output_dir", default="./output/BFPO")
    # ── Data ─────────────────────────────────────────────────────────────────
    p.add_argument("--data_dir",
                   default="/home/zjq/FZ2026/sacpo-main/data",
                   help="Directory containing the training JSONL files")
    p.add_argument("--data_files", nargs="+",
                   default=["pku_helpful.jsonl", "pku_safety.jsonl"],
                   help="JSONL filenames inside data_dir to use for training")
    p.add_argument("--safety_files", nargs="+",
                   default=["pku_safety.jsonl"],
                   help="Subset of data_files where chosen=safe, rejected=unsafe")
    # ── BFPO hypers ───────────────────────────────────────────────────────────
    p.add_argument("--b1", type=float, default=3.0,
                   help="BFPO b1 parameter (paper default=3)")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="BFPO alpha offset (paper default=0.5)")
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

    # ── model (4-bit + LoRA, identical setup to SACPO/SafeDPO) ───────────────
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

    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=bnb_cfg,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # ── dataset ───────────────────────────────────────────────────────────────
    train_ds = build_bfpo_dataset(args.data_dir, args.data_files, args.safety_files)

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

    trainer = BFPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        b1=args.b1,
        alpha=args.alpha,
    )

    logger.info(
        "Starting BFPO training (b1=%.1f, alpha=%.2f, beta=%.2f)…",
        args.b1, args.alpha, args.beta,
    )
    trainer.train()

    final_path = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info("Saved to %s", final_path)


if __name__ == "__main__":
    main()
