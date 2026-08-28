"""
CoderForge-Preview fine-tuning script for Qwen3-14B
Hardware target: 1 node × 8 × A800 (80 GB)
Parallelism: DeepSpeed ZeRO-3 (shard weights + gradients + optimizer states)
Context: 128K tokens with sequential multi-packing + boundary-aware position_ids

Usage (single node, 8 GPUs):
    deepspeed --num_gpus=8 training/train_14b.py [args]

Or via torchrun (requires --deepspeed flag to work with HF Trainer):
    torchrun --nproc_per_node=8 training/train_14b.py \\
        --deepspeed training/configs/ds_zero3_14b.json [args]
"""

import argparse
import logging
import os
import sys

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
)

from training.data_utils import CoderForgeTokenizedDataset, PackedCoderForgeDataset
from training.trainer_utils import TokenNormalizedLossTrainer

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="CoderForge-Preview SFT — Qwen3-14B")

    # Model
    p.add_argument("--model_name_or_path", type=str, default="<model_path>/Qwen/Qwen3-14B")
    p.add_argument("--attn_implementation", type=str, default="flash_attention_2",
                   choices=["flash_attention_2", "sdpa", "eager"],
                   help="Attention backend. flash_attention_2 requires flash-attn package.")

    # Data
    p.add_argument("--local_data_dir", type=str,
                   default="/path/to/CoderForge-Preview/trajectories-TQS/top_half",
                   help="Local path to trajectories-tokenized_qwencoder parquet dir. "
                        "Falls back to HuggingFace Hub if not found.")
    p.add_argument("--hf_repo", type=str, default="togethercomputer/CoderForge-Preview")
    p.add_argument("--hf_config", type=str, default="trajectories-tokenized_qwencoder")
    p.add_argument("--max_seq_length", type=int, default=65536,
                   help="Packed sequence length (128K). Lower for debugging.")

    # Training
    p.add_argument("--output_dir", type=str, default="./coderforge_14b_top_half")
    p.add_argument("--num_train_epochs", type=float, default=1,
                   help="14B best checkpoint reported at epoch 5 (analogous to 4B).")
    p.add_argument("--per_device_train_batch_size", type=int, default=1,
                   help="Keep at 1 for 128K sequences; increase only for short-context ablations.")
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=1e-5,
                   help="Not disclosed in paper; 1e-5 is standard Qwen3 SFT starting point.")
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--warmup_ratio", type=float, default=0.03,
                   help="Not disclosed in paper; 3%% warmup is standard.")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.95)
    p.add_argument("--adam_epsilon", type=float, default=1e-8)

    # Checkpointing / logging
    p.add_argument("--save_steps", type=int, default=20000,
                   help="Save checkpoints frequently to evaluate best epoch.")
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_total_limit", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)

    # DeepSpeed (pass JSON config path)
    p.add_argument("--deepspeed", type=str, default="training/configs/ds_zero3_14b.json",
                   help="Path to DeepSpeed ZeRO-3 config JSON.")

    # Repetition penalty (for inference / early-stop ablation — not used during training)
    p.add_argument("--repetition_penalty", type=float, default=1.0,
                   help="1.0 = disabled. Log the value for experiment tracking only; "
                        "repetition penalty is an inference-time parameter, not used in SFT loss.")

    # parse_known_args ignores --local_rank injected by DeepSpeed/torchrun launcher
    args, _ = p.parse_known_args()
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = local_rank == 0

    if is_main:
        logger.info(f"Args: {vars(args)}")
        if args.repetition_penalty != 1.0:
            logger.info(
                f"repetition_penalty={args.repetition_penalty} noted for inference. "
                "This parameter is NOT applied during SFT training."
            )

    # ------------------------------------------------------------------
    # 1. Load tokenizer
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # ------------------------------------------------------------------
    # 2. Load and pack dataset
    # ------------------------------------------------------------------
    if is_main:
        logger.info("Loading CoderForge-Preview tokenized dataset...")

    raw_ds = CoderForgeTokenizedDataset(
        local_dir=args.local_data_dir,
        hf_repo=args.hf_repo,
        hf_config=args.hf_config,
        max_seq_len=args.max_seq_length,
    )

    packed_ds = PackedCoderForgeDataset(
        raw_dataset=raw_ds,
        max_seq_len=args.max_seq_length,
        pad_token_id=pad_token_id,
        seed=args.seed,
    )

    if is_main:
        logger.info(f"Packed dataset: {len(packed_ds)} sequences "
                    f"(~{args.max_seq_length / 1000:.0f}K tokens each)")

    # ------------------------------------------------------------------
    # 3. Load model (ZeRO-3 shards weights across GPUs automatically)
    # ------------------------------------------------------------------
    if is_main:
        logger.info(f"Loading model: {args.model_name_or_path}")

    # With DeepSpeed ZeRO-3, we load on CPU first; DeepSpeed moves shards to GPU.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
        # Load on CPU initially; ZeRO-3 will distribute
        low_cpu_mem_usage=True,
    )
    model.gradient_checkpointing_enable()

    if is_main:
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {total_params / 1e9:.2f}B")

    # ------------------------------------------------------------------
    # 4. TrainingArguments
    # ------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        bf16=True,
        gradient_checkpointing=True,
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        remove_unused_columns=False,   # We manage columns ourselves
        dataloader_num_workers=4,
        report_to="none",              # Add "wandb" or "tensorboard" as needed
        seed=args.seed,
        deepspeed=args.deepspeed,
        # Disable HF built-in loss; we use TokenNormalizedLossTrainer
        label_names=["labels"],
    )

    # ------------------------------------------------------------------
    # 5. Trainer
    # ------------------------------------------------------------------
    trainer = TokenNormalizedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=packed_ds,
        tokenizer=tokenizer,
    )

    # ------------------------------------------------------------------
    # 6. Train
    # ------------------------------------------------------------------
    if is_main:
        logger.info("Starting training...")

    trainer.train()

    if is_main:
        logger.info(f"Saving final model to {args.output_dir}/final")
        trainer.save_model(os.path.join(args.output_dir, "final"))
        tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))
        logger.info("Done.")


if __name__ == "__main__":
    main()
