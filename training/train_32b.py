"""
CoderForge-Preview fine-tuning script for Qwen3-32B
Hardware target: 8 nodes × 8 × H100 (64 GPUs total)
Parallelism:
  - FSDP2: shard_size=8 (full shard across all 64 GPUs)
  - Ulysses sequence parallelism: size=8 (splits the 128K sequence across 8 GPUs)
  - Gradient checkpointing: enabled

Paper results: best checkpoint at epoch ~3.13, pass@1 = 59.4% on SWE-Bench Verified.

NOTE on Ulysses sequence parallelism:
  Native HuggingFace Trainer does NOT support Ulysses out of the box.
  For production, use veScale, torchtitan, or a custom training loop.
  This script provides the FSDP2 + standard distributed setup.
  To enable Ulysses, uncomment the veScale integration block (requires veScale install).

Usage (multi-node, 8 nodes × 8 H100):
    # On each node (set NODE_RANK, MASTER_ADDR, MASTER_PORT appropriately):
    torchrun \\
        --nnodes=8 \\
        --nproc_per_node=8 \\
        --node_rank=$NODE_RANK \\
        --master_addr=$MASTER_ADDR \\
        --master_port=29500 \\
        training/train_32b.py \\
        --fsdp "full_shard auto_wrap" \\
        --fsdp_config training/configs/fsdp_32b.json \\
        [other args]
"""

import argparse
import logging
import os
import sys

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


def parse_args():
    p = argparse.ArgumentParser(description="CoderForge-Preview SFT — Qwen3-32B")

    # Model
    p.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-32B")
    p.add_argument("--attn_implementation", type=str, default="flash_attention_2",
                   choices=["flash_attention_2", "sdpa", "eager"])

    # Data
    p.add_argument("--local_data_dir", type=str,
                   default="/path/to/CoderForge-Preview/trajectories-tokenized_qwencoder")
    p.add_argument("--hf_repo", type=str, default="togethercomputer/CoderForge-Preview")
    p.add_argument("--hf_config", type=str, default="trajectories-tokenized_qwencoder")
    p.add_argument("--max_seq_length", type=int, default=131072)

    # Training — paper: best at epoch ~3.13 for 32B
    p.add_argument("--output_dir", type=str, default="./coderforge_32b_output")
    p.add_argument("--num_train_epochs", type=float, default=4.0,
                   help="Train up to epoch 4; best checkpoint is at ~3.13 per paper. "
                        "Use save_steps to capture intermediate checkpoints.")
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1,
                   help="With 64 GPUs and batch=1, global batch = 64 sequences per step. "
                        "Increase for larger effective batch.")
    p.add_argument("--learning_rate", type=float, default=1e-5,
                   help="⚠️ Not disclosed in paper.")
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.95)
    p.add_argument("--adam_epsilon", type=float, default=1e-8)

    # Checkpointing
    p.add_argument("--save_steps", type=int, default=200,
                   help="Save frequently to capture best epoch (~3.13).")
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_total_limit", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)

    # FSDP (passed via TrainingArguments)
    p.add_argument("--fsdp", type=str, default="full_shard auto_wrap",
                   help="FSDP strategy for HF Trainer. 'full_shard' = FSDP2 full shard.")
    p.add_argument("--fsdp_config", type=str,
                   default="training/configs/fsdp_32b.json",
                   help="Path to FSDP config JSON (transformer_layer_cls_to_wrap, etc.).")

    # Ulysses sequence parallelism
    p.add_argument("--ulysses_size", type=int, default=8,
                   help="Ulysses sequence parallel group size. "
                        "Requires veScale or torchtitan integration. "
                        "Set to 1 to disable (standard FSDP only).")

    # Repetition penalty
    p.add_argument("--repetition_penalty", type=float, default=1.0,
                   help="1.0 = disabled. Inference-time parameter only. "
                        "Not applied during SFT training.")

    return p.parse_args()


def maybe_init_ulysses(ulysses_size: int):
    """
    Optionally initialize Ulysses sequence parallelism via veScale.
    Requires: pip install vescale
    No-op if ulysses_size <= 1 or veScale is not installed.
    """
    if ulysses_size <= 1:
        return False
    try:
        import torch.distributed as dist
        from vescale.dmodule.api import parallelize_module
        from vescale.dtensor.device_mesh import init_device_mesh

        world_size = dist.get_world_size()
        # Device mesh: (data_parallel, sequence_parallel)
        dp_size = world_size // ulysses_size
        mesh = init_device_mesh("cuda", (dp_size, ulysses_size),
                                mesh_dim_names=("dp", "sp"))
        logger.info(f"Ulysses sequence parallelism initialized: "
                    f"dp={dp_size}, sp={ulysses_size}")
        return mesh
    except ImportError:
        logger.warning(
            "veScale not installed; Ulysses sequence parallelism is disabled. "
            "Install with: pip install vescale  "
            "For standard FSDP2 without sequence parallelism, training will proceed normally."
        )
        return False


def main():
    args = parse_args()
    set_seed(args.seed)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = local_rank == 0

    if is_main:
        logger.info(f"Args: {vars(args)}")
        if args.ulysses_size > 1:
            logger.info(
                f"Ulysses sequence parallel size={args.ulysses_size}. "
                "Attempting to initialize via veScale."
            )

    # Initialize Ulysses (best-effort; falls back gracefully)
    maybe_init_ulysses(args.ulysses_size)

    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # 2. Dataset
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
        logger.info(f"Packed dataset: {len(packed_ds)} sequences")

    # 3. Model — loaded on CPU; FSDP distributes shards across GPUs
    if is_main:
        logger.info(f"Loading model: {args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.gradient_checkpointing_enable()

    if is_main:
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {total_params / 1e9:.2f}B")

    # 4. TrainingArguments with FSDP
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
        fsdp=args.fsdp,
        fsdp_config=args.fsdp_config,
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        report_to="none",
        seed=args.seed,
        label_names=["labels"],
    )

    # 5. Train
    trainer = TokenNormalizedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=packed_ds,
        tokenizer=tokenizer,
    )

    if is_main:
        logger.info("Starting training...")
    trainer.train()

    if is_main:
        trainer.save_model(os.path.join(args.output_dir, "final"))
        tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))
        logger.info("Done.")


if __name__ == "__main__":
    main()
