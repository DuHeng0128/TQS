"""
CoderForge-Preview fine-tuning script for Qwen3-4B
Hardware target: 1 node × 8 × A800 (80 GB)  [or fewer GPUs — 4B fits on 2-4 GPUs]
Parallelism: DeepSpeed ZeRO-2 (sufficient for 4B; ZeRO-3 also works)
Context: 128K tokens with sequential multi-packing + boundary-aware position_ids

Paper results: best checkpoint at epoch 5, pass@1 = 43.0% on SWE-Bench Verified.
Repetition penalty: helpful for epoch 1-3 checkpoints (use 1.05-1.1),
                    not beneficial at best checkpoint (epoch 5).

Usage (single node, 8 GPUs):
    deepspeed --num_gpus=8 training/train_4b.py [args]
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from liger_kernel.transformers import apply_liger_kernel_to_qwen3
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
    p = argparse.ArgumentParser(description="CoderForge-Preview SFT — Qwen3-4B")

    # Model
    p.add_argument("--model_name_or_path", type=str, default="<model_path>/Qwen/Qwen3-4B")
    p.add_argument("--attn_implementation", type=str, default="flash_attention_2",
                   choices=["flash_attention_2", "sdpa", "eager"])

    # Data
    p.add_argument("--local_data_dir", type=str,
                   default="/path/to/CoderForge-Preview/trajectories-TQS/top_decile")
    p.add_argument("--hf_repo", type=str, default="togethercomputer/CoderForge-Preview")
    p.add_argument("--hf_config", type=str, default="trajectories-tokenized_qwencoder")
    p.add_argument("--max_seq_length", type=int, default=65536)

    # Training — paper: best at epoch 5 for 4B
    p.add_argument("--output_dir", type=str, default="./coderforge_4b_top_decile")
    p.add_argument("--num_train_epochs", type=float, default=1.0,
                   help="4B best checkpoint is at epoch 5 per paper.")
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
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
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_total_limit", type=int, default=10)
    p.add_argument("--save_only_model", action="store_true", default=True,
                   help="Only save model weights, skip optimizer/scheduler states. "
                        "Saves ~3x disk space; cannot resume training from these checkpoints.")
    p.add_argument("--seed", type=int, default=42)

    # Resume support (for save_only_model checkpoints that lack optimizer states)
    p.add_argument("--resume_from_step", type=int, default=0,
                   help="Step number of the checkpoint loaded via --model_name_or_path. "
                        "Skips the first (resume_from_step * num_gpus * per_device_batch_size) "
                        "dataset samples and positions the LR scheduler at this step. "
                        "Use together with --max_steps to run only the remaining steps.")
    p.add_argument("--max_steps", type=int, default=-1,
                   help="If > 0, stop after this many optimizer steps (overrides num_train_epochs).")

    # DeepSpeed
    p.add_argument("--deepspeed", type=str, default="training/configs/ds_zero2_actoffload_4b.json",
                   help="ZeRO-2; activation checkpointing handled by HF TrainingArguments.")

    # Repetition penalty (inference-time ablation)
    p.add_argument("--repetition_penalty", type=float, default=1.0,
                   help="1.0 = disabled. For early checkpoints (epoch 1-3), "
                        "try 1.05-1.1 at inference time to reduce degenerate loops. "
                        "Not applied during SFT training.")

    # parse_known_args ignores --local_rank injected by DeepSpeed/torchrun launcher
    args, _ = p.parse_known_args()
    return args


def main():
    args = parse_args()
    set_seed(args.seed)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = local_rank == 0

    if is_main:
        logger.info(f"Args: {vars(args)}")
        logger.info(
            "Note: repetition_penalty is an inference parameter only. "
            f"Value {args.repetition_penalty} is logged for experiment tracking."
        )

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

    # When resuming from a save_only_model checkpoint (no optimizer states), skip
    # the samples that were already consumed in the previous run so the model sees
    # new data rather than repeating the beginning of the epoch.
    if args.resume_from_step > 0:
        # Each optimizer step consumed num_gpus × per_device_batch_size samples.
        # SequentialSampler iterates in index order, so we simply trim the front.
        num_gpus_used = int(os.environ.get("WORLD_SIZE", 1))
        samples_to_skip = args.resume_from_step * num_gpus_used * args.per_device_train_batch_size
        if samples_to_skip >= len(packed_ds):
            raise ValueError(
                f"resume_from_step={args.resume_from_step} would skip all "
                f"{len(packed_ds)} samples (need to skip {samples_to_skip})."
            )
        from torch.utils.data import Subset
        remaining_indices = list(range(samples_to_skip, len(packed_ds)))
        packed_ds = Subset(packed_ds, remaining_indices)
        if is_main:
            logger.info(
                f"Resuming from step {args.resume_from_step}: "
                f"skipped {samples_to_skip} samples, {len(packed_ds)} remaining."
            )

    # 3. Model — for 4B, ZeRO-2 keeps full model on each GPU but shards gradients/optimizer
    if is_main:
        logger.info(f"Loading model: {args.model_name_or_path}")

    # Apply Liger Kernel before model load: replaces cross-entropy + RoPE + RMSNorm with
    # fused CUDA kernels. FusedLinearCrossEntropy never materializes the full logits tensor
    # (seq_len × vocab_size), eliminating the dominant OOM source at 128K context.
    apply_liger_kernel_to_qwen3(
        rope=True,
        rms_norm=True,
        swiglu=True,
        cross_entropy=False,
        fused_linear_cross_entropy=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # 4. TrainingArguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,  # -1 = disabled; >0 overrides num_train_epochs
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
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        save_only_model=args.save_only_model,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        report_to="none",
        seed=args.seed,
        deepspeed=args.deepspeed,
        label_names=["labels"],
    )

    # 5. Train
    trainer = TokenNormalizedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=packed_ds,
        tokenizer=tokenizer,
    )

    # Reposition the LR scheduler when resuming from a save_only_model checkpoint.
    # Without this, cosine decay restarts from LR=1e-5 instead of continuing from
    # the ~2.3e-6 it was at step 1500.
    if args.resume_from_step > 0:
        # trainer.train() calls create_optimizer_and_scheduler internally.
        # We hook in after that by overriding _maybe_log_save_evaluate is too invasive;
        # instead we patch the scheduler post-creation via a callback.
        from transformers import TrainerCallback

        resume_step = args.resume_from_step

        class ResumeSchedulerCallback(TrainerCallback):
            """Advance the LR scheduler by resume_from_step steps after optimizer creation."""
            def on_train_begin(self, args, state, control, **kwargs):
                optimizer = kwargs.get("optimizer") or trainer.optimizer
                scheduler = kwargs.get("lr_scheduler") or trainer.lr_scheduler
                if scheduler is None:
                    return
                if is_main:
                    logger.info(
                        f"Advancing LR scheduler by {resume_step} steps "
                        f"to restore position after resume."
                    )
                # Step the scheduler forward without updating optimizer params
                for _ in range(resume_step):
                    scheduler.step()

        trainer.add_callback(ResumeSchedulerCallback())

    if is_main:
        logger.info("Starting training...")
    trainer.train()

    if is_main:
        trainer.save_model(os.path.join(args.output_dir, "final"))
        tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))
        logger.info("Done.")


if __name__ == "__main__":
    main()
