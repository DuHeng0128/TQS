#!/bin/bash
# Launch CoderForge-Preview SFT for Qwen3-14B
# Hardware: 1 node × 8 × A800 (80GB)
# Parallelism: DeepSpeed ZeRO-3
#
# Usage:
#   bash training/launch_14b.sh [extra args]
#   bash training/launch_14b.sh --num_train_epochs 3 --learning_rate 5e-6
#
# Optional environment variables:
#   OUTPUT_DIR   output directory (default: ./coderforge_14b_output)
#   NUM_GPUS     number of GPUs to use (default: 8)

set -euo pipefail

OUTPUT_DIR=${OUTPUT_DIR:-./coderforge_14b_output}
NUM_GPUS=${NUM_GPUS:-8}

# Navigate to project root regardless of where script is called from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}"

echo "=== CoderForge-Preview Qwen3-14B SFT ==="
echo "Project root: ${PROJECT_ROOT}"
echo "Output dir:   ${OUTPUT_DIR}"
echo "Num GPUs:     ${NUM_GPUS}"
echo ""

deepspeed --num_gpus="${NUM_GPUS}" training/train_14b.py \
    --model_name_or_path Qwen/Qwen3-14B \
    --local_data_dir "${PROJECT_ROOT}/CoderForge-Preview/trajectories-tokenized_qwencoder" \
    --max_seq_length 131072 \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --attn_implementation flash_attention_2 \
    --deepspeed training/configs/ds_zero3_14b.json \
    --save_steps 200 \
    --logging_steps 10 \
    --save_total_limit 10 \
    --repetition_penalty 1.0 \
    "$@"

echo ""
echo "Training complete. Checkpoints saved to: ${OUTPUT_DIR}"
