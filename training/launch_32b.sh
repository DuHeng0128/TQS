#!/bin/bash
# Launch CoderForge-Preview SFT for Qwen3-32B
# Hardware: 8 nodes × 8 × H100 (64 GPUs total)
# Parallelism: FSDP2 full_shard + Ulysses sequence parallelism (size=8)
#
# Environment variables that MUST be set before running:
#   NODE_RANK    rank of this node (0 to 7)
#   MASTER_ADDR  IP address of rank-0 node
#   MASTER_PORT  communication port (default: 29500)
#
# Run this script on ALL 8 nodes simultaneously.
#
# Usage:
#   NODE_RANK=0 MASTER_ADDR=10.0.0.1 bash training/launch_32b.sh          # rank-0 node
#   NODE_RANK=1 MASTER_ADDR=10.0.0.1 bash training/launch_32b.sh          # rank-1 node
#   ...

set -euo pipefail

NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}
NNODES=${NNODES:-8}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
OUTPUT_DIR=${OUTPUT_DIR:-./coderforge_32b_output}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}"

echo "=== CoderForge-Preview Qwen3-32B SFT ==="
echo "Project root: ${PROJECT_ROOT}"
echo "Node rank:    ${NODE_RANK} / ${NNODES}"
echo "Master:       ${MASTER_ADDR}:${MASTER_PORT}"
echo "Output dir:   ${OUTPUT_DIR}"
echo ""

torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    training/train_32b.py \
    --model_name_or_path Qwen/Qwen3-32B \
    --local_data_dir "${PROJECT_ROOT}/CoderForge-Preview/trajectories-tokenized_qwencoder" \
    --max_seq_length 131072 \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --attn_implementation flash_attention_2 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_config training/configs/fsdp_32b.json \
    --ulysses_size 8 \
    --save_steps 200 \
    --logging_steps 10 \
    --save_total_limit 15 \
    --repetition_penalty 1.0 \
    "$@"

echo ""
echo "Training complete. Checkpoints saved to: ${OUTPUT_DIR}"
