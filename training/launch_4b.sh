#!/bin/bash
# Launch CoderForge-Preview SFT for Qwen3-4B — two-run comparison experiment
#
# Run 1: top_decile  → ./coderforge_4b_top_decile
# Run 2: bottom_decile → ./coderforge_4b_bottom_decile
#
# Checkpoints: every 500 steps, model weights only (no optimizer states).
#
# Usage:
#   bash training/launch_4b.sh            # both runs sequentially
#   bash training/launch_4b.sh top        # top_decile only
#   bash training/launch_4b.sh bottom     # bottom_decile only
#   NUM_GPUS=4 bash training/launch_4b.sh

set -euo pipefail

NUM_GPUS=${NUM_GPUS:-8}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}"

RUN=${1:-both}

run_training() {
    local data_dir="$1"
    local output_dir="$2"
    local tag="$3"

    echo ""
    echo "=== CoderForge-Preview Qwen3-4B SFT — ${tag} ==="
    echo "  Data:   ${data_dir}"
    echo "  Output: ${output_dir}"
    echo "  GPUs:   ${NUM_GPUS}"
    echo ""

    local log_file="${output_dir}/train.log"
    mkdir -p "${output_dir}"

    deepspeed --num_gpus="${NUM_GPUS}" training/train_4b.py \
        --model_name_or_path <model_path>/Qwen/Qwen3-4B \
        --local_data_dir "${data_dir}" \
        --max_seq_length 65536 \
        --output_dir "${output_dir}" \
        --num_train_epochs 1 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --learning_rate 1e-5 \
        --lr_scheduler_type cosine \
        --warmup_ratio 0.03 \
        --weight_decay 0.01 \
        --max_grad_norm 1.0 \
        --attn_implementation flash_attention_2 \
        --deepspeed training/configs/ds_zero2_actoffload_4b.json \
        --save_steps 500 \
        --logging_steps 10 \
        --save_total_limit 10 \
        --save_only_model \
        --repetition_penalty 1.0 \
        2>&1 | tee "${log_file}"

    echo ""
    echo "Done: ${tag}. Checkpoints saved to: ${output_dir}"
}

if [[ "${RUN}" == "top" || "${RUN}" == "both" ]]; then
    run_training \
        "${PROJECT_ROOT}/CoderForge-Preview/trajectories-TQS/top_decile" \
        "${PROJECT_ROOT}/coderforge_4b_top_decile" \
        "top_decile"
fi

if [[ "${RUN}" == "bottom" || "${RUN}" == "both" ]]; then
    run_training \
        "${PROJECT_ROOT}/CoderForge-Preview/trajectories-TQS/bottom_decile" \
        "${PROJECT_ROOT}/coderforge_4b_bottom_decile" \
        "bottom_decile"
fi
