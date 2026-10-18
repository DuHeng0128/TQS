#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_PATH="${DATASET_PATH:?Set DATASET_PATH to the prepared agent-SFT JSONL file}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/lora}"
[[ -s "$DATASET_PATH" ]] || { echo "Dataset not found or empty: $DATASET_PATH" >&2; exit 1; }

export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export MASTER_PORT="${MASTER_PORT:-29520}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

swift sft \
  --model "$MODEL_PATH" \
  --template qwen3_coder --agent_template qwen3_coder \
  --loss_scale default --tuner_type lora --dataset "$DATASET_PATH" \
  --torch_dtype bfloat16 --num_train_epochs 5 \
  --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
  --learning_rate 2e-5 --lora_rank 32 --lora_alpha 64 --lora_dropout 0.05 \
  --target_modules q_proj k_proj v_proj o_proj --router_aux_loss_coef 0.0 \
  --gradient_accumulation_steps 8 --max_grad_norm 0.5 --weight_decay 0.0 \
  --warmup_ratio 0.05 --max_length 40960 --truncation_strategy left \
  --attn_impl flash_attn --gradient_checkpointing true --deepspeed zero3 \
  --packing true --dataloader_num_workers 8 --dataset_num_proc 16 --seed 42 \
  --save_steps 20 --eval_steps 0 --save_total_limit 20 --logging_steps 5 \
  --save_only_model true --output_dir "$OUTPUT_DIR" --report_to none "$@"
