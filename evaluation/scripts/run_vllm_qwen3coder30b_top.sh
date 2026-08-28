#!/usr/bin/env bash
set -euo pipefail

# Qwen3-Coder-30B-A3B-Instruct + LoRA (TQS top, v2) on port 8006
BASE_MODEL_PATH="<model_path>/Qwen/Qwen3-Coder-30B-A3B-Instruct"
LORA_PATH="<lora_checkpoint_path>"
LORA_NAME="qwen3-coder-30b-tqs-top"

HOST="0.0.0.0"
PORT="${PORT:-8006}"
SERVED_MODEL_NAME="qwen3-coder-30b-a3b-instruct"

# GPU 分配：top LoRA 固定在 rank 2,3
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

# 30B-A3B 通常需要多卡；可通过环境变量覆盖。
TP_SIZE="${TP_SIZE:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
# adapter_config.json: r=32 -> 至少 32
MAX_LORA_RANK="${MAX_LORA_RANK:-32}"

echo "Starting vLLM server (Qwen3-Coder-30B + TQS top LoRA)"
echo "  base_model : ${BASE_MODEL_PATH}"
echo "  lora_path  : ${LORA_PATH}"
echo "  lora_name  : ${LORA_NAME}"
echo "  endpoint   : http://${HOST}:${PORT}/v1"
echo "  gpus       : ${CUDA_VISIBLE_DEVICES}"
echo "  tp_size    : ${TP_SIZE}"
echo "  max_len    : ${MAX_MODEL_LEN}"

python3 -m vllm.entrypoints.openai.api_server \
  --model "${BASE_MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --dtype auto \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --enable-lora \
  --max-lora-rank "${MAX_LORA_RANK}" \
  --lora-modules "${LORA_NAME}=${LORA_PATH}" \
  --override-generation-config '{"temperature":0.2,"top_p":0.8,"top_k":20,"repetition_penalty":1.05}'
