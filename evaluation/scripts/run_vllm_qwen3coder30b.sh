#!/usr/bin/env bash
set -euo pipefail

# Qwen3-Coder-30B-A3B-Instruct (base) on port 8005
MODEL_PATH="<model_path>/Qwen/Qwen3-Coder-30B-A3B-Instruct"
SERVED_MODEL_NAME="qwen3-coder-30b-a3b-instruct"

HOST="0.0.0.0"
PORT="${PORT:-8005}"

# GPU 分配：base 模型固定在 rank 0,1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

# 30B-A3B 通常需要多卡；可通过环境变量覆盖。
TP_SIZE="${TP_SIZE:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"

echo "Starting vLLM server (Qwen3-Coder-30B base)"
echo "  model_path : ${MODEL_PATH}"
echo "  model_name : ${SERVED_MODEL_NAME}"
echo "  endpoint   : http://${HOST}:${PORT}/v1"
echo "  gpus       : ${CUDA_VISIBLE_DEVICES}"
echo "  tp_size    : ${TP_SIZE}"
echo "  max_len    : ${MAX_MODEL_LEN}"

python3 -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
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
  --override-generation-config '{"temperature":0.2,"top_p":0.8,"top_k":20,"repetition_penalty":1.05}'
