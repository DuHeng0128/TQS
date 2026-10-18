#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
SERVED_NAME=tqs-agent
LORA_ARGS=()
if [[ -n "${LORA_PATH:-}" ]]; then
  [[ -f "$LORA_PATH/adapter_config.json" ]] || { echo "Invalid adapter directory: $LORA_PATH" >&2; exit 1; }
  SERVED_NAME=tqs-base
  LORA_ARGS=(--enable-lora --max-lora-rank 32 --lora-modules "tqs-agent=$LORA_PATH")
fi

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" --served-model-name "$SERVED_NAME" \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-8005}" \
  --tensor-parallel-size "${TP_SIZE:-2}" --max-model-len "${MAX_MODEL_LEN:-65536}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.92}" --dtype auto \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --override-generation-config '{"temperature":0.2,"top_p":0.8,"top_k":20,"repetition_penalty":1.05}' \
  "${LORA_ARGS[@]}" "$@"
