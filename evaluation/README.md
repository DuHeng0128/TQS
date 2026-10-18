# Evaluation

The paper uses OpenHands v0.52.1. The runner uses an OpenAI-compatible vLLM endpoint and OpenHands with a Daytona
sandbox. Install these dependencies separately from the CPU scorer. Sandbox
execution requires a Daytona account/API key and may incur charges.

```bash
# Serve a trained adapter; omit LORA_PATH to serve the base model.
LORA_PATH=/path/to/adapter bash evaluation/serve.sh

# In another terminal, with DAYTONA_API_KEY already set:
python evaluation/run_eval.py \
  --dataset /path/to/SWE-bench_Verified/test.parquet \
  --output-dir outputs/eval/top

# Docker and the official swebench package are required for patch verification.
python evaluation/run_verify.py \
  --results-jsonl outputs/eval/top/results.jsonl \
  --output-dir outputs/verification/top --model-name top-tqs
```

[`config.toml`](config.toml) uses `tqs-agent` on port 8005 for either the base
model or selected adapter. Set `MODEL_PATH`, `TP_SIZE`, `CUDA_VISIBLE_DEVICES`,
or `PORT` when serving; update the config endpoint if the port changes.
Use `--config` for another OpenHands configuration and `OPENHANDS_ROOT` for a
local OpenHands checkout. Use independent output directories for each condition.

The verification wrapper preserves empty/failed patches, rejects duplicate
instance IDs, and delegates scoring/reporting to the official SWE-bench harness.
`--skip-harness` only exports predictions; it does not measure pass@1.
