# Training

The three manifests in [`splits/`](splits/) contain the Top-TQS, Bottom-TQS,
and Random selections, with 1,024 trajectory IDs each. Use these IDs with the
original CoderForge-Preview parquet shards; the raw corpus and model weights
are not duplicated here.

```bash
python training/prepare_dataset.py \
  --source /path/to/CoderForge-Preview/trajectories \
  --ids training/splits/top.txt \
  --output outputs/data/top.jsonl

DATASET_PATH="$PWD/outputs/data/top.jsonl" \
OUTPUT_DIR="$PWD/outputs/top-lora" \
bash training/train_lora.sh
```

Install [ms-swift](https://github.com/modelscope/ms-swift) with a CUDA-compatible
PyTorch, DeepSpeed and FlashAttention environment on Linux before training.
The converter preserves assistant text and tool calls as training targets;
user messages and tool responses are context, masked by `--loss_scale default`.
It requires every requested ID and refuses to overwrite existing output.

The launcher uses Qwen3-Coder-30B-A3B-Instruct with LoRA rank 32, alpha 64,
dropout 0.05, learning rate 2e-5, five epochs, four processes, per-device batch
size 1, accumulation 8, and ZeRO-3. The trajectory-selection cutoff is 32,768
tokens; the formatted training sequence limit is 40,960. Change the manifest
and output paths for another selection. `MODEL_PATH`, `NPROC_PER_NODE`, and
`MASTER_PORT` can be overridden; additional training arguments are forwarded.
