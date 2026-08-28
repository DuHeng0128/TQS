# Supplementary Code

This directory contains the code for reproducing the experiments in the paper.

```
supplementary/
├── tqs/          TQS scoring system
├── training/        SFT training scripts (ms-swift / DeepSpeed / FSDP)
├── evaluation/      OpenHands evaluation pipeline on SWE-bench Verified
└── analysis/        Data analysis and statistics scripts
```

---

## 1. TQS (`tqs/`)

Trajectory Quality Score — a composite quality metric for code-agent trajectories used to select SFT training data.

### Key files

| File | Description |
|------|-------------|
| `tqs.py` | Core TQS formula: 7 components (OEC, IAC, DPI, PED, PSN, TTE, SCP) with default weights |
| `trajectory_metrics/` | M1–M10 metric implementations + parsers for OpenHands / SWE-agent / CoderForge formats |
| `rerank.py` | Re-rank an existing scored parquet with TQS weights (no re-running M1–M10) |
| `split.py` | Split eligible pool into Top-K / Bottom-K / Random-K subsets for SFT |
| `report.py` | Batch TQS scoring report for OpenHands trajectory directories |

### Quick start

```bash
# Score a directory of OpenHands JSON trajectories
python tqs/report.py --workers 8

# Re-rank an existing scored parquet
python tqs/rerank.py \
    --input path/to/coderforge_scores.parquet \
    --output path/to/coderforge_scores.parquet

# Split eligible pool into Top/Bottom/Random subsets
python tqs/split.py \
    --scores path/to/coderforge_scores.parquet \
    --top-k 1024 --bottom-k 1024 --random-k 1024
```

### Component weights (default)

| Component | Weight | Source metric |
|-----------|--------|---------------|
| OEC | 0.30 | M2 min_relative_entropy (inverted) |
| IAC | 0.25 | M4 action_type_consistency |
| DPI | 0.10 | M1+M2+M10 dirty pattern indicator |
| PED | 0.10 | M5 post_error_diversity |
| PSN | 0.10 | M6 psn_late |
| TTE | 0.10 | M9 tte_score |
| SCP | 0.05 | M10 scp |

### Supported trajectory formats

- `openhands` — OpenHands parquet (role/content/tool_calls schema)
- `openhands_json` — OpenHands action-based JSON (source/action/args schema)
- `swe_agent` — SWE-agent `.traj` / JSON files
- `coderforge` — CoderForge-Preview parquet (messages/reward schema)

---

## 2. Training (`training/`)

SFT training scripts for Qwen3 models using DeepSpeed ZeRO / FSDP with ms-swift.

### Key files

| File | Description |
|------|-------------|
| `train_4b.py` | Qwen3-4B SFT — 1 node × 8 × A800, DeepSpeed ZeRO-2, 128K context |
| `train_14b.py` | Qwen3-14B SFT — 1 node × 8 × A800, DeepSpeed ZeRO-3 |
| `train_32b.py` | Qwen3-32B SFT — 1 node × 8 × A800, FSDP |
| `data_utils.py` | Dataset classes: `CoderForgeTokenizedDataset`, `PackedCoderForgeDataset` (multi-packing with boundary-aware position_ids) |
| `trainer_utils.py` | `TokenNormalizedLossTrainer` — normalizes loss by token count rather than sequence count |
| `launch_4b.sh` | Launch script for 4B training |
| `launch_14b.sh` | Launch script for 14B training |
| `launch_32b.sh` | Launch script for 32B training |
| `configs/` | DeepSpeed ZeRO-2/3 and FSDP config JSON files |

### Quick start

```bash
# 4B model (8 GPUs)
deepspeed --num_gpus=8 training/train_4b.py \
    --model_name_or_path Qwen/Qwen3-4B \
    --data_path path/to/top1024_trajectories.parquet \
    --output_dir checkpoints/qwen3-4b-coderforge

# Or use the launch script
bash training/launch_4b.sh
```

### Hardware requirements

| Model | GPUs | Memory | Parallelism |
|-------|------|--------|-------------|
| Qwen3-4B | 8 × A800 (80GB) | ~160GB | DeepSpeed ZeRO-2 |
| Qwen3-14B | 8 × A800 (80GB) | ~640GB | DeepSpeed ZeRO-3 |
| Qwen3-32B | 8 × A800 (80GB) | ~640GB | FSDP |

---

## 3. Evaluation (`evaluation/`)

OpenHands-based evaluation pipeline on SWE-bench Verified, using vLLM for model serving and Daytona for sandboxed code execution.

### Key files

| File | Description |
|------|-------------|
| `run_eval.py` | Main evaluation script — runs OpenHands agent on SWE-bench Verified instances via Daytona sandbox |
| `run_swebench_eval.py` | Parallel SWE-bench harness runner — evaluates three prediction sets (base/top/random) concurrently |
| `run_verify.py` | Verify a subset of instances for quick sanity checks |
| `prefetch_swebench_images.py` | Pre-pull Docker images for all SWE-bench instances to avoid cold-start delays |
| `scripts/` | vLLM launch scripts and OpenHands config files for each model variant |

### Evaluation workflow

```bash
# Step 1: Start vLLM server
bash evaluation/scripts/run_vllm_qwen3coder30b.sh

# Step 2: Pre-fetch Docker images (optional but recommended)
python evaluation/prefetch_swebench_images.py

# Step 3: Run OpenHands agent on SWE-bench Verified
export DAYTONA_API_KEY="your_api_key"
python evaluation/run_eval.py --instances 500 --output-dir eval_results/

# Step 4: Run SWE-bench harness to score predictions
python evaluation/run_swebench_eval.py --workers 24
```

### Prerequisites

- vLLM server running on `localhost:8000` (or configured port)
- `DAYTONA_API_KEY` environment variable set
- Docker available for SWE-bench sandbox containers
- SWE-bench Verified dataset at `SWE-bench_Verified/data/`

---

## 4. Analysis (`analysis/`)

Scripts for data statistics and result analysis used in the paper.

| File | Description |
|------|-------------|
| `tqs_eval_report.py` | Compute TQS metrics (avg steps, OEC, PSN, SCP, PED, DPI, convergence ratio) across base/top/random trajectory sets |

### Usage

```bash
# TQS metrics report across three trajectory sets
python analysis/tqs_eval_report.py --workers 8 --output results/tqs_report.json
```

---

## Dependencies

```
torch >= 2.6
transformers >= 4.50
deepspeed >= 0.16
flash-attn >= 2.8
liger-kernel
pandas
numpy
scipy
pyarrow
openhands (for evaluation)
swebench (for evaluation harness)
```

Install core dependencies:

```bash
pip install torch transformers deepspeed pandas numpy scipy pyarrow
pip install flash-attn --no-build-isolation
pip install liger-kernel
```
