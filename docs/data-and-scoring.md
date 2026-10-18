# Data and scoring

## Inputs

| `--format` | Record structure |
| --- | --- |
| `coderforge` | `trajectory_id`, `messages`, optional `reward` |
| `openhands` | `instance_id`, `trajectory` with role/content/tool_calls messages |
| `openhands_json` | A source/action/observation event list |
| `swe_agent` | `instance_id`, `trajectory` with ai/user messages |

JSON accepts a record, a list of records, or a message/event list. JSONL has one
record per line; parquet has one record per row. Directory inputs are scanned
recursively. Use `--format auto` for common schemas or specify a format.

```bash
python -m tqs --input /path/to/shards --format coderforge \
  --output outputs/scores.parquet --workers 4
```

Outputs may be `.jsonl`, `.csv`, or `.parquet`. Invalid trajectories fail
explicitly. Tool calls are parsed, never executed. Outcome labels are retained
as metadata and do not enter the score formula.

## Score fields

| Field | Meaning |
| --- | --- |
| `trajectory_id` | ID joining scores to raw trajectories |
| `total_steps` | Parsed action count |
| `total_tokens` | Supplied tokenizer-derived length; null if unavailable |
| `estimated_tokens` | Character-count estimate, diagnostic only |
| `tqs_score`, `tqs_coverage` | Composite score and fraction of defined components |
| `tqs_oec`, `tqs_iac`, `tqs_dpi`, `tqs_ped`, `tqs_psn`, `tqs_tte`, `tqs_scp` | Normalized component scores |
| `eligible_structure` | At least three steps and coverage ≥4/7 |
| `m*` | Raw metric diagnostics |

Missing components are excluded and remaining weights are renormalized. A
constant scope sequence has neutral raw PSN 0. `total_tokens` accepts a supplied
`total_tokens` / `num_tokens` field or the length of `input_ids`. Compute lengths
with your training tokenizer/template before budgeted selection; character
estimates are not substituted for missing token counts.

## Subset selection

Use the [released manifests](../training/README.md) to prepare the supplied
Top-TQS, Bottom-TQS, and Random selections. To select new subsets:

```bash
python -m tqs.split --scores outputs/scores.parquet \
  --source-dir /path/to/raw-shards --output-dir outputs/subsets \
  --k 1024 --max-tokens 32768 --seed 42
```

Selection filters success, length, step count, and coverage, then deduplicates
IDs. Source proportions are matched. Top/Bottom apply three-pass task/repository
diversity selection with a per-task cap of 30; Random samples within source
strata. Source names are shard filename prefixes before the first hyphen.
The optional `--max-ideal-frac` controls an HCR diversity quota. Newly sampled
sets need not match the fixed released manifests. Output must be an empty
directory and contains three raw parquet subsets plus `split_summary.json`.

To recompute an existing component table without parsing trajectories again:

```bash
python -m tqs.rerank --input outputs/scores.parquet \
  --output outputs/reranked.parquet
```
