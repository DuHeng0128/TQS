"""Portable, CPU-only entry point for the released TQS scorer."""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def _records(path: Path):
    """Yield rows a shard at a time, preserving source IDs and token counts."""
    files = sorted(p for p in path.rglob("*") if p.suffix in {".json", ".jsonl", ".parquet", ".traj"}) if path.is_dir() else [path]
    if not files:
        raise ValueError(f"No trajectory files found in {path}")
    for file in files:
        if file.suffix == ".parquet":
            import pyarrow.parquet as pq

            for batch in pq.ParquetFile(file).iter_batches(batch_size=256):
                for row in batch.to_pylist():
                    yield file, row
        elif file.suffix == ".jsonl":
            with file.open(encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        yield file, json.loads(line)
        else:
            value = json.loads(file.read_text(encoding="utf-8"))
            # A message/event list represents a single trajectory. A list of
            # row objects represents a collection of trajectories.
            if isinstance(value, list) and value and isinstance(value[0], dict) and any(k in value[0] for k in ("messages", "trajectory", "trajectory_id")):
                for row in value:
                    yield file, row
            else:
                yield file, value


def _clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def _score(item):
    file, raw, fmt = item
    from .trajectory_metrics.evaluate import _compute_one
    from .trajectory_metrics.parsers import CoderForgeParser, OpenHandsParser, SWEAgentParser

    row = raw if isinstance(raw, dict) else {}
    chosen = fmt
    if chosen == "auto":
        if isinstance(raw, list):
            chosen = "openhands_json" if raw and "source" in raw[0] else "coderforge"
        elif "messages" in row:
            chosen = "coderforge"
        else:
            trajectory = row.get("trajectory", [])
            chosen = "swe_agent" if file.suffix == ".traj" or (trajectory and trajectory[0].get("role") == "ai") else "openhands"
    if chosen == "openhands_json":
        from .trajectory_metrics.parsers.openhands_events import parse_json_trajectory

        if not isinstance(raw, list):
            raise ValueError(f"{file}: openhands_json expects an event list")
        traj = parse_json_trajectory(file)
    elif chosen == "coderforge":
        row = {"trajectory_id": file.stem, "messages": raw} if isinstance(raw, list) else row
        traj = CoderForgeParser().parse_row(row)
    elif chosen == "openhands":
        traj = OpenHandsParser().parse_row(row)
    else:
        traj = SWEAgentParser().parse_row(row)
    if traj is None:
        raise ValueError(f"{file}: no agent steps parsed; check --format")
    result = _compute_one(traj)
    result.pop("_full_metrics", None)
    result["trajectory_id"] = str(row.get("trajectory_id", traj.instance_id))
    result["source_file"] = str(file)
    result["estimated_tokens"] = result.pop("total_tokens")
    # Character-count estimates are not the tokenizer budget used for SFT.
    # Leave the budget unknown unless the input supplies an actual count.
    tokens = row.get("total_tokens", row.get("num_tokens"))
    if tokens is None and isinstance(row.get("input_ids"), (list, tuple)):
        tokens = len(row["input_ids"])
    result["total_tokens"] = tokens
    result["eligible_structure"] = traj.total_steps >= 3 and result["tqs_coverage"] >= 4 / 7
    return _clean(result)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Score coding-agent trajectories with TQS (CPU only)")
    parser.add_argument("--input", type=Path, required=True, help="JSON/JSONL/parquet file or trajectory directory")
    parser.add_argument("--output", type=Path, required=True, help="New .jsonl, .csv, or .parquet score table")
    parser.add_argument("--format", default="auto", choices=["auto", "coderforge", "openhands", "openhands_json", "swe_agent"])
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if not args.input.exists():
        parser.error(f"Input does not exist: {args.input}")
    if args.output.exists():
        parser.error(f"Output already exists; choose a new path: {args.output}")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.output.suffix not in {".jsonl", ".csv", ".parquet"}:
        parser.error("--output must end in .jsonl, .csv, or .parquet")
    items = ((file, row, args.format) for file, row in _records(args.input))
    try:
        if args.workers == 1:
            records = list(map(_score, items))
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                records = list(pool.map(_score, items, chunksize=16))
        if not records:
            raise ValueError("No trajectories found")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.suffix == ".jsonl":
            with args.output.open("w", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        else:
            import pandas as pd

            frame = pd.DataFrame(records)
            if args.output.suffix == ".parquet":
                frame.to_parquet(args.output, index=False)
            else:
                frame.to_csv(args.output, index=False)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"Scored {len(records)} trajectories -> {args.output}")
