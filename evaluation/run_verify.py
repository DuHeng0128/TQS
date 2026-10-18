"""Export OpenHands patches and optionally run the official SWE-bench harness."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def to_predictions(results, model_name):
    predictions, seen = [], set()
    for result in results:
        instance_id = result["instance_id"]
        if instance_id in seen:
            raise ValueError(f"Duplicate instance ID: {instance_id}")
        seen.add(instance_id)
        patch = result.get("patch") or ""
        if not isinstance(patch, str):
            raise ValueError(f"Non-text patch: {instance_id}")
        predictions.append({"instance_id": instance_id, "model_name_or_path": model_name, "model_patch": patch})
    if not predictions:
        raise ValueError("No evaluation results supplied")
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="qwen3-coder-30b-tqs")
    parser.add_argument("--dataset", default="SWE-bench/SWE-bench_Verified", help="Hub dataset ID or local JSON/JSONL/parquet dataset")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--run-id", default="tqs")
    parser.add_argument("--skip-harness", action="store_true")
    args = parser.parse_args()
    if args.max_workers < 1:
        parser.error("--max-workers must be positive")
    results = [json.loads(line) for line in args.results_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    predictions = to_predictions(results, args.model_name)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "predictions.jsonl"
    with path.open("x", encoding="utf-8") as stream:
        for prediction in predictions:
            stream.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    print(f"Exported {len(predictions)} predictions, including empty/failed patches, to {path}")
    if not args.skip_harness:
        dataset = str(Path(args.dataset).resolve()) if Path(args.dataset).is_file() else args.dataset
        subprocess.run([
            sys.executable, "-m", "swebench.harness.run_evaluation",
            "--dataset_name", dataset, "--split", "test",
            "--predictions_path", str(path), "--run_id", args.run_id,
            "--max_workers", str(args.max_workers),
        ], cwd=output, check=True)


if __name__ == "__main__":
    main()
