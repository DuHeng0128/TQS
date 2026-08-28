#!/usr/bin/env python3
"""
SWE-bench patch verification script.

Takes the results.jsonl produced by run_eval.py, converts valid patches
into the format expected by swebench, then runs the official harness to
compute the resolve rate (pass@1).

Usage:
    # First install swebench if not already:
    #   pip install swebench
    #
    # Then:
    python3 run_verify.py [--results-jsonl PATH] [--output-dir DIR] [--max-workers N]

Prerequisites:
    - Docker must be running (swebench runs tests inside Docker containers)
    - swebench package installed
    - results.jsonl from run_eval.py
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DEFAULT_RESULTS = '/path/to/eval/eval_results_bottom/results.jsonl'
DEFAULT_DATASET = '/path/to/eval/SWE-bench_Verified/data/test-00000-of-00001.parquet'
DEFAULT_OUTPUT  = '/path/to/eval/eval_results_bottom/verification'


def load_results(results_path: str) -> list[dict]:
    """Load results.jsonl and return list of dicts."""
    results = []
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def extract_patches_from_trajectories(results: list[dict]) -> list[dict]:
    """For results where patch is None, try to extract patch from trajectory.

    Looks for the last git diff output in the trajectory, or the last
    CmdRunAction that ran 'git diff'.
    """
    for r in results:
        if r.get('patch'):
            continue
        traj_path = r.get('trajectory_path')
        if not traj_path or not Path(traj_path).exists():
            continue
        try:
            with open(traj_path) as f:
                events = json.load(f)
            # Look for git diff output in reverse order
            for event in reversed(events):
                content = event.get('content', '') or ''
                args = event.get('args', {})
                output = args.get('content', '') or content
                if isinstance(output, str) and 'diff --git' in output:
                    # Extract just the diff part
                    idx = output.index('diff --git')
                    r['patch'] = output[idx:]
                    break
        except Exception:
            pass
    return results


def create_predictions_file(results: list[dict], output_path: str) -> int:
    """Create swebench-format predictions JSONL file.

    Format per line: {"instance_id": "...", "model_name_or_path": "...", "model_patch": "..."}

    Returns count of valid predictions.
    """
    count = 0
    with open(output_path, 'w') as f:
        for r in results:
            patch = r.get('patch')
            if not patch:
                continue
            pred = {
                'instance_id': r['instance_id'],
                'model_name_or_path': 'qwen3-4b',
                'model_patch': patch,
            }
            f.write(json.dumps(pred) + '\n')
            count += 1
    return count


def compute_basic_stats(results: list[dict], dataset_path: str) -> dict:
    """Compute basic statistics without running the harness."""
    total = len(results)
    completed = sum(1 for r in results if r.get('status') == 'completed')
    errors = sum(1 for r in results if r.get('status') == 'error')
    has_patch = sum(1 for r in results if r.get('patch'))
    no_patch = completed - has_patch

    # Load dataset to get total expected
    df = pd.read_parquet(dataset_path)
    dataset_total = len(df)

    avg_duration = 0
    durations = [r.get('duration_seconds', 0) for r in results if r.get('duration_seconds')]
    if durations:
        avg_duration = sum(durations) / len(durations)

    return {
        'dataset_total': dataset_total,
        'evaluated': total,
        'completed': completed,
        'errors': errors,
        'has_patch': has_patch,
        'no_patch_generated': no_patch,
        'avg_duration_seconds': round(avg_duration, 1),
    }


def run_swebench_harness(predictions_path: str, output_dir: str, max_workers: int, dataset: str):
    """Run the official swebench evaluation harness."""
    try:
        from swebench.harness.run_evaluation import main as run_evaluation
    except ImportError:
        print('\nERROR: swebench package not installed.')
        print('Install it with: pip install swebench')
        return None

    print(f'\nRunning swebench harness...')
    print(f'  Predictions : {predictions_path}')
    print(f'  Output      : {output_dir}')
    print(f'  Max workers : {max_workers}')

    import subprocess, time
    run_id = f'run_{int(time.time())}'
    report_dir = f'{output_dir}/logs'
    Path(report_dir).mkdir(parents=True, exist_ok=True)

    # swebench.harness wants the dataset *root* (HF-style), not a .parquet file.
    # If --dataset points at a parquet under <root>/data/*.parquet, recover the root.
    dataset_for_harness = dataset
    if dataset.endswith('.parquet'):
        dataset_for_harness = str(Path(dataset).parent.parent)

    cmd = [
        sys.executable, '-m', 'swebench.harness.run_evaluation',
        '--predictions_path', predictions_path,
        '--run_id', run_id,
        '--max_workers', str(max_workers),
        '--report_dir', report_dir,
        '--dataset_name', dataset_for_harness,
        '--split', 'test',
        '--namespace', 'none',
    ]
    print(f'  Running: {" ".join(cmd)}')
    subprocess.run(cmd, check=True)

    results_file = Path(report_dir) / f'{run_id}.json'
    if results_file.exists():
        with open(results_file) as f:
            return json.load(f)

    for p in Path(report_dir).rglob('*.json'):
        if 'result' in p.name.lower() or run_id in p.name:
            with open(p) as f:
                return json.load(f)
    return None


def print_report(stats: dict, harness_results: dict | None, predictions_count: int):
    """Print final evaluation report."""
    print('\n' + '=' * 60)
    print('  SWE-bench Verified Evaluation Report')
    print('=' * 60)
    print(f'  Dataset instances   : {stats["dataset_total"]}')
    print(f'  Evaluated           : {stats["evaluated"]}')
    print(f'  Completed (no err)  : {stats["completed"]}')
    print(f'  Errors              : {stats["errors"]}')
    print(f'  Generated patch     : {stats["has_patch"]}')
    print(f'  No patch            : {stats["no_patch_generated"]}')
    print(f'  Avg duration (s)    : {stats["avg_duration_seconds"]}')
    print()

    if stats["evaluated"] > 0:
        patch_rate = stats["has_patch"] / stats["evaluated"] * 100
        print(f'  Patch generation rate : {patch_rate:.1f}%')

    if harness_results:
        resolved = harness_results.get('resolved', 0)
        total_tested = harness_results.get('total', predictions_count)
        if total_tested > 0:
            resolve_rate = resolved / total_tested * 100
            print(f'  Resolved (pass@1)    : {resolved}/{total_tested} = {resolve_rate:.1f}%')
            # Also compute resolve rate over full dataset
            full_rate = resolved / stats["dataset_total"] * 100
            print(f'  Resolve rate (full)  : {resolved}/{stats["dataset_total"]} = {full_rate:.1f}%')
    else:
        print('  [Harness not run - install swebench to compute resolve rate]')

    print('=' * 60)


def main():
    parser = argparse.ArgumentParser(description='Verify SWE-bench patches')
    parser.add_argument('--results-jsonl', default=DEFAULT_RESULTS,
                        help='Path to results.jsonl from run_eval.py')
    parser.add_argument('--dataset', default=DEFAULT_DATASET,
                        help='Path to SWE-bench dataset parquet')
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT,
                        help='Directory for verification output')
    parser.add_argument('--max-workers', type=int, default=4,
                        help='Max parallel Docker containers for testing')
    parser.add_argument('--skip-harness', action='store_true',
                        help='Only generate predictions file, skip harness')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load results
    print(f'Loading results from: {args.results_jsonl}')
    results = load_results(args.results_jsonl)
    print(f'  Loaded {len(results)} results')

    # 2. Try to extract patches from trajectories where patch is None
    results = extract_patches_from_trajectories(results)

    # 3. Basic stats
    stats = compute_basic_stats(results, args.dataset)

    # 4. Create predictions file
    predictions_path = str(output_dir / 'predictions.jsonl')
    count = create_predictions_file(results, predictions_path)
    print(f'  Valid predictions with patches: {count}')

    if count == 0:
        print('\nNo valid patches found. Cannot run harness.')
        print_report(stats, None, 0)
        return

    # 5. Run harness (or skip)
    harness_results = None
    if not args.skip_harness:
        harness_results = run_swebench_harness(
            predictions_path, str(output_dir), args.max_workers, args.dataset
        )

    # 6. Report
    print_report(stats, harness_results, count)

    # Save stats
    stats_path = output_dir / 'stats.json'
    with open(stats_path, 'w') as f:
        json.dump({**stats, 'predictions_with_patch': count,
                   'harness_results': harness_results}, f, indent=2)
    print(f'\nStats saved to: {stats_path}')


if __name__ == '__main__':
    main()
