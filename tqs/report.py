"""report.py — TQS scoring report for Code_Trajectory_Eval datasets.

Scores three trajectory sets:
  - eval_results        (base)
  - eval_results_top    (top)
  - eval_results_random (random)

Each set is a directory of per-instance JSON files in OpenHands action-based format.

Reports per dataset:
  - avg steps
  - avg TQS score
  - avg OEC, PSN, SCP, PED component scores
  - convergence ratio (PSN > 0.3)

Usage:
    python report.py [--workers N]
"""

from __future__ import annotations

import json
import sys
import warnings
import argparse
import multiprocessing
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
TRAJ_METRICS_ROOT = REPO_ROOT / "Code_Trajectory"
TQS_V2_ROOT = Path(__file__).parent

sys.path.insert(0, str(TRAJ_METRICS_ROOT))
sys.path.insert(0, str(TQS_V2_ROOT))

from trajectory_metrics.parsers.base import ParsedStep, ParsedTrajectory
from trajectory_metrics.metrics import registry
import trajectory_metrics.metrics.m1_rbs   # noqa
import trajectory_metrics.metrics.m2_oec   # noqa
import trajectory_metrics.metrics.m3_fld   # noqa
import trajectory_metrics.metrics.m4_dcs   # noqa
import trajectory_metrics.metrics.m5_bus   # noqa
import trajectory_metrics.metrics.m6_psn   # noqa
import trajectory_metrics.metrics.m7_hcr   # noqa
import trajectory_metrics.metrics.m8_tgcr  # noqa
import trajectory_metrics.metrics.m9_tte   # noqa
import trajectory_metrics.metrics.m10_scp  # noqa
from tqs.tqs import compute_tqs

# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------

EVAL_BASE = Path(__file__).parent.parent / "Code_Trajectory_Eval"

DATASETS = {
    "base":   EVAL_BASE / "eval_results"        / "trajectories",
    "top":    EVAL_BASE / "eval_results_top"    / "trajectories",
    "random": EVAL_BASE / "eval_results_random" / "trajectories",
}

# ---------------------------------------------------------------------------
# Parser for OpenHands action-based JSON format
# ---------------------------------------------------------------------------

_ACTION_MAP = {
    "run":    "execute_bash",
    "edit":   "write_file",
    "think":  "think",
    "finish": "submit",
    "read":   "read_file",
    "browse": "search",
    "recall": "search",
}

_WORKSPACE_PREFIXES = ("/workspace/", "/repo/", "/testbed/")


def _strip_prefix(path: str) -> str:
    for p in _WORKSPACE_PREFIXES:
        if path.startswith(p):
            return path[len(p):]
    return path


def parse_json_trajectory(json_path: Path, resolved: Optional[bool] = None) -> Optional[ParsedTrajectory]:
    """Parse an OpenHands action-based JSON trajectory file into ParsedTrajectory."""
    try:
        with open(json_path) as f:
            raw = json.load(f)
    except Exception as e:
        warnings.warn(f"Failed to load {json_path}: {e}")
        return None

    # Build observation map: action_id -> content
    obs_map: dict[int, str] = {}
    for entry in raw:
        if entry.get("action") is None and entry.get("observation") is not None:
            cause = entry.get("cause")
            if cause is not None:
                obs_map[cause] = entry.get("content", "") or ""

    steps = []
    for entry in raw:
        if entry.get("source") != "agent":
            continue
        action = entry.get("action")
        if action is None or action == "system":
            continue

        args = entry.get("args") or {}
        thought = args.get("thought", "") or entry.get("message", "") or ""
        obs = obs_map.get(entry["id"], "")
        action_type = _ACTION_MAP.get(action, "other")

        path_arg = args.get("path", "") or ""
        files = [_strip_prefix(path_arg)] if path_arg and "/" in path_arg else []

        steps.append(ParsedStep(
            step_id=len(steps),
            thought=thought,
            action_type=action_type,
            action_args=args,
            observation=obs,
            target_files=files,
            token_count=max(1, (len(thought) + len(str(args)) + len(obs)) // 4),
        ))

    if not steps:
        return None

    return ParsedTrajectory(
        instance_id=json_path.stem,
        resolved=resolved,
        steps=steps,
        total_steps=len(steps),
        source_format="openhands_json",
    )


# ---------------------------------------------------------------------------
# Per-trajectory scoring
# ---------------------------------------------------------------------------

def _score_one(json_path: Path) -> Optional[dict]:
    traj = parse_json_trajectory(json_path)
    if traj is None:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            all_metrics = registry.compute_all(traj)
        tqs = compute_tqs(all_metrics)

        comps = tqs["components"]
        m6 = all_metrics.get("m6_psn") or {}
        psn_late_raw = m6.get("psn_late")  # raw [-1, 1] value for convergence check

        return {
            "instance_id": traj.instance_id,
            "total_steps": traj.total_steps,
            "tqs_score":   tqs["tqs_score"],
            "oec":         comps.get("oec"),
            "psn":         comps.get("psn"),
            "scp":         comps.get("scp"),
            "ped":         comps.get("ped"),
            "dpi":         comps.get("dpi"),
            "psn_late_raw": psn_late_raw,  # raw value for convergence threshold
        }
    except Exception as e:
        warnings.warn(f"Scoring failed for {json_path.stem}: {e}")
        return None


def _score_one_safe(json_path: Path) -> Optional[dict]:
    try:
        return _score_one(json_path)
    except Exception as e:
        warnings.warn(f"Unexpected error for {json_path}: {e}")
        return None


# ---------------------------------------------------------------------------
# Dataset-level aggregation
# ---------------------------------------------------------------------------

def score_dataset(traj_dir: Path, n_workers: int = 1) -> list[dict]:
    json_files = sorted(traj_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {traj_dir}")

    print(f"  Scoring {len(json_files)} trajectories from {traj_dir.parent.name}/trajectories ...")

    if n_workers > 1:
        with multiprocessing.Pool(n_workers) as pool:
            results = pool.map(_score_one_safe, json_files)
    else:
        results = [_score_one_safe(p) for p in json_files]

    return [r for r in results if r is not None]


def aggregate(records: list[dict]) -> dict:
    def _mean(key):
        vals = [r[key] for r in records if r.get(key) is not None]
        return float(np.mean(vals)) if vals else float("nan")

    # PSN convergence: psn_late_raw > 0.3 means converging (raw value in [-1,1])
    # component psn = (psn_late_raw + 1) / 2, so psn_late_raw > 0.3 <=> psn_component > 0.65
    # But the user asked for PSN > 0.3 on the component score, so use component value
    psn_vals = [r["psn"] for r in records if r.get("psn") is not None]
    convergence_ratio = float(np.mean([v > 0.3 for v in psn_vals])) if psn_vals else float("nan")

    return {
        "n":                  len(records),
        "avg_steps":          _mean("total_steps"),
        "avg_tqs":            _mean("tqs_score"),
        "avg_oec":            _mean("oec"),
        "avg_psn":            _mean("psn"),
        "avg_scp":            _mean("scp"),
        "avg_ped":            _mean("ped"),
        "avg_dpi":            _mean("dpi"),
        "convergence_ratio":  convergence_ratio,
    }


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def print_report(label: str, agg: dict) -> None:
    print(f"\n{'='*50}")
    print(f"  Dataset: {label}  (n={agg['n']})")
    print(f"{'='*50}")
    print(f"  Avg steps          : {agg['avg_steps']:.2f}")
    print(f"  Avg TQS         : {agg['avg_tqs']:.4f}")
    print(f"  Avg OEC            : {agg['avg_oec']:.4f}")
    print(f"  Avg PSN            : {agg['avg_psn']:.4f}")
    print(f"  Avg SCP            : {agg['avg_scp']:.4f}")
    print(f"  Avg PED            : {agg['avg_ped']:.4f}")
    print(f"  Avg DPI            : {agg['avg_dpi']:.4f}")
    print(f"  Convergence (PSN>0.3): {agg['convergence_ratio']*100:.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TQS report for Code_Trajectory_Eval datasets")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers (default: 4)")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    all_results = {}
    for label, traj_dir in DATASETS.items():
        records = score_dataset(traj_dir, n_workers=args.workers)
        agg = aggregate(records)
        all_results[label] = agg
        print_report(label, agg)

    # Comparison table
    print(f"\n{'='*70}")
    print(f"  {'Metric':<25} {'base':>10} {'top':>10} {'random':>10}")
    print(f"  {'-'*55}")
    metrics_order = [
        ("avg_steps",         "Avg steps"),
        ("avg_tqs",           "Avg TQS"),
        ("avg_oec",           "Avg OEC"),
        ("avg_psn",           "Avg PSN"),
        ("avg_scp",           "Avg SCP"),
        ("avg_ped",           "Avg PED"),
        ("avg_dpi",           "Avg DPI"),
        ("convergence_ratio", "Convergence (PSN>0.3)"),
    ]
    for key, name in metrics_order:
        vals = {ds: all_results[ds][key] for ds in ("base", "top", "random")}
        if key == "convergence_ratio":
            row = f"  {name:<25} {vals['base']*100:>9.1f}% {vals['top']*100:>9.1f}% {vals['random']*100:>9.1f}%"
        elif key == "avg_steps":
            row = f"  {name:<25} {vals['base']:>10.2f} {vals['top']:>10.2f} {vals['random']:>10.2f}"
        else:
            row = f"  {name:<25} {vals['base']:>10.4f} {vals['top']:>10.4f} {vals['random']:>10.4f}"
        print(row)
    print(f"{'='*70}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
