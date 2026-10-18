"""
evaluate.py · Batch evaluation entry point for trajectory metrics.

Usage:
    python -m trajectory_metrics.evaluate \\
        --input nebius/SWE-agent-trajectories/ \\
        --format swe_agent \\
        --output results/metrics_summary.csv

    python -m trajectory_metrics.evaluate \\
        --input nebius/SWE-rebench-openhands-trajectories/trajectories.parquet \\
        --format openhands \\
        --output results/openhands_metrics.csv
"""

import os
import json
import logging
import warnings
import argparse
import multiprocessing
from pathlib import Path
from typing import Optional

import pandas as pd

from .parsers.base import ParsedTrajectory
from .parsers.swe_agent import SWEAgentParser
from .parsers.openhands import OpenHandsParser
from .parsers.coderforge import CoderForgeParser
from .metrics import registry
# Import all metrics to trigger registry.register() calls
from .metrics import m1_rbs, m2_oec, m3_fld, m4_dcs, m5_bus  # noqa: F401
from .metrics import m6_psn, m7_hcr, m8_tgcr, m9_tte, m10_scp  # noqa: F401
from ..tqs import compute_tqs

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def _parquet_columns(path: str) -> list:
    """Read parquet column names without loading data."""
    try:
        import pyarrow.parquet as pq
        return pq.read_schema(path).names
    except Exception:
        try:
            return pd.read_parquet(path, nrows=0).columns.tolist()
        except Exception:
            return []


def detect_format(input_path: str) -> str:
    """Auto-detect trajectory format from file/directory."""
    path = Path(input_path)

    if path.is_file() and path.suffix == ".parquet":
        cols = _parquet_columns(input_path)
        if "messages" in cols and "reward" in cols:
            return "coderforge"
        if "resolved" in cols:
            return "openhands"
        if "target" in cols or "exit_status" in cols:
            return "swe_agent"
        return "openhands"

    if path.is_dir():
        parquets = list(path.glob("**/*.parquet"))
        if parquets:
            cols = _parquet_columns(str(parquets[0]))
            if "messages" in cols and "reward" in cols:
                return "coderforge"
            if "resolved" in cols:
                return "openhands"
            return "swe_agent"
        return "swe_agent"

    return "swe_agent"


# ---------------------------------------------------------------------------
# Loading trajectories
# ---------------------------------------------------------------------------

def load_trajectories(input_path: str, fmt: str) -> list:
    """Load trajectories from path using the appropriate parser."""
    path = Path(input_path)

    if fmt == "swe_agent":
        parser = SWEAgentParser()
    elif fmt == "openhands":
        parser = OpenHandsParser()
    elif fmt == "coderforge":
        parser = CoderForgeParser()
    else:
        raise ValueError(f"Unknown format: {fmt}")

    trajs = []
    if path.is_file() and path.suffix == ".parquet":
        df = pd.read_parquet(input_path)
        for _, row in df.iterrows():
            t = parser.parse_row(row)
            if t is not None:
                trajs.append(t)

    elif path.is_dir():
        parquet_files = sorted(path.glob("**/*.parquet"))
        if parquet_files:
            for pf in parquet_files:
                df = pd.read_parquet(str(pf))
                for _, row in df.iterrows():
                    t = parser.parse_row(row)
                    if t is not None:
                        trajs.append(t)
        else:
            # Try JSON/traj files
            traj_files = sorted(path.glob("**/*.traj")) + sorted(path.glob("**/*.json"))
            import json as _json
            for tf in traj_files:
                try:
                    with open(tf) as f:
                        data = _json.load(f)
                    # Wrap as a pseudo-row for the parser
                    row = {
                        "instance_id": tf.stem,
                        "target": data.get("info", {}).get("exit_status") == "submitted",
                        "trajectory": data.get("trajectory", []),
                        "exit_status": data.get("info", {}).get("exit_status", ""),
                        "generated_patch": data.get("info", {}).get("submission", ""),
                    }
                    t = parser.parse_row(type("Row", (), row)())
                    if t is not None:
                        trajs.append(t)
                except Exception as e:
                    warnings.warn(f"Failed to load {tf}: {e}")
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")

    logger.info(f"Loaded {len(trajs)} trajectories from {input_path}")
    return trajs


# ---------------------------------------------------------------------------
# Per-trajectory computation
# ---------------------------------------------------------------------------

def _compute_one(traj: ParsedTrajectory) -> dict:
    """Compute all metrics for a single trajectory. Safe for multiprocessing."""
    result = {
        "instance_id": traj.instance_id,
        "resolved": traj.resolved,
        "total_steps": traj.total_steps,
        "source_format": traj.source_format,
        "total_tokens": sum(s.token_count for s in traj.steps),
    }

    all_metrics = registry.compute_all(traj)

    # Flatten metrics to scalar summary fields
    m1 = all_metrics.get("m1_rbs") or {}
    result["m1_rbs_global"] = m1.get("rbs_global")
    result["m1_rbs_mean"] = m1.get("rbs_mean")
    result["m1_rbs_loop_fraction"] = m1.get("rbs_loop_fraction")
    result["m1_rbs_max_loop_frac"] = m1.get("rbs_max_loop_frac")
    result["m1_rbs_n_loops"] = m1.get("rbs_n_loops")
    result["m1_early_stop"] = m1.get("early_stop_triggered")
    result["m1_early_stop_pct"] = m1.get("early_stop_step_pct")

    m2 = all_metrics.get("m2_oec") or {}
    result["m2_collapse"] = m2.get("collapse_detected")
    result["m2_min_rel_entropy"] = m2.get("min_relative_entropy")
    result["m2_collapse_pct"] = m2.get("collapse_step_pct")
    result["m2_error_repeat_rate"] = m2.get("error_repeat_rate")
    result["m2_error_rate"] = m2.get("error_rate")

    m3 = all_metrics.get("m3_fld") or {}
    result["m3_fld_global"] = m3.get("fld_global")
    result["m3_fld_early"] = m3.get("fld_early")
    result["m3_fld_late"] = m3.get("fld_late")
    result["m3_fld_convergence"] = m3.get("fld_convergence")
    result["m3_file_breadth_ratio"] = m3.get("file_breadth_ratio")
    result["m3_accel_drift"] = m3.get("accelerating_drift")
    result["m3_revisit_rate"] = m3.get("files_revisit_rate")

    m4 = all_metrics.get("m4_dcs") or {}
    result["m4_dcs_global"] = m4.get("dcs_global")
    result["m4_dcs_combined"] = m4.get("dcs_combined")
    result["m4_dcs_file_align"] = m4.get("dcs_file_align")
    result["m4_action_type_consistency"] = m4.get("action_type_consistency")

    m5 = all_metrics.get("m5_bus") or {}
    result["m5_bus_score"] = m5.get("bus_score")
    result["m5_total_backtracks"] = m5.get("total_backtracks")
    result["m5_post_error_diversity"] = m5.get("post_error_diversity")
    result["m5_post_error_type_change_rate"] = m5.get("post_error_type_change_rate")

    m6 = all_metrics.get("m6_psn") or {}
    result["m6_psn_score"] = m6.get("psn_score")
    result["m6_psn_early"] = m6.get("psn_early")
    result["m6_psn_late"] = m6.get("psn_late")
    result["m6_pattern"] = m6.get("pattern")

    m7 = all_metrics.get("m7_hcr") or {}
    result["m7_hcr_global"] = m7.get("hcr_global")
    result["m7_ideal_pattern"] = m7.get("ideal_pattern")
    result["m7_first_write_pct"] = m7.get("first_write_pct")

    m8 = all_metrics.get("m8_tgcr") or {}
    result["m8_tgcr"] = m8.get("tgcr")
    result["m8_back_edges"] = m8.get("back_edges")
    result["m8_oscillation_rate"] = m8.get("oscillation_rate")
    result["m8_repeated_transition_rate"] = m8.get("repeated_transition_rate")

    m9 = all_metrics.get("m9_tte") or {}
    result["m9_tte_score"] = m9.get("tte_score")
    result["m9_dominant_tool"] = m9.get("dominant_tool")
    result["m9_tool_diversity"] = m9.get("tool_diversity")

    m10 = all_metrics.get("m10_scp") or {}
    result["m10_scp"] = m10.get("scp")
    result["m10_never_committed"] = m10.get("never_committed")
    result["m10_write_failure_rate"] = m10.get("write_failure_rate")

    # TQS composite
    tqs = compute_tqs(all_metrics)
    result["tqs_score"] = tqs.get("tqs_score")
    result["tqs_coverage"] = tqs["coverage"]
    for component in ("oec", "iac", "dpi", "ped", "psn", "tte", "scp"):
        result[f"tqs_{component}"] = tqs["components"].get(component)

    # Store full metric dicts as JSON strings in a separate dict (for series data)
    result["_full_metrics"] = all_metrics

    return result


def _compute_one_safe(traj: ParsedTrajectory) -> dict:
    try:
        return _compute_one(traj)
    except Exception as e:
        warnings.warn(f"Failed to compute metrics for {traj.instance_id}: {e}")
        return {
            "instance_id": traj.instance_id,
            "resolved": traj.resolved,
            "total_steps": traj.total_steps,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_dataset(
    input_path: str,
    fmt: str = "auto",
    output_path: str = "results/metrics_summary.csv",
    metrics: list = None,
    n_workers: int = 1,
    max_trajectories: int = None,
) -> pd.DataFrame:
    """
    Batch compute all metrics for a dataset.

    Returns a DataFrame with one row per trajectory.
    Also writes:
      - <output_path>                  : summary CSV
      - <output_dir>/metrics_full.csv  : full scalar fields
      - <output_dir>/series_data/      : per-trajectory JSON with time series
    """
    if fmt == "auto":
        fmt = detect_format(input_path)
        logger.info(f"Auto-detected format: {fmt}")

    trajs = load_trajectories(input_path, fmt)
    if max_trajectories:
        trajs = trajs[:max_trajectories]

    logger.info(f"Computing metrics for {len(trajs)} trajectories (workers={n_workers})")

    if n_workers > 1:
        with multiprocessing.Pool(n_workers) as pool:
            results = pool.map(_compute_one_safe, trajs)
    else:
        results = [_compute_one_safe(t) for t in trajs]

    # Save series data
    output_dir = Path(output_path).parent
    series_dir = output_dir / "series_data"
    series_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for r in results:
        full_metrics = r.pop("_full_metrics", {})
        summary_rows.append(r)
        # Save series JSON
        series_path = series_dir / f"{r['instance_id']}.json"
        try:
            with open(series_path, "w") as f:
                json.dump(full_metrics, f, default=str)
        except Exception:
            pass

    df = pd.DataFrame(summary_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved summary to {output_path}")

    # Also save full CSV
    full_path = output_dir / "metrics_full.csv"
    df.to_csv(str(full_path), index=False)

    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Batch evaluate trajectory metrics")
    parser.add_argument("--input", required=True, help="Input path (directory or parquet file)")
    parser.add_argument("--format", default="auto", choices=["auto", "swe_agent", "openhands", "coderforge"])
    parser.add_argument("--output", default="results/metrics_summary.csv")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max", type=int, default=None, help="Max trajectories to process")
    args = parser.parse_args()

    df = evaluate_dataset(
        input_path=args.input,
        fmt=args.format,
        output_path=args.output,
        n_workers=args.workers,
        max_trajectories=args.max,
    )
    print(f"\nProcessed {len(df)} trajectories")
    print(df[["instance_id", "tqs_score", "total_steps"]].head(10).to_string())


if __name__ == "__main__":
    main()
