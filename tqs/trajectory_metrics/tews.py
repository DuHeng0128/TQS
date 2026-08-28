"""
TEWS · Trajectory Early Warning System
Online early-stop simulation using M1/M2/M3 signals.
"""

import warnings
from typing import Optional

from .parsers.base import ParsedTrajectory
from .metrics.m1_rbs import RBSMetric
from .metrics.m2_oec import OECMetric
from .metrics.m3_fld import FLDMetric


def simulate_early_stopping(
    trajectory: ParsedTrajectory,
    cutoff_pct: float = 0.3,
    strategy: str = "any",
    rbs_threshold: float = 0.85,
    oec_collapse_threshold: float = 0.3,
    fld_accel_required: bool = True,
) -> dict:
    """
    Simulate early stopping at cutoff_pct of the trajectory.

    Args:
        trajectory:  full ParsedTrajectory
        cutoff_pct:  fraction [0,1] at which to truncate and check signals
        strategy:    "any" = stop if any signal fires; "all" = stop only if all fire
        rbs_threshold:           M1 threshold for triggering
        oec_collapse_threshold:  M2 collapse_threshold param
        fld_accel_required:      if True, M3 only triggers when accelerating_drift=True

    Returns:
        dict with should_stop, triggered_by, tokens_saved_pct, prediction_correct
    """
    n = trajectory.total_steps
    cutoff_step = max(1, int(n * cutoff_pct))

    # Truncate trajectory at cutoff
    from dataclasses import replace
    truncated = ParsedTrajectory(
        instance_id=trajectory.instance_id,
        resolved=trajectory.resolved,
        steps=trajectory.steps[:cutoff_step],
        total_steps=cutoff_step,
        source_format=trajectory.source_format,
    )

    triggered_by = []

    # M1: RBS
    try:
        m1 = RBSMetric(threshold=rbs_threshold)
        r1 = m1.compute(truncated)
        if r1 and isinstance(r1, dict) and r1.get("early_stop_triggered"):
            triggered_by.append("m1_rbs")
    except Exception as e:
        warnings.warn(f"TEWS M1 failed: {e}")

    # M2: OEC
    try:
        m2 = OECMetric(collapse_threshold=oec_collapse_threshold)
        r2 = m2.compute(truncated)
        if r2 and isinstance(r2, dict) and r2.get("collapse_detected"):
            triggered_by.append("m2_oec")
    except Exception as e:
        warnings.warn(f"TEWS M2 failed: {e}")

    # M3: FLD
    try:
        m3 = FLDMetric()
        r3 = m3.compute(truncated)
        if r3 and isinstance(r3, dict) and r3.get("status") != "insufficient_data":
            if fld_accel_required:
                if r3.get("accelerating_drift"):
                    triggered_by.append("m3_fld")
            else:
                if r3.get("fld_global", 0) > 0.7:
                    triggered_by.append("m3_fld")
    except Exception as e:
        warnings.warn(f"TEWS M3 failed: {e}")

    # Combine signals
    if strategy == "any":
        should_stop = len(triggered_by) > 0
    elif strategy == "all":
        should_stop = len(triggered_by) == 3
    else:
        should_stop = len(triggered_by) > 0

    tokens_saved_pct = (1.0 - cutoff_pct) if should_stop else 0.0

    # Correctness (if ground truth available)
    prediction_correct = None
    if trajectory.resolved is not None:
        # We predict failure if should_stop=True
        predicted_failure = should_stop
        actual_failure = not trajectory.resolved
        prediction_correct = (predicted_failure == actual_failure)

    return {
        "should_stop": should_stop,
        "triggered_by": triggered_by,
        "cutoff_pct": cutoff_pct,
        "cutoff_step": cutoff_step,
        "total_steps": n,
        "tokens_saved_pct": float(tokens_saved_pct),
        "prediction_correct": prediction_correct,
        "strategy": strategy,
    }


def evaluate_tews_over_cutoffs(
    trajectory: ParsedTrajectory,
    cutoffs: list = None,
    strategy: str = "any",
) -> list:
    """Run TEWS simulation at multiple cutoff points."""
    if cutoffs is None:
        cutoffs = [0.1, 0.2, 0.3, 0.4, 0.5]
    return [
        simulate_early_stopping(trajectory, cutoff_pct=c, strategy=strategy)
        for c in cutoffs
    ]
