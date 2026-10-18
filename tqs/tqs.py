"""TQS — revised composite quality score for code-agent trajectories.

Changes vs V1 (motivated by V1 diagnostic on 38 K reward=1 / ≤32 K eligible pool):

  • Remove HCR as a weighted component
      V1 saw 99.9% of top1024 with ideal_pattern=True vs 15.1% in bot1024.
      The component is too binary and would mode-collapse the SFT data.
      ideal_pattern is preserved as a diagnostic. An optional historical
      diversity quota is available in the sampler, but disabled by default.

  • Add DPI (Dirty Pattern Indicator)
      Penalises trajectories with hard quality red flags from M1/M2/M10
      (early_stop, never_committed, repetitive loops, repeated errors).
      Replaces the role HCR was failing to play.

  • Re-weight remaining components
      OEC↑, IAC↑, PED↑   (length-robust, clean AUC≥0.92 even step-controlled)
      PSN↓, TTE↓, SCP↓   (PSN ↔ step length confound; SCP saturated; TTE weak)

  • Same fail-soft semantics as V1
      Component returns None → component skipped, weights renormalised over
      the *available* set.

Component set / weights (sum=1.00):

      OEC  0.30   M2 min_relative_entropy   (focus, inverted)
      IAC  0.25   M4 action_type_consistency (intent ↔ action)
      DPI  0.10   M1+M2+M10 dirty patterns   (new)
      PED  0.10   M5 post_error_diversity
      PSN  0.10   M6 psn_late                (down-weighted from 0.15)
      TTE  0.10   M9 tte_score               (down-weighted from 0.15)
      SCP  0.05   M10 scp                    (down-weighted from 0.15)

Public API mirrors V1 so the rest of the pipeline (score_coderforge.py-style
batch driver, plotting, etc.) can drop V2 in by importing compute_tqs.
"""

from __future__ import annotations

import math
from typing import Optional


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS = {
    "oec": 0.30,
    "iac": 0.25,
    "dpi": 0.10,
    "ped": 0.10,
    "psn": 0.10,
    "tte": 0.10,
    "scp": 0.05,
}

ALL_COMPONENTS = set(DEFAULT_WEIGHTS.keys())

# OEC: rel-entropy ≤ 1.0 by construction (we still clamp).
_OEC_MAX_REL = 1.0

# TTE: theoretical max for 7 action-types = log2(7*7).
_MAX_TTE = math.log2(7 * 7)

# SCP sweet spot.
_SCP_SWEET_LOW, _SCP_SWEET_HIGH, _SCP_SIGMA = 0.2, 0.5, 0.2

# DPI penalty knobs — calibrated against V1 dirty-flag rates so that bot1024-
# style trajectories land near 0 and top-quality trajectories near 1.
_DPI_EARLY_STOP_PENALTY = 0.40        # m1_early_stop_triggered = True
_DPI_NEVER_COMMIT_PENALTY = 0.40      # m10_never_committed = True
_DPI_LOOP_FRAC_SCALE = 3.0            # mapped to penalty via min(0.30, frac*scale)
_DPI_LOOP_FRAC_CAP = 0.30
_DPI_ERR_REPEAT_SCALE = 0.60          # m2_error_repeat_rate * scale
_DPI_ERR_REPEAT_CAP = 0.30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gaussian_penalty(x: float, low: float, high: float, sigma: float) -> float:
    """1.0 inside [low, high]; Gaussian decay outside."""
    if low <= x <= high:
        return 1.0
    dist = min(abs(x - low), abs(x - high))
    return math.exp(-(dist ** 2) / (2 * sigma ** 2))


def _safe_get(metrics: dict, metric_name: str, key: str, default=None):
    m = metrics.get(metric_name)
    if not m or not isinstance(m, dict):
        return default
    if m.get("status") == "insufficient_data":
        return default
    return m.get(key, default)


def _dpi_score(metrics: dict) -> Optional[float]:
    """Compute DPI from M1, M2, M10. Returns None if no input is available."""
    m1 = metrics.get("m1_rbs") or {}
    m2 = metrics.get("m2_oec") or {}
    m10 = metrics.get("m10_scp") or {}

    early_stop = m1.get("early_stop_triggered")
    never_commit = m10.get("never_committed")
    loop_frac = m1.get("rbs_loop_fraction")
    err_repeat = m2.get("error_repeat_rate")

    # Need at least one signal to even attempt a DPI score.
    if all(x is None for x in (early_stop, never_commit, loop_frac, err_repeat)):
        return None

    penalty = 0.0
    if early_stop is True:
        penalty += _DPI_EARLY_STOP_PENALTY
    if never_commit is True:
        penalty += _DPI_NEVER_COMMIT_PENALTY
    if loop_frac is not None:
        penalty += min(_DPI_LOOP_FRAC_CAP, max(0.0, float(loop_frac)) * _DPI_LOOP_FRAC_SCALE)
    if err_repeat is not None:
        penalty += min(_DPI_ERR_REPEAT_CAP, max(0.0, float(err_repeat)) * _DPI_ERR_REPEAT_SCALE)

    return float(max(0.0, min(1.0, 1.0 - penalty)))


# Tabular convenience: same DPI on a flat-dict of pre-extracted columns.
def dpi_from_flat(row) -> Optional[float]:
    """Compute DPI from a pandas row / dict that has the V1 flat-score columns
    (m1_early_stop, m1_rbs_loop_fraction, m2_error_repeat_rate, m10_never_committed).
    Used by rerank.py to recompute V2 without rerunning M1-M10."""
    early_stop = row.get("m1_early_stop") if hasattr(row, "get") else getattr(row, "m1_early_stop", None)
    never_commit = row.get("m10_never_committed") if hasattr(row, "get") else getattr(row, "m10_never_committed", None)
    loop_frac = row.get("m1_rbs_loop_fraction") if hasattr(row, "get") else getattr(row, "m1_rbs_loop_fraction", None)
    err_repeat = row.get("m2_error_repeat_rate") if hasattr(row, "get") else getattr(row, "m2_error_repeat_rate", None)

    fake = {
        "m1_rbs": {"early_stop_triggered": early_stop, "rbs_loop_fraction": loop_frac},
        "m2_oec": {"error_repeat_rate": err_repeat},
        "m10_scp": {"never_committed": never_commit},
    }
    return _dpi_score(fake)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_tqs(
    metrics: dict,
    weights: Optional[dict] = None,
    enable_components: Optional[set] = None,
) -> dict:
    """Compute TQS V2 from the dict returned by MetricRegistry.compute_all.

    Args:
        metrics: {metric_name: result_dict}, same shape as V1.
        weights: optional override; renormalised over enabled & available set.
        enable_components: subset of ALL_COMPONENTS; defaults to all.

    Returns dict with:
        tqs_score, components, weights_used, enabled, coverage.
    """
    if enable_components is None:
        enabled = set(ALL_COMPONENTS)
    else:
        unknown = set(enable_components) - ALL_COMPONENTS
        if unknown:
            raise ValueError(f"Unknown component names: {unknown}")
        enabled = set(enable_components)

    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    components: dict[str, float] = {}

    if "oec" in enabled:
        v = _safe_get(metrics, "m2_oec", "min_relative_entropy")
        if v is not None:
            components["oec"] = float(max(0.0, min(1.0, 1.0 - v / _OEC_MAX_REL)))

    if "iac" in enabled:
        v = _safe_get(metrics, "m4_dcs", "action_type_consistency")
        if v is not None:
            components["iac"] = float(max(0.0, min(1.0, v)))

    if "psn" in enabled:
        v = _safe_get(metrics, "m6_psn", "psn_late")
        if v is not None:
            components["psn"] = float(max(0.0, min(1.0, (v + 1.0) / 2.0)))

    if "tte" in enabled:
        v = _safe_get(metrics, "m9_tte", "tte_score")
        if v is not None:
            components["tte"] = float(max(0.0, min(1.0, v / _MAX_TTE)))

    if "scp" in enabled:
        scp = metrics.get("m10_scp")
        if scp and isinstance(scp, dict) and scp.get("status") != "insufficient_data":
            never = scp.get("never_committed", True)
            x = scp.get("scp")
            if never:
                components["scp"] = 0.0
            elif x is not None:
                components["scp"] = float(_gaussian_penalty(x, _SCP_SWEET_LOW, _SCP_SWEET_HIGH, _SCP_SIGMA))

    if "ped" in enabled:
        v = _safe_get(metrics, "m5_bus", "post_error_diversity")
        if v is not None:
            components["ped"] = float(max(0.0, min(1.0, v)))

    if "dpi" in enabled:
        v = _dpi_score(metrics)
        if v is not None:
            components["dpi"] = v

    total_w = 0.0
    weighted = 0.0
    weights_used: dict[str, float] = {}
    for name, val in components.items():
        wt = w.get(name, 0.0)
        weighted += wt * val
        total_w += wt
        weights_used[name] = wt

    score = float(max(0.0, min(1.0, weighted / total_w))) if total_w > 0 else 0.0

    return {
        "tqs_score": score,
        "components": components,
        "weights_used": weights_used,
        "enabled": enabled,
        "coverage": len(components) / len(enabled) if enabled else 0.0,
    }
