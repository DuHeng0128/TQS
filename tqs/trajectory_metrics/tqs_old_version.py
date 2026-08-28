"""
TQS · Trajectory Quality Score
Weighted composite of selected metrics into a single [0, 1] quality score.

Design is informed by V2 CoderForge-Preview validity experiment (413K trajectories):
  - Components and weights chosen to maximise discriminative validity (V2a concordance)
    while avoiding step-length confound (|ρ| < 0.5)
  - BUS (concordance=0.202) and TGCR (concordance=0.479, not sig) removed
  - DCS jaccard removed (ρ=-0.729, pure step-length proxy); replaced by IAC
  - OEC added as top component (concordance=0.640, ρ=0.103)
  - PSN uses psn_late variant (τ=0.129 vs global τ=0.066)
  - HCR promoted from bonus to weighted component

Default component set and weights:
  oec   0.25   M2 Observation Entropy Collapse (min_relative_entropy, inverted)
  iac   0.20   M4 Intent-Action Consistency     (action_type_consistency)
  psn   0.15   M6 Progressive Scope Narrowing   (psn_late)
  tte   0.15   M9 Tool Transition Entropy        (tte_score)
  scp   0.15   M10 Solution Commitment Point     (scp)
  hcr   0.05   M7 Hypothesis-Commit Ratio        (ideal_pattern bonus)
  ped   0.05   M5 Post-Error Diversity           (post_error_diversity)
"""

import math
import warnings
from typing import Optional


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS = {
    "oec": 0.25,
    "iac": 0.20,
    "psn": 0.15,
    "tte": 0.15,
    "scp": 0.15,
    "hcr": 0.05,
    "ped": 0.05,
}

# All recognised component names — used to validate enable_components.
ALL_COMPONENTS = set(DEFAULT_WEIGHTS.keys())

# OEC: min_relative_entropy upper bound for normalisation.
# At baseline, entropy equals itself → rel=1.0; practically never exceeds 1.
# We clamp to [0,1] so values above 1 (expansion) are treated as 1.
_OEC_MAX_REL = 1.0

# TTE: theoretical max for 7 distinct action-types = log2(7*7)
_MAX_TTE = math.log2(7 * 7)

# SCP sweet spot: commit between 20 % and 50 % through the trajectory
_SCP_SWEET_LOW = 0.2
_SCP_SWEET_HIGH = 0.5
_SCP_SIGMA = 0.2

# HCR: ideal_pattern=True earns a normalised score of 1.0 for its weight
_HCR_IDEAL_SCORE = 1.0
_HCR_DEFAULT_SCORE = 0.0


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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_tqs(
    metrics: dict,
    weights: Optional[dict] = None,
    enable_components: Optional[set] = None,
) -> dict:
    """
    Compute TQS from the output dict returned by MetricRegistry.

    Args:
        metrics:
            Mapping of metric name → result dict, e.g.
            ``{"m2_oec": {...}, "m4_dcs": {...}, ...}``
        weights:
            Override the default per-component weights.  Missing keys fall back
            to DEFAULT_WEIGHTS; extra keys are silently ignored.  Weights are
            re-normalised over the *enabled* components, so you do not need to
            ensure they sum to 1.0 when disabling components.
        enable_components:
            Subset of component names to include.  Defaults to ALL_COMPONENTS.
            Any name not in ALL_COMPONENTS raises ValueError.
            Example: ``{"oec", "iac", "tte"}`` disables psn / scp / hcr / ped.

    Returns:
        dict with keys:
            tqs_score       float in [0, 1]
            components      {name: normalised_score} for each computed component
            weights_used    {name: effective_weight} after renormalisation
            enabled         set of requested component names
            coverage        fraction of enabled components that had valid data
    """
    # --- Resolve enabled set ---
    if enable_components is None:
        enabled = set(ALL_COMPONENTS)
    else:
        unknown = set(enable_components) - ALL_COMPONENTS
        if unknown:
            raise ValueError(
                f"Unknown component names: {unknown}. "
                f"Valid names: {ALL_COMPONENTS}"
            )
        enabled = set(enable_components)

    # --- Resolve weights (merge defaults with overrides) ---
    w = {**DEFAULT_WEIGHTS, **(weights or {})}

    # --- Extract and normalise each component ---
    components: dict[str, float] = {}

    # OEC: min_relative_entropy — lower means more focused (better)
    # Inverted: score = 1 - clamp(min_rel / _OEC_MAX_REL, 0, 1)
    if "oec" in enabled:
        oec_raw = _safe_get(metrics, "m2_oec", "min_relative_entropy")
        if oec_raw is not None:
            components["oec"] = float(max(0.0, min(1.0, 1.0 - oec_raw / _OEC_MAX_REL)))

    # IAC: action_type_consistency — already [0, 1], higher is better
    if "iac" in enabled:
        iac_raw = _safe_get(metrics, "m4_dcs", "action_type_consistency")
        if iac_raw is not None:
            components["iac"] = float(max(0.0, min(1.0, iac_raw)))

    # PSN: psn_late in [-1, 1], higher (converging) is better → map to [0, 1]
    if "psn" in enabled:
        psn_raw = _safe_get(metrics, "m6_psn", "psn_late")
        if psn_raw is not None:
            components["psn"] = float(max(0.0, min(1.0, (psn_raw + 1.0) / 2.0)))

    # TTE: tte_score ≥ 0, normalised by theoretical max; moderate diversity desired
    if "tte" in enabled:
        tte_raw = _safe_get(metrics, "m9_tte", "tte_score")
        if tte_raw is not None:
            components["tte"] = float(max(0.0, min(1.0, tte_raw / _MAX_TTE)))

    # SCP: commit-point fraction — penalise deviation from sweet spot [0.2, 0.5]
    # Only included when the m10_scp metric was actually computed.
    if "scp" in enabled:
        scp_result = metrics.get("m10_scp")
        if scp_result and isinstance(scp_result, dict) and scp_result.get("status") != "insufficient_data":
            never_committed = scp_result.get("never_committed", True)
            scp_raw = scp_result.get("scp")
            if never_committed:
                components["scp"] = 0.0
            elif scp_raw is not None:
                components["scp"] = float(
                    _gaussian_penalty(scp_raw, _SCP_SWEET_LOW, _SCP_SWEET_HIGH, _SCP_SIGMA)
                )

    # HCR: ideal_pattern (explore early, commit late) → binary [0, 1]
    if "hcr" in enabled:
        ideal = _safe_get(metrics, "m7_hcr", "ideal_pattern")
        if ideal is not None:
            components["hcr"] = _HCR_IDEAL_SCORE if ideal else _HCR_DEFAULT_SCORE

    # PED: post_error_diversity — already [0, 1], higher means better adaptation
    if "ped" in enabled:
        ped_raw = _safe_get(metrics, "m5_bus", "post_error_diversity")
        if ped_raw is not None:
            components["ped"] = float(max(0.0, min(1.0, ped_raw)))

    # --- Weighted average over available components ---
    total_weight = 0.0
    weighted_sum = 0.0
    weights_used: dict[str, float] = {}

    for name, val in components.items():
        weight = w.get(name, 0.0)
        weighted_sum += weight * val
        total_weight += weight
        weights_used[name] = weight

    if total_weight == 0.0:
        tqs_score = 0.0
    else:
        tqs_score = float(max(0.0, min(1.0, weighted_sum / total_weight)))

    return {
        "tqs_score": tqs_score,
        "components": components,
        "weights_used": weights_used,
        "enabled": enabled,
        "coverage": len(components) / len(enabled) if enabled else 0.0,
    }
