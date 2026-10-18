"""
M6 · Progressive Scope Narrowing (PSN)
Measures whether the agent's file focus converges monotonically over time.
"""

import warnings
from typing import Optional

import numpy as np

from .base import BaseMetric, registry
from ..parsers.base import ParsedTrajectory


class PSNMetric(BaseMetric):
    name = "m6_psn"

    def __init__(self, window: int = 5):
        self.window = window

    def _compute(self, trajectory: ParsedTrajectory) -> dict:
        steps = trajectory.steps
        n = len(steps)

        # Cumulative unique files
        seen = set()
        cumulative_unique = []
        for step in steps:
            seen.update(step.target_files)
            cumulative_unique.append(len(seen))

        # Active scope: files in sliding window
        active_scope = []
        for t in range(n):
            window_files = set()
            for s in steps[max(0, t - self.window + 1): t + 1]:
                window_files.update(s.target_files)
            active_scope.append(len(window_files))

        def _kendall_score(scope_slice: list) -> tuple:
            """Return (psn_score, tau, p_value, pattern) for a scope slice."""
            if len(scope_slice) < 3:
                return 0.0, 0.0, 1.0, "oscillating"
            try:
                from scipy.stats import kendalltau
                indices = list(range(len(scope_slice)))
                tau_val, p_val = kendalltau(indices, scope_slice)
                # Constant file scope has no rank trend. scipy returns NaN;
                # use a neutral score instead of propagating it into TQS.
                if not np.isfinite(tau_val):
                    return 0.0, 0.0, 1.0, "oscillating"
                score = float(-tau_val)
                if score > 0.3:
                    pattern = "converging"
                elif score < -0.3:
                    pattern = "diverging"
                else:
                    pattern = "oscillating"
                return score, float(tau_val), float(p_val), pattern
            except Exception as e:
                warnings.warn(f"PSN scipy failed: {e}")
                return 0.0, 0.0, 1.0, "oscillating"

        # Global (full trajectory)
        psn_score, tau_val, p_value, pattern = _kendall_score(active_scope)

        # Phase-aware: early (first half) and late (second half)
        # psn_late is the primary quality signal: good agents converge in the
        # back half even if they expand scope early.
        mid = n // 2
        psn_early, _, _, _ = _kendall_score(active_scope[:mid])
        psn_late, _, _, _ = _kendall_score(active_scope[mid:])

        return {
            "psn_score": float(psn_score),
            "psn_early": float(psn_early),
            "psn_late": float(psn_late),
            "kendall_tau": float(tau_val),
            "p_value": float(p_value),
            "pattern": pattern,
            "active_scope_series": active_scope,
            "cumulative_unique_files": int(cumulative_unique[-1]) if cumulative_unique else 0,
            "peak_scope": int(max(active_scope)) if active_scope else 0,
            "final_scope": int(active_scope[-1]) if active_scope else 0,
        }


registry.register(PSNMetric())
