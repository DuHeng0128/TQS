"""
M3 · Fault Localization Drift (FLD)
Detects whether the agent's file focus is drifting away from the relevant area.
"""

import warnings
from typing import Optional

import numpy as np

from .base import BaseMetric, registry
from ..parsers.base import ParsedTrajectory


def jaccard_dist(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return 1.0 - len(a & b) / len(union)


class FLDMetric(BaseMetric):
    name = "m3_fld"

    def __init__(self, min_files_per_step: int = 1):
        self.min_files_per_step = min_files_per_step

    def _compute(self, trajectory: ParsedTrajectory) -> dict:
        # Only consider steps that interact with files
        file_steps = [
            set(s.target_files)
            for s in trajectory.steps
            if s.action_type in ("read_file", "write_file")
            and len(s.target_files) >= self.min_files_per_step
        ]

        if len(file_steps) < 3:
            return {"status": "insufficient_data", "reason": "fewer than 3 file-interaction steps"}

        drift_series = [
            jaccard_dist(file_steps[i], file_steps[i + 1])
            for i in range(len(file_steps) - 1)
        ]

        drift_velocity = [
            drift_series[i + 1] - drift_series[i]
            for i in range(len(drift_series) - 1)
        ]

        drift_acceleration = [
            drift_velocity[i + 1] - drift_velocity[i]
            for i in range(len(drift_velocity) - 1)
        ]

        fld_global = float(np.mean(drift_series))

        if len(drift_acceleration) >= 3:
            accelerating_drift = bool(float(np.mean(drift_acceleration[-3:])) > 0)
        else:
            accelerating_drift = False

        # ------------------------------------------------------------------
        # Phase-aware FLD: split drift_series into early (first 1/3) and
        # late (last 1/3).  A high-quality trajectory should have:
        #   early_fld  high  — agent explores broadly at the start
        #   late_fld   low   — agent converges on specific files at the end
        # fld_convergence = early_fld - late_fld  (positive = converged, good)
        # ------------------------------------------------------------------
        n_drift = len(drift_series)
        third = max(1, n_drift // 3)
        early_fld = float(np.mean(drift_series[:third]))
        late_fld = float(np.mean(drift_series[n_drift - third:]))
        fld_convergence = early_fld - late_fld

        # File revisit stats
        all_files = [f for step in file_steps for f in step]
        unique_files = len(set(all_files))
        total_visits = len(all_files)
        revisit_rate = 1.0 - (unique_files / total_visits) if total_visits > 0 else 0.0

        # --- File breadth ratio: scope convergence without target files ---
        # Count how many NEW files appear in second half vs first half
        mid = len(file_steps) // 2
        first_half_files = set()
        for fs in file_steps[:mid]:
            first_half_files |= fs
        second_half_new = set()
        for fs in file_steps[mid:]:
            second_half_new |= (fs - first_half_files)
        file_breadth_ratio = (
            len(second_half_new) / len(first_half_files)
            if first_half_files else 0.0
        )

        return {
            "fld_global": fld_global,
            "fld_early": early_fld,
            "fld_late": late_fld,
            "fld_convergence": fld_convergence,
            "file_breadth_ratio": float(file_breadth_ratio),
            "drift_series": drift_series,
            "drift_velocity": drift_velocity,
            "drift_acceleration": drift_acceleration,
            "accelerating_drift": accelerating_drift,
            "unique_files_visited": unique_files,
            "files_revisit_rate": float(revisit_rate),
        }


registry.register(FLDMetric())
