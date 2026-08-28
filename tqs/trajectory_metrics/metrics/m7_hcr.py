"""
M7 · Hypothesis-Commit Ratio (HCR)
Characterizes the exploration vs. execution dynamic over the trajectory.
"""

import warnings
from typing import Optional

import numpy as np

from .base import BaseMetric, registry
from ..parsers.base import ParsedTrajectory

_REASONING_TYPES = {"think", "read_file", "search"}
_EXECUTION_TYPES = {"write_file", "execute_bash"}


def _hcr(steps: list) -> float:
    r = sum(1 for s in steps if s.action_type in _REASONING_TYPES)
    e = sum(1 for s in steps if s.action_type in _EXECUTION_TYPES)
    if e == 0:
        return float("inf")
    return r / e


class HCRMetric(BaseMetric):
    name = "m7_hcr"

    def _compute(self, trajectory: ParsedTrajectory) -> dict:
        steps = trajectory.steps
        n = len(steps)

        reasoning_steps = [s for s in steps if s.action_type in _REASONING_TYPES]
        execution_steps = [s for s in steps if s.action_type in _EXECUTION_TYPES]

        hcr_global = _hcr(steps)
        never_committed = hcr_global == float("inf")

        # Phase HCR
        t1, t2 = n // 3, 2 * n // 3
        early_hcr = _hcr(steps[:t1]) if t1 > 0 else 0.0
        mid_hcr = _hcr(steps[t1:t2]) if t2 > t1 else 0.0
        late_hcr = _hcr(steps[t2:]) if n > t2 else 0.0

        # Ideal: decreasing HCR (explore first, commit later)
        def finite(x):
            return x if x != float("inf") else 999.0

        ideal_pattern = finite(early_hcr) >= finite(mid_hcr) >= finite(late_hcr)

        # First write step
        write_steps = [s for s in steps if s.action_type == "write_file"]
        if write_steps:
            first_write_pct = write_steps[0].step_id / n
        else:
            first_write_pct = 1.0

        return {
            "hcr_global": hcr_global if not never_committed else None,
            "hcr_early": early_hcr if early_hcr != float("inf") else None,
            "hcr_mid": mid_hcr if mid_hcr != float("inf") else None,
            "hcr_late": late_hcr if late_hcr != float("inf") else None,
            "ideal_pattern": ideal_pattern,
            "first_write_pct": float(first_write_pct),
            "never_committed": never_committed,
            "reasoning_steps_count": len(reasoning_steps),
            "execution_steps_count": len(execution_steps),
        }


registry.register(HCRMetric())
