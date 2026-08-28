"""
M10 · Solution Commitment Point (SCP)
Identifies when the agent first makes a substantive code modification.
"""

import warnings
from typing import Optional

from .base import BaseMetric, registry
from ..parsers.base import ParsedTrajectory

_FAILURE_SIGNALS = {
    "error", "no such file", "permission denied", "not found",
    "failed", "invalid", "traceback", "exception",
}


def _is_substantive_write(step) -> bool:
    if step.action_type != "write_file":
        return False
    obs_lower = step.observation.lower()
    return not any(sig in obs_lower for sig in _FAILURE_SIGNALS)


class SCPMetric(BaseMetric):
    name = "m10_scp"

    def _compute(self, trajectory: ParsedTrajectory) -> dict:
        steps = trajectory.steps
        n = trajectory.total_steps

        write_steps = [s for s in steps if s.action_type == "write_file"]
        successful_write_steps = [s for s in write_steps if _is_substantive_write(s)]

        total_writes = len(write_steps)
        successful_writes = len(successful_write_steps)
        write_failure_rate = (
            1.0 - successful_writes / total_writes if total_writes > 0 else 0.0
        )

        scp = None
        never_committed = True
        last_write_pct = None
        write_spread = None

        if successful_write_steps:
            never_committed = False
            scp = successful_write_steps[0].step_id / n
            last_write_pct = successful_write_steps[-1].step_id / n
            write_spread = last_write_pct - scp

        return {
            "scp": float(scp) if scp is not None else None,
            "never_committed": never_committed,
            "total_writes": total_writes,
            "successful_writes": successful_writes,
            "write_failure_rate": float(write_failure_rate),
            "last_write_pct": float(last_write_pct) if last_write_pct is not None else None,
            "write_spread": float(write_spread) if write_spread is not None else None,
        }


registry.register(SCPMetric())
