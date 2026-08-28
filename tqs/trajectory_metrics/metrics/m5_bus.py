"""
M5 · Backtrack Utility Score (BUS) + Post-Error Diversity
Distinguishes productive backtracks from unproductive ones.
Also measures whether the agent adapts after encountering errors.
"""

import re
import warnings
from typing import Optional

import numpy as np

from .base import BaseMetric, registry
from ..parsers.base import ParsedTrajectory


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


_ERROR_INDICATORS = re.compile(
    r'Traceback \(most recent call last\)'
    r'|(?:^|\n)\s*(?:Error|ERROR|Exception|FAILED|FAILURE)'
    r'|AssertionError|TypeError|ValueError|KeyError|AttributeError'
    r'|ImportError|NameError|IndexError|RuntimeError|FileNotFoundError'
    r'|SyntaxError|IndentationError|ModuleNotFoundError'
    r'|command not found|Permission denied|No such file or directory'
    r'|exit (?:code|status)\s*[1-9]|returned non-zero exit status',
    re.MULTILINE | re.IGNORECASE,
)


def _has_error(obs: str) -> bool:
    """Check if an observation contains error indicators."""
    if not obs:
        return False
    return bool(_ERROR_INDICATORS.search(obs))


class BUSMetric(BaseMetric):
    name = "m5_bus"

    def __init__(self, lookback_window: int = 3, lookahead_window: int = 5):
        self.lookback_window = lookback_window
        self.lookahead_window = lookahead_window

    def _compute(self, trajectory: ParsedTrajectory) -> dict:
        steps = trajectory.steps
        n = len(steps)

        file_steps = [s for s in steps if s.action_type in ("read_file", "write_file")]

        backtrack_steps = []
        effective_backtrack_steps = []

        for idx, step in enumerate(steps):
            if step.action_type not in ("read_file", "write_file"):
                continue
            if step.step_id < self.lookback_window:
                continue

            curr_files = set(step.target_files)
            if not curr_files:
                continue

            # Past files in lookback window (by step_id)
            past_steps = [s for s in steps if s.step_id >= step.step_id - self.lookback_window
                          and s.step_id < step.step_id
                          and s.action_type in ("read_file", "write_file")]
            past_files = set(f for s in past_steps for f in s.target_files)

            # Backtrack = current files completely disjoint from past
            if past_files and _jaccard(curr_files, past_files) == 0.0:
                backtrack_steps.append(step.step_id)

                # Lookahead effectiveness
                future_steps = [s for s in steps
                                if s.step_id > step.step_id
                                and s.step_id <= step.step_id + self.lookahead_window]

                post_files = set(f for s in future_steps for f in s.target_files)
                pre_files = set(f for s in past_steps for f in s.target_files) | curr_files

                scope_reduced = len(post_files) <= len(pre_files)
                has_write = any(s.action_type == "write_file" for s in future_steps)

                if scope_reduced or has_write:
                    effective_backtrack_steps.append(step.step_id)

        total_bt = len(backtrack_steps)
        effective_bt = len(effective_backtrack_steps)
        ineffective_bt = total_bt - effective_bt

        if total_bt == 0:
            bus_score = 1.0
        else:
            bus_score = effective_bt / total_bt

        return {
            "bus_score": float(bus_score),
            "total_backtracks": total_bt,
            "effective_backtracks": effective_bt,
            "ineffective_backtracks": ineffective_bt,
            "backtrack_steps": backtrack_steps,
            "effective_backtrack_steps": effective_backtrack_steps,
            **self._post_error_diversity(trajectory),
        }

    def _post_error_diversity(self, trajectory: ParsedTrajectory) -> dict:
        """Measure how much the agent changes approach after errors."""
        steps = trajectory.steps
        n = len(steps)

        diversities = []
        type_changes = 0
        file_changes = 0
        n_transitions = 0

        for i in range(n - 1):
            prev = steps[i]
            curr = steps[i + 1]

            if not _has_error(prev.observation):
                continue

            n_transitions += 1

            # 1. Action type changed?
            type_changed = int(curr.action_type != prev.action_type)
            type_changes += type_changed

            # 2. Target files changed?
            prev_files = set(prev.target_files)
            curr_files = set(curr.target_files)
            if prev_files or curr_files:
                file_overlap = _jaccard(curr_files, prev_files)
                file_changed = 1.0 - file_overlap
            else:
                file_changed = float(type_changed)

            diversity = (type_changed + file_changed) / 2.0
            diversities.append(diversity)

        if not diversities:
            return {
                "post_error_diversity": None,
                "post_error_type_change_rate": None,
                "n_error_transitions": 0,
            }

        return {
            "post_error_diversity": float(np.mean(diversities)),
            "post_error_type_change_rate": type_changes / n_transitions,
            "n_error_transitions": n_transitions,
        }


registry.register(BUSMetric())
