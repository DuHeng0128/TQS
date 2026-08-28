"""
M9 · Tool Transition Entropy (TTE)
Quantifies diversity of tool usage and the information content of the transition pattern.
"""

import math
import warnings
from collections import Counter, defaultdict
from typing import Optional

import numpy as np

from .base import BaseMetric, registry
from ..parsers.base import ParsedTrajectory

_ALL_TOOL_TYPES = {"read_file", "write_file", "execute_bash", "search", "think", "submit", "other"}


def _transition_entropy(transitions: dict) -> float:
    total = sum(sum(v.values()) for v in transitions.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for from_tool, to_counts in transitions.items():
        for to_tool, count in to_counts.items():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
    return entropy


class TTEMetric(BaseMetric):
    name = "m9_tte"

    def _compute(self, trajectory: ParsedTrajectory) -> dict:
        tool_seq = [s.action_type for s in trajectory.steps]
        n = len(tool_seq)

        if n < 2:
            return {"status": "insufficient_data", "reason": "too few steps"}

        # Transition matrix
        transitions = defaultdict(lambda: defaultdict(int))
        for i in range(n - 1):
            transitions[tool_seq[i]][tool_seq[i + 1]] += 1

        tte_score = _transition_entropy(transitions)

        # Tool distribution
        counts = Counter(tool_seq)
        tool_dist = {t: c / n for t, c in counts.items()}

        # Transition matrix as plain dict
        transition_matrix = {k: dict(v) for k, v in transitions.items()}

        dominant_tool = counts.most_common(1)[0][0] if counts else "other"

        unique_tools = len(set(tool_seq) & _ALL_TOOL_TYPES)
        tool_diversity = unique_tools / len(_ALL_TOOL_TYPES)

        return {
            "tte_score": float(tte_score),
            "tool_distribution": tool_dist,
            "transition_matrix": transition_matrix,
            "dominant_tool": dominant_tool,
            "tool_diversity": float(tool_diversity),
        }


registry.register(TTEMetric())
