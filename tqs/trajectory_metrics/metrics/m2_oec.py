"""
M2 · Observation Entropy Collapse (OEC)
Detects if the agent has stopped receiving informative observations.
"""

import math
import re
import warnings
from collections import Counter
from typing import Optional

import numpy as np

from .base import BaseMetric, registry
from ..parsers.base import ParsedTrajectory


def char_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


_ERROR_PATTERNS = [
    re.compile(r'Traceback \(most recent call last\)', re.IGNORECASE),
    re.compile(r'(?:^|\n)\s*(?:Error|ERROR|Exception|FAILED|FAILURE)', re.MULTILINE),
    re.compile(r'AssertionError|TypeError|ValueError|KeyError|AttributeError|ImportError|NameError|IndexError|RuntimeError|FileNotFoundError|SyntaxError|IndentationError|ModuleNotFoundError|OSError|PermissionError'),
    re.compile(r'command not found|Permission denied|No such file or directory'),
    re.compile(r'exit (?:code|status)\s*[1-9]|returned non-zero exit status'),
]


def _error_signature(obs: str) -> Optional[str]:
    """Extract a coarse error signature from an observation.

    Returns None if no error pattern detected.
    Returns a string fingerprint grouping similar errors together.
    """
    if not obs:
        return None
    for i, pat in enumerate(_ERROR_PATTERNS):
        m = pat.search(obs)
        if m:
            # Use last few non-empty lines as fingerprint (captures the actual error)
            lines = [l.strip() for l in obs.strip().split('\n') if l.strip()]
            sig_lines = lines[-3:] if lines else [m.group()]
            return f"e{i}:" + "|".join(sig_lines)
    return None


class OECMetric(BaseMetric):
    name = "m2_oec"

    def __init__(self, window_size: int = 3, collapse_threshold: float = 0.3):
        self.window_size = window_size
        self.collapse_threshold = collapse_threshold

    def _compute(self, trajectory: ParsedTrajectory) -> dict:
        steps = trajectory.steps
        n = len(steps)

        entropy_series = [char_entropy(s.observation) for s in steps]
        has_empty_obs = sum(1 for s in steps if not s.observation)

        # Baseline: mean of first min(5, n) steps
        baseline_n = min(5, n)
        baseline = float(np.mean(entropy_series[:baseline_n])) if baseline_n > 0 else 0.0

        # Smoothed entropy
        smooth = []
        for t in range(n):
            window = entropy_series[max(0, t - self.window_size + 1): t + 1]
            smooth.append(float(np.mean(window)))

        # Relative entropy series
        if baseline == 0:
            rel_series = [0.0] * n
            min_rel = 0.0
        else:
            rel_series = [s / baseline for s in smooth]
            min_rel = float(min(rel_series)) if rel_series else 0.0

        # Detect collapse: consecutive window_size steps below threshold
        collapse_level = 1.0 - self.collapse_threshold
        collapse_step = None
        consecutive = 0
        for t, rel in enumerate(rel_series):
            if rel < collapse_level:
                consecutive += 1
                if consecutive >= self.window_size and collapse_step is None:
                    collapse_step = steps[t - self.window_size + 1].step_id
            else:
                consecutive = 0

        total = trajectory.total_steps

        # --- Error repetition detection ---
        error_sigs = [_error_signature(s.observation) for s in steps]
        n_errors = sum(1 for e in error_sigs if e is not None)
        error_rate = n_errors / n if n > 0 else 0.0

        consecutive_same = 0
        consecutive_total = 0
        for i in range(n - 1):
            if error_sigs[i] is not None and error_sigs[i + 1] is not None:
                consecutive_total += 1
                if error_sigs[i] == error_sigs[i + 1]:
                    consecutive_same += 1

        error_repeat_rate = consecutive_same / consecutive_total if consecutive_total > 0 else None

        return {
            "oec_baseline": baseline,
            "entropy_series": entropy_series,
            "entropy_smooth_series": smooth,
            "collapse_detected": collapse_step is not None,
            "collapse_step": collapse_step,
            "collapse_step_pct": (collapse_step / total) if collapse_step is not None else None,
            "min_relative_entropy": min_rel,
            "has_empty_obs": has_empty_obs,
            "error_repeat_rate": float(error_repeat_rate) if error_repeat_rate is not None else None,
            "error_rate": float(error_rate),
            "n_errors": n_errors,
        }


registry.register(OECMetric())
