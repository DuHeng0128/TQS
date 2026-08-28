"""
M4 · Decision Coherence Score (DCS)
Measures how well intent → action → observation are semantically aligned.

Two sub-scores:
  dcs_file_align  — fraction of steps where files mentioned in thought match
                    actual target_files (path-level intent verification)
  dcs_followup    — Jaccard token overlap: observation → next step's thought
                    (observation-to-intent continuity, unchanged from v1)

The old dcs_global (Jaccard between thought text and action_args text) is
retained for backward compatibility but is known to be near-zero; prefer
dcs_file_align as the primary coherence signal.
"""

import re
import warnings
from typing import Optional, Callable

import numpy as np

from .base import BaseMetric, registry
from ..parsers.base import ParsedTrajectory

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "in", "it",
    "that", "this", "for", "with", "on", "at", "from", "by", "as", "or",
    "and", "be", "i", "we", "you", "he", "she", "they", "do", "does", "did",
    "have", "has", "had", "will", "would", "could", "should", "may", "might",
    "not", "no", "but", "if", "so", "then", "can", "my", "your", "its",
}

# Regex to pull file paths out of free-form thought text.
# Matches common patterns like: "look at auth/views.py", "/repo/src/foo.py",
# "testbed/lib/bar.py", or any token containing "/" and ending in a known ext.
_FILE_PATH_RE = re.compile(
    r'(?:'
    r'(?:/(?:repo|workspace|testbed)/[\w./\-]+)'  # absolute repo paths
    r'|(?:[\w./\-]+/[\w.\-]+\.(?:py|js|ts|java|c|cpp|h|rb|go|rs|kt|scala|sh|yaml|yml|json|toml|cfg|ini|txt|md))'  # relative paths with extension
    r')'
)


def default_jaccard_similarity(text_a: str, text_b: str) -> float:
    ta = set(text_a.lower().split()) - _STOPWORDS
    tb = set(text_b.lower().split()) - _STOPWORDS
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _normalize_path(p: str) -> str:
    """Strip repo prefix and normalise to relative path."""
    for prefix in ("/repo/", "/workspace/", "/testbed/"):
        if p.startswith(prefix):
            return p[len(prefix):]
    return p.lstrip("/")


# Keyword sets for detecting intent from thought text
_INTENT_KEYWORDS = {
    "read_file": {"read", "look", "check", "examine", "open", "view", "see",
                  "cat", "inspect", "review", "content"},
    "write_file": {"write", "edit", "modify", "fix", "change", "update", "add",
                   "replace", "patch", "create", "insert", "remove", "delete",
                   "correct", "adjust"},
    "execute_bash": {"run", "execute", "test", "try", "install", "pip", "python",
                     "bash", "command", "compile", "build", "pytest", "make"},
    "search": {"search", "find", "grep", "locate", "where", "rg", "ag"},
}


def _detect_intent(thought: str) -> Optional[str]:
    """Infer intended action type from thought text keywords."""
    if not thought:
        return None
    thought_lower = thought.lower()
    scores = {}
    for action_type, keywords in _INTENT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in thought_lower)
        if score > 0:
            scores[action_type] = score
    if not scores:
        return None
    return max(scores, key=scores.get)


def _file_align_score(thought: str, target_files: list) -> Optional[float]:
    """Precision of file-path mentions in thought vs actual target_files.

    Returns None when thought mentions no file paths (step is not claiming to
    access a specific file — no signal either way, excluded from mean).
    Returns 1.0 when all mentioned files were actually accessed.
    Returns 0.0 when no mentioned file was accessed.
    """
    if not thought:
        return None
    mentioned = {_normalize_path(m) for m in _FILE_PATH_RE.findall(thought)}
    if not mentioned:
        return None
    actual = set(target_files)
    hits = len(mentioned & actual)
    return hits / len(mentioned)


class DCSCalculator:
    def __init__(self, similarity_fn: Optional[Callable] = None):
        self.similarity_fn = similarity_fn or default_jaccard_similarity

    def compute(self, trajectory: ParsedTrajectory) -> dict:
        steps = trajectory.steps
        n = len(steps)

        dcs_series = []
        followup_series = []
        file_align_series = []
        low_coherence_steps = []
        intent_matches = 0
        intent_total = 0

        for t, step in enumerate(steps):
            # Original: Jaccard between thought text and action args text
            action_text = f"{step.action_type} {' '.join(str(v) for v in step.action_args.values())}"
            align = self.similarity_fn(step.thought, action_text)
            dcs_series.append(float(align))
            if align < 0.1:
                low_coherence_steps.append(step.step_id)

            # New: file-path alignment (only for steps that access files)
            if step.action_type in ("read_file", "write_file"):
                fa = _file_align_score(step.thought, step.target_files)
                if fa is not None:
                    file_align_series.append(fa)

            # New: action type consistency (intent vs actual)
            if step.action_type != "think" and step.thought:
                intent = _detect_intent(step.thought)
                if intent is not None:
                    intent_total += 1
                    if intent == step.action_type:
                        intent_matches += 1

            if t + 1 < n:
                followup = self.similarity_fn(
                    step.observation[:500],
                    steps[t + 1].thought[:500],
                )
                followup_series.append(float(followup))

        dcs_global = float(np.mean(dcs_series)) if dcs_series else 0.0
        dcs_followup = float(np.mean(followup_series)) if followup_series else 0.0
        dcs_combined = (dcs_global + dcs_followup) / 2.0
        dcs_file_align = float(np.mean(file_align_series)) if file_align_series else None
        action_type_consistency = (
            intent_matches / intent_total if intent_total > 0 else None
        )

        return {
            "dcs_global": dcs_global,
            "dcs_followup": dcs_followup,
            "dcs_combined": dcs_combined,
            "dcs_file_align": dcs_file_align,
            "dcs_file_align_n": len(file_align_series),
            "action_type_consistency": float(action_type_consistency) if action_type_consistency is not None else None,
            "action_type_consistency_n": intent_total,
            "dcs_series": dcs_series,
            "low_coherence_steps": low_coherence_steps,
        }


class DCSMetric(BaseMetric):
    name = "m4_dcs"

    def __init__(self, similarity_fn: Optional[Callable] = None):
        self._calc = DCSCalculator(similarity_fn)

    def _compute(self, trajectory: ParsedTrajectory) -> dict:
        return self._calc.compute(trajectory)


registry.register(DCSMetric())
