"""
M1 · Repetitive Behavior Score (RBS)
Detects if the agent is stuck in a repetitive loop.
"""

import warnings
from itertools import combinations
from typing import Optional

import numpy as np

from .base import BaseMetric, registry
from ..parsers.base import ParsedTrajectory


class RBSMetric(BaseMetric):
    name = "m1_rbs"

    def __init__(self, window_size: int = 5, threshold: float = 0.85, skip_think: bool = False):
        self.window_size = window_size
        self.threshold = threshold
        self.skip_think = skip_think

    def _compute(self, trajectory: ParsedTrajectory) -> dict:
        steps = trajectory.steps
        if self.skip_think:
            steps = [s for s in steps if s.action_type != "think"]
        if len(steps) < 2:
            return {"status": "insufficient_data", "reason": "too few non-think steps"}

        features = [
            f"{s.action_type}|{','.join(sorted(s.target_files))}|{s.thought[:100]}"
            for s in steps
        ]

        # Vectorize with TF-IDF (fallback to simple token overlap)
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
            matrix = vec.fit_transform(features)

            rbs_series = []
            for t in range(len(features)):
                if t < self.window_size - 1:
                    rbs_series.append(0.0)
                    continue
                window_idx = list(range(t - self.window_size + 1, t + 1))
                if len(window_idx) < 2:
                    rbs_series.append(0.0)
                    continue
                window_vecs = matrix[window_idx]
                sims = []
                for i, j in combinations(range(len(window_idx)), 2):
                    sim = cosine_similarity(window_vecs[i], window_vecs[j])[0, 0]
                    sims.append(float(sim))
                rbs_series.append(float(np.mean(sims)) if sims else 0.0)

        except ImportError:
            # Fallback: Jaccard token overlap
            rbs_series = self._jaccard_rbs(features)

        rbs_global = float(max(rbs_series)) if rbs_series else 0.0

        early_stop_step = None
        for t, val in enumerate(rbs_series):
            if val > self.threshold:
                early_stop_step = steps[t].step_id
                break

        # --- Loop detection: consecutive runs of high similarity ---
        loop_threshold = 0.50
        loop_steps_count = 0
        current_run = 0
        max_loop = 0
        n_loops = 0
        for val in rbs_series:
            if val > loop_threshold:
                current_run += 1
            else:
                if current_run >= 2:
                    loop_steps_count += current_run
                    max_loop = max(max_loop, current_run)
                    n_loops += 1
                current_run = 0
        if current_run >= 2:
            loop_steps_count += current_run
            max_loop = max(max_loop, current_run)
            n_loops += 1

        n_steps = len(rbs_series) if rbs_series else 1
        rbs_loop_fraction = loop_steps_count / n_steps
        rbs_max_loop_frac = max_loop / n_steps

        # Continuous variant: mean of rbs_series (step-length invariant)
        rbs_mean = float(np.mean(rbs_series)) if rbs_series else 0.0

        total = trajectory.total_steps
        return {
            "rbs_global": rbs_global,
            "rbs_mean": rbs_mean,
            "rbs_loop_fraction": float(rbs_loop_fraction),
            "rbs_max_loop_frac": float(rbs_max_loop_frac),
            "rbs_n_loops": n_loops,
            "rbs_series": rbs_series,
            "early_stop_triggered": early_stop_step is not None,
            "early_stop_step": early_stop_step,
            "early_stop_step_pct": (early_stop_step / total) if early_stop_step is not None else None,
        }

    def _jaccard_rbs(self, features: list) -> list:
        def tokenize(text):
            return set(text.lower().split())

        token_sets = [tokenize(f) for f in features]
        rbs_series = []
        for t in range(len(features)):
            if t < self.window_size - 1:
                rbs_series.append(0.0)
                continue
            window = token_sets[t - self.window_size + 1: t + 1]
            sims = []
            for a, b in combinations(window, 2):
                if not a and not b:
                    sims.append(1.0)
                elif not a or not b:
                    sims.append(0.0)
                else:
                    sims.append(len(a & b) / len(a | b))
            rbs_series.append(float(np.mean(sims)) if sims else 0.0)
        return rbs_series


registry.register(RBSMetric())
