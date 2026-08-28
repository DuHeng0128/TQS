"""
M8 · Trajectory Graph Cycle Ratio (TGCR)
Builds a directed file-transition graph and measures how many back-edges exist.
"""

import sys
import warnings
from typing import Optional

import numpy as np

from .base import BaseMetric, registry
from ..parsers.base import ParsedTrajectory


def _count_back_edges(G) -> int:
    """DFS-based back-edge detection."""
    visited = set()
    in_stack = set()
    back_edges = 0

    # Iterative DFS to avoid Python recursion limit
    for start in list(G.nodes()):
        if start in visited:
            continue
        stack = [(start, iter(G.successors(start)))]
        visited.add(start)
        in_stack.add(start)
        while stack:
            node, children = stack[-1]
            try:
                child = next(children)
                if child in in_stack:
                    back_edges += 1
                elif child not in visited:
                    visited.add(child)
                    in_stack.add(child)
                    stack.append((child, iter(G.successors(child))))
            except StopIteration:
                stack.pop()
                in_stack.discard(node)

    return back_edges


class TGCRMetric(BaseMetric):
    name = "m8_tgcr"

    def _compute(self, trajectory: ParsedTrajectory) -> dict:
        try:
            import networkx as nx
        except ImportError:
            warnings.warn("networkx not installed; TGCR skipped")
            return None

        # Collect file sequence from read/write steps
        file_seq = []
        self_loops = 0
        for step in trajectory.steps:
            if step.action_type in ("read_file", "write_file") and step.target_files:
                for f in step.target_files:
                    if file_seq and file_seq[-1] == f:
                        self_loops += 1
                    else:
                        file_seq.append(f)

        if len(file_seq) < 2:
            return {"status": "insufficient_data", "reason": "too few file transitions"}

        G = nx.DiGraph()
        edges = []
        for i in range(len(file_seq) - 1):
            src, dst = file_seq[i], file_seq[i + 1]
            if src != dst:
                G.add_edge(src, dst)
                edges.append((src, dst))

        total_edges = len(edges)
        if total_edges == 0:
            return {
                "tgcr": 0.0,
                "total_edges": 0,
                "back_edges": 0,
                "self_loops": self_loops,
                "unique_nodes": G.number_of_nodes(),
                "graph_density": 0.0,
                "strongly_connected_components": 0,
            }

        back_edges = _count_back_edges(G)
        tgcr = back_edges / total_edges

        density = nx.density(G)
        scc = nx.number_strongly_connected_components(G)

        # --- File oscillation: A→B→A patterns ---
        oscillations = 0
        for i in range(len(file_seq) - 2):
            if file_seq[i] == file_seq[i + 2] and file_seq[i] != file_seq[i + 1]:
                oscillations += 1
        oscillation_rate = oscillations / max(1, len(file_seq) - 2)

        # --- Repeated transitions: same edge used multiple times ---
        from collections import Counter
        edge_counts = Counter(edges)
        repeated_uses = sum(c - 1 for c in edge_counts.values() if c > 1)
        repeated_transition_rate = repeated_uses / total_edges if total_edges > 0 else 0.0

        return {
            "tgcr": float(tgcr),
            "total_edges": total_edges,
            "back_edges": back_edges,
            "self_loops": self_loops,
            "unique_nodes": G.number_of_nodes(),
            "graph_density": float(density),
            "strongly_connected_components": scc,
            "oscillation_rate": float(oscillation_rate),
            "repeated_transition_rate": float(repeated_transition_rate),
        }


registry.register(TGCRMetric())
