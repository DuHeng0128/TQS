"""Base metric class and registry."""

import warnings
import logging
from abc import ABC, abstractmethod
from typing import Optional

from ..parsers.base import ParsedTrajectory

logger = logging.getLogger(__name__)

MIN_STEPS = 3


def insufficient_data(reason: str = "") -> dict:
    return {"status": "insufficient_data", "reason": reason}


class BaseMetric(ABC):
    name: str = ""

    def compute(self, trajectory: ParsedTrajectory) -> Optional[dict]:
        """Safe wrapper around _compute with exception handling."""
        try:
            if trajectory.total_steps < MIN_STEPS:
                return insufficient_data(f"only {trajectory.total_steps} steps")
            return self._compute(trajectory)
        except Exception as e:
            warnings.warn(f"Metric {self.name} failed on {trajectory.instance_id}: {e}")
            return None

    @abstractmethod
    def _compute(self, trajectory: ParsedTrajectory) -> Optional[dict]:
        pass


class MetricRegistry:
    def __init__(self):
        self._metrics: dict[str, BaseMetric] = {}

    def register(self, metric: BaseMetric):
        self._metrics[metric.name] = metric

    def get(self, name: str) -> Optional[BaseMetric]:
        return self._metrics.get(name)

    def all_names(self) -> list:
        return list(self._metrics.keys())

    def compute_all(self, trajectory: ParsedTrajectory, names: list = None) -> dict:
        names = names or self.all_names()
        return {n: self._metrics[n].compute(trajectory) for n in names if n in self._metrics}


registry = MetricRegistry()
