"""
Parser for CoderForge-Preview trajectories stored in parquet format.

Schema (parquet):
  Columns: trajectory_id, finish_reason, image, messages, reward, tools, license

  messages: JSON string → list of dicts with keys:
    role          : "system" | "user" | "assistant" | "tool"
    content       : str
    tool_calls    : list[dict] | None   (assistant messages; each dict has "function")
    function_call : dict | None         (legacy field, ignored)
    name          : str | None          (tool messages — the tool that was called)
    tool_call_id  : str | None          (tool messages — matches tool_calls[i].id)

  tool_calls[i].function.arguments may contain a "thought" key (e.g., for the
  "think" tool).  The openhands._parse_trajectory handles this transparently once
  thought extraction from tool_call args is enabled.

  reward: float — 1.0 = success, 0.0 = failure, -1.0 = error.
  Repo path: /testbed/ (handled by updated openhands path regex).
"""

import json
import logging
import warnings
from typing import Optional

from .base import ParsedTrajectory
from .openhands import OpenHandsParser

logger = logging.getLogger(__name__)


class CoderForgeParser(OpenHandsParser):
    """Parse CoderForge-Preview trajectories.

    Inherits all message-parsing logic from OpenHandsParser (which already
    handles /testbed/ paths and thought-from-args extraction).  Only overrides
    parse_row to map CoderForge column names and reward → resolved.
    """

    def parse_row(self, row) -> Optional[ParsedTrajectory]:
        try:
            trajectory_id = str(row.get("trajectory_id", "unknown"))

            reward_raw = row.get("reward", None)
            if reward_raw is None:
                resolved = None
            else:
                resolved = bool(float(reward_raw) >= 1.0)

            raw_messages = row.get("messages", [])
            if raw_messages is None:
                return None
            if isinstance(raw_messages, str):
                try:
                    raw_messages = json.loads(raw_messages)
                except Exception as e:
                    warnings.warn(f"CoderForgeParser: failed to parse messages JSON for {trajectory_id}: {e}")
                    return None

            steps = self._parse_trajectory(raw_messages)
            if not steps:
                return None

            return ParsedTrajectory(
                instance_id=trajectory_id,
                resolved=resolved,
                steps=steps,
                total_steps=len(steps),
                source_format="coderforge",
            )
        except Exception as e:
            warnings.warn(f"CoderForgeParser.parse_row failed for {row.get('trajectory_id', '?')}: {e}")
            return None

    def parse_parquet(self, path: str) -> list:
        """Parse all rows from a parquet file."""
        import pandas as pd
        df = pd.read_parquet(path)
        results = []
        for _, row in df.iterrows():
            traj = self.parse_row(row)
            if traj is not None:
                results.append(traj)
        return results
