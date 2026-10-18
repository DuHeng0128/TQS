"""Parser for OpenHands source/action/observation event JSON files."""

import json
from pathlib import Path

from .base import ParsedStep, ParsedTrajectory

ACTION_MAP = {"run": "execute_bash", "edit": "write_file", "think": "think",
              "finish": "submit", "read": "read_file", "browse": "search", "recall": "search"}


def parse_json_trajectory(path, resolved=None):
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    observations = {
        event["cause"]: event.get("content") or ""
        for event in raw if event.get("action") is None
        and event.get("observation") is not None and event.get("cause") is not None
    }
    steps = []
    for event in raw:
        action = event.get("action")
        if event.get("source") != "agent" or action in (None, "system"):
            continue
        args = event.get("args") or {}
        thought = args.get("thought") or event.get("message") or ""
        observation = observations.get(event["id"], "")
        if not isinstance(observation, str):
            observation = json.dumps(observation, ensure_ascii=False)
        target = args.get("path") or ""
        for prefix in ("/workspace/", "/repo/", "/testbed/"):
            if target.startswith(prefix):
                target = target[len(prefix):]
                break
        steps.append(ParsedStep(
            step_id=len(steps), thought=thought,
            action_type=ACTION_MAP.get(action, "other"), action_args=args,
            observation=observation, target_files=[target] if target else [],
            token_count=max(1, (len(thought) + len(str(args)) + len(observation)) // 4),
        ))
    if not steps:
        return None
    return ParsedTrajectory(path.stem, resolved, steps, len(steps), "openhands_json")
