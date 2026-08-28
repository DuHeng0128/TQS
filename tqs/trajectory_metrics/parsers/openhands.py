"""
Parser for OpenHands trajectories stored in parquet format.

Schema:
  Columns: trajectory_id, instance_id, repo, trajectory, tools,
           model_patch, exit_status, resolved, gen_tests_correct,
           pred_passes_gen_tests

  trajectory: numpy array of dicts with keys:
    role       : "system" | "assistant" | "tool"
    content    : str
    tool_calls : str (JSON) | None   (on assistant messages)
    name       : str | None          (on tool messages)
    tool_call_id: str | None

  Each (assistant, tool*) group forms one logical step.
  An assistant message may have multiple tool_calls.
"""

import re
import json
import logging
import warnings
from typing import Optional

from .base import ParsedStep, ParsedTrajectory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Action type mapping from tool/function names
# ---------------------------------------------------------------------------

_TOOL_TO_ACTION = {
    # read
    "str_replace_editor": None,   # depends on subcommand
    "view_file": "read_file",
    "read_file": "read_file",
    "open": "read_file",
    # write
    "write_file": "write_file",
    "edit_file": "write_file",
    "create_file": "write_file",
    # bash
    "execute_bash": "execute_bash",
    "bash": "execute_bash",
    "run_command": "execute_bash",
    "terminal": "execute_bash",
    # search
    "search_code": "search",
    "search_files": "search",
    "grep": "search",
    "ripgrep": "search",
    "web_search": "search",
    # think
    "think": "think",
    "analyze": "think",
    # submit
    "finish": "submit",
    "submit": "submit",
}

_REPO_PATH_RE = re.compile(r'/(?:repo|workspace|testbed)/[\w./\-]+')
_FILE_BRACKET_RE = re.compile(r'\[File:\s*(/[^\s(]+)')
_OPEN_FILE_RE = re.compile(r'\(Open file:\s*(/[^\s)]+)\)')


def _strip_repo_prefix(path: str) -> str:
    for prefix in ("/repo/", "/workspace/", "/testbed/"):
        if path.startswith(prefix):
            return path[len(prefix):]
    return path


def _classify_tool_call(fn_name: str, args: dict) -> tuple[str, dict]:
    """Map a function call to (action_type, action_args)."""
    name_lower = fn_name.lower()

    # str_replace_editor: check command arg
    if "str_replace_editor" in name_lower or name_lower == "str_replace_editor":
        cmd = args.get("command", "")
        if cmd in ("view",):
            return "read_file", args
        if cmd in ("str_replace", "create", "write", "insert", "create_file"):
            return "write_file", args
        # default to read
        return "read_file", args

    # git commands
    if name_lower == "execute_bash" or name_lower == "bash":
        command_str = args.get("command", "") or args.get("cmd", "")
        if command_str.strip().startswith("git "):
            parts = command_str.strip().split()
            sub = parts[1].lower() if len(parts) > 1 else ""
            if sub in ("log", "diff", "show", "blame", "grep"):
                return "search", args
        return "execute_bash", args

    mapped = _TOOL_TO_ACTION.get(name_lower)
    if mapped:
        return mapped, args

    # fallback heuristics
    if any(k in name_lower for k in ("read", "view", "open", "cat", "show")):
        return "read_file", args
    if any(k in name_lower for k in ("write", "edit", "create", "replace", "patch")):
        return "write_file", args
    if any(k in name_lower for k in ("search", "grep", "find", "rg")):
        return "search", args
    if any(k in name_lower for k in ("bash", "exec", "run", "terminal", "shell")):
        return "execute_bash", args

    return "other", args


def _parse_args(raw_args) -> dict:
    """Deserialize function arguments, handling nested JSON strings."""
    if not raw_args:
        return {}
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        try:
            return json.loads(raw_args)
        except Exception:
            return {"raw": raw_args}
    return {}


def _extract_files_from_args(args: dict) -> list:
    files = []
    for key in ("path", "file_path", "filename", "file", "target_file"):
        val = args.get(key, "")
        if val and isinstance(val, str):
            if "/" in val:
                files.append(_strip_repo_prefix(val))
    return files


def _extract_files_from_text(text: str) -> list:
    if not text:
        return []
    paths = _REPO_PATH_RE.findall(text)
    return [_strip_repo_prefix(p) for p in paths]


def _extract_files_from_observation(obs: str) -> list:
    if not obs:
        return []
    files = []
    for m in _FILE_BRACKET_RE.finditer(obs):
        files.append(_strip_repo_prefix(m.group(1)))
    for m in _OPEN_FILE_RE.finditer(obs):
        files.append(_strip_repo_prefix(m.group(1)))
    return files


def _dedupe_files(files: list) -> list:
    seen = set()
    result = []
    for f in files:
        f = f.strip("/").strip()
        if f and f not in seen and not f.endswith("/"):
            seen.add(f)
            result.append(f)
    return result


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

class OpenHandsParser:
    """Parse OpenHands trajectories from parquet rows."""

    def parse_row(self, row) -> Optional[ParsedTrajectory]:
        try:
            instance_id = str(row.get("instance_id", "unknown"))
            resolved_raw = row.get("resolved", None)
            if resolved_raw is None:
                resolved = None
            else:
                resolved = bool(int(resolved_raw))

            raw_traj = row.get("trajectory", [])
            if raw_traj is None:
                return None

            steps = self._parse_trajectory(raw_traj)
            if not steps:
                return None

            return ParsedTrajectory(
                instance_id=instance_id,
                resolved=resolved,
                steps=steps,
                total_steps=len(steps),
                source_format="openhands",
            )
        except Exception as e:
            warnings.warn(f"OpenHandsParser.parse_row failed for {row.get('instance_id', '?')}: {e}")
            return None

    def _parse_trajectory(self, raw_traj) -> list:
        steps = []
        step_id = 0

        # Collect messages by role
        messages = []
        for m in raw_traj:
            d = dict(m) if not isinstance(m, dict) else m
            role = d.get("role", "")
            if role in ("assistant", "tool", "user"):
                messages.append(d)

        i = 0
        while i < len(messages):
            msg = messages[i]
            role = msg.get("role", "")

            if role == "assistant":
                raw_tool_calls = msg.get("tool_calls")

                # Extract thought: prefer tool_call args (CoderForge uses think tool
                # with arguments.thought); fall back to message content.
                thought = str(msg.get("content", "") or "")
                if raw_tool_calls:
                    _tc0 = None
                    if isinstance(raw_tool_calls, str):
                        try:
                            _tc_list = json.loads(raw_tool_calls)
                            _tc0 = _tc_list[0] if _tc_list else None
                        except Exception:
                            pass
                    elif isinstance(raw_tool_calls, (list, tuple)) and raw_tool_calls:
                        _tc0 = raw_tool_calls[0]
                    if _tc0 and isinstance(_tc0, dict):
                        _tc0_args = _parse_args(_tc0.get("function", {}).get("arguments", {}))
                        if isinstance(_tc0_args, dict) and _tc0_args.get("thought"):
                            thought = str(_tc0_args["thought"])

                # Parse tool_calls
                tool_calls = []
                if raw_tool_calls:
                    if isinstance(raw_tool_calls, str):
                        try:
                            tool_calls = json.loads(raw_tool_calls)
                        except Exception:
                            tool_calls = []
                    elif hasattr(raw_tool_calls, '__iter__'):
                        tool_calls = list(raw_tool_calls)

                if not tool_calls:
                    # Pure thought step
                    steps.append(ParsedStep(
                        step_id=step_id,
                        thought=thought,
                        action_type="think",
                        action_args={},
                        observation="",
                        target_files=[],
                        token_count=max(1, len(thought) // 4),
                    ))
                    step_id += 1
                    i += 1
                    continue

                # Each tool call becomes a step (collect all following tool responses)
                # Collect tool responses
                j = i + 1
                tool_responses = {}
                while j < len(messages) and messages[j].get("role") == "tool":
                    t = messages[j]
                    tid = t.get("tool_call_id", "")
                    tool_responses[tid] = str(t.get("content", "") or "")
                    j += 1

                for tc in tool_calls:
                    try:
                        fn = tc.get("function", {})
                        fn_name = fn.get("name", "other")
                        raw_args = fn.get("arguments", {})
                        args = _parse_args(raw_args)

                        action_type, action_args = _classify_tool_call(fn_name, args)

                        tc_id = tc.get("id", "")
                        obs = tool_responses.get(tc_id, "")

                        files = _dedupe_files(
                            _extract_files_from_args(action_args)
                            + _extract_files_from_text(str(action_args.get("command", ""))
                                                       + str(action_args.get("path", "")))
                            + _extract_files_from_observation(obs)
                        )

                        token_count = max(1, (len(thought) + len(str(raw_args)) + len(obs)) // 4)

                        steps.append(ParsedStep(
                            step_id=step_id,
                            thought=thought,
                            action_type=action_type,
                            action_args=action_args,
                            observation=obs,
                            target_files=files,
                            token_count=token_count,
                        ))
                        step_id += 1
                        thought = ""  # only first step gets the thought
                    except Exception as e:
                        warnings.warn(f"Error parsing tool_call: {e}")

                i = j  # skip consumed tool responses
                continue

            i += 1

        return steps

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
