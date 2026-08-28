"""
Parser for SWE-agent trajectories stored in parquet format.

Schema:
  Columns: instance_id, model_name, target, trajectory, exit_status,
           generated_patch, eval_logs
  trajectory: numpy array of dicts with keys:
    role       : "system" | "user" | "ai"
    text       : message content
    mask       : bool
    cutoff_date: str | None
    system_prompt: str | None

  Pairing: system(0) -> user(1)[task] -> [ai, user]* pairs
    ai step  -> thought + command (in ``` fence)
    user step -> observation
"""

import re
import logging
import warnings
from typing import Optional

from .base import ParsedStep, ParsedTrajectory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Action type classification
# ---------------------------------------------------------------------------

# Maps command prefixes/names to action_type
_READ_CMDS = {
    "open", "view", "cat", "less", "head", "tail", "scroll_down", "scroll_up",
    "goto", "str_replace_editor",
}
_WRITE_CMDS = {
    "edit", "create", "write", "edit_file",
}
_SEARCH_CMDS = {
    "grep", "find_file", "search_dir", "search_file", "search_code",
    "ripgrep", "rg",
}
_BASH_CMDS = {
    "bash", "execute_bash", "run", "python", "python3", "pytest", "pip",
    "ls", "find", "cd", "mkdir", "cp", "mv", "rm", "touch", "echo",
    "cat",  # ambiguous but treat as bash when not open/view
    "git", "curl", "wget", "make", "sh",
}
_SUBMIT_CMDS = {"submit", "finish"}

# Regexes for file path extraction
_REPO_PATH_RE = re.compile(r'/repo/[\w./\-]+')
_FILE_BRACKET_RE = re.compile(r'\[File:\s*(/[^\s(]+)')
_OPEN_FILE_RE = re.compile(r'\(Open file:\s*(/[^\s)]+)\)')
_CURRENT_DIR_RE = re.compile(r'\(Current directory:\s*(/[^\s)]+)\)')


def _classify_action(command: str) -> tuple[str, dict]:
    """Return (action_type, action_args) from a raw command string."""
    if not command or not command.strip():
        return "think", {}

    cmd = command.strip()
    first_token = cmd.split()[0].lower()

    # git subcommands: log/diff/show -> search, else execute_bash
    if first_token == "git":
        parts = cmd.split()
        sub = parts[1].lower() if len(parts) > 1 else ""
        if sub in ("log", "diff", "show", "blame", "grep"):
            return "search", {"command": cmd}
        return "execute_bash", {"command": cmd}

    if first_token in _SUBMIT_CMDS:
        return "submit", {}

    if first_token in _SEARCH_CMDS:
        return "search", {"command": cmd}

    # str_replace_editor: classify by subcommand
    if first_token == "str_replace_editor":
        parts = cmd.split()
        sub = parts[1].lower() if len(parts) > 1 else ""
        if sub in ("view",):
            path = parts[2] if len(parts) > 2 else ""
            return "read_file", {"command": sub, "path": path}
        if sub in ("str_replace", "create", "write", "insert"):
            path = parts[2] if len(parts) > 2 else ""
            return "write_file", {"command": sub, "path": path}
        return "read_file", {"command": cmd}

    if first_token in _READ_CMDS:
        parts = cmd.split()
        path = parts[1] if len(parts) > 1 else ""
        return "read_file", {"command": first_token, "path": path}

    if first_token in _WRITE_CMDS:
        parts = cmd.split()
        path = parts[1] if len(parts) > 1 else ""
        return "write_file", {"command": first_token, "path": path}

    if first_token in _BASH_CMDS:
        return "execute_bash", {"command": cmd}

    # Fallback: treat as execute_bash if looks like a command
    return "execute_bash", {"command": cmd}


def _extract_command_from_text(text: str) -> str:
    """Extract the command from inside the last ``` block in ai text."""
    # Find all code blocks (``` or `...`)
    blocks = re.findall(r'```(?:\w+\n)?(.*?)```', text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    # Single-line backtick code
    inline = re.findall(r'`([^`\n]+)`', text)
    if inline:
        return inline[-1].strip()
    return ""


def _extract_thought_from_text(text: str) -> str:
    """Extract thought (text before the last code block)."""
    # Remove last code block
    idx = text.rfind("```")
    if idx == -1:
        return text.strip()
    # Find the opening ``` before idx
    open_idx = text.rfind("```", 0, idx)
    if open_idx == -1:
        return text.strip()
    return text[:open_idx].strip()


def _strip_repo_prefix(path: str) -> str:
    """Remove /repo/ prefix to get relative path."""
    if path.startswith("/repo/"):
        return path[6:]
    if path == "/repo":
        return ""
    return path


def _extract_files_from_args(action_args: dict) -> list:
    files = []
    for key in ("path", "file_path", "filename"):
        val = action_args.get(key, "")
        if val and isinstance(val, str) and val.startswith("/"):
            files.append(_strip_repo_prefix(val))
    return files


def _extract_files_from_text(text: str) -> list:
    """Extract /repo/... paths from text."""
    paths = _REPO_PATH_RE.findall(text)
    return [_strip_repo_prefix(p) for p in paths]


def _extract_files_from_observation(obs: str) -> list:
    files = []
    # [File: /path ...]
    for m in _FILE_BRACKET_RE.finditer(obs):
        files.append(_strip_repo_prefix(m.group(1)))
    # (Open file: /path)
    for m in _OPEN_FILE_RE.finditer(obs):
        files.append(_strip_repo_prefix(m.group(1)))
    return files


def _dedupe_files(files: list) -> list:
    seen = set()
    result = []
    for f in files:
        f = f.strip("/").strip()
        if f and f not in seen:
            seen.add(f)
            result.append(f)
    return result


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

class SWEAgentParser:
    """Parse SWE-agent trajectories from parquet rows."""

    def parse_row(self, row) -> Optional[ParsedTrajectory]:
        """Parse a single pandas row into ParsedTrajectory."""
        try:
            instance_id = str(row.get("instance_id", "unknown"))
            resolved = bool(row.get("target", None))
            if row.get("target") is None:
                resolved = None

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
                source_format="swe_agent",
            )
        except Exception as e:
            warnings.warn(f"SWEAgentParser.parse_row failed for {row.get('instance_id', '?')}: {e}")
            return None

    def _parse_trajectory(self, raw_traj) -> list:
        """Convert raw trajectory array to list[ParsedStep]."""
        # Collect (ai_step, user_step) pairs
        steps = []
        step_id = 0

        # Filter to non-system messages
        messages = [dict(m) for m in raw_traj if dict(m).get("role") in ("ai", "user")]

        i = 0
        while i < len(messages):
            msg = messages[i]
            role = msg.get("role", "")

            if role == "ai":
                text = msg.get("text", "") or ""
                command = _extract_command_from_text(text)
                thought = _extract_thought_from_text(text)
                action_type, action_args = _classify_action(command)

                # Look for the next user message as observation
                obs = ""
                if i + 1 < len(messages) and messages[i + 1].get("role") == "user":
                    obs = messages[i + 1].get("text", "") or ""
                    i += 1  # consume the user step

                # Extract target files
                files = _dedupe_files(
                    _extract_files_from_args(action_args)
                    + _extract_files_from_text(command)
                    + _extract_files_from_observation(obs)
                )

                token_count = (len(thought) + len(command) + len(obs)) // 4

                steps.append(ParsedStep(
                    step_id=step_id,
                    thought=thought,
                    action_type=action_type,
                    action_args=action_args,
                    observation=obs,
                    target_files=files,
                    token_count=max(1, token_count),
                ))
                step_id += 1

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
