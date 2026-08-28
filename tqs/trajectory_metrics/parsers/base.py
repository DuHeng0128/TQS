from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ParsedStep:
    step_id: int
    thought: str                 # agent reasoning text
    action_type: str             # read_file / write_file / execute_bash / search / think / submit / other
    action_args: dict            # tool parameters
    observation: str             # raw tool output text
    target_files: list           # extracted file paths (relative, /repo/ stripped)
    token_count: int             # estimated tokens = len(all text) // 4


@dataclass
class ParsedTrajectory:
    instance_id: str
    resolved: Optional[bool]     # ground truth label from dataset
    steps: list                  # list[ParsedStep]
    total_steps: int
    source_format: str           # "swe_agent" or "openhands"
