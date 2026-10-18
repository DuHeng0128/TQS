#!/usr/bin/env python3
"""
SWE-bench Verified evaluation script using OpenHands + Daytona sandbox.

Usage:
    export DAYTONA_API_KEY="your_api_key_here"
    export DAYTONA_API_URL="https://app.daytona.io/api"  # optional
    python3 run_eval.py [--instances N] [--output-dir DIR] [--start-idx IDX]

Prerequisites:
    1. vLLM server must be running:
       bash evaluation/serve.sh
    2. DAYTONA_API_KEY must be set
    3. OpenHands with Daytona support installed (or set OPENHANDS_ROOT)
"""

import asyncio
import json
import os
import sys
import time
import uuid
import argparse
import traceback
from pathlib import Path
from datetime import datetime

import pandas as pd
# Ensure OpenHands is importable
if os.getenv('OPENHANDS_ROOT'):
    sys.path.insert(0, os.environ['OPENHANDS_ROOT'])

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATASET_PATH = os.getenv('SWEBENCH_DATASET', 'data/SWE-bench_Verified/data/test-00000-of-00001.parquet')
CONFIG_FILE = str(Path(__file__).resolve().parent / 'config.toml')
OUTPUT_DIR = 'outputs/eval'

# Model inference parameters (injected via config.toml / env)
TEMPERATURE   = 0.2
TOP_P         = 0.8
MAX_TOKENS    = 16384
MAX_ITERATIONS = 100

# Optional external parser LLM for malformed textual tool calls
PARSER_BASE_URL = None
PARSER_MODEL = None
PARSER_API_KEY = None

# Ensure Daytona sandboxes are deleted (not just stopped) after each instance,
# otherwise stopped sandboxes accumulate and hit the CPU limit.
os.environ.setdefault('DAYTONA_DELETE_ON_CLOSE', 'true')
os.environ.setdefault('DAYTONA_SANDBOX_CPU', '3')
os.environ.setdefault('DAYTONA_SANDBOX_MEMORY_GIB', '3')
os.environ.setdefault('DAYTONA_SANDBOX_DISK_GIB', '8')
REQUIRED_ENV = {"DAYTONA_API_KEY"}

# ---------------------------------------------------------------------------
# SWE-bench task prompt template
# ---------------------------------------------------------------------------
SWE_BENCH_PROMPT = """\
Please fix the following issue in the repository.

<issue>
{problem_statement}
</issue>

You are working in the repository `{repo}` at commit `{base_commit}`.
The repository has already been cloned and is available at `/workspace/{repo_name}`.

Editing guidance:
- Prefer the `str_replace` mode of the file editor for small, targeted edits.
  Pass a unique `old_str` (a few lines around the change) and the replacement
  `new_str`. This avoids the "range of lines to edit is too long" error that
  whole-file rewrites trigger on files larger than ~300 lines.
- Only fall back to a full-file rewrite when the change genuinely spans the
  whole file.
- Do NOT create new files (test scripts, scratch files, etc.) unless the fix
  itself requires them.

Please make the necessary code changes to fix the issue.
When you are done, make sure all changes are saved to disk.
Do NOT run the tests - just make the code changes to fix the issue.

As the final step, before calling the `finish` tool, run exactly this command so the patch is captured in the trajectory:
  cd /workspace/{repo_name} && git --no-pager diff --no-color HEAD
"""


def build_prompt(instance: dict) -> str:
    repo = instance['repo']
    repo_name = repo.split('/')[-1]
    return SWE_BENCH_PROMPT.format(
        problem_statement=instance['problem_statement'],
        repo=repo,
        base_commit=instance['base_commit'],
        repo_name=repo_name,
    )


# ---------------------------------------------------------------------------
# Single instance runner
# ---------------------------------------------------------------------------
async def run_single_instance(instance: dict, output_dir: Path) -> dict:
    """Run one SWE-bench instance through OpenHands and return result dict."""
    from openhands.core.config.utils import load_openhands_config
    from openhands.core.main import run_controller
    from openhands.core.setup import generate_sid
    from openhands.events.action import MessageAction

    instance_id = instance['instance_id']
    result = {
        'instance_id': instance_id,
        'repo': instance['repo'],
        'base_commit': instance['base_commit'],
        'status': 'pending',
        'error': None,
        'patch': None,
        'trajectory_path': None,
        'start_time': datetime.utcnow().isoformat(),
        'end_time': None,
        'duration_seconds': None,
    }

    traj_dir = output_dir / 'trajectories'
    traj_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    try:
        # Load config fresh for each instance (avoids state leakage)
        config = load_openhands_config(config_file=CONFIG_FILE)

        # Per-instance trajectory file
        traj_path = str(traj_dir / f'{instance_id}.json')
        config.save_trajectory_path = traj_path

        # Max iterations
        config.max_iterations = MAX_ITERATIONS

        # Set repository for this instance via sandbox selected_repo
        # OpenHands will clone it into /workspace
        config.sandbox.selected_repo = instance['repo']

        # Pass base_commit via env so DaytonaRuntime can checkout the right commit
        os.environ['SWE_BENCH_BASE_COMMIT'] = instance['base_commit']

        # Optional external parser LLM configuration for malformed textual tool calls
        if PARSER_BASE_URL:
            os.environ['OPENHANDS_TOOL_CALL_PARSER_BASE_URL'] = PARSER_BASE_URL
        else:
            os.environ.pop('OPENHANDS_TOOL_CALL_PARSER_BASE_URL', None)
        if PARSER_MODEL:
            os.environ['OPENHANDS_TOOL_CALL_PARSER_MODEL'] = PARSER_MODEL
        else:
            os.environ.pop('OPENHANDS_TOOL_CALL_PARSER_MODEL', None)
        if PARSER_API_KEY:
            os.environ['OPENHANDS_TOOL_CALL_PARSER_API_KEY'] = PARSER_API_KEY
        else:
            os.environ.pop('OPENHANDS_TOOL_CALL_PARSER_API_KEY', None)

        sid = generate_sid(
            config,
            session_name=f'{instance_id[:20]}-{uuid.uuid4().hex[:8]}',
        )

        task = build_prompt(instance)
        initial_action = MessageAction(content=task)

        state = await run_controller(
            config=config,
            initial_user_action=initial_action,
            sid=sid,
            exit_on_message=True,
            headless_mode=True,
        )

        result['status'] = 'completed'
        result['trajectory_path'] = traj_path

        # Try to extract git diff / patch from trajectory
        if state is not None and state.history:
            patch = _extract_patch(traj_path)
            result['patch'] = patch

    except Exception as e:
        result['status'] = 'error'
        result['error'] = f'{type(e).__name__}: {e}\n{traceback.format_exc()}'

    elapsed = time.time() - start
    result['end_time'] = datetime.utcnow().isoformat()
    result['duration_seconds'] = round(elapsed, 1)
    return result


def _extract_patch(traj_path: str) -> str | None:
    """Extract the final git diff patch from a trajectory file.

    The agent is instructed to run `git diff` as its last action; that produces
    a run-observation whose `content` contains the diff text. The diff may be
    preceded by shell prologue (e.g. a `cd` echoing) so we search for the
    `diff --git` marker inside the content rather than requiring it at the
    start.
    """
    try:
        with open(traj_path) as f:
            events = json.load(f)
        for event in reversed(events):
            content = event.get('content') or event.get('output') or ''
            if not isinstance(content, str):
                continue
            idx = content.find('diff --git ')
            if idx >= 0:
                return content[idx:].rstrip() + '\n'
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
async def run_evaluation(
    instances: list[dict],
    output_dir: Path,
    max_concurrent: int = 1,
) -> None:
    results_path = output_dir / 'results.jsonl'
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: skip already-done instances
    done_ids = set()
    if results_path.exists():
        with open(results_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done_ids.add(r['instance_id'])
                except Exception:
                    pass
        print(f'Resuming: {len(done_ids)} instances already done, skipping.')

    pending = [i for i in instances if i['instance_id'] not in done_ids]
    print(f'Total to process: {len(pending)} instances '
          f'(concurrency={max_concurrent})')

    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_with_sem(inst):
        async with semaphore:
            print(f'[{datetime.utcnow().isoformat()}] START {inst["instance_id"]}')
            res = await run_single_instance(inst, output_dir)
            # Append result immediately (crash-safe)
            with open(results_path, 'a') as f:
                f.write(json.dumps(res) + '\n')
            status = res['status']
            elapsed = res.get('duration_seconds', '?')
            print(f'[{datetime.utcnow().isoformat()}] DONE  {inst["instance_id"]} '
                  f'status={status} elapsed={elapsed}s')
            return res

    tasks = [run_with_sem(inst) for inst in pending]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Summary
    completed = sum(1 for r in results
                    if isinstance(r, dict) and r.get('status') == 'completed')
    errors    = sum(1 for r in results
                    if isinstance(r, dict) and r.get('status') == 'error')
    print(f'\n=== Evaluation complete ===')
    print(f'  Completed : {completed}')
    print(f'  Errors    : {errors}')
    print(f'  Results   : {results_path}')


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    global CONFIG_FILE, DATASET_PATH, PARSER_BASE_URL, PARSER_MODEL, PARSER_API_KEY
    parser = argparse.ArgumentParser(description='Run SWE-bench evaluation with OpenHands+Daytona')
    parser.add_argument('--dataset', default=DATASET_PATH, help='Local SWE-bench Verified parquet')
    parser.add_argument('--instances', type=int, default=None,
                        help='Number of instances to evaluate (default: all 500)')
    parser.add_argument('--start-idx', type=int, default=0,
                        help='Start index in dataset (for sharding)')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR,
                        help='Directory to store results and trajectories')
    parser.add_argument('--concurrency', type=int, default=1,
                        help='Max concurrent evaluations (default: 1)')
    parser.add_argument('--instance-id', type=str, default=None,
                        help='Run a single specific instance by ID')
    parser.add_argument('--config', type=str, default=CONFIG_FILE,
                        help='Path to config.toml file')
    parser.add_argument('--parser-base-url', type=str, default=None,
                        help='Base URL for external parser LLM (OpenAI-compatible). '
                             'Leave unset to disable external recovery; the controller '
                             'will fall back to retrying the main model.')
    parser.add_argument('--parser-model', type=str, default=None,
                        help='Model name for external parser LLM (set together with --parser-base-url to enable)')
    parser.add_argument('--parser-api-key', type=str, default=None,
                        help='API key for external parser LLM (set together with --parser-base-url to enable)')
    args = parser.parse_args()

    CONFIG_FILE = args.config
    DATASET_PATH = args.dataset
    if not Path(DATASET_PATH).is_file():
        parser.error(f'Dataset not found: {DATASET_PATH}; pass --dataset')
    if not Path(CONFIG_FILE).is_file():
        parser.error(f'OpenHands config not found: {CONFIG_FILE}')
    PARSER_BASE_URL = args.parser_base_url
    PARSER_MODEL = args.parser_model
    PARSER_API_KEY = args.parser_api_key

    # Validate environment
    missing = [k for k in REQUIRED_ENV if not os.getenv(k)]
    if missing:
        print(f'ERROR: Missing required environment variables: {missing}')
        print('Set DAYTONA_API_KEY before running.')
        sys.exit(1)

    # Load dataset
    df = pd.read_parquet(DATASET_PATH)
    instances = df.to_dict('records')

    if args.instance_id:
        instances = [i for i in instances if i['instance_id'] == args.instance_id]
        if not instances:
            print(f'ERROR: instance_id {args.instance_id!r} not found in dataset')
            sys.exit(1)
    else:
        instances = instances[args.start_idx:]
        if args.instances is not None:
            instances = instances[:args.instances]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'SWE-bench Verified Evaluation')
    print(f'  Dataset      : {DATASET_PATH}')
    print(f'  Instances    : {len(instances)}')
    print(f'  Config       : {CONFIG_FILE}')
    print(f'  Output dir   : {output_dir}')
    print(f'  Runtime      : daytona')
    print(f'  Sandbox CPU  : {os.getenv("DAYTONA_SANDBOX_CPU")} vCPU')
    print(f'  Sandbox Mem  : {os.getenv("DAYTONA_SANDBOX_MEMORY_GIB")} GiB')
    print(f'  Sandbox Disk : {os.getenv("DAYTONA_SANDBOX_DISK_GIB")} GiB')
    print(f'  Model        : qwen3-coder-30b-a3b-instruct (local vLLM)')
    print(f'  Temperature  : {TEMPERATURE}')
    print(f'  Top-p        : {TOP_P}')
    print(f'  Max tokens   : {MAX_TOKENS}')
    print(f'  Max iters    : {MAX_ITERATIONS}')
    print(f'  Parser URL   : {PARSER_BASE_URL or "disabled"}')
    print(f'  Parser model : {PARSER_MODEL or "disabled"}')
    print()

    asyncio.run(run_evaluation(instances, output_dir, args.concurrency))


if __name__ == '__main__':
    main()
