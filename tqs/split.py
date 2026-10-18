"""Source-stratified Top/Bottom/Random selection from a TQS score table.

Inputs are supplied through --scores and --source-dir. The default recipe
filters by success, true tokenized length, step count, and component coverage.
Top/Bottom preserve task/repository diversity; Random samples within sources.
The historical HCR quota is opt-in, not part of the default arXiv recipe.

Outputs under --output-dir: top/, bottom/, random/, and split_summary.json.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ───────────────────────────── config ──────────────────────────────────────
SCORES = "/path/to/tqs/results/coderforge_scores.parquet"
SOURCE_DIR = "/path/to/CoderForge-Preview/trajectories"
OUT_BASE = Path("/path/to/tqs/data/trajectories-TQS-32k-1k")

TOP_K = 1024
MAX_TRAJ_LEN = 32768
RANDOM_SEED = 42

# Optional historical HCR diversity quota for Top-K.
# The arXiv sampler does not impose an HCR quota. Retain it as an opt-in
# experiment for compatibility with earlier exploratory runs.
MAX_IDEAL_FRAC = None

# Per-task hard cap inside a single split — prevents Pass-3 fill from stacking
# many runs of the same task once unique tasks are exhausted.
MAX_PER_TASK = 30

# Per-source proportion: lock to eligible-pool ratios (computed at runtime).
# Override here to force a different mix (e.g. {"SWE_Smith":1/3,"SWE_Rebench":1/3,"R2E_Gym":1/3}).
SOURCE_QUOTA: dict[str, float] | None = None

_TRAJ_RE = re.compile(r"^(?P<prefix>.+?)_[0-9a-f]{40}_run\d+$")
_RUN_RE = re.compile(r"_run\d+$")


def parse_task_repo(trajectory_id: str) -> tuple[str, str]:
    s = str(trajectory_id)
    m = _TRAJ_RE.match(s)
    task_key = m.group("prefix") if m else _RUN_RE.sub("", s)
    parts = task_key.split("_")
    repo_key = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else task_key
    return task_key, repo_key


# ───────────────────────────── selectors ───────────────────────────────────

def select_diverse_in_source(
    pool: pd.DataFrame,
    k: int,
    score_col: str,
    ideal_col: str,
    descending: bool,
    max_ideal_frac: float | None = None,
) -> pd.DataFrame:
    """Pick k rows from `pool` (already source-filtered) with:
        Pass 1: per-task uniqueness (best score in each task).
        Pass 2: per-repo coverage on remaining slots.
        Pass 3: fill by score.
    With an optional cap on the fraction of ideal_pattern=True (only meaningful
    for top selection — bottom usually has very few ideal_pattern=True rows).
    """
    if pool.empty or k <= 0:
        return pool.head(0)

    pool = pool.sort_values([score_col, "trajectory_id"], ascending=[not descending, True]).reset_index(drop=True)
    parsed = pool["trajectory_id"].map(parse_task_repo)
    pool = pool.assign(
        task_key=parsed.map(lambda x: x[0]),
        repo_key=parsed.map(lambda x: x[1]),
    )

    cap_ideal = int(round(k * max_ideal_frac)) if max_ideal_frac is not None else None
    n_ideal = 0
    per_task_count: dict[str, int] = {}

    chosen_idx: list[int] = []
    chosen_set: set[str] = set()
    seen_tasks: set[str] = set()
    seen_repos: set[str] = set()

    def _can_take(row) -> bool:
        if row.trajectory_id in chosen_set:
            return False
        if cap_ideal is not None and bool(getattr(row, ideal_col)) and n_ideal >= cap_ideal:
            return False
        if MAX_PER_TASK is not None and per_task_count.get(row.task_key, 0) >= MAX_PER_TASK:
            return False
        return True

    # Pass 1: task uniqueness (in score order). Cap on ideal_pattern is enforced
    # by skipping new ideal rows once the budget is reached, which lets non-ideal
    # rows take the spillover slots — but we do NOT pre-emptively prefer them.
    order = list(pool.itertuples(index=False))

    for row in order:
        if len(chosen_idx) >= k:
            break
        if row.task_key in seen_tasks:
            continue
        if not _can_take(row):
            continue
        chosen_set.add(row.trajectory_id)
        seen_tasks.add(row.task_key)
        seen_repos.add(row.repo_key)
        per_task_count[row.task_key] = per_task_count.get(row.task_key, 0) + 1
        n_ideal += 1 if bool(getattr(row, ideal_col)) else 0
        chosen_idx.append(row.Index if hasattr(row, "Index") else None)

    # Pass 2: repo coverage on remaining
    if len(chosen_set) < k:
        for row in pool.itertuples(index=False):
            if len(chosen_set) >= k:
                break
            if not _can_take(row):
                continue
            if row.repo_key in seen_repos:
                continue
            chosen_set.add(row.trajectory_id)
            seen_tasks.add(row.task_key)
            seen_repos.add(row.repo_key)
            per_task_count[row.task_key] = per_task_count.get(row.task_key, 0) + 1
            n_ideal += 1 if bool(getattr(row, ideal_col)) else 0

    # Pass 3: fill by score
    if len(chosen_set) < k:
        for row in pool.itertuples(index=False):
            if len(chosen_set) >= k:
                break
            if not _can_take(row):
                continue
            chosen_set.add(row.trajectory_id)
            per_task_count[row.task_key] = per_task_count.get(row.task_key, 0) + 1
            n_ideal += 1 if bool(getattr(row, ideal_col)) else 0

    # Pass 4 fallback: if cap prevented us from filling k, relax the cap.
    if len(chosen_set) < k:
        logger.warning("ideal_pattern cap left selection short of k=%d (have %d). "
                       "Relaxing the cap.", k, len(chosen_set))
        for row in pool.itertuples(index=False):
            if len(chosen_set) >= k:
                break
            if row.trajectory_id in chosen_set:
                continue
            if MAX_PER_TASK is not None and per_task_count.get(row.task_key, 0) >= MAX_PER_TASK:
                continue
            chosen_set.add(row.trajectory_id)
            per_task_count[row.task_key] = per_task_count.get(row.task_key, 0) + 1

    return pool[pool["trajectory_id"].isin(chosen_set)].drop(columns=["task_key", "repo_key"])


def stratified_select(
    eligible: pd.DataFrame,
    k_total: int,
    score_col: str,
    ideal_col: str,
    descending: bool,
    source_col: str = "src",
    max_ideal_frac: float | None = None,
    quota: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Pick k_total rows total, with a fixed per-source quota."""
    if quota is None:
        share = eligible[source_col].value_counts(normalize=True).to_dict()
    else:
        share = quota

    # Normalise quota
    s = sum(share.values())
    share = {k: v / s for k, v in share.items()}

    # Per-source target counts (largest-remainder rounding)
    raw = {k: share[k] * k_total for k in share}
    base = {k: int(np.floor(v)) for k, v in raw.items()}
    rem = k_total - sum(base.values())
    leftovers = sorted(share.keys(), key=lambda k: raw[k] - base[k], reverse=True)
    for k in leftovers[:rem]:
        base[k] += 1

    parts = []
    for src, n in base.items():
        if n <= 0:
            continue
        sub = eligible[eligible[source_col] == src]
        if sub.empty:
            logger.warning("source %s has 0 eligible rows; skipping", src)
            continue
        picked = select_diverse_in_source(
            sub, n, score_col=score_col, ideal_col=ideal_col,
            descending=descending, max_ideal_frac=max_ideal_frac,
        )
        if len(picked) < n:
            logger.warning("source %s short: requested %d got %d", src, n, len(picked))
        parts.append(picked)

    return pd.concat(parts, ignore_index=True) if parts else eligible.head(0)


# ───────────────────────────── pipeline ────────────────────────────────────

def main() -> None:
    global SCORES, SOURCE_DIR, OUT_BASE, TOP_K, MAX_TRAJ_LEN, RANDOM_SEED, MAX_IDEAL_FRAC
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Construct source-stratified Top/Bottom/Random TQS subsets")
    parser.add_argument("--scores", required=True, help="Scored parquet from python -m tqs")
    parser.add_argument("--source-dir", required=True, help="Directory containing source trajectory parquet shards")
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory")
    parser.add_argument("--k", type=int, default=1024, help="Number of trajectories per subset")
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-ideal-frac", type=float, default=None, help="Optional historical HCR quota; disabled in the paper recipe")
    args = parser.parse_args()
    if args.k < 1 or args.max_tokens < 1:
        parser.error("--k and --max-tokens must be positive")
    if args.max_ideal_frac is not None and not 0 <= args.max_ideal_frac <= 1:
        parser.error("--max-ideal-frac must be between zero and one")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("--output-dir must be empty to avoid mixing selections from different runs")
    SCORES, SOURCE_DIR, OUT_BASE = args.scores, args.source_dir, args.output_dir
    TOP_K, MAX_TRAJ_LEN, RANDOM_SEED = args.k, args.max_tokens, args.seed
    MAX_IDEAL_FRAC = args.max_ideal_frac

    # Source map (filename prefix → source name)
    src_files = sorted(glob.glob(os.path.join(SOURCE_DIR, "*.parquet")))
    if not src_files:
        raise RuntimeError(f"No source shards in {SOURCE_DIR}")

    tid2src: dict[str, str] = {}
    reward1: set[str] = set()
    for p in src_files:
        name = os.path.basename(p)
        src_name = name.split("-")[0]
        d = pd.read_parquet(p, columns=["trajectory_id", "reward"])
        for t in d["trajectory_id"].tolist():
            tid2src.setdefault(t, src_name)
        reward1.update(d.loc[d["reward"] == 1, "trajectory_id"].tolist())

    logger.info("source map: %d ids; reward=1: %d ids", len(tid2src), len(reward1))

    # Load scores
    s = pd.read_parquet(SCORES)
    needed = {"trajectory_id", "tqs_score", "total_tokens", "total_steps", "tqs_coverage"}
    missing = needed - set(s.columns)
    if missing:
        raise RuntimeError(f"Score table missing columns: {missing}")

    e = s.dropna(subset=["trajectory_id", "tqs_score", "total_tokens"]).copy()
    e = e[e["trajectory_id"].isin(reward1)]
    e = e[e["total_tokens"] <= MAX_TRAJ_LEN]
    e = e[(e["total_steps"] >= 3) & (e["tqs_coverage"] >= 4 / 7)]
    if len(e) < TOP_K:
        raise ValueError(f"Only {len(e)} eligible rows for k={TOP_K}. Supply true token counts and check coverage/length filters.")
    e["src"] = e["trajectory_id"].map(tid2src).fillna("OTHER")
    e = e.sort_values(["tqs_score", "trajectory_id"], ascending=[False, True]).drop_duplicates("trajectory_id", keep="first").reset_index(drop=True)

    # Use False if ideal_pattern is missing — diversity quota then doesn't bite.
    if "m7_ideal_pattern" not in e:
        e["m7_ideal_pattern"] = False
    e["m7_ideal_pattern"] = e["m7_ideal_pattern"].fillna(False).astype(bool)

    logger.info("eligible pool: %d trajectories", len(e))
    logger.info("eligible source mix: %s", e["src"].value_counts(normalize=True).round(3).to_dict())

    # Top: stratified + ideal-pattern cap
    top = stratified_select(
        e, TOP_K, "tqs_score", "m7_ideal_pattern",
        descending=True, max_ideal_frac=MAX_IDEAL_FRAC, quota=SOURCE_QUOTA,
    )
    # Bottom: stratified, no ideal cap
    bot = stratified_select(
        e, TOP_K, "tqs_score", "m7_ideal_pattern",
        descending=False, max_ideal_frac=None, quota=SOURCE_QUOTA,
    )
    # Random: stratified, sampled within each source
    quota_for_random = SOURCE_QUOTA or e["src"].value_counts(normalize=True).to_dict()
    s_norm = sum(quota_for_random.values())
    quota_for_random = {k: v / s_norm for k, v in quota_for_random.items()}
    rng = np.random.default_rng(RANDOM_SEED)
    rnd_parts = []
    raw = {k: quota_for_random[k] * TOP_K for k in quota_for_random}
    base = {k: int(np.floor(v)) for k, v in raw.items()}
    rem = TOP_K - sum(base.values())
    for k in sorted(quota_for_random.keys(), key=lambda k: raw[k] - base[k], reverse=True)[:rem]:
        base[k] += 1
    for src, n in base.items():
        if n <= 0:
            continue
        sub = e[e["src"] == src]
        if len(sub) < n:
            n = len(sub)
        idx = rng.choice(len(sub), size=n, replace=False)
        rnd_parts.append(sub.iloc[idx])
    rnd = pd.concat(rnd_parts, ignore_index=True)
    for name, selected in [("top", top), ("bottom", bot), ("random", rnd)]:
        if len(selected) != TOP_K:
            raise ValueError(f"{name} selection has {len(selected)} rows instead of {TOP_K}; lower --k or inspect source quotas/task caps")

    # Reporting
    summary = {
        "TOP_K": TOP_K,
        "MAX_TRAJ_LEN": MAX_TRAJ_LEN,
        "MAX_IDEAL_FRAC": MAX_IDEAL_FRAC,
        "SOURCE_QUOTA": SOURCE_QUOTA or "matches eligible pool",
        "eligible": {
            "n": int(len(e)),
            "tqs_mean": float(e["tqs_score"].mean()),
            "tqs_p10": float(e["tqs_score"].quantile(0.10)),
            "tqs_p90": float(e["tqs_score"].quantile(0.90)),
            "src_mix": e["src"].value_counts(normalize=True).round(3).to_dict(),
            "ideal_pattern_rate": float(e["m7_ideal_pattern"].mean()),
        },
    }
    for name, d in [(f"top{TOP_K}", top), (f"bottom{TOP_K}", bot), (f"random{TOP_K}", rnd)]:
        summary[name] = {
            "n": int(len(d)),
            "tqs_mean": float(d["tqs_score"].mean()),
            "tqs_min": float(d["tqs_score"].min()),
            "tqs_max": float(d["tqs_score"].max()),
            "src_mix": d["src"].value_counts(normalize=True).round(3).to_dict(),
            "ideal_pattern_rate": float(d["m7_ideal_pattern"].mean()),
            "mean_total_steps": float(d["total_steps"].mean()),
            "mean_total_tokens": float(d["total_tokens"].mean()),
        }
    logger.info("\n%s", json.dumps(summary, indent=2, ensure_ascii=False))

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    with open(OUT_BASE / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Materialise per-shard parquet rows
    target_dirs = {
        "top": (top, OUT_BASE / "top"),
        "bottom": (bot, OUT_BASE / "bottom"),
        "random": (rnd, OUT_BASE / "random"),
    }
    for _, (_, p) in target_dirs.items():
        p.mkdir(parents=True, exist_ok=True)

    id_sets = {name: set(d["trajectory_id"]) for name, (d, _) in target_dirs.items()}
    written = {name: 0 for name in target_dirs}

    for src_path in src_files:
        fname = os.path.basename(src_path)
        df = pd.read_parquet(src_path)
        df_unique = df.drop_duplicates(subset=["trajectory_id"], keep="first")
        for name, (sel, out_dir) in target_dirs.items():
            keep = df_unique[df_unique["trajectory_id"].isin(id_sets[name])]
            if not keep.empty:
                keep.to_parquet(out_dir / fname, index=False)
                written[name] += len(keep)
                # mark these IDs as written so a later shard with the same row
                # (CoderForge has dup trajectory_ids across shards) doesn't re-emit it.
                id_sets[name] -= set(keep["trajectory_id"])

    print("\n=== written ===")
    for name, n in written.items():
        print(f"  {name}: {n}")
    print("\nOutput root:", OUT_BASE)


if __name__ == "__main__":
    main()
