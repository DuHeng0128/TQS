"""rerank.py — Recompute TQS-V2 over the V1 score table without rerunning M1–M10.

The V1 scoring run wrote `Code_Trajectory/results/coderforge_scores.parquet`
with one row per trajectory, containing every raw metric we need:

    tqs_oec, tqs_iac, tqs_psn, tqs_tte, tqs_scp, tqs_ped       (V1 normalised)
    m1_early_stop, m1_rbs_loop_fraction, m2_error_repeat_rate, m10_never_committed

V2 keeps the same per-component normalisation as V1 (so we can reuse tqs_*
columns directly), drops HCR, and adds a DPI component derived from M1/M2/M10.
We renormalise over the components that are non-null per row, matching the
fail-soft semantics of compute_tqs.

Outputs:
    TQS/results/coderforge_scores.parquet     (full table + tqs_score, tqs_dpi)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tqs import DEFAULT_WEIGHTS, _DPI_EARLY_STOP_PENALTY, _DPI_NEVER_COMMIT_PENALTY, \
    _DPI_LOOP_FRAC_SCALE, _DPI_LOOP_FRAC_CAP, _DPI_ERR_REPEAT_SCALE, _DPI_ERR_REPEAT_CAP

logger = logging.getLogger(__name__)


def compute_dpi_vec(df: pd.DataFrame) -> pd.Series:
    """Vectorised DPI from V1 flat columns. Returns float in [0,1] or NaN."""
    es = df["m1_early_stop"].astype("boolean")
    nc = df["m10_never_committed"].astype("boolean")
    lf = pd.to_numeric(df["m1_rbs_loop_fraction"], errors="coerce").clip(lower=0)
    er = pd.to_numeric(df["m2_error_repeat_rate"], errors="coerce").clip(lower=0)

    has_signal = es.notna() | nc.notna() | lf.notna() | er.notna()

    penalty = pd.Series(0.0, index=df.index, dtype="float64")
    penalty = penalty + np.where(es.fillna(False), _DPI_EARLY_STOP_PENALTY, 0.0)
    penalty = penalty + np.where(nc.fillna(False), _DPI_NEVER_COMMIT_PENALTY, 0.0)
    penalty = penalty + np.minimum(_DPI_LOOP_FRAC_CAP, lf.fillna(0.0) * _DPI_LOOP_FRAC_SCALE)
    penalty = penalty + np.minimum(_DPI_ERR_REPEAT_CAP, er.fillna(0.0) * _DPI_ERR_REPEAT_SCALE)

    dpi = (1.0 - penalty).clip(lower=0.0, upper=1.0)
    dpi[~has_signal] = np.nan
    return dpi


def recompute(df: pd.DataFrame) -> pd.DataFrame:
    """Add tqs_dpi and tqs_score columns to df."""
    df = df.copy()
    df["tqs_dpi"] = compute_dpi_vec(df)

    comps = {
        "oec": df["tqs_oec"].astype(float),
        "iac": df["tqs_iac"].astype(float),
        "psn": df["tqs_psn"].astype(float),
        "tte": df["tqs_tte"].astype(float),
        "scp": df["tqs_scp"].astype(float),
        "ped": df["tqs_ped"].astype(float),
        "dpi": df["tqs_dpi"].astype(float),
    }
    weighted_sum = pd.Series(0.0, index=df.index)
    weight_total = pd.Series(0.0, index=df.index)
    for name, w in DEFAULT_WEIGHTS.items():
        v = comps[name]
        avail = v.notna()
        weighted_sum = weighted_sum + np.where(avail, v.fillna(0.0) * w, 0.0)
        weight_total = weight_total + np.where(avail, w, 0.0)

    score = np.where(weight_total > 0, weighted_sum / weight_total.replace(0, np.nan), np.nan)
    df["tqs_score"] = pd.Series(score, index=df.index).clip(lower=0.0, upper=1.0)
    return df


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Recompute TQS V2 without rerunning M1-M10")
    ap.add_argument("--input", default="/path/to/results/coderforge_scores.parquet")
    ap.add_argument("--output", default="/path/to/tqs/results/coderforge_scores.parquet")
    args = ap.parse_args()

    inp = Path(args.input); out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Loading {inp}")
    df = pd.read_parquet(inp)
    logger.info(f"  {len(df)} rows, {len(df.columns)} cols")

    df = recompute(df)
    df.to_parquet(out, index=False)
    logger.info(f"Saved {out}")

    s = df["tqs_score"].dropna()
    print("\n=== TQS-V2 distribution (full 413K) ===")
    print(f"n={len(s)}  mean={s.mean():.4f}  std={s.std():.4f}  min={s.min():.4f}  max={s.max():.4f}")
    print("percentiles:", {p: round(float(s.quantile(p/100)), 4) for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]})


if __name__ == "__main__":
    main()
