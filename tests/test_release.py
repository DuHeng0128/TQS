import copy
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tqs import DEFAULT_WEIGHTS, compute_tqs
from tqs.cli import _score
from tqs.rerank import recompute
from tqs.trajectory_metrics.metrics.m6_psn import PSNMetric
from tqs.trajectory_metrics.parsers.base import ParsedStep, ParsedTrajectory

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/trajectory.json"


class ScoreTests(unittest.TestCase):
    def test_paper_example_component_aggregation(self):
        # Component values from arXiv Table 7, Run 2 (rounded at source).
        components = dict(oec=0.7384, iac=0.6667, psn=0.8491, tte=0.5718, scp=0.9904, ped=1.0, dpi=1.0)
        score = sum(DEFAULT_WEIGHTS[name] * value for name, value in components.items())
        self.assertAlmostEqual(sum(DEFAULT_WEIGHTS.values()), 1.0)
        self.assertAlmostEqual(score, 0.7798, places=4)

    def test_missing_components_renormalize(self):
        result = compute_tqs({"m2_oec": {"min_relative_entropy": 0.2}})
        self.assertAlmostEqual(result["tqs_score"], 0.8)
        self.assertAlmostEqual(result["coverage"], 1 / 7)
        self.assertNotIn("ped", result["components"])

    def test_scalar_and_vector_agree(self):
        raw = {
            "m1_rbs": {"early_stop_triggered": True, "rbs_loop_fraction": 0.02},
            "m2_oec": {"min_relative_entropy": 0.3, "error_repeat_rate": 0.1},
            "m4_dcs": {"action_type_consistency": 0.7},
            "m5_bus": {"post_error_diversity": 0.6},
            "m6_psn": {"psn_late": 0.4},
            "m9_tte": {"tte_score": 2.0},
            "m10_scp": {"scp": 0.3, "never_committed": False},
        }
        scalar = compute_tqs(raw)
        row = {f"tqs_{name}": value for name, value in scalar["components"].items()}
        row.update(m1_early_stop=True, m1_rbs_loop_fraction=0.02, m2_error_repeat_rate=0.1, m10_never_committed=False)
        vector = recompute(pd.DataFrame([row])).iloc[0]
        self.assertAlmostEqual(scalar["tqs_score"], vector["tqs_score"])

    def test_constant_scope_is_neutral(self):
        steps = [ParsedStep(i, "read", "read_file", {}, "text", ["src/a.py"], 10) for i in range(12)]
        trajectory = ParsedTrajectory("constant", None, steps, len(steps), "test")
        result = PSNMetric().compute(trajectory)
        self.assertEqual(result["psn_late"], 0.0)

    def test_outcome_label_does_not_change_score(self):
        row = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        passing = copy.deepcopy(row)
        passing["reward"] = 1.0
        failing = copy.deepcopy(row)
        failing["reward"] = 0.0
        first = _score((EXAMPLE, passing, "coderforge"))
        second = _score((EXAMPLE, failing, "coderforge"))
        self.assertEqual(first["tqs_score"], second["tqs_score"])
        self.assertNotEqual(first["resolved"], second["resolved"])
        self.assertIsNone(first["total_tokens"])
        self.assertGreater(first["estimated_tokens"], 0)

    def test_json_and_parquet_cli_match(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            json_output = temp / "scores.jsonl"
            subprocess.run([sys.executable, "-m", "tqs", "--input", str(EXAMPLE), "--output", str(json_output)], cwd=ROOT, check=True, capture_output=True, text=True)
            json_score = json.loads(json_output.read_text(encoding="utf-8"))
            self.assertTrue(math.isfinite(json_score["tqs_score"]))
            row = json.loads(EXAMPLE.read_text(encoding="utf-8"))
            row["total_tokens"] = 512  # synthetic test metadata, not a tokenizer measurement
            source = temp / "source.parquet"
            pd.DataFrame([row]).to_parquet(source, index=False)
            table_output = temp / "scores.parquet"
            subprocess.run([sys.executable, "-m", "tqs", "--input", str(source), "--output", str(table_output), "--workers", "2"], cwd=ROOT, check=True, capture_output=True, text=True)
            table_score = pd.read_parquet(table_output).iloc[0]
            self.assertAlmostEqual(table_score["tqs_score"], json_score["tqs_score"])
            self.assertEqual(table_score["total_tokens"], 512)
            retry = subprocess.run([sys.executable, "-m", "tqs", "--input", str(EXAMPLE), "--output", str(json_output)], cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(retry.returncode, 0)

    def test_split_filters_and_materializes_source_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            source = temp / "source"
            source.mkdir()
            ids = [f"owner_repo{i}-1_run1" for i in range(14)]
            rows = pd.DataFrame({"trajectory_id": ids, "reward": [1] * 14, "messages": ["[]"] * 14})
            rows.to_parquet(source / "SWE_Smith-0000.parquet", index=False)
            scores = pd.DataFrame({
                "trajectory_id": ids,
                "tqs_score": np.linspace(0.1, 0.9, 14),
                "total_tokens": [512] * 13 + [40000],
                "total_steps": [8] * 12 + [1, 8],
                "tqs_coverage": [1.0] * 14,
            })
            scores.to_parquet(temp / "scores.parquet", index=False)
            output = temp / "subsets"
            subprocess.run([sys.executable, "-m", "tqs.split", "--scores", str(temp / "scores.parquet"), "--source-dir", str(source), "--output-dir", str(output), "--k", "4"], cwd=ROOT, check=True, capture_output=True, text=True)
            selections = {}
            for name in ("top", "bottom", "random"):
                selected = pd.concat([pd.read_parquet(path) for path in (output / name).glob("*.parquet")])
                self.assertEqual(len(selected), 4)
                self.assertFalse(set(selected["trajectory_id"]) & set(ids[12:]))
                selections[name] = set(selected["trajectory_id"])
            self.assertEqual(selections["top"], set(ids[8:12]))
            self.assertEqual(selections["bottom"], set(ids[:4]))
            self.assertTrue((output / "split_summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
