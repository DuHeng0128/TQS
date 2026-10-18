import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "evaluation"))
from prepare_dataset import convert_row, prepare, read_ids
from run_verify import to_predictions
from tqs.cli import _score


class TrainingTests(unittest.TestCase):
    def test_manifests_match_supplied_release(self):
        checksums = {
            "top": "56e93901cf5e18519147c0c6f42d68fe3ad5d706815206b82504df8d0761a5ff",
            "bottom": "f5a16c1459ca9a7894477b95ba462efaf4e6c85b6127b5023a76735d6e1a1d3e",
            "random": "94a49d3a52514f9ed93eaec2cebaec632469c28d107e80f80e3f9dfa33d064f2",
        }
        for name, checksum in checksums.items():
            path = ROOT / "training/splits" / f"{name}.txt"
            self.assertEqual(len(read_ids(path)), 1024)
            # Normalize Git's platform-specific text checkout convention.
            data = path.read_bytes().replace(b"\r\n", b"\n")
            self.assertEqual(hashlib.sha256(data).hexdigest(), checksum)

    def test_bundled_data_matches_manifest_order_and_schema(self):
        expected = {"top.parquet", "bottom.parquet", "random.parquet"}
        self.assertEqual({path.name for path in (ROOT / "data").glob("*.parquet")}, expected)
        for split in ("top", "bottom", "random"):
            path = ROOT / "data" / f"{split}.parquet"
            self.assertLess(path.stat().st_size, 50 * 1024 * 1024)
            schema = pq.ParquetFile(path).schema_arrow
            self.assertEqual(schema.names, ["trajectory_id", "finish_reason", "image", "messages", "reward", "tools", "license"])
            rows = pq.read_table(path, columns=["trajectory_id", "reward", "license"]).to_pylist()
            self.assertEqual([row["trajectory_id"] for row in rows], read_ids(ROOT / "training/splits" / f"{split}.txt"))
            self.assertEqual(len(rows), 1024)
            self.assertTrue(all(row["reward"] == 1.0 and row["license"] for row in rows))

    def test_assistant_and_tool_call_conversion(self):
        row = {"trajectory_id": "demo", "messages": [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Fix"},
            {"role": "assistant", "content": "Inspect", "tool_calls": [
                {"function": {"name": "read_file", "arguments": '{"path":"a.py"}'}}]},
            {"role": "tool", "content": "file"},
            {"role": "assistant", "content": "Done"}],
            "tools": json.dumps([{"type": "function", "function": {"name": "read_file"}}])}
        sample = convert_row(row)
        self.assertEqual([item["role"] for item in sample["messages"]],
                         ["system", "user", "assistant", "tool_call", "tool_response", "assistant"])
        self.assertEqual(json.loads(sample["messages"][3]["content"])["arguments"], {"path": "a.py"})
        self.assertEqual(sample["tools"][0]["function"]["name"], "read_file")

    def test_parquet_selection_order_dedup_and_missing_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [{"trajectory_id": key, "messages": json.dumps([
                {"role": "user", "content": "Fix"}, {"role": "assistant", "content": key}])}
                for key in ("a", "b", "a", "unselected")]
            pq.write_table(pa.Table.from_pylist(rows), root / "raw.parquet", row_group_size=2)
            ids = root / "ids.txt"
            ids.write_text("b\na\n", encoding="utf-8")
            output = root / "out.jsonl"
            self.assertEqual(prepare(root / "raw.parquet", output, ids), 2)
            selected = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["meta"]["trajectory_id"] for row in selected], ["b", "a"])
            with self.assertRaises(FileExistsError):
                prepare(root / "raw.parquet", output, ids)
            ids.write_text("missing\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                prepare(root / "raw.parquet", root / "missing.jsonl", ids)
            self.assertFalse((root / "missing.jsonl").exists())

    def test_verification_keeps_failures_and_rejects_duplicates(self):
        rows = [{"instance_id": "a", "patch": "diff --git ..."}, {"instance_id": "b", "patch": None}]
        predictions = to_predictions(rows, "top-tqs")
        self.assertEqual(len(predictions), 2)
        self.assertEqual(predictions[1]["model_patch"], "")
        self.assertEqual(predictions[0]["model_name_or_path"], "top-tqs")
        with self.assertRaises(ValueError):
            to_predictions(rows + rows, "top-tqs")

    def test_openhands_event_parser(self):
        events = [{"id": 1, "source": "agent", "action": "read", "args": {"path": "/testbed/a.py"}},
                  {"id": 2, "source": "environment", "observation": "read", "cause": 1, "content": "code"},
                  {"id": 3, "source": "agent", "action": "finish", "args": {}}]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "events.json"
            path.write_text(json.dumps(events), encoding="utf-8")
            row = _score((path, events, "openhands_json"))
            self.assertEqual(row["total_steps"], 2)
            self.assertEqual(row["trajectory_id"], "events")


if __name__ == "__main__":
    unittest.main()
