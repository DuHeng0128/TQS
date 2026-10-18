"""Convert selected CoderForge parquet rows to ms-swift agent-SFT JSONL."""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def decoded(value):
    return json.loads(value) if isinstance(value, str) else value


def convert_messages(raw):
    raw = decoded(raw)
    if not isinstance(raw, list) or not raw:
        raise ValueError("Expected a nonempty messages list")
    messages = []
    for message in raw:
        role = message["role"]
        content = message.get("content") or ""
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        if role in ("system", "user", "assistant"):
            if content.strip():
                messages.append({"role": role, "content": content})
            if role == "assistant":
                for call in decoded(message.get("tool_calls")) or []:
                    function = call.get("function") or call
                    name = function.get("name")
                    if not name:
                        raise ValueError("Tool call has no function name")
                    arguments = function.get("arguments")
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            pass
                    payload = {"name": name, "arguments": arguments if arguments is not None else {}}
                    messages.append({"role": "tool_call", "content": json.dumps(payload, ensure_ascii=False)})
        elif role in ("tool", "tool_response", "function"):
            messages.append({"role": "tool_response", "content": content})
        elif role == "tool_call":
            messages.append({"role": role, "content": content})
        else:
            raise ValueError(f"Unsupported message role: {role}")
    if not any(message["role"] in ("assistant", "tool_call") for message in messages):
        raise ValueError("Trajectory has no assistant training target")
    return messages


def convert_row(row):
    sample = {"messages": convert_messages(row.get("messages"))}
    tools = decoded(row.get("tools"))
    if tools:
        if not isinstance(tools, list):
            tools = [tools]
        tools = [decoded(tool) for tool in tools]
        if any(not isinstance(tool, dict) for tool in tools):
            raise ValueError("Invalid tool schema")
        sample["tools"] = tools
    sample["meta"] = {"trajectory_id": str(row["trajectory_id"])}
    return sample


def read_ids(path):
    ids = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("ID manifest must be nonempty and contain no duplicates")
    return ids


def prepare(source, output, ids_file=None):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    paths = sorted(source.rglob("*.parquet")) if source.is_dir() else [source]
    if not paths or any(not path.is_file() for path in paths):
        raise FileNotFoundError(f"No parquet inputs found: {source}")
    requested = read_ids(ids_file) if ids_file else None
    wanted = set(requested) if requested else None
    records = {}
    for path in paths:
        parquet = pq.ParquetFile(path)
        columns = [key for key in ("trajectory_id", "messages", "tools") if key in parquet.schema_arrow.names]
        if "trajectory_id" not in columns or "messages" not in columns:
            raise ValueError(f"Not a raw trajectory shard: {path}")
        # Read IDs first to avoid loading unrelated, potentially large histories.
        for group in range(parquet.num_row_groups):
            group_ids = parquet.read_row_group(group, columns=["trajectory_id"]).column(0).to_pylist()
            indexes = [i for i, value in enumerate(group_ids)
                       if (wanted is None or str(value) in wanted) and str(value) not in records]
            if not indexes:
                continue
            offset = 0
            for batch in parquet.iter_batches(batch_size=32, row_groups=[group], columns=columns):
                local = [index - offset for index in indexes if offset <= index < offset + batch.num_rows]
                offset += batch.num_rows
                if not local:
                    continue
                for row in batch.take(local).to_pylist():
                    trajectory_id = str(row["trajectory_id"])
                    if row["trajectory_id"] is None or not trajectory_id.strip():
                        raise ValueError(f"Empty trajectory ID in {path}")
                    if trajectory_id not in records:
                        records[trajectory_id] = json.dumps(convert_row(row), ensure_ascii=False, allow_nan=False)
    if requested:
        missing = wanted - records.keys()
        if missing:
            raise ValueError(f"Missing {len(missing)} requested trajectories (first: {sorted(missing)[0]})")
        order = requested
    else:
        order = sorted(records)
    if not order:
        raise ValueError("No trajectories selected")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for trajectory_id in order:
            stream.write(records[trajectory_id] + "\n")
    return len(order)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Raw parquet shard or directory")
    parser.add_argument("--ids", type=Path, help="Optional ordered trajectory ID manifest")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(f"Wrote {prepare(args.source, args.output, args.ids)} trajectories to {args.output}")


if __name__ == "__main__":
    main()
