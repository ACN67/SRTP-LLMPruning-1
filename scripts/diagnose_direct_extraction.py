#!/usr/bin/env python3
"""Read old Pilot records and report extraction differences without modifying them."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.direct_evaluation import get_benchmark  # noqa: E402


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.input_root.rglob("generations.jsonl")):
        benchmark_name = path.relative_to(args.input_root).parts[0]
        benchmark = get_benchmark(benchmark_name)
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            old = record["processed_generation"]
            new = benchmark.postprocess_generation(record["raw_generation"])
            rows.append({
                "file": str(path), "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "task_id": record["task_id"], "raw_sha256": sha(record["raw_generation"]),
                "old_sha256": sha(old), "new_sha256": sha(new),
                "changed": old != new, "old_nonempty": bool(old), "new_nonempty": bool(new),
                "new_head": new[:160],
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(rows), "changed": sum(x["changed"] for x in rows),
                      "new_nonempty": sum(x["new_nonempty"] for x in rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
