#!/usr/bin/env python3
"""Compare aligned dense/pruned Direct benchmark outcomes."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.analysis import paired_bootstrap_ci  # noqa: E402


PAIRING_PROVENANCE_FIELDS = (
    "benchmark",
    "project_model_id",
    "evaluation_profile_id",
    "prompt_sha256",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-outcomes", type=Path, required=True)
    parser.add_argument("--pruned-outcomes", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser


def _read_outcomes(
    path: Path,
) -> tuple[dict[str, float], dict[tuple[str, int], dict[str, str]]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_identity: dict[tuple[str, int], float] = {}
    provenance_by_identity: dict[tuple[str, int], dict[str, str]] = {}
    by_task: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Every outcome row must be a JSON object")
        if not isinstance(row.get("passed"), bool):
            raise ValueError("Outcome field 'passed' must be boolean")
        identity = (str(row["task_id"]), int(row["trial_index"]))
        if not identity[0] or identity[1] < 0:
            raise ValueError("Outcome identity requires a non-empty task_id and non-negative trial_index")
        if identity in by_identity:
            raise ValueError(f"Duplicate outcome identity: {identity}")
        provenance: dict[str, str] = {}
        for field in PAIRING_PROVENANCE_FIELDS:
            value = row.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"Outcome field {field!r} must be a non-empty string")
            provenance[field] = value
        value = float(row["passed"])
        by_identity[identity] = value
        provenance_by_identity[identity] = provenance
        by_task[identity[0]].append(value)
    return (
        {task_id: statistics.fmean(values) for task_id, values in by_task.items()},
        provenance_by_identity,
    )


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    dense, dense_provenance = _read_outcomes(args.dense_outcomes)
    pruned, pruned_provenance = _read_outcomes(args.pruned_outcomes)
    if set(dense_provenance) != set(pruned_provenance):
        raise ValueError("Dense/pruned (task_id, trial_index) identities must match exactly")
    for identity in dense_provenance:
        for field in PAIRING_PROVENANCE_FIELDS:
            if dense_provenance[identity][field] != pruned_provenance[identity][field]:
                raise ValueError(
                    "Dense/pruned provenance must match exactly for "
                    f"{identity}: field {field!r} differs"
                )
    result = paired_bootstrap_ci(
        dense,
        pruned,
        resamples=args.resamples,
        seed=args.seed,
    ).to_dict()
    result["unit"] = "task"
    result["trial_identity_alignment"] = "exact"
    result["provenance_alignment"] = "exact"
    return result


def main() -> int:
    args = _parser().parse_args()
    result = analyze(args)
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.expanduser().resolve().write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
