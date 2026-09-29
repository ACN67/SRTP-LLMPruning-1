"""Pinned SWT-Bench inference/evaluation dataset identities and derivation."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class SWTEvaluationDatasetSpec:
    repo_id: str
    revision: str
    split: str
    source_expected_task_count: int
    expected_task_count: int
    source_local_path: str
    local_path: str
    filter_path: str
    filter_expected_count: int
    filter_sha256: str
    output_sha256: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SWTEvaluationDatasetSpec":
        spec = cls(**dict(value))
        if not re.fullmatch(r"[0-9a-f]{40}", spec.revision):
            raise ValueError("SWT evaluation dataset revision must be immutable")
        if not re.fullmatch(r"[0-9a-f]{64}", spec.filter_sha256):
            raise ValueError("SWT filter SHA256 must be pinned")
        if not re.fullmatch(r"[0-9a-f]{64}", spec.output_sha256):
            raise ValueError("SWT derived output SHA256 must be pinned")
        if spec.source_expected_task_count - spec.filter_expected_count != spec.expected_task_count:
            raise ValueError("SWT source/filter/derived counts are inconsistent")
        return spec


def dataset_payload(rows: Sequence[Mapping[str, Any]]) -> bytes:
    """Serialize snapshots exactly like setup's Hugging Face materializer."""
    return json.dumps(list(rows), ensure_ascii=False, sort_keys=True).encode("utf-8")


def instance_ids(rows: Sequence[Mapping[str, Any]], label: str) -> list[str]:
    ids = [str(row["instance_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label} contains duplicate instance_id values")
    return ids


def read_filter_ids(path: Path, expected_count: int, expected_sha256: str) -> set[str]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"SWT filter SHA256 is {digest}, expected {expected_sha256}")
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(values) != len(set(values)):
        raise ValueError("SWT filter contains duplicate instance IDs")
    if len(values) != expected_count:
        raise ValueError(f"SWT filter contains {len(values)} IDs, expected {expected_count}")
    return set(values)


def derive_evaluation_rows(
    source_rows: Sequence[Mapping[str, Any]],
    inference_rows: Sequence[Mapping[str, Any]],
    filter_ids: set[str],
    spec: SWTEvaluationDatasetSpec,
) -> list[Mapping[str, Any]]:
    source_ids = instance_ids(source_rows, "SWE-bench Verified source")
    inference_ids = instance_ids(inference_rows, "SWT ZSP inference snapshot")
    if len(source_rows) != spec.source_expected_task_count:
        raise ValueError(f"SWT evaluation source has {len(source_rows)} rows, expected {spec.source_expected_task_count}")
    if not filter_ids <= set(source_ids):
        raise ValueError("SWT filter contains IDs absent from the evaluation source")
    derived = [row for row in source_rows if str(row["instance_id"]) not in filter_ids]
    derived_ids = instance_ids(derived, "SWT derived evaluation snapshot")
    if len(derived) != spec.expected_task_count:
        raise ValueError(f"SWT derived evaluation snapshot has {len(derived)} rows, expected {spec.expected_task_count}")
    if set(derived_ids) != set(source_ids) - filter_ids:
        raise ValueError("SWT derived IDs do not equal source IDs minus filter IDs")
    if set(derived_ids) != set(inference_ids):
        raise ValueError("SWT derived evaluation IDs do not equal ZSP inference IDs")
    digest = hashlib.sha256(dataset_payload(derived)).hexdigest()
    if digest != spec.output_sha256:
        raise ValueError(f"SWT derived output SHA256 is {digest}, expected {spec.output_sha256}")
    return derived
