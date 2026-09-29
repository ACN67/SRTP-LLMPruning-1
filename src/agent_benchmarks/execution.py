"""Dataset loading, deterministic selection and benchmark generation lifecycle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .base import DatasetSpec, repository_root


def _load_local(path: Path, split: str) -> list[dict[str, Any]]:
    candidate = path / f"{split}.jsonl" if path.is_dir() else path
    if candidate.suffix == ".jsonl":
        return [json.loads(line) for line in candidate.read_text(encoding="utf-8").splitlines() if line.strip()]
    if candidate.suffix == ".json":
        value = json.loads(candidate.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise ValueError("Local benchmark JSON must contain a list")
        return [dict(item) for item in value]
    try:
        from datasets import load_from_disk
    except ImportError as error:
        raise RuntimeError("datasets is required for saved-to-disk benchmark snapshots") from error
    loaded = load_from_disk(str(path))
    dataset = loaded[split] if hasattr(loaded, "keys") and split in loaded else loaded
    return [dict(item) for item in dataset]


def load_dataset_records(spec: DatasetSpec, *, offline: bool, local_path: Path | None = None, validate_expected_count: bool = True) -> list[dict[str, Any]]:
    path = local_path or ((repository_root() / spec.local_path).expanduser() if spec.local_path else None)
    if path is not None and path.exists():
        records = _load_local(path, spec.split)
    elif offline:
        raise FileNotFoundError("Offline mode requires an existing local dataset snapshot")
    else:
        try:
            from datasets import load_dataset
        except ImportError as error:
            raise RuntimeError("datasets is required to load a Hugging Face benchmark") from error
        records = [dict(item) for item in load_dataset(spec.repo_id, revision=spec.revision, split=spec.split)]
    if validate_expected_count and len(records) != spec.expected_task_count:
        raise ValueError(f"Expected {spec.expected_task_count} instances, loaded {len(records)}")
    return records


def select_records(records: Iterable[Mapping[str, Any]], *, instance_ids: Iterable[str] = (), limit: int | None = None) -> list[dict[str, Any]]:
    values = [dict(item) for item in records]
    ids = [str(item.get("instance_id", "")) for item in values]
    if any(not item for item in ids) or len(ids) != len(set(ids)):
        raise ValueError("Dataset instance_id values must be non-empty and unique")
    requested = tuple(dict.fromkeys(instance_ids))
    if requested:
        by_id = dict(zip(ids, values))
        missing = [item for item in requested if item not in by_id]
        if missing:
            raise KeyError(f"Unknown benchmark instance IDs: {missing}")
        values = [by_id[item] for item in requested]
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        values = values[:limit]
    return values
