"""Deterministic, validated benchmark prediction persistence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


REQUIRED_FIELDS = {"instance_id", "model_name_or_path", "model_patch"}


def validate_prediction(value: Mapping[str, Any], *, allow_full_output: bool = True) -> dict[str, Any]:
    missing = REQUIRED_FIELDS - set(value)
    if missing:
        raise ValueError(f"Prediction is missing required fields: {sorted(missing)}")
    allowed = REQUIRED_FIELDS | ({"full_output"} if allow_full_output else set())
    extra = set(value) - allowed
    if extra:
        raise ValueError(f"Prediction has unsupported fields: {sorted(extra)}")
    result = dict(value)
    if not isinstance(result["model_patch"], str):
        raise TypeError("model_patch must be a string")
    if not result["instance_id"] or not result["model_name_or_path"]:
        raise ValueError("Prediction identities cannot be empty")
    return result


def prediction_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_predictions(path: Path, predictions: Iterable[Mapping[str, Any]]) -> str:
    records = [validate_prediction(item) for item in predictions]
    ids = [item["instance_id"] for item in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Each instance must occur exactly once in predictions")
    records.sort(key=lambda item: item["instance_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(item, sort_keys=True, ensure_ascii=False) + "\n" for item in records)
    path.write_text(content, encoding="utf-8")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def read_predictions(path: Path) -> list[dict[str, Any]]:
    records = [validate_prediction(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len({item["instance_id"] for item in records}) != len(records):
        raise ValueError("Duplicate instance_id in predictions file")
    return records
