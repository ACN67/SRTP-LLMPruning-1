"""Canonical artifact manifest serialization and content identity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .base import ModelArtifact


ARTIFACT_MANIFEST_NAME = "artifact_manifest.json"
SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_inventory(path: Path) -> tuple[list[dict[str, Any]], str]:
    entries = []
    for item in sorted(entry for entry in path.rglob("*") if entry.is_file() and not entry.is_symlink()):
        relative = item.relative_to(path).as_posix()
        if relative in {ARTIFACT_MANIFEST_NAME, "recovery_manifest.json"}:
            continue
        entries.append({"path": relative, "size": item.stat().st_size, "sha256": sha256_file(item)})
    digest = hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return entries, digest


def write_artifact_manifest(path: Path, artifact: ModelArtifact) -> Path:
    destination = path / ARTIFACT_MANIFEST_NAME
    payload = {"schema_version": SCHEMA_VERSION, "artifact": artifact.to_dict()}
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return destination


def read_artifact_manifest(path: Path) -> ModelArtifact:
    manifest_path = path / ARTIFACT_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing canonical artifact manifest: {manifest_path}")
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid artifact manifest JSON: {manifest_path}") from error
    if not isinstance(raw, Mapping):
        raise ValueError("Artifact manifest must contain a JSON object")
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported artifact manifest schema: {raw.get('schema_version')!r}")
    artifact_raw = raw.get("artifact")
    if not isinstance(artifact_raw, Mapping):
        raise ValueError("Artifact manifest is missing artifact mapping")
    artifact = ModelArtifact.from_dict(artifact_raw)
    if Path(artifact.path).expanduser().resolve() != path.expanduser().resolve():
        artifact = ModelArtifact.from_dict({**artifact.to_dict(), "path": str(path.resolve())})
    return artifact
