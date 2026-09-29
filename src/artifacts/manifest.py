"""Canonical artifact manifest serialization and content identity."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from src.utils.identity import canonical_sha256

from .base import ModelArtifact


ARTIFACT_MANIFEST_NAME = "artifact_manifest.json"
SCHEMA_VERSION = 2


def artifact_provenance_payload(artifact: ModelArtifact) -> dict[str, Any]:
    """Return path-independent scientific identity for a canonical manifest."""

    def without_location_hints(item: Any) -> Any:
        if isinstance(item, Mapping):
            artifact_mapping = "content_sha256" in item
            return {
                key: without_location_hints(child)
                for key, child in item.items()
                if key != "base_artifact_path"
                and not (key == "path" and artifact_mapping)
                and key != "manifest_provenance_sha256"
            }
        if isinstance(item, (list, tuple)):
            return [without_location_hints(child) for child in item]
        return item

    value = without_location_hints(artifact.to_dict())
    return {"schema_version": SCHEMA_VERSION, "artifact": value}


def artifact_provenance_sha256(artifact: ModelArtifact) -> str:
    return canonical_sha256(artifact_provenance_payload(artifact))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_inventory(path: Path) -> tuple[list[dict[str, Any]], str]:
    entries = []
    for item in sorted(path.rglob("*")):
        if item.is_symlink():
            raise ValueError(f"Canonical artifact directories must not contain symlinks: {item}")
        if not item.is_file():
            continue
        relative = item.relative_to(path).as_posix()
        if relative in {
            ARTIFACT_MANIFEST_NAME,
            "recovery_manifest.json",
            ".srtp_model_source.json",
        }:
            continue
        entries.append({"path": relative, "size": item.stat().st_size, "sha256": sha256_file(item)})
    digest = hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return entries, digest


def write_artifact_manifest(path: Path, artifact: ModelArtifact) -> Path:
    destination = path / ARTIFACT_MANIFEST_NAME
    artifact_value = artifact.to_dict()
    artifact_value.pop("manifest_provenance_sha256", None)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact": artifact_value,
        "artifact_provenance_sha256": artifact_provenance_sha256(artifact),
    }
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
        raise ValueError(
            f"Unsupported artifact manifest schema: {raw.get('schema_version')!r}; "
            "schema 1 artifacts lack manifest provenance protection and must be "
            "regenerated with the current writer before use"
        )
    artifact_raw = raw.get("artifact")
    if not isinstance(artifact_raw, Mapping):
        raise ValueError("Artifact manifest is missing artifact mapping")
    artifact = ModelArtifact.from_dict(artifact_raw)
    expected = artifact_provenance_sha256(artifact)
    actual = raw.get("artifact_provenance_sha256")
    if actual != expected:
        raise ValueError(
            "Artifact manifest provenance hash mismatch: lineage or identity metadata changed"
        )
    artifact = replace(artifact, manifest_provenance_sha256=expected)
    if Path(artifact.path).expanduser().resolve() != path.expanduser().resolve():
        artifact = replace(artifact, path=str(path.resolve()))
    return artifact
