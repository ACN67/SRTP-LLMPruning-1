"""Runtime snapshot manifest loading and verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from src.downloads import verify_file

DEFAULT_SNAPSHOT_MANIFEST_DIR = (
    Path(__file__).resolve().parents[2] / "configs" / "model_snapshots"
)


def _validate_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("Runtime file path must be a non-empty string")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Runtime file path must be repository-relative: {value!r}")
    return value


def _validate_runtime_file(item: object) -> dict:
    if not isinstance(item, dict):
        raise ValueError("Runtime file entries must be mappings")
    required = ("path", "size", "hash_type", "expected_hash")
    missing = [key for key in required if key not in item]
    if missing:
        raise ValueError(f"Runtime file entry is missing keys: {', '.join(missing)}")
    path = _validate_relative_path(item["path"])
    size = int(item["size"])
    if size < 0:
        raise ValueError(f"Runtime file size must be non-negative: {path}")
    hash_type = str(item["hash_type"])
    if hash_type not in {"sha256", "git_blob_sha1"}:
        raise ValueError(f"Unsupported runtime file hash type: {hash_type!r}")
    expected_hash = str(item["expected_hash"])
    if not expected_hash:
        raise ValueError(f"Runtime file expected hash is empty: {path}")
    return {
        "path": path,
        "size": size,
        "hash_type": hash_type,
        "expected_hash": expected_hash,
    }


def _validate_manifest(model_id: str, data: object) -> dict:
    if not isinstance(data, dict):
        raise ValueError("Snapshot manifest must be a mapping")
    required = (
        "schema_version",
        "project_model_id",
        "canonical_hf_repo",
        "canonical_hf_revision",
        "domestic_modelscope_repo",
        "required_runtime_files",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Snapshot manifest is missing keys: {', '.join(missing)}")
    if data["schema_version"] != 1:
        raise ValueError(f"Unsupported snapshot manifest schema: {data['schema_version']!r}")
    if data["project_model_id"] != model_id:
        raise ValueError(
            f"Snapshot manifest project_model_id {data['project_model_id']!r} "
            f"does not match {model_id!r}"
        )
    files = data["required_runtime_files"]
    if not isinstance(files, list) or not files:
        raise ValueError("Snapshot manifest must declare runtime files")
    normalized = [_validate_runtime_file(item) for item in files]
    paths = [item["path"] for item in normalized]
    if len(paths) != len(set(paths)):
        raise ValueError("Snapshot manifest contains duplicate runtime file paths")
    result = dict(data)
    result["required_runtime_files"] = normalized
    return result


def load_snapshot_manifest(
    model_id: str,
    manifest_dir: Path = DEFAULT_SNAPSHOT_MANIFEST_DIR,
) -> dict:
    """Load and validate the pinned runtime snapshot manifest for one model."""

    path = manifest_dir / f"{model_id}.json"
    if not path.is_file():
        available = ", ".join(sorted(item.stem for item in manifest_dir.glob("*.json"))) or "none"
        raise KeyError(f"Unknown snapshot manifest {model_id!r}; available: {available}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return _validate_manifest(model_id, data)


def snapshot_manifest_sha256(
    model_id: str,
    manifest_dir: Path = DEFAULT_SNAPSHOT_MANIFEST_DIR,
) -> str:
    """Return the SHA256 of the manifest file bytes used for provenance."""

    path = manifest_dir / f"{model_id}.json"
    if not path.is_file():
        raise KeyError(f"Unknown snapshot manifest {model_id!r}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_runtime_files(manifest: dict, paths: Iterable[str] | None = None) -> list[dict]:
    """Select runtime file records from a manifest, preserving manifest order."""

    files = list(manifest.get("required_runtime_files", ()))
    if paths is None:
        return files
    requested = tuple(paths)
    requested_set = set(requested)
    selected = [item for item in files if item["path"] in requested_set]
    found = {item["path"] for item in selected}
    missing = [path for path in requested if path not in found]
    if missing:
        raise KeyError(f"Snapshot manifest does not contain runtime files: {', '.join(missing)}")
    return selected


def verify_runtime_snapshot(
    model_id: str,
    root: Path,
    *,
    paths: Iterable[str] | None = None,
) -> list[dict]:
    """Verify all or selected runtime files under a local model snapshot."""

    manifest = load_snapshot_manifest(model_id)
    selected = select_runtime_files(manifest, paths)
    root = root.expanduser().resolve()
    for item in selected:
        verify_file(
            root / item["path"],
            size=int(item["size"]),
            hash_type=item["hash_type"],
            expected_hash=item["expected_hash"],
        )
    return selected
