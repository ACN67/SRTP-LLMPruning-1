"""Reproducibility manifest construction for model validation and experiments."""

from __future__ import annotations

import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping

from .base import BaseModelAdapter, ModelSpec
from .loader import LoadOptions, LoadedModel


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _cuda_version() -> str | None:
    try:
        import torch
    except ImportError:
        return None
    return getattr(getattr(torch, "version", None), "cuda", None)


def _git_metadata(repository_root: Path) -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repository_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def _serializable_device_map(device_map: Any) -> Any:
    if isinstance(device_map, Mapping):
        return {str(key): str(value) for key, value in device_map.items()}
    return device_map


def build_model_manifest(
    spec: ModelSpec,
    adapter: BaseModelAdapter,
    *,
    repository_root: Path,
    mode: str,
    status: str,
    options: LoadOptions | None = None,
    loaded: LoadedModel | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable record without guessing unavailable values."""

    options = options or LoadOptions()
    git_commit, git_dirty = _git_metadata(repository_root)
    local_path = options.local_path or spec.local_path
    is_local = local_path is not None
    structure = dict(loaded.structure) if loaded else {
        **spec.expected_structure(),
        "actual_block_count": None,
    }
    resolved_revision = loaded.resolved_revision if loaded else (
        None if is_local else spec.revision
    )
    config = loaded.config if loaded else None
    model = loaded.model if loaded else None
    tokenizer = loaded.tokenizer if loaded else None

    return {
        "schema_version": 2,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "status": status,
        "project_model_id": spec.project_model_id,
        "display_name": spec.display_name,
        "hf_repo": spec.huggingface_repo_id,
        "model_source": loaded.source if loaded else (
            str(local_path) if local_path is not None else spec.huggingface_repo_id
        ),
        "requested_revision": spec.revision,
        "resolved_revision": resolved_revision,
        "architecture": spec.architecture,
        "model_type": spec.model_type,
        "adapter": adapter.adapter_id,
        "trust_remote_code": spec.trust_remote_code,
        "dtype": loaded.dtype if loaded else (options.dtype or spec.default_dtype),
        "device": options.device,
        "device_map": _serializable_device_map(options.device_map),
        "cache_dir": str(options.cache_dir) if options.cache_dir else None,
        "local_files_only": options.local_files_only,
        "torch_version": _package_version("torch"),
        "transformers_version": _package_version("transformers"),
        "cuda_version": _cuda_version(),
        "python_version": platform.python_version(),
        "tokenizer_class": type(tokenizer).__name__ if tokenizer is not None else None,
        "model_class": type(model).__name__ if model is not None else None,
        "expected_model_class": spec.expected_model_class,
        "num_hidden_layers": structure.get("num_hidden_layers"),
        "actual_block_count": structure.get("actual_block_count"),
        "hidden_size": structure.get("hidden_size"),
        "intermediate_size": structure.get("intermediate_size"),
        "num_attention_heads": structure.get("num_attention_heads"),
        "num_key_value_heads": structure.get("num_key_value_heads"),
        "config_class": type(config).__name__ if config is not None else None,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
    }
