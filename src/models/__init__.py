"""Model configuration and architecture adapters."""

from .base import BaseModelAdapter, ModelSpec
from .loader import LoadOptions, LoadedModel, load_dense_model
from .validation_manifest import build_model_manifest
from .registry import get_model_adapter, list_model_ids, load_model_spec
from .snapshots import (
    load_snapshot_manifest,
    snapshot_manifest_sha256,
    verify_runtime_snapshot,
)

__all__ = [
    "BaseModelAdapter",
    "LoadOptions",
    "LoadedModel",
    "ModelSpec",
    "build_model_manifest",
    "get_model_adapter",
    "list_model_ids",
    "load_dense_model",
    "load_model_spec",
    "load_snapshot_manifest",
    "snapshot_manifest_sha256",
    "verify_runtime_snapshot",
]
