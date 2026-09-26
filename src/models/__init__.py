"""Model configuration and architecture adapters."""

from .base import BaseModelAdapter, ModelSpec
from .artifacts import load_model_artifact
from .loader import LoadOptions, LoadedModel, load_dense_model
from .manifest import build_model_manifest
from .replay import (
    BlockCallContext,
    NativeCalibrationCapture,
    capture_native_calibration,
    replay_captured_block,
)
from .registry import get_model_adapter, list_model_ids, load_model_spec
from .snapshots import (
    load_snapshot_manifest,
    snapshot_manifest_sha256,
    verify_runtime_snapshot,
)

__all__ = [
    "BaseModelAdapter",
    "BlockCallContext",
    "LoadOptions",
    "LoadedModel",
    "ModelSpec",
    "NativeCalibrationCapture",
    "build_model_manifest",
    "capture_native_calibration",
    "get_model_adapter",
    "list_model_ids",
    "load_dense_model",
    "load_model_artifact",
    "load_model_spec",
    "load_snapshot_manifest",
    "replay_captured_block",
    "snapshot_manifest_sha256",
    "verify_runtime_snapshot",
]
