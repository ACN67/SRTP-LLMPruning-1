"""Model configuration and architecture adapters."""

from .base import BaseModelAdapter, ModelSpec
from .loader import LoadOptions, LoadedModel, load_dense_model
from .manifest import build_model_manifest
from .replay import (
    BlockCallContext,
    NativeCalibrationCapture,
    capture_native_calibration,
    replay_captured_block,
)
from .registry import get_model_adapter, list_model_ids, load_model_spec

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
    "load_model_spec",
    "replay_captured_block",
]
