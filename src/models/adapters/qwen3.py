"""Qwen3 structural adapter."""

from typing import Any, Sequence

from ..base import BaseModelAdapter


class Qwen3Adapter(BaseModelAdapter):
    adapter_id = "qwen3"
    architecture = "Qwen3"
    model_type = "qwen3"

    def get_backbone(self, model: Any) -> Any:
        return model.model

    def normalize_block_output(self, output: Any) -> Any:
        if isinstance(output, tuple):
            raise TypeError("Qwen3 decoder block unexpectedly returned a tuple")
        return output

    def capture_block_removal_metadata(self, model: Any) -> Any:
        layer_types = getattr(model.config, "layer_types", None)
        return tuple(layer_types) if layer_types is not None else None

    def finalize_block_removal(
        self,
        model: Any,
        retained_original_indices: Sequence[int],
        metadata: Any = None,
    ) -> None:
        super().finalize_block_removal(model, retained_original_indices, metadata)
        if metadata is not None:
            model.config.layer_types = [
                metadata[index] for index in retained_original_indices
            ]
