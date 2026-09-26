"""Granite structural adapter."""

from typing import Any

from ..base import BaseModelAdapter


class GraniteAdapter(BaseModelAdapter):
    adapter_id = "granite"
    architecture = "Granite"
    model_type = "granite"

    def get_backbone(self, model: Any) -> Any:
        return model.model

    def normalize_block_output(self, output: Any) -> Any:
        if not isinstance(output, tuple) or not output:
            raise TypeError("Granite decoder block must return a non-empty tuple")
        return output[0]
