"""Qwen3 structural adapter."""

from typing import Any

from ..base import BaseModelAdapter


class Qwen3Adapter(BaseModelAdapter):
    adapter_id = "qwen3"
    architecture = "Qwen3"
    model_type = "qwen3"

    def get_backbone(self, model: Any) -> Any:
        return model.model
