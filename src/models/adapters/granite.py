"""Granite structural adapter."""

from typing import Any

from ..base import BaseModelAdapter


class GraniteAdapter(BaseModelAdapter):
    adapter_id = "granite"
    architecture = "Granite"
    model_type = "granite"

    def get_backbone(self, model: Any) -> Any:
        return model.model
