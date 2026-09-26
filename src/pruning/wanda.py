"""Wanda placeholder."""

from typing import Any

from src.models.base import BaseModelAdapter

from .base import BasePruner, PruningRequest


class WandaPruner(BasePruner):
    method_id = "wanda"

    def prune(
        self, model: Any, adapter: BaseModelAdapter, request: PruningRequest
    ) -> Any:
        raise NotImplementedError("Wanda has not yet been verified and adapted.")
