"""SparseGPT placeholder."""

from typing import Any

from src.models.base import BaseModelAdapter

from .base import BasePruner, PruningRequest


class SparseGPTPruner(BasePruner):
    method_id = "sparsegpt"

    def prune(
        self, model: Any, adapter: BaseModelAdapter, request: PruningRequest
    ) -> Any:
        raise NotImplementedError("SparseGPT has not yet been verified and adapted.")
