"""SLEB placeholder."""

from typing import Any

from src.models.base import BaseModelAdapter

from .base import BasePruner, PruningRequest


class SLEBPruner(BasePruner):
    method_id = "sleb"

    def prune(
        self,
        model: Any,
        adapter: BaseModelAdapter,
        request: PruningRequest,
        context: Any | None = None,
    ) -> Any:
        raise NotImplementedError("SLEB has not yet been verified and adapted.")
