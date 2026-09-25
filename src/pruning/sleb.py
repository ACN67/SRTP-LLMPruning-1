"""SLEB placeholder."""

from typing import Any

from .base import BasePruner, PruningRequest


class SLEBPruner(BasePruner):
    method_id = "sleb"

    def prune(self, model: Any, request: PruningRequest) -> Any:
        raise NotImplementedError("SLEB has not yet been verified and adapted.")
