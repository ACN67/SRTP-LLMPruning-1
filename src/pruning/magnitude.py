"""Magnitude Pruning placeholder."""

from typing import Any

from .base import BasePruner, PruningRequest


class MagnitudePruner(BasePruner):
    method_id = "magnitude"

    def prune(self, model: Any, request: PruningRequest) -> Any:
        raise NotImplementedError("Magnitude Pruning has not yet been verified and adapted.")
