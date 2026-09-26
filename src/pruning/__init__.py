"""Pruning interfaces and registered methods."""

from .base import BasePruner, ModulePruningStats, PruningRequest, PruningSummary
from .magnitude import MagnitudePruner
from .sleb import SLEBPruner
from .sparsegpt import SparseGPTPruner
from .wanda import WandaPruner

PRUNER_REGISTRY: dict[str, type[BasePruner]] = {
    "magnitude": MagnitudePruner,
    "wanda": WandaPruner,
    "sparsegpt": SparseGPTPruner,
    "sleb": SLEBPruner,
}


def get_pruner(method_id: str) -> BasePruner:
    try:
        return PRUNER_REGISTRY[method_id]()
    except KeyError as error:
        available = ", ".join(PRUNER_REGISTRY)
        raise KeyError(f"Unknown pruner {method_id!r}; available: {available}") from error


__all__ = [
    "BasePruner",
    "ModulePruningStats",
    "PRUNER_REGISTRY",
    "PruningRequest",
    "PruningSummary",
    "get_pruner",
]
