"""Pruning interfaces and registered methods."""

from .base import BasePruner, ModulePruningStats, PruningRequest, PruningSummary
from .calibration import (
    C4CalibrationProvider,
    CalibrationContext,
    CalibrationConfig,
    CalibrationSample,
    WandaPruningContext,
    get_calibration_provider,
)
from .magnitude import MagnitudePruner
from .sleb import (
    SLEBPruner,
    SLEBPruningSummary,
    SLEBSearchResult,
    greedy_block_search,
    ratio_to_remove_count,
    sleb_get_loss,
    temporary_block_removal,
)
from .sleb_calibration import (
    SLEBCalibrationConfig,
    SLEBCalibrationContext,
    WikiText2SLEBCalibrationProvider,
    get_sleb_calibration_provider,
)
from .sparsegpt import (
    SparseGPTCoreResult,
    SparseGPTHessian,
    SparseGPTPruner,
    SparseGPTPruningSummary,
    sparsegpt_reconstruct,
)
from .wanda import WandaActivationStats, WandaPruner, WandaPruningSummary, wanda_mask

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
    "C4CalibrationProvider",
    "CalibrationContext",
    "CalibrationConfig",
    "CalibrationSample",
    "ModulePruningStats",
    "PRUNER_REGISTRY",
    "PruningRequest",
    "PruningSummary",
    "SLEBCalibrationConfig",
    "SLEBCalibrationContext",
    "SLEBPruner",
    "SLEBPruningSummary",
    "SLEBSearchResult",
    "SparseGPTCoreResult",
    "SparseGPTHessian",
    "SparseGPTPruner",
    "SparseGPTPruningSummary",
    "WandaActivationStats",
    "WandaPruningContext",
    "WandaPruningSummary",
    "WikiText2SLEBCalibrationProvider",
    "get_sleb_calibration_provider",
    "greedy_block_search",
    "get_pruner",
    "get_calibration_provider",
    "sparsegpt_reconstruct",
    "ratio_to_remove_count",
    "sleb_get_loss",
    "temporary_block_removal",
    "wanda_mask",
]
