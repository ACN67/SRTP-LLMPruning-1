"""Pruning interfaces and registered methods."""

from .base import BasePruner, ModulePruningStats, PruningRequest, PruningSummary
from .block_removal import ratio_to_block_count, remove_blocks, retained_indices
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
from .tabp import (
    MEASURE_DIRECTION,
    SUPPORTED_SSN_MEASURES,
    SUPPORTED_TABP_MEASURES,
    TaBPPruner,
    TaBPPruningSummary,
    TaBPRanking,
    aggregate_ddf_scores,
    aggregate_ssn_scores,
    rank_blocks_ddf,
    rank_blocks_ssn,
)
from .tabp_calibration import (
    ARC_EASY_REVISION,
    ARCEasyTaBPCalibrationProvider,
    TaBPCalibrationConfig,
    TaBPCalibrationContext,
    TaBPCalibrationSample,
    WikiTextTaBPCalibrationProvider,
    format_arc_easy,
    get_tabp_calibration_provider,
)
from .wanda import WandaActivationStats, WandaPruner, WandaPruningSummary, wanda_mask

PRUNER_REGISTRY: dict[str, type[BasePruner]] = {
    "magnitude": MagnitudePruner,
    "wanda": WandaPruner,
    "sparsegpt": SparseGPTPruner,
    "sleb": SLEBPruner,
    "tabp": TaBPPruner,
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
    "SUPPORTED_SSN_MEASURES",
    "SUPPORTED_TABP_MEASURES",
    "MEASURE_DIRECTION",
    "SparseGPTCoreResult",
    "SparseGPTHessian",
    "SparseGPTPruner",
    "SparseGPTPruningSummary",
    "TaBPCalibrationConfig",
    "TaBPCalibrationContext",
    "TaBPCalibrationSample",
    "TaBPPruner",
    "TaBPPruningSummary",
    "TaBPRanking",
    "aggregate_ddf_scores",
    "aggregate_ssn_scores",
    "WandaActivationStats",
    "WandaPruningContext",
    "WandaPruningSummary",
    "WikiText2SLEBCalibrationProvider",
    "ARC_EASY_REVISION",
    "ARCEasyTaBPCalibrationProvider",
    "WikiTextTaBPCalibrationProvider",
    "format_arc_easy",
    "get_sleb_calibration_provider",
    "get_tabp_calibration_provider",
    "greedy_block_search",
    "get_pruner",
    "get_calibration_provider",
    "sparsegpt_reconstruct",
    "rank_blocks_ssn",
    "rank_blocks_ddf",
    "ratio_to_block_count",
    "remove_blocks",
    "retained_indices",
    "ratio_to_remove_count",
    "sleb_get_loss",
    "temporary_block_removal",
    "wanda_mask",
]
