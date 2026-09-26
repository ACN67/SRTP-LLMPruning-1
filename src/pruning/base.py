"""Shared pruning interfaces and serializable statistics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any

from src.models.base import BaseModelAdapter


@dataclass(frozen=True)
class PruningRequest:
    project_model_id: str
    method: str
    sparsity: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.sparsity < 1.0:
            raise ValueError("sparsity must be in the half-open interval [0, 1)")


@dataclass(frozen=True)
class ModulePruningStats:
    """Exact accounting for one targeted weight matrix."""

    module: str
    shape: tuple[int, ...]
    targeted_weights: int
    requested_mask_count: int
    preexisting_zeros: int
    newly_zeroed_weights: int
    post_pruning_zeros: int
    achieved_mask_sparsity: float
    achieved_zero_sparsity: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PruningSummary:
    """Aggregate pruning result suitable for an experiment manifest."""

    pruner: str
    implementation_version: str
    pruning_type: str
    scope: str
    sparsity_ratio: float
    target_policy: str
    excluded_components: tuple[str, ...]
    number_of_target_modules: int
    targeted_weights: int
    requested_mask_count: int
    preexisting_zeros: int
    newly_zeroed_weights: int
    post_pruning_zeros: int
    achieved_mask_sparsity: float
    achieved_zero_sparsity: float
    per_module: tuple[ModulePruningStats, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["excluded_components"] = list(self.excluded_components)
        result["per_module"] = [stats.to_dict() for stats in self.per_module]
        return result


class BasePruner(ABC):
    """Contract implemented by each verified pruning method."""

    method_id: str

    @abstractmethod
    def prune(
        self,
        model: Any,
        adapter: BaseModelAdapter,
        request: PruningRequest,
    ) -> PruningSummary:
        """Prune the model in place and return exact statistics."""

        raise NotImplementedError
