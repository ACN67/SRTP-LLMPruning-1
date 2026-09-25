"""Shared pruning interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PruningRequest:
    project_model_id: str
    method: str
    sparsity: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.sparsity < 1.0:
            raise ValueError("sparsity must be in the half-open interval [0, 1)")


class BasePruner(ABC):
    """Contract implemented by each verified pruning method."""

    method_id: str

    @abstractmethod
    def prune(self, model: Any, request: PruningRequest) -> Any:
        """Return a pruned model or artifact."""

        raise NotImplementedError
