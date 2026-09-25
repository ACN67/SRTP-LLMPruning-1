"""Common model metadata and architecture-adapter contracts."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ModelSpec:
    """Stable project metadata for one upstream model repository."""

    project_model_id: str
    display_name: str
    huggingface_repo_id: str
    architecture: str
    adapter: str
    revision: str
    trust_remote_code: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ModelSpec":
        required = (
            "project_model_id",
            "display_name",
            "huggingface_repo_id",
            "architecture",
            "adapter",
            "revision",
        )
        missing = [key for key in required if not data.get(key)]
        if missing:
            raise ValueError(f"Missing model configuration keys: {', '.join(missing)}")
        return cls(
            project_model_id=str(data["project_model_id"]),
            display_name=str(data["display_name"]),
            huggingface_repo_id=str(data["huggingface_repo_id"]),
            architecture=str(data["architecture"]),
            adapter=str(data["adapter"]),
            revision=str(data["revision"]),
            trust_remote_code=bool(data.get("trust_remote_code", False)),
        )


class BaseModelAdapter(ABC):
    """Architecture-specific model access contract.

    Implementations must be checked against the concrete upstream model and the
    selected pruning implementation before loading weights.
    """

    adapter_id: str
    architecture: str

    def validate(self, spec: ModelSpec) -> None:
        if spec.adapter != self.adapter_id:
            raise ValueError(
                f"Model {spec.project_model_id!r} requests adapter {spec.adapter!r}, "
                f"not {self.adapter_id!r}."
            )

    @abstractmethod
    def load_model(
        self,
        spec: ModelSpec,
        *,
        model_root: Path,
        cache_dir: Path,
    ) -> Any:
        """Load a model after compatibility is verified."""

        raise NotImplementedError
