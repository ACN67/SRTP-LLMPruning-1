"""Stable artifact identity shared by transformations and evaluation paths."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


def _is_sha256(value: str) -> bool:
    if len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True



@dataclass(frozen=True)
class LineageOperation:
    operation: str
    method: str
    config_hash: str
    input_artifact_hash: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.operation or not self.method:
            raise ValueError("Lineage operation and method must be non-empty")
        if not _is_sha256(self.config_hash) or not _is_sha256(self.input_artifact_hash):
            raise ValueError("Lineage config/input hashes must be SHA256 hex identities")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LineageOperation":
        return cls(**dict(value))


@dataclass(frozen=True)
class ModelArtifact:
    path: str
    kind: str
    representation: str
    standalone: bool
    project_model_id: str
    architecture: str
    model_type: str
    adapter_id: str
    num_hidden_layers: int
    content_sha256: str
    lineage: tuple[LineageOperation, ...] = ()
    capabilities: Mapping[str, bool] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in {"dense", "pruned", "recovered"}:
            raise ValueError(f"Unsupported artifact kind: {self.kind!r}")
        if self.representation not in {"full_checkpoint", "peft_adapter"}:
            raise ValueError(f"Unsupported artifact representation: {self.representation!r}")
        if self.representation == "peft_adapter" and self.standalone:
            raise ValueError("A PEFT adapter cannot be marked standalone")
        if self.num_hidden_layers <= 0 or not _is_sha256(self.content_sha256):
            raise ValueError("Artifact depth/hash is invalid")
        operations = tuple(item.operation for item in self.lineage)
        if self.kind == "dense" and operations:
            raise ValueError("Dense artifacts cannot contain transformation lineage")
        if self.kind == "pruned" and "pruning" not in operations:
            raise ValueError("Pruned artifacts require a pruning lineage operation")
        if self.kind == "recovered" and "recovery" not in operations:
            raise ValueError("Recovered artifacts require a recovery lineage operation")
        if self.representation == "peft_adapter" and self.kind != "recovered":
            raise ValueError("PEFT adapter artifacts must be recovered artifacts")

    @property
    def is_weight_sparse(self) -> bool:
        return any(
            item.operation == "pruning"
            and item.provenance.get("structure_effect") == "weight_sparse"
            for item in self.lineage
        )

    @property
    def pruning_structure_effect(self) -> str:
        for item in reversed(self.lineage):
            if item.operation == "pruning":
                value = item.provenance.get("structure_effect")
                if isinstance(value, str) and value:
                    return value
        return "none"

    @property
    def pruning_method(self) -> str:
        for item in reversed(self.lineage):
            if item.operation == "pruning":
                return item.method
        return "dense"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["lineage"] = [asdict(item) for item in self.lineage]
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ModelArtifact":
        data = dict(value)
        data["lineage"] = tuple(LineageOperation.from_dict(item) for item in data.get("lineage", ()))
        return cls(**data)
