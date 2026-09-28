"""Resolve raw and canonical model artifacts without loading weights."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.models.base import BaseModelAdapter, ModelSpec

from .base import ModelArtifact
from .manifest import ARTIFACT_MANIFEST_NAME, artifact_inventory, read_artifact_manifest


def _config(path: Path) -> dict[str, Any]:
    config_path = path / "config.json"
    if not config_path.is_file():
        raise ValueError(f"Artifact is missing config.json: {path}")
    value = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("config.json must contain an object")
    return value


def _validate_structure(config: dict[str, Any], spec: ModelSpec, depth: int) -> None:
    expected = {
        "model_type": spec.model_type,
        "hidden_size": spec.expected_hidden_size,
        "intermediate_size": spec.expected_intermediate_size,
        "num_attention_heads": spec.expected_num_attention_heads,
        "num_key_value_heads": spec.expected_num_key_value_heads,
        "num_hidden_layers": depth,
    }
    mismatch = [
        f"{key}={config.get(key)!r}, expected {value!r}"
        for key, value in expected.items()
        if config.get(key) != value
    ]
    if mismatch:
        raise ValueError("Artifact model/depth mismatch: " + "; ".join(mismatch))


def _validate_lineage_structure(artifact: ModelArtifact, spec: ModelSpec) -> None:
    effect = artifact.pruning_structure_effect
    if artifact.kind == "pruned" and effect not in {"weight_sparse", "reduced_depth"}:
        raise ValueError(
            "Pruned artifact lineage must declare provenance.structure_effect "
            "as 'weight_sparse' or 'reduced_depth'"
        )
    if effect == "weight_sparse" and artifact.num_hidden_layers != spec.expected_num_hidden_layers:
        raise ValueError(
            "Weight-sparse pruning cannot change model depth: "
            f"{artifact.num_hidden_layers} != {spec.expected_num_hidden_layers}"
        )
    if effect == "reduced_depth" and artifact.num_hidden_layers >= spec.expected_num_hidden_layers:
        raise ValueError(
            "Reduced-depth pruning must reduce model depth: "
            f"{artifact.num_hidden_layers} >= {spec.expected_num_hidden_layers}"
        )


def resolve_model_artifact(
    path: Path,
    spec: ModelSpec,
    adapter: BaseModelAdapter,
) -> ModelArtifact:
    root = path.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Artifact directory does not exist: {root}")
    _inventory, content_hash = artifact_inventory(root)

    if (root / ARTIFACT_MANIFEST_NAME).is_file():
        artifact = read_artifact_manifest(root)
        if artifact.content_sha256 != content_hash:
            raise ValueError("Artifact content hash does not match canonical manifest")
        if (
            artifact.project_model_id,
            artifact.architecture,
            artifact.model_type,
            artifact.adapter_id,
        ) != (
            spec.project_model_id,
            spec.architecture,
            spec.model_type,
            adapter.adapter_id,
        ):
            raise ValueError("Canonical artifact model identity mismatch")
        _validate_lineage_structure(artifact, spec)
        if artifact.representation == "full_checkpoint":
            _validate_structure(_config(root), spec, artifact.num_hidden_layers)
        return artifact

    config = _config(root)
    _validate_structure(config, spec, spec.expected_num_hidden_layers)
    weights = list(root.glob("*.safetensors")) + list(root.glob("*.bin"))
    if not weights:
        raise ValueError("Dense artifact contains no model weights")
    return ModelArtifact(
        str(root),
        "dense",
        "full_checkpoint",
        True,
        spec.project_model_id,
        spec.architecture,
        spec.model_type,
        adapter.adapter_id,
        spec.expected_num_hidden_layers,
        content_hash,
        (),
        {"direct_evaluation": True, "vllm_serving": True},
        {"manifest_status": "raw_dense_checkpoint"},
    )
