"""Manifest-aware loading for dense and pruned model artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .base import BaseModelAdapter, ModelSpec
from .loader import LoadOptions, LoadedModel, _resolve_dtype, load_dense_model

PRUNED_MANIFEST_NAME = "pruning_manifest.json"
SAME_DEPTH_PRUNERS = {"magnitude", "wanda", "sparsegpt"}
REDUCED_DEPTH_PRUNERS = {"sleb", "tabp"}


def _runtime_imports() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Artifact loading requires torch and transformers; install requirements.txt first."
        ) from error
    return torch, AutoConfig, AutoModelForCausalLM, AutoTokenizer


def _read_manifest(path: Path) -> dict:
    manifest_path = path / PRUNED_MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValueError(f"Pruned artifact requires {PRUNED_MANIFEST_NAME}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{PRUNED_MANIFEST_NAME} must be a mapping")
    if data.get("schema_version") != 3:
        raise ValueError(f"Unsupported pruning manifest schema: {data.get('schema_version')!r}")
    return data


def _manifest_model(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    model = manifest.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("Pruning manifest is missing model record")
    return model


def _manifest_pruning(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    pruning = manifest.get("pruning")
    if not isinstance(pruning, Mapping):
        raise ValueError("Pruning manifest is missing pruning record")
    return pruning


def _validate_manifest_identity(
    manifest: Mapping[str, Any],
    spec: ModelSpec,
    adapter: BaseModelAdapter,
) -> tuple[str, int]:
    model = _manifest_model(manifest)
    expected = {
        "project_model_id": spec.project_model_id,
        "architecture": spec.architecture,
        "model_type": spec.model_type,
        "adapter": adapter.adapter_id,
    }
    for key, value in expected.items():
        if model.get(key) != value:
            raise ValueError(
                f"Pruning manifest {key}={model.get(key)!r} does not match {value!r}"
            )

    pruning = _manifest_pruning(manifest)
    pruner = pruning.get("pruner") or pruning.get("method")
    if not isinstance(pruner, str) or not pruner:
        raise ValueError("Pruning manifest is missing pruner")
    recorded_depth = pruning.get("post_pruning_actual_block_count")
    if recorded_depth is None:
        recorded_depth = model.get("actual_block_count")
    if recorded_depth is None:
        raise ValueError("Pruning manifest is missing post-pruning depth")
    recorded_depth = int(recorded_depth)
    if recorded_depth <= 0:
        raise ValueError("Pruning manifest depth must be positive")

    if pruner in SAME_DEPTH_PRUNERS and recorded_depth != spec.expected_num_hidden_layers:
        raise ValueError(
            f"Same-depth pruner {pruner!r} recorded depth {recorded_depth}, "
            f"expected {spec.expected_num_hidden_layers}"
        )
    if pruner not in SAME_DEPTH_PRUNERS | REDUCED_DEPTH_PRUNERS:
        raise ValueError(f"Unsupported pruning manifest pruner: {pruner!r}")
    return pruner, recorded_depth


def _validate_pruned_config(
    config: Any,
    spec: ModelSpec,
    *,
    recorded_depth: int,
) -> None:
    observed = {
        "model_type": getattr(config, "model_type", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "intermediate_size": getattr(config, "intermediate_size", None),
        "num_attention_heads": getattr(config, "num_attention_heads", None),
        "num_key_value_heads": getattr(config, "num_key_value_heads", None),
    }
    expected = {
        "model_type": spec.model_type,
        "hidden_size": spec.expected_hidden_size,
        "intermediate_size": spec.expected_intermediate_size,
        "num_attention_heads": spec.expected_num_attention_heads,
        "num_key_value_heads": spec.expected_num_key_value_heads,
    }
    mismatches = [
        f"{key}={observed[key]!r}, expected {value!r}"
        for key, value in expected.items()
        if observed[key] != value
    ]
    config_depth = getattr(config, "num_hidden_layers", None)
    if config_depth != recorded_depth:
        mismatches.append(f"num_hidden_layers={config_depth!r}, expected depth {recorded_depth!r}")
    if mismatches:
        raise ValueError("Loaded artifact config mismatch: " + "; ".join(mismatches))


def _load_pruned_artifact(
    spec: ModelSpec,
    adapter: BaseModelAdapter,
    artifact_path: Path,
    options: LoadOptions,
    *,
    recorded_depth: int,
) -> LoadedModel:
    if options.device is not None and options.device_map is not None:
        raise ValueError("Specify either device or device_map, not both")

    torch, AutoConfig, AutoModelForCausalLM, AutoTokenizer = _runtime_imports()
    dtype_name = options.dtype or spec.default_dtype
    resolved_dtype = _resolve_dtype(torch, dtype_name)
    source = str(artifact_path)
    common_kwargs: dict[str, Any] = {
        "trust_remote_code": spec.trust_remote_code,
        "local_files_only": options.local_files_only,
    }
    if options.cache_dir is not None:
        common_kwargs["cache_dir"] = str(options.cache_dir.expanduser())

    config = AutoConfig.from_pretrained(source, **common_kwargs)
    _validate_pruned_config(config, spec, recorded_depth=recorded_depth)
    tokenizer = AutoTokenizer.from_pretrained(source, **common_kwargs)
    model_kwargs = {
        **common_kwargs,
        "config": config,
        "dtype": resolved_dtype,
        "low_cpu_mem_usage": True,
    }
    if options.device_map is not None:
        model_kwargs["device_map"] = options.device_map
    model = AutoModelForCausalLM.from_pretrained(source, **model_kwargs)
    if type(model).__name__ != spec.expected_model_class:
        raise ValueError(
            f"Loaded model class {type(model).__name__!r}, "
            f"expected {spec.expected_model_class!r}"
        )
    if options.device is not None:
        model = model.to(options.device)
    model.eval()
    structure = adapter.get_structure(model)
    if structure["actual_block_count"] != recorded_depth:
        raise ValueError(
            f"Pruned artifact depth {structure['actual_block_count']!r} does not match "
            f"manifest depth {recorded_depth!r}"
        )
    runtime_dtype = getattr(model, "dtype", None)
    recorded_dtype = (
        str(runtime_dtype).removeprefix("torch.")
        if runtime_dtype is not None
        else dtype_name
    )
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        config=config,
        source=source,
        is_local=True,
        requested_revision=spec.revision,
        resolved_revision=None,
        dtype=recorded_dtype,
        structure=structure,
    )


def load_model_artifact(
    spec: ModelSpec,
    adapter: BaseModelAdapter,
    artifact_path: Path,
    *,
    kind: str = "dense",
    options: LoadOptions | None = None,
) -> LoadedModel:
    """Load a dense or pruned artifact with manifest-backed structure checks."""

    options = options or LoadOptions()
    path = artifact_path.expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Model artifact directory does not exist: {path}")
    if kind == "dense":
        return load_dense_model(spec, adapter, options=LoadOptions(
            local_path=path,
            cache_dir=options.cache_dir,
            dtype=options.dtype,
            device=options.device,
            device_map=options.device_map,
            local_files_only=options.local_files_only,
        ))
    if kind != "pruned":
        raise ValueError(f"Unsupported artifact kind: {kind!r}")

    manifest = _read_manifest(path)
    _pruner, recorded_depth = _validate_manifest_identity(manifest, spec, adapter)
    return _load_pruned_artifact(
        spec, adapter, path, options, recorded_depth=recorded_depth,
    )
