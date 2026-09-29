"""Load full checkpoints or PEFT overlays through one artifact boundary."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from src.models.base import BaseModelAdapter, ModelSpec
from src.models.loader import LoadOptions, LoadedModel, _resolve_dtype, load_dense_model

from .validation import resolve_model_artifact


def _load_reduced(spec: ModelSpec, adapter: BaseModelAdapter, path: Path, options: LoadOptions, depth: int) -> LoadedModel:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    common: dict[str, Any] = {"trust_remote_code": spec.trust_remote_code, "local_files_only": options.local_files_only}
    if options.cache_dir: common["cache_dir"] = str(options.cache_dir.expanduser())
    config = AutoConfig.from_pretrained(path, **common)
    expected = {"model_type": spec.model_type, "hidden_size": spec.expected_hidden_size, "intermediate_size": spec.expected_intermediate_size, "num_attention_heads": spec.expected_num_attention_heads, "num_key_value_heads": spec.expected_num_key_value_heads, "num_hidden_layers": depth}
    mismatch = [key for key, value in expected.items() if getattr(config, key, None) != value]
    if mismatch: raise ValueError(f"Reduced artifact config mismatch: {mismatch}")
    tokenizer = AutoTokenizer.from_pretrained(path, **common)
    kwargs = {**common, "config": config, "dtype": _resolve_dtype(torch, options.dtype or spec.default_dtype), "low_cpu_mem_usage": True}
    if options.device_map is not None: kwargs["device_map"] = options.device_map
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    if options.device is not None: model = model.to(options.device)
    model.eval(); structure = adapter.get_structure(model)
    if structure["actual_block_count"] != depth: raise ValueError("Loaded reduced artifact depth mismatch")
    return LoadedModel(model, tokenizer, config, str(path), True, spec.revision, None, str(model.dtype).removeprefix("torch."), structure)


def load_model_artifact(
    spec: ModelSpec,
    adapter: BaseModelAdapter,
    artifact_path: Path,
    *,
    options: LoadOptions | None = None,
    kind: str | None = None,
    require_verified_dense: bool = False,
    base_artifact_path: Path | None = None,
) -> LoadedModel:
    options = options or LoadOptions()
    artifact = resolve_model_artifact(
        artifact_path, spec, adapter, require_verified_dense=require_verified_dense,
    )
    if kind is not None and kind != artifact.kind:
        raise ValueError(f"Requested artifact kind {kind!r} does not match {artifact.kind!r}")
    if artifact.representation == "full_checkpoint":
        if artifact.num_hidden_layers == spec.expected_num_hidden_layers:
            return load_dense_model(spec, adapter, replace(options, local_path=Path(artifact.path)))
        return _load_reduced(spec, adapter, Path(artifact.path), options, artifact.num_hidden_layers)
    if artifact.representation != "peft_adapter": raise ValueError(f"Unsupported artifact representation: {artifact.representation}")
    base_path = base_artifact_path or artifact.metadata.get("base_artifact_path")
    base_hash = artifact.metadata.get("base_artifact_hash")
    if not isinstance(base_path, (str, Path)):
        raise ValueError(
            "PEFT adapter base artifact location is unavailable; pass --base-artifact-path"
        )
    if not isinstance(base_hash, str) or len(base_hash) != 64:
        raise ValueError("PEFT adapter manifest lacks a valid base_artifact_hash")
    resolved_base = resolve_model_artifact(
        Path(base_path), spec, adapter, require_verified_dense=require_verified_dense,
    )
    if resolved_base.content_sha256 != base_hash:
        raise ValueError(
            "PEFT adapter base artifact hash mismatch: "
            f"{resolved_base.content_sha256} != {base_hash}"
        )
    expected_base_provenance = artifact.metadata.get(
        "base_artifact_manifest_provenance_sha256"
    )
    if (
        expected_base_provenance is not None
        and resolved_base.manifest_provenance_sha256 != expected_base_provenance
    ):
        raise ValueError(
            "PEFT adapter base artifact manifest provenance mismatch: "
            f"{resolved_base.manifest_provenance_sha256} != {expected_base_provenance}"
        )
    recovery_ops = [item for item in artifact.lineage if item.operation == "recovery"]
    if not recovery_ops or recovery_ops[-1].input_artifact_hash != base_hash:
        raise ValueError("PEFT adapter recovery lineage does not match its base artifact hash")
    base = load_model_artifact(
        spec, adapter, Path(base_path), options=options,
        require_verified_dense=require_verified_dense,
    )
    from peft import PeftModel
    model = PeftModel.from_pretrained(base.model, artifact.path)
    model.eval()
    return replace(base, model=model)
