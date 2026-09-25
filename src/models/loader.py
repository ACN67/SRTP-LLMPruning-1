"""Unified dense Hugging Face model loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .base import IMMUTABLE_REVISION_PATTERN, BaseModelAdapter, ModelSpec


@dataclass(frozen=True)
class LoadOptions:
    """Runtime choices kept separate from stable model identity."""

    local_path: Path | None = None
    cache_dir: Path | None = None
    dtype: str | None = None
    device: str | None = None
    device_map: str | Mapping[str, Any] | None = None
    local_files_only: bool = False


@dataclass(frozen=True)
class LoadedModel:
    model: Any
    tokenizer: Any
    config: Any
    source: str
    is_local: bool
    requested_revision: str
    resolved_revision: str | None
    dtype: str
    structure: Mapping[str, Any]


def _runtime_imports() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Dense model loading requires torch and transformers; "
            "install requirements.txt first."
        ) from error
    return torch, AutoConfig, AutoModelForCausalLM, AutoTokenizer


def _resolve_source(spec: ModelSpec, options: LoadOptions) -> tuple[str, bool]:
    local_path = options.local_path or spec.local_path
    if local_path is None:
        return spec.huggingface_repo_id, False
    resolved = local_path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Local model directory does not exist: {resolved}")
    return str(resolved), True


def _resolve_dtype(torch: Any, dtype_name: str) -> Any:
    normalized = dtype_name.lower()
    if normalized == "auto":
        return "auto"
    aliases = {
        "float16": "float16",
        "fp16": "float16",
        "bfloat16": "bfloat16",
        "bf16": "bfloat16",
        "float32": "float32",
        "fp32": "float32",
    }
    try:
        attribute = aliases[normalized]
    except KeyError as error:
        raise ValueError(f"Unsupported dtype: {dtype_name!r}") from error
    return getattr(torch, attribute)


def _resolved_revision(config: Any, spec: ModelSpec, *, is_local: bool) -> str | None:
    commit_hash = getattr(config, "_commit_hash", None)
    if isinstance(commit_hash, str) and IMMUTABLE_REVISION_PATTERN.fullmatch(commit_hash):
        return commit_hash
    if not is_local:
        return spec.revision
    return None


def load_dense_model(
    spec: ModelSpec,
    adapter: BaseModelAdapter,
    options: LoadOptions | None = None,
) -> LoadedModel:
    """Load and validate a dense model and tokenizer at one revision."""

    options = options or LoadOptions()
    if options.device is not None and options.device_map is not None:
        raise ValueError("Specify either device or device_map, not both")

    torch, AutoConfig, AutoModelForCausalLM, AutoTokenizer = _runtime_imports()
    source, is_local = _resolve_source(spec, options)
    dtype_name = options.dtype or spec.default_dtype
    resolved_dtype = _resolve_dtype(torch, dtype_name)

    common_kwargs: dict[str, Any] = {
        "trust_remote_code": spec.trust_remote_code,
        "local_files_only": options.local_files_only,
    }
    if options.cache_dir is not None:
        common_kwargs["cache_dir"] = str(options.cache_dir.expanduser())
    if not is_local:
        common_kwargs["revision"] = spec.revision

    config = AutoConfig.from_pretrained(source, **common_kwargs)
    adapter.validate_config(config, spec)
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
    structure = adapter.validate_loaded_model(model, spec)
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
        is_local=is_local,
        requested_revision=spec.revision,
        resolved_revision=_resolved_revision(config, spec, is_local=is_local),
        dtype=recorded_dtype,
        structure=structure,
    )
