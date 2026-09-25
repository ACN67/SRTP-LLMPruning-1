"""Configuration-backed model and architecture-adapter registry."""

from pathlib import Path

import yaml

from .adapters.granite import GraniteAdapter
from .adapters.qwen3 import Qwen3Adapter
from .base import BaseModelAdapter, ModelSpec


DEFAULT_MODEL_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "models"

_ADAPTERS: dict[str, type[BaseModelAdapter]] = {
    "qwen3": Qwen3Adapter,
    "granite": GraniteAdapter,
}


def list_model_ids(config_dir: Path = DEFAULT_MODEL_CONFIG_DIR) -> tuple[str, ...]:
    return tuple(sorted(path.stem for path in config_dir.glob("*.yaml")))


def load_model_spec(
    project_model_id: str,
    config_dir: Path = DEFAULT_MODEL_CONFIG_DIR,
) -> ModelSpec:
    config_path = config_dir / f"{project_model_id}.yaml"
    if not config_path.is_file():
        available = ", ".join(list_model_ids(config_dir)) or "none"
        raise KeyError(f"Unknown model ID {project_model_id!r}; available: {available}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Model config must be a mapping: {config_path}")
    spec = ModelSpec.from_mapping(raw)
    if spec.project_model_id != project_model_id:
        raise ValueError(
            f"Config filename ID {project_model_id!r} does not match "
            f"project_model_id {spec.project_model_id!r}."
        )
    return spec


def get_model_adapter(spec: ModelSpec) -> BaseModelAdapter:
    try:
        adapter_type = _ADAPTERS[spec.adapter]
    except KeyError as error:
        raise KeyError(f"No architecture adapter registered for {spec.adapter!r}") from error
    adapter = adapter_type()
    adapter.validate(spec)
    return adapter
