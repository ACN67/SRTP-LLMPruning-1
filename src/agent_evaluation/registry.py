"""Configuration-backed registry for planned model-specific Agent systems."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.models import list_model_ids

from .base import AgentSystemSpec


DEFAULT_AGENT_SYSTEM_DIR = Path(__file__).resolve().parents[2] / "configs" / "systems"


def list_agent_system_ids(
    config_dir: Path = DEFAULT_AGENT_SYSTEM_DIR,
) -> tuple[str, ...]:
    return tuple(sorted(path.stem for path in config_dir.glob("*.yaml")))


def load_agent_system_spec(
    system_id: str,
    config_dir: Path = DEFAULT_AGENT_SYSTEM_DIR,
) -> AgentSystemSpec:
    path = config_dir / f"{system_id}.yaml"
    if not path.is_file():
        available = ", ".join(list_agent_system_ids(config_dir)) or "none"
        raise KeyError(f"Unknown Agent system {system_id!r}; available: {available}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Agent system config must be a mapping: {path}")
    spec = AgentSystemSpec.from_mapping(raw)
    if spec.system_id != system_id:
        raise ValueError(f"Agent system filename identity mismatch: {path}")
    if spec.project_model_id not in list_model_ids():
        raise ValueError(
            f"Agent system references unknown model {spec.project_model_id!r}"
        )
    return spec
