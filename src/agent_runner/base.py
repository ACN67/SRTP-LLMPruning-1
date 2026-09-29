"""Executable, non-null configuration schema for Agent systems."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping


PROVENANCE_STATUSES = {
    "official_score_recipe", "official_agent_config", "official_serving_recipe",
    "checkpoint_native", "official_framework_default", "project_baseline",
    "score_recipe_undisclosed",
}


def _validate_parameter_provenance(values: Mapping[str, str], scope: str) -> None:
    if not values:
        raise ValueError(f"{scope}.parameter_provenance cannot be empty")
    invalid = {value for value in values.values() if value not in PROVENANCE_STATUSES}
    if invalid:
        raise ValueError(f"Invalid {scope} parameter provenance: {sorted(invalid)}")


def _require_revision(name: str, value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError(f"{name} must be a full Git commit")
    return value


def _reject_incomplete(value: Any, path: str = "config") -> None:
    if value is None:
        raise ValueError(f"Active runtime field cannot be null: {path}")
    if isinstance(value, str) and value.strip().lower() in {"unknown", "planned", "todo", "placeholder"}:
        raise ValueError(f"Active runtime field is unresolved: {path}")
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _reject_incomplete(nested, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _reject_incomplete(nested, f"{path}[{index}]")


@dataclass(frozen=True)
class ProvenanceSpec:
    status: str
    sources: tuple[str, ...]
    notes: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ProvenanceSpec":
        status = str(raw["status"])
        if status not in PROVENANCE_STATUSES:
            raise ValueError(f"Unsupported provenance status: {status!r}")
        sources = tuple(str(item) for item in raw["sources"])
        if not sources or any(not item.startswith("https://") for item in sources):
            raise ValueError("Provenance requires HTTPS sources")
        return cls(status, sources, tuple(str(item) for item in raw.get("notes", ())))


@dataclass(frozen=True)
class ServingSpec:
    backend: str
    version: str
    executable: str
    host: str
    port: int
    served_model_name: str
    tensor_parallel_size: int
    gpu_memory_utilization: float
    max_model_len: int
    max_num_seqs: int
    dtype: str
    api_key: str
    extra_args: tuple[str, ...]
    parameter_provenance: Mapping[str, str]
    provenance: ProvenanceSpec

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ServingSpec":
        values = dict(raw)
        provenance = ProvenanceSpec.from_mapping(values.pop("provenance"))
        values["extra_args"] = tuple(str(item) for item in values.get("extra_args", ()))
        spec = cls(provenance=provenance, **values)
        if spec.backend != "vllm":
            raise ValueError("Only vLLM serving is supported")
        if not re.fullmatch(r"\d+\.\d+\.\d+", spec.version):
            raise ValueError("vLLM version must be exact")
        if not 1 <= spec.port <= 65535 or spec.tensor_parallel_size <= 0:
            raise ValueError("Serving port/TP must be positive")
        if not 0 < spec.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if spec.max_model_len <= 0 or spec.max_num_seqs <= 0:
            raise ValueError("Serving capacities must be positive")
        _validate_parameter_provenance(spec.parameter_provenance, "serving")
        _reject_incomplete(asdict(spec), "serving")
        return spec

    def with_overrides(self, overrides: Mapping[str, Any]) -> "ServingSpec":
        allowed = {"host", "port", "tensor_parallel_size", "gpu_memory_utilization", "max_num_seqs"}
        invalid = set(overrides) - allowed
        if invalid:
            raise ValueError(f"Unsupported serving overrides: {sorted(invalid)}")
        updated = replace(self, **dict(overrides))
        return ServingSpec.from_mapping({**asdict(updated), "provenance": asdict(updated.provenance)})


@dataclass(frozen=True)
class GenerationSpec:
    context_length: int
    max_input_tokens: int
    max_output_tokens: int
    request_parameters: Mapping[str, Any]
    thinking_mode: str
    history_thinking: str
    parameter_provenance: Mapping[str, str]
    provenance: ProvenanceSpec

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "GenerationSpec":
        values = dict(raw)
        provenance = ProvenanceSpec.from_mapping(values.pop("provenance"))
        spec = cls(provenance=provenance, **values)
        if min(spec.context_length, spec.max_input_tokens, spec.max_output_tokens) <= 0:
            raise ValueError("Generation token budgets must be positive")
        if spec.max_input_tokens + spec.max_output_tokens > spec.context_length:
            raise ValueError("Input plus output budget exceeds context length")
        if spec.thinking_mode not in {"enabled", "disabled", "checkpoint_default"}:
            raise ValueError("Invalid thinking_mode")
        if spec.history_thinking not in {"truncate", "preserve", "checkpoint_default"}:
            raise ValueError("Invalid history_thinking")
        _validate_parameter_provenance(spec.parameter_provenance, "generation")
        _reject_incomplete(asdict(spec), "generation")
        return spec


@dataclass(frozen=True)
class MiniSweAgentPlusSpec:
    framework: str
    version: str
    upstream_repo: str
    upstream_revision: str
    runtime_python: str
    source_checkout: str
    config_path: str
    runs: int
    worker_count: int
    step_limit: int
    cost_limit: float
    environment_cwd: str
    command_timeout_seconds: int
    tools: tuple[str, ...]
    model_parameters: Mapping[str, Any]
    parameter_provenance: Mapping[str, str]
    provenance: ProvenanceSpec

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "MiniSweAgentPlusSpec":
        values = dict(raw)
        provenance = ProvenanceSpec.from_mapping(values.pop("provenance"))
        values["tools"] = tuple(str(item) for item in values.get("tools", ()))
        spec = cls(provenance=provenance, **values)
        if spec.framework != "mini_swe_agent_plus":
            raise ValueError("Invalid mini-swe-agent-plus framework identity")
        _require_revision("agent.upstream_revision", spec.upstream_revision)
        if min(spec.runs, spec.worker_count, spec.step_limit, spec.command_timeout_seconds) <= 0 or spec.cost_limit <= 0:
            raise ValueError("mini-swe-agent-plus limits must be positive")
        _validate_parameter_provenance(spec.parameter_provenance, "agent")
        _reject_incomplete(asdict(spec), "agent")
        return spec


@dataclass(frozen=True)
class OpenHandsSpec:
    framework: str
    version: str
    upstream_repo: str
    upstream_revision: str
    runtime_python: str
    source_checkout: str
    agent_class: str
    max_iterations: int
    runs: int
    mode: str
    worker_count: int
    benchmark_worker_count: int
    use_hint_text: bool
    instruction_template_name: str
    enable_plan_mode: bool
    add_in_context_learning_example: bool
    runtime: str
    command_timeout_seconds: int
    native_tool_calling: bool
    parameter_provenance: Mapping[str, str]
    provenance: ProvenanceSpec

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "OpenHandsSpec":
        values = dict(raw)
        provenance = ProvenanceSpec.from_mapping(values.pop("provenance"))
        spec = cls(provenance=provenance, **values)
        if spec.framework != "openhands" or spec.agent_class != "CodeActAgent":
            raise ValueError("OpenHands systems require CodeActAgent")
        _require_revision("agent.upstream_revision", spec.upstream_revision)
        if min(spec.max_iterations, spec.runs, spec.worker_count, spec.benchmark_worker_count, spec.command_timeout_seconds) <= 0:
            raise ValueError("OpenHands limits must be positive")
        _validate_parameter_provenance(spec.parameter_provenance, "agent")
        _reject_incomplete(asdict(spec), "agent")
        return spec


AgentSpec = MiniSweAgentPlusSpec | OpenHandsSpec


@dataclass(frozen=True)
class ScoreIdentitySpec:
    benchmark_id: str
    reported_resolved_rate: float
    recipe_status: str
    undisclosed_fields: tuple[str, ...]
    provenance: ProvenanceSpec

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ScoreIdentitySpec":
        values = dict(raw)
        provenance = ProvenanceSpec.from_mapping(values.pop("provenance"))
        values["undisclosed_fields"] = tuple(str(item) for item in values.get("undisclosed_fields", ()))
        spec = cls(provenance=provenance, **values)
        if spec.recipe_status not in {"official_score_recipe", "score_recipe_undisclosed"}:
            raise ValueError("Invalid score recipe status")
        if not 0 <= spec.reported_resolved_rate <= 1:
            raise ValueError("reported_resolved_rate must be in [0, 1]")
        return spec


@dataclass(frozen=True)
class AgentSystemSpec:
    system_id: str
    project_model_id: str
    implementation_status: str
    serving: ServingSpec
    generation: GenerationSpec
    agent: AgentSpec
    score_identity: ScoreIdentitySpec

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "AgentSystemSpec":
        agent_raw = raw["agent"]
        agent = (
            MiniSweAgentPlusSpec.from_mapping(agent_raw)
            if agent_raw["framework"] == "mini_swe_agent_plus"
            else OpenHandsSpec.from_mapping(agent_raw)
        )
        spec = cls(
            system_id=str(raw["system_id"]), project_model_id=str(raw["project_model_id"]),
            implementation_status=str(raw["implementation_status"]),
            serving=ServingSpec.from_mapping(raw["serving"]),
            generation=GenerationSpec.from_mapping(raw["generation"]), agent=agent,
            score_identity=ScoreIdentitySpec.from_mapping(raw["score_identity"]),
        )
        if spec.implementation_status != "ready":
            raise ValueError("Executable Agent systems must have implementation_status='ready'")
        if spec.system_id != spec.project_model_id:
            raise ValueError("system_id must equal project_model_id")
        _reject_incomplete({
            "system_id": spec.system_id, "project_model_id": spec.project_model_id,
            "implementation_status": spec.implementation_status,
            "serving": asdict(spec.serving), "generation": asdict(spec.generation),
            "agent": asdict(spec.agent),
        })
        return spec

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def canonical_config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
