"""Validated configuration schema for planned Agent evaluation systems."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping


SUPPORTED_FRAMEWORKS = {"openhands", "mini_swe_agent_plus"}
SUPPORTED_SERVING_BACKENDS = {"vllm"}
PROVENANCE_STATUSES = {
    "checkpoint_native",
    "official_agent_config",
    "official_benchmark_release",
    "official_framework_identity",
    "official_score_recipe",
    "official_serving_recipe",
    "official_supported_backend",
    "score_recipe_undisclosed",
    "upstream_not_disclosed",
}


def _optional_positive(name: str, value: int | float | None) -> None:
    if value is not None and value <= 0:
        raise ValueError(f"{name} must be positive when disclosed")


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
        sources = tuple(str(value) for value in raw.get("sources", ()))
        if not sources or any(not value.startswith("https://") for value in sources):
            raise ValueError("Provenance requires at least one HTTPS source")
        return cls(
            status=status,
            sources=sources,
            notes=tuple(str(value) for value in raw.get("notes", ())),
        )


@dataclass(frozen=True)
class AgentRuntimeSpec:
    framework: str
    upstream_repo: str
    upstream_revision: str | None
    version: str | None
    config_path: str | None
    agent_class: str | None
    max_iterations: int | None
    step_limit: int | None
    cost_limit: float | None
    runs: int | None
    mode: str | None
    worker_count: int | None
    use_hint_text: bool | None
    instruction_template_name: str | None
    enable_plan_mode: bool | None
    add_in_context_learning_example: bool | None
    environment_cwd: str | None
    command_timeout_seconds: int | None
    config_model_temperature: float | None
    config_drop_params: bool | None
    additional_tools: tuple[str, ...]
    provenance: ProvenanceSpec

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "AgentRuntimeSpec":
        framework = str(raw["framework"])
        if framework not in SUPPORTED_FRAMEWORKS:
            raise ValueError(f"Unsupported Agent framework: {framework!r}")
        revision = raw.get("upstream_revision")
        if revision is not None and not re.fullmatch(r"[0-9a-f]{40}", str(revision)):
            raise ValueError("Agent upstream_revision must be a full Git commit")
        values = dict(raw)
        provenance = ProvenanceSpec.from_mapping(values.pop("provenance"))
        values["framework"] = framework
        values["upstream_revision"] = None if revision is None else str(revision)
        values["additional_tools"] = tuple(
            str(value) for value in values.get("additional_tools", ())
        )
        spec = cls(provenance=provenance, **values)
        for name in (
            "max_iterations", "step_limit", "cost_limit", "runs",
            "worker_count", "command_timeout_seconds",
        ):
            _optional_positive(f"agent.{name}", getattr(spec, name))
        if spec.config_model_temperature is not None and spec.config_model_temperature < 0:
            raise ValueError("agent.config_model_temperature must be non-negative")
        return spec


@dataclass(frozen=True)
class ServingSpec:
    backend: str
    host: str | None
    port: int | None
    openai_compatible: bool | None
    served_model_name: str | None
    tensor_parallel_size: int | None
    gpu_memory_utilization: float | None
    max_model_len: int | None
    max_num_seqs: int | None
    api_key_semantics: str | None
    dtype: str | None
    reasoning_parser: str | None
    reasoning_parser_plugin: str | None
    minimum_backend_version: str | None
    tool_call_parser: str | None
    enable_auto_tool_choice: bool | None
    provenance: ProvenanceSpec

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ServingSpec":
        backend = str(raw["backend"])
        if backend not in SUPPORTED_SERVING_BACKENDS:
            raise ValueError(f"Unsupported serving backend: {backend!r}")
        values = dict(raw)
        provenance = ProvenanceSpec.from_mapping(values.pop("provenance"))
        spec = cls(provenance=provenance, **values)
        for name in ("port", "tensor_parallel_size", "max_model_len", "max_num_seqs"):
            _optional_positive(f"serving.{name}", getattr(spec, name))
        if (
            spec.gpu_memory_utilization is not None
            and not 0 < spec.gpu_memory_utilization <= 1
        ):
            raise ValueError("serving.gpu_memory_utilization must be in (0, 1]")
        return spec


@dataclass(frozen=True)
class GenerationRuntimeSpec:
    context_length: int | None
    max_input_tokens: int | None
    max_output_tokens: int | None
    do_sample: bool | None
    temperature: float | None
    top_p: float | None
    top_k: int | None
    enable_thinking: bool | None
    truncate_history_thinking: bool | None
    provenance: ProvenanceSpec

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "GenerationRuntimeSpec":
        values = dict(raw)
        provenance = ProvenanceSpec.from_mapping(values.pop("provenance"))
        spec = cls(provenance=provenance, **values)
        for name in ("context_length", "max_input_tokens", "max_output_tokens", "top_k"):
            _optional_positive(f"generation.{name}", getattr(spec, name))
        if spec.temperature is not None and spec.temperature < 0:
            raise ValueError("generation.temperature must be non-negative")
        if spec.top_p is not None and not 0 < spec.top_p <= 1:
            raise ValueError("generation.top_p must be in (0, 1]")
        return spec


@dataclass(frozen=True)
class ScoreIdentitySpec:
    benchmark_id: str
    reported_resolved_rate: float | None
    recipe_status: str
    unknown_fields: tuple[str, ...]
    provenance: ProvenanceSpec

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ScoreIdentitySpec":
        values = dict(raw)
        provenance = ProvenanceSpec.from_mapping(values.pop("provenance"))
        values["unknown_fields"] = tuple(
            str(value) for value in values.get("unknown_fields", ())
        )
        spec = cls(provenance=provenance, **values)
        if spec.recipe_status not in {"official_score_recipe", "score_recipe_undisclosed"}:
            raise ValueError(f"Unsupported score recipe status: {spec.recipe_status!r}")
        if (
            spec.reported_resolved_rate is not None
            and not 0 <= spec.reported_resolved_rate <= 1
        ):
            raise ValueError("reported_resolved_rate must be in [0, 1]")
        return spec


@dataclass(frozen=True)
class AgentSystemSpec:
    system_id: str
    project_model_id: str
    implementation_status: str
    agent: AgentRuntimeSpec
    serving: ServingSpec
    generation: GenerationRuntimeSpec
    score_identity: ScoreIdentitySpec
    known_unknowns: tuple[str, ...]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "AgentSystemSpec":
        if raw.get("implementation_status") != "planned":
            raise ValueError("Agent execution is not implemented; status must be 'planned'")
        spec = cls(
            system_id=str(raw["system_id"]),
            project_model_id=str(raw["project_model_id"]),
            implementation_status="planned",
            agent=AgentRuntimeSpec.from_mapping(raw["agent"]),
            serving=ServingSpec.from_mapping(raw["serving"]),
            generation=GenerationRuntimeSpec.from_mapping(raw["generation"]),
            score_identity=ScoreIdentitySpec.from_mapping(raw["score_identity"]),
            known_unknowns=tuple(str(value) for value in raw.get("known_unknowns", ())),
        )
        if spec.system_id != spec.project_model_id:
            raise ValueError("system_id must equal project_model_id in the v1 registry")
        return spec

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
