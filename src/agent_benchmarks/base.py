"""Validated configuration and shared instance identity for Agent benchmarks."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


def _revision(name: str, value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError(f"{name} must be an immutable 40-character revision")
    return value


@dataclass(frozen=True)
class DatasetSpec:
    repo_id: str
    revision: str
    split: str
    expected_task_count: int
    local_path: str = ""

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DatasetSpec":
        spec = cls(**dict(value))
        _revision("dataset.revision", spec.revision)
        if not spec.repo_id or not spec.split or spec.expected_task_count <= 0:
            raise ValueError("Dataset identity, split and expected count are required")
        return spec


@dataclass(frozen=True)
class HarnessSpec:
    repository: str
    release: str
    revision: str
    package_version: str
    runtime_python: str
    source_checkout: str
    evaluator_entry: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "HarnessSpec":
        spec = cls(**dict(value))
        _revision("harness.revision", spec.revision)
        if not spec.repository.startswith("https://") or not spec.package_version:
            raise ValueError("Harness repository and package version must be pinned")
        return spec


@dataclass(frozen=True)
class PromptSpec:
    protocol_id: str
    source: str
    template: str
    template_sha256: str
    provenance_status: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PromptSpec":
        spec = cls(**dict(value))
        if not spec.source.startswith("https://"):
            raise ValueError("Prompt source must be an HTTPS provenance URL")
        if not re.fullmatch(r"[0-9a-f]{64}", spec.template_sha256):
            raise ValueError("prompt.template_sha256 must be SHA256")
        actual = hashlib.sha256(spec.template.encode("utf-8")).hexdigest()
        if actual != spec.template_sha256:
            raise ValueError("prompt.template_sha256 does not match the configured template")
        return spec


@dataclass(frozen=True)
class AgentBenchmarkSpec:
    benchmark_id: str
    implementation_status: str
    task_type: str
    primary_metric: str
    execution_mode: str
    result_parser_id: str
    dataset: DatasetSpec
    harness: HarnessSpec
    prompt: PromptSpec
    metadata: Mapping[str, Any]
    provenance: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AgentBenchmarkSpec":
        raw = dict(value)
        raw["dataset"] = DatasetSpec.from_mapping(raw["dataset"])
        raw["harness"] = HarnessSpec.from_mapping(raw["harness"])
        raw["prompt"] = PromptSpec.from_mapping(raw["prompt"])
        spec = cls(**raw)
        if spec.implementation_status != "software_ready":
            raise ValueError("Agent benchmark must have implementation_status='software_ready'")
        if not spec.benchmark_id or not spec.task_type or not spec.primary_metric:
            raise ValueError("Benchmark identity, task type and metric are required")
        return spec

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgentBenchmarkInstance:
    benchmark_id: str
    instance_id: str
    problem_statement: str
    repo: str
    base_commit: str
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not all((self.benchmark_id, self.instance_id, self.problem_statement, self.repo, self.base_commit)):
            raise ValueError("Benchmark instance identity/task/repository fields cannot be empty")
        if "patch" in self.metadata or "test_patch" in self.metadata:
            raise ValueError("Gold patch fields must not enter Agent benchmark instance metadata")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]
