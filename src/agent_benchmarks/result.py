"""Benchmark-level result, intentionally separate from one Agent execution result."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class AgentBenchmarkResult:
    benchmark_id: str
    instance_id: str
    system_id: str
    artifact_identity: Mapping[str, Any]
    agent_result_path: str
    prediction_path: str
    prediction_sha256: str
    evaluator_status: str
    success: bool | None
    evaluator_raw_result_path: str
    provenance: Mapping[str, Any]
    started_at: str
    ended_at: str
    runtime_seconds: float
    error_stage: str = "none"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
