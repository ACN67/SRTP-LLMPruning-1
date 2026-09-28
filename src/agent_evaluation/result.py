"""Serializable result of one local repository Agent run."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class AgentResult:
    system_id: str
    project_model_id: str
    artifact_provenance: Mapping[str, Any]
    task_id: str
    status: str
    started_at: str
    ended_at: str
    runtime_seconds: float
    framework: str
    framework_version: str
    framework_revision: str
    effective_agent_config: Mapping[str, Any]
    endpoint: str
    serving_manifest_path: str
    trajectory_path: str
    stdout_path: str
    stderr_path: str
    patch_path: str
    patch_sha256: str
    changed_files: tuple[str, ...]
    repository_before: Mapping[str, Any]
    repository_after: Mapping[str, Any]
    framework_metadata: Mapping[str, Any] = field(default_factory=dict)
    error_type: str = "none"
    error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["changed_files"] = list(self.changed_files)
        return result
