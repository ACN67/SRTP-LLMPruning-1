"""Stable benchmark contracts and result records."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class BenchmarkSpec:
    benchmark: str
    display_name: str
    source_revision: str
    expected_task_count: int
    metric: str
    prompt_protocol: str
    code_extraction_protocol: str
    dataset_path: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkTask:
    benchmark: str
    task_id: str
    prompt: str
    data: Mapping[str, Any]


@dataclass(frozen=True)
class TaskEvaluation:
    task_id: str
    passed: bool
    status: str
    error: str | None = None


class BaseBenchmark(ABC):
    def __init__(self, spec: BenchmarkSpec) -> None:
        self.spec = spec

    @abstractmethod
    def load_tasks(self) -> list[BenchmarkTask]:
        """Load the pinned task set."""

    def validate_tasks(self, tasks: Sequence[BenchmarkTask]) -> None:
        ids = [task.task_id for task in tasks]
        if len(ids) != self.spec.expected_task_count:
            raise ValueError(
                f"{self.spec.benchmark} expected {self.spec.expected_task_count} tasks, "
                f"got {len(ids)}"
            )
        if len(set(ids)) != len(ids):
            raise ValueError(f"{self.spec.benchmark} contains duplicate task IDs")

    @abstractmethod
    def build_prompt(self, task: BenchmarkTask) -> str:
        """Build versioned model input content."""

    def build_messages(self, task: BenchmarkTask) -> list[dict[str, str]]:
        return [{"role": "user", "content": self.build_prompt(task)}]

    @abstractmethod
    def postprocess_generation(self, text: str) -> str:
        """Apply the benchmark-specific extraction protocol."""

    @abstractmethod
    def evaluate(
        self, task: BenchmarkTask, completion: str, *, timeout: float = 3.0
    ) -> TaskEvaluation:
        """Execute one generated answer against benchmark tests."""

def task_id_hash(tasks: Sequence[BenchmarkTask]) -> str:
    import hashlib

    payload = "\n".join(task.task_id for task in tasks).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
