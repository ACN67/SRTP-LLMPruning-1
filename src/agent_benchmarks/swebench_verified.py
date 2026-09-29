"""SWE-bench Verified adapter for the pinned official v5 evaluator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.agent_runner import AgentResult, RepositoryTask

from .base import AgentBenchmarkInstance, AgentBenchmarkSpec, repository_root
from .common.predictions import validate_prediction


FORBIDDEN_AGENT_FIELDS = {"patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS"}


class SWEbenchVerifiedAdapter:
    benchmark_id = "swebench_verified"

    def __init__(self, spec: AgentBenchmarkSpec) -> None:
        if spec.benchmark_id != self.benchmark_id:
            raise ValueError("SWE-bench Verified adapter/config mismatch")
        if spec.prompt.template != "{problem_statement}":
            raise ValueError("SWE-bench adapters require the non-leaking problem_statement template")
        self.spec = spec

    def load_instance(self, row: Mapping[str, Any]) -> AgentBenchmarkInstance:
        required = {"instance_id", "repo", "base_commit", "problem_statement"}
        missing = required - set(row)
        if missing:
            raise ValueError(f"SWE-bench row missing fields: {sorted(missing)}")
        metadata = {key: row[key] for key in ("version",) if key in row}
        return AgentBenchmarkInstance(self.benchmark_id, str(row["instance_id"]), str(row["problem_statement"]), str(row["repo"]), str(row["base_commit"]), metadata)

    def prepare_task(self, instance: AgentBenchmarkInstance, repo_path: Path) -> RepositoryTask:
        return RepositoryTask(instance.instance_id, repo_path, instance.problem_statement, instance.base_commit)

    def build_prediction(self, result: AgentResult) -> dict[str, Any]:
        patch = Path(result.patch_path).read_text(encoding="utf-8") if result.status == "patch_generated" else ""
        return validate_prediction({"instance_id": result.task_id, "model_name_or_path": result.system_id, "model_patch": patch}, allow_full_output=False)

    def evaluator_command(self, predictions: Path, run_id: str, workers: int, *, instance_ids: Sequence[str] = (), task_repo: Path | None = None, report_dir: Path | None = None, gold: bool = False, dataset_path: Path | None = None) -> tuple[str, ...]:
        dataset = str(dataset_path.resolve()) if dataset_path else (str((repository_root() / self.spec.dataset.local_path).resolve()) if self.spec.dataset.local_path else "verified")
        executable = str((repository_root() / self.spec.harness.runtime_python).with_name(self.spec.harness.evaluator_entry))
        command = [executable, "eval", dataset]
        command += ["--gold"] if gold else ["--predictions", str(predictions)]
        command += ["--run-id", run_id, "--workers", str(workers), "--split", self.spec.dataset.split]
        if report_dir is not None:
            command += ["--report-dir", str(report_dir)]
        if task_repo is not None:
            command += ["--task-repo", str(task_repo)]
        for item in instance_ids:
            command += ["--instance", item]
        return tuple(command)

    def parse_report(self, path: Path) -> dict[str, Any]:
        value = json.loads(path.read_text(encoding="utf-8"))
        required = {"resolved_ids", "unresolved_ids", "error_ids", "submitted_ids"}
        if required - set(value):
            raise ValueError("SWE-bench v5 report schema is incomplete")
        resolved, submitted = set(value["resolved_ids"]), list(value["submitted_ids"])
        return {"attempted": len(submitted), "evaluated": len(value.get("completed_ids", [])), "resolved": len(resolved), "resolved_rate": len(resolved) / len(submitted) if submitted else 0.0, "harness_errors": len(value["error_ids"]), "missing": len(value.get("incomplete_ids", [])), "per_instance": {item: {"resolved": item in resolved} for item in submitted}}
