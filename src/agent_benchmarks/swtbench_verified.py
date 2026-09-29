"""SWT-Bench Verified test-generation adapter and official harness wrapper."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.agent_runner import AgentResult, RepositoryTask

from .base import AgentBenchmarkInstance, AgentBenchmarkSpec, repository_root
from .common.predictions import validate_prediction
from .swtbench_data import SWTEvaluationDatasetSpec


TASK_TEMPLATE = """Your task is to add or modify tests that reproduce the issue below.
Do not fix the product bug. Produce a focused test patch that fails on the buggy checkout and passes after the official fix.

Issue:
{problem_statement}
"""


class SWTbenchVerifiedAdapter:
    benchmark_id = "swtbench_verified"

    def __init__(self, spec: AgentBenchmarkSpec) -> None:
        if spec.benchmark_id != self.benchmark_id:
            raise ValueError("SWT-Bench Verified adapter/config mismatch")
        if spec.prompt.template != TASK_TEMPLATE:
            raise ValueError("SWT-Bench prompt template/config mismatch")
        self.spec = spec
        self.evaluation_dataset = SWTEvaluationDatasetSpec.from_mapping(spec.metadata["evaluation_dataset"])
        if self.evaluation_dataset.repo_id == spec.dataset.repo_id:
            raise ValueError("SWT inference and evaluation dataset identities must differ")

    def load_instance(self, row: Mapping[str, Any]) -> AgentBenchmarkInstance:
        required = {"instance_id", "repo", "base_commit", "problem_statement"}
        missing = required - set(row)
        if missing:
            raise ValueError(f"SWT-Bench row missing fields: {sorted(missing)}")
        metadata = {key: row[key] for key in ("version", "difficulty") if key in row}
        return AgentBenchmarkInstance(self.benchmark_id, str(row["instance_id"]), TASK_TEMPLATE.format(problem_statement=row["problem_statement"]), str(row["repo"]), str(row["base_commit"]), metadata)

    def prepare_task(self, instance: AgentBenchmarkInstance, repo_path: Path) -> RepositoryTask:
        return RepositoryTask(instance.instance_id, repo_path, instance.problem_statement, instance.base_commit)

    def build_prediction(self, result: AgentResult) -> dict[str, Any]:
        patch = Path(result.patch_path).read_text(encoding="utf-8") if result.status == "success" else ""
        full_output = ""
        trajectory = Path(result.trajectory_path)
        if trajectory.is_file():
            full_output = trajectory.read_text(encoding="utf-8")
        return validate_prediction({"instance_id": result.task_id, "model_name_or_path": result.system_id, "model_patch": patch, "full_output": full_output})

    def evaluator_command(self, predictions: Path, run_id: str, workers: int, *, instance_ids: Sequence[str] = (), gold: bool = False, dataset_path: Path | None = None) -> tuple[str, ...]:
        python = str(repository_root() / self.spec.harness.runtime_python)
        evaluation = self.evaluation_dataset
        configured_dataset = (repository_root() / evaluation.local_path).resolve()
        if dataset_path is not None and dataset_path.resolve() != configured_dataset:
            raise ValueError("SWT evaluation must use the configured derived original-SWE snapshot")
        dataset = str(configured_dataset)
        command = [python, "-m", self.spec.harness.evaluator_entry, "--dataset_name", dataset, "--split", evaluation.split, "--predictions_path", "gold" if gold else str(predictions), "--max_workers", str(workers), "--run_id", run_id, "--exec_mode", self.spec.execution_mode, "--patch_types", "vanilla"]
        if instance_ids:
            command += ["--instance_ids", *instance_ids]
        return tuple(command)

    def parse_report(self, path: Path) -> dict[str, Any]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("SWT evaluator report must be a mapping")
        if "resolved_ids" in value:
            completed = set(value.get("completed_ids", ()))
            resolved = set(value["resolved_ids"])
            all_ids = completed | resolved | set(value.get("unresolved_ids", ())) | set(value.get("error_ids", ()))
            per_instance = {
                item: {
                    "success": item in resolved,
                    "applicable": item in completed,
                    "raw": {"official_resolved": item in resolved, "official_error": item in set(value.get("error_ids", ()))},
                }
                for item in sorted(all_ids)
            }
            return {"attempted": int(value.get("total_instances", len(all_ids))), "applicable": len(completed), "successful": len(resolved), "success_rate": len(resolved) / len(all_ids) if all_ids else 0.0, "per_instance": per_instance, "official_aggregate": value}
        per_instance = value.get("per_instance", value)
        normalized: dict[str, dict[str, Any]] = {}
        for instance_id, verdict in per_instance.items():
            if not isinstance(verdict, Mapping):
                continue
            success = bool(verdict.get("success", verdict.get("resolved", False)))
            applicable = bool(verdict.get("applicable", verdict.get("patch_successfully_applied", False)))
            normalized[str(instance_id)] = {"success": success, "applicable": applicable, "raw": dict(verdict)}
        if not normalized:
            raise ValueError("SWT report contains no per-instance verdicts")
        return {"attempted": len(normalized), "applicable": sum(int(v["applicable"]) for v in normalized.values()), "successful": sum(int(v["success"]) for v in normalized.values()), "success_rate": sum(int(v["success"]) for v in normalized.values()) / len(normalized), "per_instance": normalized}
