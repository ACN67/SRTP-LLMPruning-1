"""HumanEval, MBPP, and LiveCodeBench code-generation adapters."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Iterable

from .base import BaseBenchmark, BenchmarkSpec, BenchmarkTask, TaskEvaluation
from .execution import run_checked, run_lcb_checked
from .lcb_protocol import (
    chat_messages as lcb_chat_messages,
    evaluation_sample as lcb_evaluation_sample,
    extract_lcb_code,
    generic_question_prompt,
    load_verified_v6,
)


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


class HumanEvalBenchmark(BaseBenchmark):
    def load_tasks(self) -> list[BenchmarkTask]:
        path = Path(self.spec.dataset_path)
        if not path.is_file():
            raise FileNotFoundError(f"HumanEval asset missing: {path}; run prefetch_assets.py")
        tasks = [
            BenchmarkTask("humaneval", str(row["task_id"]), row["prompt"], row)
            for row in _jsonl(path)
        ]
        self.validate_tasks(tasks)
        return tasks

    def build_prompt(self, task: BenchmarkTask) -> str:
        return (
            "Complete the following Python task. Return the complete runnable solution "
            "inside one Python fenced code block.\n\n```python\n"
            + task.prompt.rstrip()
            + "\n```"
        )

    def postprocess_generation(self, text: str) -> str:
        return _last_fenced_or_plain(text)

    def evaluate(self, task: BenchmarkTask, completion: str, *, timeout: float = 3.0) -> TaskEvaluation:
        from .vendor.human_eval_execution import check_correctness

        problem = dict(task.data)
        problem["task_id"] = task.task_id
        if f"def {task.data['entry_point']}" in completion:
            problem["prompt"] = ""
        else:
            problem["prompt"] = task.prompt
        result = check_correctness(problem, completion, timeout)
        status = (
            "passed" if result["passed"] else
            "timeout" if result["result"] == "timed out" else
            "wrong"
        )
        return TaskEvaluation(task.task_id, result["passed"], status, None if result["passed"] else result["result"])


class MBPPBenchmark(BaseBenchmark):
    def load_tasks(self) -> list[BenchmarkTask]:
        path = Path(self.spec.dataset_path)
        if not path.is_file():
            raise FileNotFoundError(f"MBPP asset missing: {path}; run prefetch_assets.py")
        tasks = []
        for row in _jsonl(path):
            task_id = int(row["task_id"])
            if 11 <= task_id <= 510:
                tasks.append(BenchmarkTask("mbpp", str(task_id), row["text"], row))
        self.validate_tasks(tasks)
        return tasks

    def build_prompt(self, task: BenchmarkTask) -> str:
        first_test = task.data["test_list"][0]
        return f'"""\n{task.data["text"]}\n{first_test}\n"""\n'

    def postprocess_generation(self, text: str) -> str:
        return _last_fenced_or_plain(text)

    def evaluate(self, task: BenchmarkTask, completion: str, *, timeout: float = 3.0) -> TaskEvaluation:
        imports = task.data.get("test_imports") or []
        if isinstance(imports, str):
            imports = [imports]
        setup = str(task.data.get("test_setup_code") or "")
        # BigCode/MBPP semantics: candidate definitions precede setup objects that
        # may instantiate candidate-defined classes (task 367 in the pinned data).
        code = "\n".join([*imports, completion, setup, *task.data["test_list"]])
        return run_checked(task.task_id, code, timeout=timeout)


def _last_fenced_or_plain(text: str) -> str:
    lines = text.split("\n")
    indices = [index for index, line in enumerate(lines) if "```" in line]
    if len(indices) >= 2:
        return "\n".join(lines[indices[-2] + 1 : indices[-1]]).strip()
    return text.strip()


class LiveCodeBenchBenchmark(BaseBenchmark):
    def load_tasks(self) -> list[BenchmarkTask]:
        path = Path(self.spec.dataset_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"LiveCodeBench v6 asset missing: {path}; run prefetch_assets.py"
            )
        tasks = load_verified_v6(path, self.spec.metadata["dataset_sha256"])
        self.validate_tasks(tasks)
        return tasks

    def build_prompt(self, task: BenchmarkTask) -> str:
        return generic_question_prompt(task.prompt, str(task.data.get("starter_code") or ""))

    def build_messages(self, task: BenchmarkTask) -> list[dict[str, str]]:
        return lcb_chat_messages(task.prompt, str(task.data.get("starter_code") or ""))

    def postprocess_generation(self, text: str) -> str:
        return extract_lcb_code(text)

    def evaluate(self, task: BenchmarkTask, completion: str, *, timeout: float = 3.0) -> TaskEvaluation:
        return run_lcb_checked(
            task.task_id, lcb_evaluation_sample(task), completion, timeout=timeout
        )
