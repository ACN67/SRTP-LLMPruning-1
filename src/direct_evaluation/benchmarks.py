"""HumanEval, MBPP, and LiveCodeBench code-generation adapters."""

from __future__ import annotations

import gzip
import json
import re
import ast
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
        return ("Write the final executable Python solution. Return only one Python "
                "code block, with no reasoning or printed examples.\n\n"
                f'"""\n{task.data["text"]}\n{first_test}\n"""\n')

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


def _valid_python(code: str) -> bool:
    if not code.strip():
        return False
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # HumanEval may return only the indented continuation of its function.
        if not code.startswith((" ", "\t")):
            return False
        try:
            tree = ast.parse("def _continuation():\n" + code)
        except SyntaxError:
            return False
    return any(not isinstance(node, (ast.Expr, ast.Pass)) or
               isinstance(getattr(node, "value", None), ast.Call)
               for node in tree.body)


def _last_fenced_or_plain(text: str) -> str:
    """Select the last valid explicit Python fence, then valid unlabelled code.

    Display fences (text/output/etc.) and unfinished reasoning never become code.
    Syntax is a format check only; benchmark tests never select a candidate.
    """
    if "<think>" in text:
        if "</think>" not in text.rsplit("<think>", 1)[-1]:
            return ""
        text = text.rsplit("</think>", 1)[-1]
    lines = text.splitlines()
    candidates: list[tuple[int, int, str]] = []
    opening: tuple[int, str] | None = None
    for index, line in enumerate(lines):
        match = re.match(r"^\s*```\s*([\w+-]*)\s*$", line)
        if not match:
            continue
        if opening is None:
            opening = (index, match.group(1).lower())
        else:
            start, language = opening
            code = "\n".join(lines[start + 1:index]).strip("\n")
            if language in ("python", "py", "python3", "") and _valid_python(code):
                candidates.append((1 if language else 0, index, code.strip("\n")))
            opening = None
    if opening is not None:
        start, language = opening
        code = "\n".join(lines[start + 1:]).strip("\n")
        if language in ("python", "py", "python3", "") and _valid_python(code):
            candidates.append((1 if language else 0, len(lines), code.strip("\n")))
    if candidates:
        return max(candidates) [2]
    if "```" in text:
        return ""
    code = text.strip("\n")
    return code if _valid_python(code) else ""


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
