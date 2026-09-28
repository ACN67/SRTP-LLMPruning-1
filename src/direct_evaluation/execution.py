"""Minimal benchmark code execution helpers; reliability guard, not a sandbox."""

from __future__ import annotations

import json
import multiprocessing
import os
import shutil
from typing import Any

from .base import TaskEvaluation


def _worker(code: str, queue) -> None:
    try:
        _reliability_guard()
        exec(code, {})
        queue.put(("passed", None))
    except AssertionError as error:
        queue.put(("wrong", f"AssertionError: {error}"))
    except BaseException as error:
        queue.put(("execution_error", f"{type(error).__name__}: {error}"))


def _reliability_guard() -> None:
    """Disable common destructive operations, following HumanEval's intent.

    This reduces accidental damage but is explicitly not a security sandbox.
    """

    for name in (
        "kill", "system", "putenv", "remove", "removedirs", "rmdir", "fchdir",
        "setuid", "fork", "forkpty", "killpg", "rename", "renames", "truncate",
        "replace", "unlink", "fchmod", "fchown", "chmod", "chown", "chroot",
        "lchflags", "lchmod", "lchown", "getcwd",
    ):
        if hasattr(os, name):
            setattr(os, name, None)
    for name in ("rmtree", "move", "chown"):
        if hasattr(shutil, name):
            setattr(shutil, name, None)


def run_checked(task_id: str, code: str, *, timeout: float = 3.0) -> TaskEvaluation:
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_worker, args=(code, queue))
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join()
        return TaskEvaluation(task_id, False, "timeout", "execution timed out")
    if queue.empty():
        return TaskEvaluation(task_id, False, "execution_error", "worker exited without result")
    status, error = queue.get()
    return TaskEvaluation(task_id, status == "passed", status, error)


def _lcb_worker(sample: dict[str, str], code: str, timeout: float, queue: Any) -> None:
    try:
        from .vendor.livecodebench import run_test

        results, metadata = run_test(sample, test=code, timeout=timeout)
        passed = bool(results) and all(
            (bool(item) if isinstance(item, bool) else item > 0) for item in results
        )
        error_code = metadata.get("error_code") if isinstance(metadata, dict) else None
        if passed:
            status = "passed"
        elif error_code == -2:
            status = "wrong"
        elif error_code == -3:
            status = "timeout"
        else:
            status = "execution_error"
        queue.put((passed, status, None if passed else json.dumps(metadata, default=str)))
    except BaseException as error:
        queue.put((False, "execution_error", f"{type(error).__name__}: {error}"))


def run_lcb_checked(
    task_id: str, sample: dict[str, str], code: str, *, timeout: float = 6.0
) -> TaskEvaluation:
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_lcb_worker, args=(sample, code, timeout, queue))
    process.start()
    test_count = len(json.loads(sample["input_output"])["inputs"])
    process.join((timeout + 1) * test_count + 5)
    if process.is_alive():
        process.kill()
        process.join()
        return TaskEvaluation(task_id, False, "timeout", "global execution timeout")
    if queue.empty():
        return TaskEvaluation(task_id, False, "execution_error", "LCB worker exited without result")
    passed, status, error = queue.get()
    return TaskEvaluation(task_id, passed, status, error)
