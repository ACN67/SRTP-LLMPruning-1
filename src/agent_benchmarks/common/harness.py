"""Thin subprocess lifecycle shared by official evaluator wrappers."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class HarnessRun:
    command: tuple[str, ...]
    cwd: str
    stdout_path: str
    stderr_path: str
    exit_code: int | None
    dry_run: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_harness(
    command: Sequence[str],
    *,
    cwd: Path,
    output_dir: Path,
    timeout: float,
    dry_run: bool,
    environment: Mapping[str, str] | None = None,
) -> HarnessRun:
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path, stderr_path = output_dir / "evaluator.stdout.log", output_dir / "evaluator.stderr.log"
    record = HarnessRun(tuple(str(item) for item in command), str(cwd.resolve()), str(stdout_path), str(stderr_path), None, dry_run)
    (output_dir / "evaluator_command.json").write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")
    if dry_run:
        return record
    env = os.environ.copy()
    env.update(environment or {})
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        completed = subprocess.run(tuple(command), cwd=cwd, env=env, stdout=stdout, stderr=stderr, timeout=timeout, check=False)
    record = HarnessRun(record.command, record.cwd, record.stdout_path, record.stderr_path, completed.returncode, False)
    (output_dir / "evaluator_command.json").write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"Official evaluator exited with code {completed.returncode}; see {stderr_path}")
    return record
