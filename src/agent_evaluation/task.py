"""Benchmark-agnostic local repository task validation."""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ("git", "-C", str(repo), *args), check=True, capture_output=True, text=True,
    ).stdout.strip()


@dataclass(frozen=True)
class RepositoryTask:
    task_id: str
    repo_path: Path
    problem_statement: str
    base_commit: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self, *, allow_dirty: bool = False) -> "RepositoryState":
        repo = self.repo_path.expanduser().resolve()
        if not self.task_id.strip() or not self.problem_statement.strip():
            raise ValueError("Repository task requires non-empty task_id and problem_statement")
        if not repo.is_dir() or _git(repo, "rev-parse", "--is-inside-work-tree") != "true":
            raise ValueError(f"Not a Git working tree: {repo}")
        commit = _git(repo, "rev-parse", "HEAD")
        if self.base_commit is not None and _git(repo, "rev-parse", self.base_commit) != commit:
            raise ValueError(f"Repository HEAD {commit} does not match requested base_commit")
        status = _git(repo, "status", "--porcelain=v1", "--untracked-files=all")
        if status and not allow_dirty:
            raise ValueError("Repository is dirty; pass --allow-dirty only when intentional")
        return RepositoryState(str(repo), commit, bool(status), status.splitlines())

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["repo_path"] = str(self.repo_path.expanduser().resolve())
        return result


@dataclass(frozen=True)
class RepositoryState:
    repo_path: str
    head_commit: str
    dirty: bool
    status_lines: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def repository_state(repo_path: Path) -> RepositoryState:
    repo = repo_path.expanduser().resolve()
    status = _git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    return RepositoryState(str(repo), _git(repo, "rev-parse", "HEAD"), bool(status), status.splitlines())


def extract_patch(repo_path: Path) -> tuple[str, tuple[str, ...]]:
    repo = repo_path.expanduser().resolve()
    tracked_patch = _git(repo, "diff", "--binary", "HEAD")
    tracked_names = [
        line for line in _git(repo, "diff", "--name-only", "HEAD").splitlines() if line
    ]
    untracked_names = [
        line
        for line in _git(repo, "ls-files", "--others", "--exclude-standard").splitlines()
        if line
    ]
    untracked_patches: list[str] = []
    for relative in untracked_names:
        completed = subprocess.run(
            ("git", "diff", "--no-index", "--binary", "--", "/dev/null", relative),
            cwd=repo,
            capture_output=True,
            text=True,
        )
        if completed.returncode not in {0, 1}:
            raise subprocess.CalledProcessError(
                completed.returncode, completed.args, completed.stdout, completed.stderr
            )
        if completed.stdout:
            untracked_patches.append(completed.stdout.rstrip())
    patch_parts = [part for part in (tracked_patch, *untracked_patches) if part]
    patch = "\n".join(patch_parts)
    names = tuple(dict.fromkeys((*tracked_names, *untracked_names)))
    return patch, names
