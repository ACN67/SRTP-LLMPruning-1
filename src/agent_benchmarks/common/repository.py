"""Safe per-instance Git worktree provisioning outside the project repository."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


def _git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(("git", *args), cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def _slug(value: str) -> str:
    readable = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in value)[:80]
    return f"{readable}-{hashlib.sha256(value.encode()).hexdigest()[:10]}"


@dataclass
class ProvisionedRepository:
    repo: str
    base_commit: str
    mirror_path: Path
    worktree_path: Path

    def cleanup(self) -> None:
        if self.worktree_path.exists():
            subprocess.run(("git", "-C", str(self.mirror_path), "worktree", "remove", "--force", str(self.worktree_path)), check=False, capture_output=True)
        if self.worktree_path.exists():
            shutil.rmtree(self.worktree_path)

    def manifest(self) -> dict[str, str]:
        return {"repo": self.repo, "base_commit": self.base_commit, "mirror_path": str(self.mirror_path), "worktree_path": str(self.worktree_path)}


def provision_repository(
    repo: str,
    base_commit: str,
    instance_id: str,
    *,
    repo_cache_root: Path,
    workspace_root: Path,
    offline: bool,
) -> ProvisionedRepository:
    repo_cache_root = repo_cache_root.expanduser().resolve()
    workspace_root = workspace_root.expanduser().resolve()
    repo_cache_root.mkdir(parents=True, exist_ok=True)
    workspace_root.mkdir(parents=True, exist_ok=True)
    source = Path(repo).expanduser()
    mirror = repo_cache_root / f"{_slug(repo)}.git"
    if not mirror.is_dir():
        if offline and not source.exists():
            raise FileNotFoundError(f"Offline repository mirror is missing for {repo}")
        clone_source = str(source.resolve()) if source.exists() else f"https://github.com/{repo}.git"
        _git("clone", "--mirror", clone_source, str(mirror))
    elif not offline and not source.exists():
        _git("-C", str(mirror), "fetch", "--prune", "origin")
    try:
        resolved = _git("-C", str(mirror), "rev-parse", f"{base_commit}^{{commit}}")
    except subprocess.CalledProcessError as error:
        raise ValueError(f"Base commit {base_commit} is unavailable in mirror {mirror}") from error
    worktree = workspace_root / _slug(instance_id)
    if worktree.exists():
        raise FileExistsError(f"Instance workspace already exists: {worktree}")
    _git("-C", str(mirror), "worktree", "add", "--detach", str(worktree), resolved)
    if _git("-C", str(worktree), "status", "--porcelain=v1"):
        raise ValueError(f"Provisioned repository is not clean: {worktree}")
    if _git("-C", str(worktree), "rev-parse", "HEAD") != resolved:
        raise ValueError("Provisioned repository revision mismatch")
    return ProvisionedRepository(repo, resolved, mirror, worktree)
