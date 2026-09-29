"""Subprocess adapters for pinned official Agent frameworks."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .base import AgentSystemSpec, MiniSweAgentPlusSpec, OpenHandsSpec
from .result import AgentResult
from .task import RepositoryTask, extract_patch, repository_state


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolved(path: str) -> Path:
    value = Path(path).expanduser()
    # Do not resolve the final symlink: venv/bin/python normally points at the
    # base interpreter, and resolving it would silently discard the venv.
    return value.absolute() if value.is_absolute() else (REPOSITORY_ROOT / value).absolute()


class AgentFrameworkError(RuntimeError):
    pass


class AgentRunner:
    def __init__(self, system: AgentSystemSpec) -> None:
        self.system = system

    def command(
        self, task: RepositoryTask, endpoint: str, output_dir: Path,
    ) -> tuple[str, ...]:
        agent = self.system.agent
        bridge = (
            REPOSITORY_ROOT / "scripts" / "agent_runner" /
            ("miniswe_bridge.py" if isinstance(agent, MiniSweAgentPlusSpec) else "openhands_bridge.py")
        )
        command = [
            str(_resolved(agent.runtime_python)), str(bridge),
            "--repo-path", str(task.repo_path.expanduser().resolve()),
            "--task-file", str((output_dir / "task.md").resolve()),
            "--output-dir", str(output_dir.resolve()),
            "--endpoint", endpoint,
            "--model", self.system.serving.served_model_name,
            "--max-input-tokens", str(self.system.generation.max_input_tokens),
            "--max-output-tokens", str(self.system.generation.max_output_tokens),
        ]
        if isinstance(agent, MiniSweAgentPlusSpec):
            command.extend((
                "--config", str(_resolved(agent.source_checkout) / agent.config_path),
                "--step-limit", str(agent.step_limit),
                "--cost-limit", str(agent.cost_limit),
                "--command-timeout", str(agent.command_timeout_seconds),
            ))
        else:
            command.extend((
                "--max-iterations", str(agent.max_iterations),
                "--command-timeout", str(agent.command_timeout_seconds),
                "--runtime", agent.runtime,
                "--native-tool-calling", str(agent.native_tool_calling).lower(),
                "--mode", agent.mode,
                "--instruction-template", agent.instruction_template_name,
                "--use-hint-text", str(agent.use_hint_text).lower(),
                "--enable-plan-mode", str(agent.enable_plan_mode).lower(),
                "--add-icl-example", str(agent.add_in_context_learning_example).lower(),
            ))
        return tuple(command)

    def validate_installation(self) -> None:
        agent = self.system.agent
        python = _resolved(agent.runtime_python)
        source = _resolved(agent.source_checkout)
        if not python.is_file():
            raise FileNotFoundError(f"Agent runner Python is missing: {python}")
        if not source.is_dir():
            raise FileNotFoundError(f"Pinned Agent source checkout is missing: {source}")
        revision = subprocess.run(
            ("git", "-C", str(source), "rev-parse", "HEAD"), check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        if revision != agent.upstream_revision:
            raise ValueError(
                f"Agent checkout revision {revision} does not match {agent.upstream_revision}"
            )
        tracked_status = subprocess.run(
            (
                "git", "-C", str(source), "status", "--porcelain=v1",
                "--untracked-files=no",
            ),
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        if tracked_status:
            raise ValueError(
                f"Pinned Agent checkout has tracked local modifications: {source}"
            )

    def run(
        self,
        task: RepositoryTask,
        endpoint: str,
        output_dir: Path,
        *,
        timeout: float,
        allow_dirty: bool = False,
        artifact_provenance: Mapping[str, Any] | None = None,
    ) -> AgentResult:
        self.validate_installation()
        before = task.validate(allow_dirty=allow_dirty)
        output = output_dir.expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        (output / "task.md").write_text(task.problem_statement, encoding="utf-8")
        stdout_path = output / "agent.stdout.log"
        stderr_path = output / "agent.stderr.log"
        started_at = _now()
        started = time.monotonic()
        returncode = -1
        error_type = "none"
        error_message = ""
        env = os.environ.copy()
        env.update({
            "OPENAI_API_KEY": self.system.serving.api_key,
            "SRTP_GENERATION_CONFIG": json.dumps(self.system.generation.request_parameters),
            "MSWEA_SILENT_STARTUP": "1",
        })
        try:
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                process = subprocess.Popen(
                    self.command(task, endpoint, output), cwd=REPOSITORY_ROOT,
                    env=env, stdout=stdout, stderr=stderr, start_new_session=True,
                )
                try:
                    returncode = process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    if os.name == "posix":
                        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    else:
                        process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        if os.name == "posix":
                            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                        else:
                            process.kill()
                        process.wait(timeout=5)
                    raise
            if returncode != 0:
                error_type = "AgentFrameworkExit"
                error_message = f"Agent bridge exited with code {returncode}"
        except subprocess.TimeoutExpired:
            error_type = "AgentTimeout"
            error_message = f"Agent exceeded timeout {timeout}s"

        runtime = time.monotonic() - started
        patch, changed_files = extract_patch(task.repo_path)
        patch_path = output / "changes.patch"
        patch_path.write_text(patch, encoding="utf-8")
        patch_hash = hashlib.sha256(patch.encode("utf-8")).hexdigest()
        after = repository_state(task.repo_path)
        bridge_path = output / "bridge_result.json"
        framework_metadata: dict[str, Any] = {
            "reporting_status": "not_reported_by_framework", "returncode": returncode,
        }
        if bridge_path.is_file():
            framework_metadata = json.loads(bridge_path.read_text(encoding="utf-8"))
        status = "patch_generated" if returncode == 0 and bool(patch) else "generation_failed"
        if returncode == 0 and not patch:
            error_type, error_message = "EmptyPatch", "Agent completed without repository changes"
        trajectory = output / "trajectory.json"
        return AgentResult(
            system_id=self.system.system_id,
            project_model_id=self.system.project_model_id,
            artifact_provenance=artifact_provenance or {},
            task_id=task.task_id, status=status,
            started_at=started_at, ended_at=_now(), runtime_seconds=runtime,
            framework=self.system.agent.framework,
            framework_version=self.system.agent.version,
            framework_revision=self.system.agent.upstream_revision,
            effective_agent_config=asdict(self.system.agent), endpoint=endpoint,
            serving_manifest_path=str(output / "serving_manifest.json"),
            trajectory_path=str(trajectory), stdout_path=str(stdout_path),
            stderr_path=str(stderr_path), patch_path=str(patch_path),
            patch_sha256=patch_hash, changed_files=changed_files,
            repository_before=before.to_dict(), repository_after=after.to_dict(),
            framework_metadata=framework_metadata, error_type=error_type,
            error_message=error_message,
        )


def get_agent_runner(system: AgentSystemSpec) -> AgentRunner:
    return AgentRunner(system)
