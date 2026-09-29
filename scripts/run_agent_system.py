#!/usr/bin/env python3
"""Run one canonical Agent system on a local Git repository task."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent_runner import (  # noqa: E402
    RepositoryTask, VLLMServer, get_agent_runner, list_agent_system_ids,
    load_agent_system_spec, normalize_endpoint, resolve_artifact, serving_provenance,
)
from src.agent_runner.serving import (  # noqa: E402
    ServingStartupError, build_vllm_command, resolve_granite_parser,
)
from src.agent_runner.task import extract_patch  # noqa: E402
from src.utils.identity import build_resume_identity, require_matching_resume_identity  # noqa: E402


SCHEMA_VERSION = "2.0"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(("git", "-C", str(ROOT), *args), check=True, capture_output=True, text=True).stdout.strip()


def _hardware() -> dict[str, Any]:
    result: dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform()}
    command = ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]
    try:
        result["gpus"] = subprocess.run(command, check=True, capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
    except (FileNotFoundError, subprocess.SubprocessError):
        result["gpus"] = []
    result["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", "not_set")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", required=True, choices=list_agent_system_ids())
    parser.add_argument("--artifact-path", type=Path, required=True)
    parser.add_argument("--repo-path", type=Path, required=True)
    parser.add_argument("--task-file", type=Path, required=True)
    parser.add_argument("--task-id")
    parser.add_argument("--base-commit")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--endpoint", help="Reuse an existing OpenAI-compatible vLLM endpoint")
    parser.add_argument(
        "--allow-unverified-external-endpoint", action="store_true",
        help="Explicitly accept that an external endpoint's checkpoint digest is unverifiable",
    )
    parser.add_argument("--keep-server", action="store_true", help="Leave a successfully managed server running")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--tensor-parallel-size", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--agent-timeout", type=float, default=3600)
    parser.add_argument("--parser-plugin", type=Path)
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument("--overwrite", action="store_true")
    output_mode.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-unverified-model", action="store_true")
    return parser


def _write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _sanitized_serving(serving: object) -> dict[str, Any]:
    value = asdict(serving)  # type: ignore[arg-type]
    value.pop("api_key", None)
    value["api_key_status"] = "configured_dummy_value"
    return value


def main() -> int:
    args = _parser().parse_args()
    if args.endpoint and not args.allow_unverified_external_endpoint:
        raise SystemExit(
            "--endpoint requires --allow-unverified-external-endpoint because the "
            "remote checkpoint identity cannot be verified"
        )
    if args.allow_unverified_external_endpoint and not args.endpoint:
        raise SystemExit("--allow-unverified-external-endpoint requires --endpoint")
    if args.resume and args.allow_dirty:
        raise SystemExit("--resume cannot be combined with --allow-dirty")
    output = args.output_dir.expanduser().resolve()
    manifest_path = output / "run_manifest.json"
    previous_resume_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if args.resume and manifest_path.is_file() else None
    )
    if args.resume and previous_resume_manifest is None:
        raise SystemExit("Cannot resume single Agent run: existing run_manifest.json is missing")
    if output.exists() and any(output.iterdir()) and not (args.overwrite or args.resume):
        print(f"Output directory is not empty: {output}; use --overwrite or --resume", file=sys.stderr)
        return 2
    output.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for name in (
            "run_manifest.json", "serving_manifest.json", "trajectory.json",
            "bridge_result.json", "agent.stdout.log", "agent.stderr.log",
            "vllm.stdout.log", "vllm.stderr.log", "changes.patch", "task.md",
        ):
            path = output / name
            if path.is_file():
                path.unlink()

    server: VLLMServer | None = None
    preserve_previous_manifest = False
    manifest: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "started_at": _now()}
    try:
        system = load_agent_system_spec(args.system)
        overrides = {
            name: value for name, value in {
                "host": args.host, "port": args.port,
                "tensor_parallel_size": args.tensor_parallel_size,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "max_num_seqs": args.max_num_seqs,
            }.items() if value is not None
        }
        serving = system.serving.with_overrides(overrides)
        artifact = resolve_artifact(
            system, args.artifact_path,
            allow_unverified_model=args.allow_unverified_model,
        )
        external_endpoint = normalize_endpoint(args.endpoint) if args.endpoint else None
        serving_identity = serving_provenance(
            artifact,
            external=external_endpoint is not None,
            allow_unverified_external_endpoint=args.allow_unverified_external_endpoint,
        )
        parser_plugin = resolve_granite_parser(system, args.parser_plugin)
        task = RepositoryTask(
            task_id=args.task_id or args.task_file.stem,
            repo_path=args.repo_path,
            problem_statement=args.task_file.read_text(encoding="utf-8"),
            base_commit=args.base_commit,
        )
        before = task.validate(allow_dirty=args.allow_dirty or args.resume)
        runner = get_agent_runner(system)
        git_status = _git("status", "--porcelain=v1", "--untracked-files=all")
        manifest.update({
            "project": {"git_commit": _git("rev-parse", "HEAD"), "dirty": bool(git_status)},
            "system_id": system.system_id,
            "system_config_hash": system.canonical_config_hash,
            "artifact": artifact.to_dict(),
            "serving_provenance": serving_identity,
            "effective_serving_config": _sanitized_serving(serving),
            "effective_generation_config": asdict(system.generation),
            "effective_agent_config": asdict(system.agent),
            "runtime_overrides": overrides,
            "runtime_overrides_provenance": "cli_explicit" if overrides else "canonical_config",
            "task": task.to_dict(), "repository_before": before.to_dict(),
            "runtime": _hardware(),
            "pins": {"vllm": serving.version, "agent_version": system.agent.version, "agent_revision": system.agent.upstream_revision},
            "dry_run": args.dry_run,
            "model_provenance_policy": (
                "explicit_unverified_opt_in" if args.allow_unverified_model else "verified_dense_or_canonical_artifact_required"
            ),
        })
        resume_identity = build_resume_identity(
            artifact_content_sha256=artifact.inventory_sha256,
            artifact_manifest_provenance_sha256=artifact.manifest_provenance_sha256,
            system_id=system.system_id,
            system_config_hash=system.canonical_config_hash,
            agent_timeout_seconds=args.agent_timeout,
            serving_mode="external_endpoint" if external_endpoint else "managed_subprocess",
            external_endpoint=external_endpoint,
            allow_unverified_external_endpoint=args.allow_unverified_external_endpoint,
            allow_dirty=args.allow_dirty,
            dry_run=args.dry_run,
            task={
                "task_id": task.task_id,
                "problem_statement_sha256": hashlib.sha256(
                    task.problem_statement.encode("utf-8")
                ).hexdigest(),
                "base_commit": task.base_commit or before.head_commit,
                "repository_head": before.head_commit,
            },
            serving_config=_sanitized_serving(serving),
        )
        manifest["resume_identity"] = resume_identity
        if previous_resume_manifest is not None:
            try:
                require_matching_resume_identity(
                    previous_resume_manifest, resume_identity, scope="single Agent run",
                )
            except ValueError:
                preserve_previous_manifest = True
                raise
            if previous_resume_manifest.get("result", {}).get("status") == "patch_generated":
                current_patch, _changed_files = extract_patch(task.repo_path)
                expected_patch_sha256 = previous_resume_manifest["result"].get("patch_sha256")
                actual_patch_sha256 = hashlib.sha256(current_patch.encode("utf-8")).hexdigest()
                if actual_patch_sha256 != expected_patch_sha256:
                    preserve_previous_manifest = True
                    raise ValueError(
                        "Cannot resume completed single Agent run: repository patch changed"
                    )
                preserve_previous_manifest = True
                print(json.dumps(previous_resume_manifest, indent=2, ensure_ascii=False))
                return 0
            if before.dirty:
                preserve_previous_manifest = True
                raise ValueError(
                    "Cannot continue an incomplete single Agent run from a dirty repository"
                )
        if args.dry_run:
            endpoint = external_endpoint or f"http://127.0.0.1:{serving.port}"
            manifest["endpoint"] = endpoint
            manifest["vllm_command"] = (
                None if external_endpoint else
                list(build_vllm_command(serving, artifact, parser_plugin=parser_plugin))
            )
            manifest["agent_command"] = list(runner.command(task, endpoint, output))
            manifest["result"] = {"status": "dry_run_validated"}
            manifest["ended_at"] = _now()
            _write(manifest_path, manifest)
            print(json.dumps(manifest, indent=2, ensure_ascii=False))
            return 0

        runner.validate_installation()
        server = VLLMServer(
            system, artifact, output, serving=serving, endpoint=external_endpoint,
            allow_unverified_external_endpoint=args.allow_unverified_external_endpoint,
            parser_plugin=parser_plugin, startup_timeout=args.startup_timeout,
        ).start()
        _write(output / "serving_manifest.json", server.manifest())
        result = runner.run(
            task, server.endpoint, output, timeout=args.agent_timeout,
            allow_dirty=args.allow_dirty,
            artifact_provenance={
                "requested_local_artifact": artifact.to_dict(),
                "serving_provenance": server.manifest()["serving_provenance"],
            },
        )
        manifest["endpoint"] = server.endpoint
        manifest["result"] = result.to_dict()
        exit_code = 0 if result.status == "patch_generated" else 4
        if not (args.keep_server and not server.external and exit_code == 0):
            server.stop()
        manifest["serving"] = server.manifest()
        _write(output / "serving_manifest.json", server.manifest())
        hashes = {}
        for name in ("trajectory.json", "agent.stdout.log", "agent.stderr.log", "changes.patch", "vllm.stdout.log", "vllm.stderr.log"):
            path = output / name
            if path.is_file():
                hashes[name] = _sha256(path)
        manifest["output_sha256"] = hashes
        manifest["ended_at"] = _now()
        _write(manifest_path, manifest)
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return exit_code
    except ServingStartupError as error:
        code = 3
        manifest["error"] = {"type": type(error).__name__, "message": str(error)}
    except Exception as error:
        code = 2
        manifest["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        if server is not None and not (args.keep_server and server.process is not None and server.process.poll() is None and "error" not in manifest):
            server.stop()
        manifest["ended_at"] = _now()
        if not preserve_previous_manifest:
            output.mkdir(parents=True, exist_ok=True)
            _write(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False), file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
