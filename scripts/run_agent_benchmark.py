#!/usr/bin/env python3
"""Generate Agent benchmark predictions or run the pinned official evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent_benchmarks import AgentBenchmarkInstance, AgentBenchmarkResult, get_agent_benchmark, list_agent_benchmark_ids  # noqa: E402
from src.agent_benchmarks.common import prediction_sha256, provision_repository, read_predictions, run_harness, write_predictions  # noqa: E402
from src.agent_benchmarks.execution import load_dataset_records, select_records  # noqa: E402
from src.agent_runner import RepositoryTask, VLLMServer, get_agent_runner, list_agent_system_ids, load_agent_system_spec, normalize_endpoint, resolve_artifact, serving_provenance  # noqa: E402
from src.agent_runner.serving import build_vllm_command, resolve_granite_parser  # noqa: E402
from src.utils.identity import build_resume_identity, canonical_sha256, require_matching_resume_identity  # noqa: E402


RUN_MANIFEST_SCHEMA_VERSION = "3.0"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ids(values: list[str]) -> tuple[str, ...]:
    return tuple(item for value in values for item in value.split(",") if item)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("generate", "evaluate"), required=True)
    parser.add_argument("--benchmark", choices=list_agent_benchmark_ids(), required=True)
    parser.add_argument("--system", choices=list_agent_system_ids(), required=True)
    parser.add_argument("--artifact-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--instance-id", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--repo-cache-root", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--allow-incomplete-dataset", action="store_true", help="Only for tiny software smoke fixtures")
    parser.add_argument("--endpoint")
    parser.add_argument(
        "--allow-unverified-external-endpoint", action="store_true",
        help="Explicitly accept that an external endpoint's checkpoint digest is unverifiable",
    )
    parser.add_argument("--evaluator-workers", type=int, default=1)
    parser.add_argument("--evaluator-timeout", type=float, default=7200)
    parser.add_argument("--agent-timeout", type=float, default=3600)
    parser.add_argument("--task-repo", type=Path)
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gold", action="store_true")
    parser.add_argument("--check-runtime", action="store_true")
    parser.add_argument("--allow-unverified-model", action="store_true")
    return parser


def _generation_contract(
    *, artifact: Any, adapter: Any, system: Any, records: list[dict[str, Any]],
) -> dict[str, Any]:
    return build_resume_identity(
        artifact_content_sha256=artifact.inventory_sha256,
        artifact_manifest_provenance_sha256=artifact.manifest_provenance_sha256,
        benchmark_id=adapter.spec.benchmark_id,
        benchmark_config_sha256=canonical_sha256(adapter.spec.to_dict()),
        system_id=system.system_id,
        system_config_hash=system.canonical_config_hash,
        task_set_sha256=canonical_sha256(records),
    )


def _read_manifest(path: Path, *, label: str) -> tuple[dict[str, object], bytes]:
    if not path.is_file():
        raise SystemExit(f"{label} is missing: {path}")
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as error:
        raise SystemExit(f"{label} is invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise SystemExit(f"{label} must contain a JSON object: {path}")
    return value, payload


def _validate_generation_upstream(
    generation_path: Path,
    predictions_path: Path,
    *,
    expected: dict[str, Any],
) -> dict[str, Any]:
    generation_manifest, generation_bytes = _read_manifest(
        generation_path, label="Generate phase manifest",
    )
    mismatches = [
        key for key, value in expected.items()
        if generation_manifest.get(key) != value
    ]
    if not isinstance(generation_manifest.get("ended_at"), str):
        mismatches.append("ended_at")
    if mismatches:
        raise SystemExit(
            "Generate phase manifest identity/status mismatch: "
            + ", ".join(mismatches)
        )
    if not predictions_path.is_file():
        raise SystemExit(f"Generated predictions file is missing: {predictions_path}")
    predictions_hash = hashlib.sha256(predictions_path.read_bytes()).hexdigest()
    if generation_manifest.get("prediction_file_sha256") != predictions_hash:
        raise SystemExit(
            "Generated predictions file hash does not match generate phase manifest"
        )
    return {
        "mode": "generated_predictions",
        "generation_manifest_path": str(generation_path),
        "generation_manifest_sha256": hashlib.sha256(generation_bytes).hexdigest(),
        "prediction_file_sha256": predictions_hash,
        "generation_contract_identity": expected["generation_contract_identity"],
    }


def main() -> int:
    args = _parser().parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_id) or args.run_id in {".", ".."}:
        raise SystemExit("--run-id must be a safe path component")
    if args.resume and args.overwrite:
        raise SystemExit("--resume and --overwrite are mutually exclusive")
    if args.endpoint and not args.allow_unverified_external_endpoint:
        raise SystemExit(
            "--endpoint requires --allow-unverified-external-endpoint because the "
            "remote checkpoint identity cannot be verified"
        )
    if args.allow_unverified_external_endpoint and not args.endpoint:
        raise SystemExit("--allow-unverified-external-endpoint requires --endpoint")
    adapter = get_agent_benchmark(args.benchmark)
    system = load_agent_system_spec(args.system)
    artifact = resolve_artifact(
        system, args.artifact_path,
        allow_unverified_model=args.allow_unverified_model,
    )
    external_endpoint = normalize_endpoint(args.endpoint) if args.endpoint else None
    run_dir = (args.output_root / args.benchmark / args.system / args.run_id).expanduser().resolve()
    phase_manifest_path = run_dir / f"{args.phase}_run_manifest.json"
    previous_manifest = None
    previous_path = phase_manifest_path
    if previous_path.is_file():
        previous_manifest = json.loads(previous_path.read_text(encoding="utf-8"))
    if args.overwrite and args.phase == "generate" and run_dir.exists():
        shutil.rmtree(run_dir)
    if args.overwrite and args.phase == "evaluate" and (run_dir / "evaluation").exists():
        shutil.rmtree(run_dir / "evaluation")
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume and args.phase == "generate":
        raise SystemExit(f"Run output already exists: {run_dir}; use --resume or --overwrite")
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = run_dir / "predictions.jsonl"
    instance_ids = _ids(args.instance_id)
    records = []
    resume_identity = None
    generation_contract = None
    if args.phase == "generate" or not args.gold:
        records = select_records(
            load_dataset_records(
                adapter.spec.dataset, offline=args.offline, local_path=args.dataset_path,
                validate_expected_count=not args.allow_incomplete_dataset,
            ),
            instance_ids=instance_ids, limit=args.limit,
        )
        generation_contract = _generation_contract(
            artifact=artifact, adapter=adapter, system=system, records=records,
        )
    if args.phase == "generate":
        resume_identity = build_resume_identity(
            generation_contract_sha256=generation_contract["sha256"],
            agent_timeout=args.agent_timeout,
            serving_mode="external_endpoint" if external_endpoint else "managed_subprocess",
            external_endpoint=external_endpoint,
            allow_unverified_external_endpoint=args.allow_unverified_external_endpoint,
            dry_run=args.dry_run,
        )
        if args.resume:
            if previous_manifest is None:
                raise SystemExit("Cannot resume Agent benchmark: existing run_manifest.json is missing")
            try:
                require_matching_resume_identity(
                    previous_manifest, resume_identity, scope="Agent benchmark generation",
                )
            except ValueError as error:
                raise SystemExit(str(error)) from error
    manifest = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION, "benchmark": adapter.spec.to_dict(), "system_id": system.system_id,
        "system_config_hash": system.canonical_config_hash,
        "artifact": artifact.to_dict(), "run_id": args.run_id, "phase": args.phase,
        "selection": {"instance_ids": list(instance_ids), "limit": args.limit},
        "execution_options": {"offline": args.offline, "resume": args.resume, "overwrite": args.overwrite, "agent_timeout": args.agent_timeout, "evaluator_timeout": args.evaluator_timeout, "evaluator_workers": args.evaluator_workers, "serving_mode": "external_endpoint" if external_endpoint else "managed_subprocess", "external_endpoint": external_endpoint, "allow_unverified_external_endpoint": args.allow_unverified_external_endpoint, "dataset_path": str(args.dataset_path) if args.dataset_path else "configured_exact_snapshot", "task_repo": str(args.task_repo) if args.task_repo else ("configured_pinned_task_repo" if args.benchmark.startswith("swebench_") else "not_applicable"), "allow_incomplete_dataset": args.allow_incomplete_dataset, "gold": args.gold},
        "paths": {"run_dir": str(run_dir), "predictions": str(predictions_path)}, "started_at": _now(),
        "model_provenance_policy": "explicit_unverified_opt_in" if args.allow_unverified_model else "verified_dense_or_canonical_artifact_required",
    }
    if resume_identity is not None:
        manifest["resume_identity"] = resume_identity
    if generation_contract is not None:
        manifest["generation_contract_identity"] = generation_contract
    if args.phase == "generate":
        manifest["serving_provenance"] = serving_provenance(
            artifact,
            external=external_endpoint is not None,
            allow_unverified_external_endpoint=args.allow_unverified_external_endpoint,
        )
    def write_manifest() -> None:
        _write(phase_manifest_path, manifest)
        _write(run_dir / "run_manifest.json", manifest)

    if args.phase == "generate":
        manifest["status"] = "running"
        if not args.resume:
            write_manifest()
    if args.phase == "evaluate":
        if args.gold:
            manifest["evaluation_input"] = {
                "mode": "gold_evaluator_only",
                "prediction_provenance_required": False,
            }
        else:
            assert generation_contract is not None
            generation_path = run_dir / "generate_run_manifest.json"
            required_equal = {
                "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
                "phase": "generate",
                "status": "completed",
                "run_id": args.run_id,
                "benchmark": adapter.spec.to_dict(),
                "system_id": system.system_id,
                "system_config_hash": system.canonical_config_hash,
                "selection": {"instance_ids": list(instance_ids), "limit": args.limit},
                "generation_contract_identity": generation_contract,
            }
            manifest["evaluation_input"] = _validate_generation_upstream(
                generation_path, predictions_path, expected=required_equal,
            )
        manifest["status"] = "running"
        if args.check_runtime:
            if shutil.which("docker") is None:
                raise SystemExit("Docker executable is unavailable")
            docker = subprocess.run(("docker", "info"), check=False, capture_output=True, text=True, timeout=15)
            if docker.returncode:
                raise SystemExit(f"Docker daemon check failed: {docker.stderr.strip()}")
        predictions = []
        if not args.gold:
            predictions = read_predictions(predictions_path)
        if args.benchmark.startswith("swebench_"):
            task_repo = args.task_repo or (ROOT / ".agent_benchmarks" / Path(adapter.spec.metadata["task_repo"]).stem)
            if not args.dry_run and not task_repo.is_dir():
                raise SystemExit(f"Pinned SWE-bench task repository is missing: {task_repo}; run setup_agent_benchmarks.py")
            command = adapter.evaluator_command(predictions_path, args.run_id, args.evaluator_workers, instance_ids=instance_ids, task_repo=task_repo, report_dir=run_dir / "evaluation", gold=args.gold, dataset_path=args.dataset_path)
            evaluator_cwd = ROOT / adapter.spec.harness.source_checkout
            evaluator_env = None
        else:
            command = adapter.evaluator_command(predictions_path, args.run_id, args.evaluator_workers, instance_ids=instance_ids, gold=args.gold, dataset_path=args.dataset_path)
            evaluator_cwd = run_dir / "evaluation"
            checkout = (ROOT / adapter.spec.harness.source_checkout).resolve()
            evaluator_env = {"PYTHONPATH": f"{checkout}:{checkout / 'src'}"}
        evaluation_started_at = _now()
        evaluation_started = time.monotonic()
        harness = run_harness(command, cwd=evaluator_cwd, output_dir=run_dir / "evaluation", timeout=args.evaluator_timeout, dry_run=args.dry_run, environment=evaluator_env)
        evaluation_runtime = time.monotonic() - evaluation_started
        manifest["evaluator"] = harness.to_dict()
        if not args.dry_run:
            report_path = args.report_path
            if report_path is None:
                pattern_root = run_dir / "evaluation" if args.benchmark.startswith("swebench_") else run_dir / "evaluation" / "evaluation_results"
                candidates = sorted(pattern_root.glob(f"*.{args.run_id}.json"))
                if len(candidates) != 1:
                    raise SystemExit(f"Could not uniquely locate official report under {pattern_root}; pass --report-path")
                report_path = candidates[0]
            summary = adapter.parse_report(report_path)
            if args.benchmark == "swebench_multilingual":
                instances = {}
                for path in (run_dir / "instances").glob("*/generation_manifest.json"):
                    raw = json.loads(path.read_text(encoding="utf-8"))["instance"]
                    item = AgentBenchmarkInstance(**raw)
                    instances[item.instance_id] = item
                summary = adapter.aggregate(summary, instances)
            _write(run_dir / "evaluation" / "aggregate.json", summary)
            manifest["aggregate"] = summary
            ended = _now()
            results = []
            for prediction in predictions:
                instance_id = prediction["instance_id"]
                verdict = summary.get("per_instance", {}).get(instance_id)
                success = None if verdict is None else bool(verdict.get("resolved", verdict.get("success", False)))
                result = AgentBenchmarkResult(
                    args.benchmark, instance_id, args.system, artifact.to_dict(),
                    str(run_dir / "instances" / instance_id.replace("/", "__") / "agent_result.json"),
                    str(predictions_path), prediction_sha256(prediction),
                    "missing" if verdict is None else "completed", success, str(report_path),
                    {"harness_repository": adapter.spec.harness.repository, "harness_revision": adapter.spec.harness.revision, "parser_id": adapter.spec.result_parser_id},
                    evaluation_started_at, ended, evaluation_runtime,
                    "evaluation" if verdict is None else "none",
                    {"language": verdict.get("language")} if verdict and verdict.get("language") else {},
                )
                results.append(result.to_dict())
            results_path = run_dir / "evaluation" / "benchmark_results.jsonl"
            results_path.write_text("".join(json.dumps(item, sort_keys=True, ensure_ascii=False) + "\n" for item in results), encoding="utf-8")
        manifest["status"] = "completed"
        manifest["ended_at"] = _now()
        write_manifest()
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return 0

    prior = {item["instance_id"]: item for item in read_predictions(predictions_path)} if args.resume and predictions_path.is_file() else {}
    if prior and previous_manifest is not None:
        expected_predictions_hash = previous_manifest.get("prediction_file_sha256")
        actual_predictions_hash = hashlib.sha256(predictions_path.read_bytes()).hexdigest()
        if expected_predictions_hash != actual_predictions_hash:
            raise SystemExit(
                "Cannot resume Agent benchmark generation: predictions file hash changed"
            )
    if args.resume:
        write_manifest()
    runner = get_agent_runner(system)
    parser_plugin = resolve_granite_parser(system)
    if args.dry_run:
        endpoint = external_endpoint or f"http://127.0.0.1:{system.serving.port}"
        manifest["vllm_command"] = (
            None if external_endpoint else list(
                build_vllm_command(system.serving, artifact, parser_plugin=parser_plugin)
            )
        )
        manifest["instances"] = []
        for row in records:
            instance = adapter.load_instance(row)
            expected_workspace = args.workspace_root.expanduser().resolve() / instance.instance_id.replace("/", "-")
            task = RepositoryTask(instance.instance_id, expected_workspace, instance.problem_statement, instance.base_commit)
            manifest["instances"].append({"instance": instance.to_dict(), "agent_command": list(runner.command(task, endpoint, run_dir / "instances" / instance.instance_id / "agent"))})
        manifest["status"] = "dry_run_validated"
        manifest["ended_at"] = _now()
        write_manifest()
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return 0
    server = VLLMServer(
        system, artifact, run_dir / "serving", endpoint=external_endpoint,
        allow_unverified_external_endpoint=args.allow_unverified_external_endpoint,
        parser_plugin=parser_plugin,
    ).start()
    manifest["serving"] = server.manifest()
    _write(run_dir / "serving" / "serving_manifest.json", server.manifest())
    outcomes: dict[str, object] = {}
    try:
        for row in records:
            instance = adapter.load_instance(row)
            if instance.instance_id in prior:
                outcomes[instance.instance_id] = {"status": "resumed_patch_generated"}
                continue
            instance_dir = run_dir / "instances" / instance.instance_id.replace("/", "__")
            provisioned = None
            try:
                provisioned = provision_repository(instance.repo, instance.base_commit, instance.instance_id, repo_cache_root=args.repo_cache_root, workspace_root=args.workspace_root, offline=args.offline)
                result = runner.run(
                    adapter.prepare_task(instance, provisioned.worktree_path),
                    server.endpoint,
                    instance_dir / "agent",
                    timeout=args.agent_timeout,
                    artifact_provenance={
                        "requested_local_artifact": artifact.to_dict(),
                        "serving_provenance": server.manifest()["serving_provenance"],
                    },
                )
                _write(instance_dir / "agent_result.json", result.to_dict())
                if result.status == "patch_generated":
                    prior[instance.instance_id] = adapter.build_prediction(result)
                outcomes[instance.instance_id] = {"status": result.status, "error_stage": "none" if result.status == "patch_generated" else "agent"}
                _write(instance_dir / "generation_manifest.json", {"benchmark_id": args.benchmark, "instance": instance.to_dict(), "repository": provisioned.manifest(), "agent_result": str(instance_dir / "agent_result.json"), "prediction_sha256": prediction_sha256(prior[instance.instance_id]) if instance.instance_id in prior else "", "status": result.status})
            except Exception as error:
                outcomes[instance.instance_id] = {"status": "failed", "error_stage": "generation", "error": f"{type(error).__name__}: {error}"}
                _write(instance_dir / "generation_manifest.json", {"benchmark_id": args.benchmark, "instance": instance.to_dict(), "status": "failed", "error_stage": "generation", "error": f"{type(error).__name__}: {error}"})
            finally:
                if provisioned is not None:
                    provisioned.cleanup()
            write_predictions(predictions_path, prior.values())
            manifest["prediction_file_sha256"] = hashlib.sha256(
                predictions_path.read_bytes()
            ).hexdigest()
            manifest["outcomes"] = outcomes
            write_manifest()
    finally:
        server.stop()
        manifest["serving"] = server.manifest()
        _write(run_dir / "serving" / "serving_manifest.json", server.manifest())
    manifest["prediction_file_sha256"] = hashlib.sha256(predictions_path.read_bytes()).hexdigest() if predictions_path.is_file() else ""
    manifest["outcomes"] = outcomes
    manifest["status"] = "completed"
    manifest["ended_at"] = _now()
    write_manifest()
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0 if all(item["status"] in {"patch_generated", "resumed_patch_generated"} for item in outcomes.values()) else 4


if __name__ == "__main__":
    raise SystemExit(main())
