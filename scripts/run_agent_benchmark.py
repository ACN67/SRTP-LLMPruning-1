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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent_benchmarks import AgentBenchmarkInstance, AgentBenchmarkResult, get_agent_benchmark, list_agent_benchmark_ids  # noqa: E402
from src.agent_benchmarks.common import prediction_sha256, provision_repository, read_predictions, run_harness, write_predictions  # noqa: E402
from src.agent_benchmarks.execution import load_dataset_records, select_records  # noqa: E402
from src.agent_runner import RepositoryTask, VLLMServer, get_agent_runner, list_agent_system_ids, load_agent_system_spec, resolve_artifact  # noqa: E402
from src.agent_runner.serving import build_vllm_command  # noqa: E402


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
    parser.add_argument("--evaluator-workers", type=int, default=1)
    parser.add_argument("--evaluator-timeout", type=float, default=7200)
    parser.add_argument("--agent-timeout", type=float, default=3600)
    parser.add_argument("--task-repo", type=Path)
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gold", action="store_true")
    parser.add_argument("--check-runtime", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_id) or args.run_id in {".", ".."}:
        raise SystemExit("--run-id must be a safe path component")
    if args.resume and args.overwrite:
        raise SystemExit("--resume and --overwrite are mutually exclusive")
    adapter = get_agent_benchmark(args.benchmark)
    system = load_agent_system_spec(args.system)
    artifact = resolve_artifact(system, args.artifact_path)
    run_dir = (args.output_root / args.benchmark / args.system / args.run_id).expanduser().resolve()
    if args.overwrite and args.phase == "generate" and run_dir.exists():
        shutil.rmtree(run_dir)
    if args.overwrite and args.phase == "evaluate" and (run_dir / "evaluation").exists():
        shutil.rmtree(run_dir / "evaluation")
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume and args.phase == "generate":
        raise SystemExit(f"Run output already exists: {run_dir}; use --resume or --overwrite")
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = run_dir / "predictions.jsonl"
    instance_ids = _ids(args.instance_id)
    manifest = {
        "schema_version": "1.0", "benchmark": adapter.spec.to_dict(), "system_id": system.system_id,
        "artifact": artifact.to_dict(), "run_id": args.run_id, "phase": args.phase,
        "selection": {"instance_ids": list(instance_ids), "limit": args.limit},
        "execution_options": {"offline": args.offline, "resume": args.resume, "overwrite": args.overwrite, "agent_timeout": args.agent_timeout, "evaluator_timeout": args.evaluator_timeout, "evaluator_workers": args.evaluator_workers, "endpoint_reused": bool(args.endpoint), "dataset_path": str(args.dataset_path) if args.dataset_path else "configured_exact_snapshot", "task_repo": str(args.task_repo) if args.task_repo else ("configured_pinned_task_repo" if args.benchmark.startswith("swebench_") else "not_applicable"), "allow_incomplete_dataset": args.allow_incomplete_dataset, "gold": args.gold},
        "paths": {"run_dir": str(run_dir), "predictions": str(predictions_path)}, "started_at": _now(),
    }
    if args.phase == "evaluate":
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
        manifest["ended_at"] = _now()
        _write(run_dir / "run_manifest.json", manifest)
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return 0

    records = select_records(load_dataset_records(adapter.spec.dataset, offline=args.offline, local_path=args.dataset_path, validate_expected_count=not args.allow_incomplete_dataset), instance_ids=instance_ids, limit=args.limit)
    prior = {item["instance_id"]: item for item in read_predictions(predictions_path)} if args.resume and predictions_path.is_file() else {}
    runner = get_agent_runner(system)
    if args.dry_run:
        endpoint = args.endpoint or f"http://127.0.0.1:{system.serving.port}"
        manifest["vllm_command"] = list(build_vllm_command(system.serving, artifact))
        manifest["instances"] = []
        for row in records:
            instance = adapter.load_instance(row)
            expected_workspace = args.workspace_root.expanduser().resolve() / instance.instance_id.replace("/", "-")
            task = RepositoryTask(instance.instance_id, expected_workspace, instance.problem_statement, instance.base_commit)
            manifest["instances"].append({"instance": instance.to_dict(), "agent_command": list(runner.command(task, endpoint, run_dir / "instances" / instance.instance_id / "agent"))})
        manifest["status"] = "dry_run_validated"
        manifest["ended_at"] = _now()
        _write(run_dir / "run_manifest.json", manifest)
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return 0
    server = VLLMServer(system, artifact, run_dir / "serving", endpoint=args.endpoint).start()
    outcomes: dict[str, object] = {}
    try:
        for row in records:
            instance = adapter.load_instance(row)
            if instance.instance_id in prior:
                outcomes[instance.instance_id] = {"status": "resumed_success"}
                continue
            instance_dir = run_dir / "instances" / instance.instance_id.replace("/", "__")
            provisioned = None
            try:
                provisioned = provision_repository(instance.repo, instance.base_commit, instance.instance_id, repo_cache_root=args.repo_cache_root, workspace_root=args.workspace_root, offline=args.offline)
                result = runner.run(adapter.prepare_task(instance, provisioned.worktree_path), server.endpoint, instance_dir / "agent", timeout=args.agent_timeout, artifact_provenance=artifact.to_dict())
                _write(instance_dir / "agent_result.json", result.to_dict())
                if result.status == "success":
                    prior[instance.instance_id] = adapter.build_prediction(result)
                outcomes[instance.instance_id] = {"status": result.status, "error_stage": "none" if result.status == "success" else "agent"}
                _write(instance_dir / "generation_manifest.json", {"benchmark_id": args.benchmark, "instance": instance.to_dict(), "repository": provisioned.manifest(), "agent_result": str(instance_dir / "agent_result.json"), "prediction_sha256": prediction_sha256(prior[instance.instance_id]) if instance.instance_id in prior else "", "status": result.status})
            except Exception as error:
                outcomes[instance.instance_id] = {"status": "failed", "error_stage": "generation", "error": f"{type(error).__name__}: {error}"}
                _write(instance_dir / "generation_manifest.json", {"benchmark_id": args.benchmark, "instance": instance.to_dict(), "status": "failed", "error_stage": "generation", "error": f"{type(error).__name__}: {error}"})
            finally:
                if provisioned is not None:
                    provisioned.cleanup()
            write_predictions(predictions_path, prior.values())
    finally:
        server.stop()
    manifest["prediction_file_sha256"] = hashlib.sha256(predictions_path.read_bytes()).hexdigest() if predictions_path.is_file() else ""
    manifest["outcomes"] = outcomes
    manifest["ended_at"] = _now()
    _write(run_dir / "run_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0 if all(item["status"] in {"success", "resumed_success"} for item in outcomes.values()) else 4


if __name__ == "__main__":
    raise SystemExit(main())
