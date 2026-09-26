#!/usr/bin/env python3
"""Generate and evaluate pinned HumanEval, MBPP, or LiveCodeBench runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.evaluation import (  # noqa: E402
    get_benchmark, list_benchmarks, load_evaluation_profile, task_id_hash,
)
from src.evaluation.generation import generate_one  # noqa: E402
from src.models import (  # noqa: E402
    LoadOptions, get_model_adapter, list_model_ids, load_model_artifact, load_model_spec,
)


def _safe_component(value: str) -> str:
    if value in {".", ".."} or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value) is None:
        raise argparse.ArgumentTypeError("value must be one safe path component")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("generate", "evaluate"), required=True)
    parser.add_argument("--benchmark", choices=list_benchmarks(), required=True)
    parser.add_argument("--model", choices=list_model_ids(), required=True)
    parser.add_argument("--artifact-path", type=Path)
    parser.add_argument("--artifact-kind", choices=("dense", "pruned"), default="dense")
    parser.add_argument("--artifact-label", type=_safe_component, default="dense")
    parser.add_argument("--output-root", type=Path, default=Path("/data/results"))
    parser.add_argument("--run-id", type=_safe_component, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--num-trials", type=int)
    parser.add_argument("--device")
    parser.add_argument("--device-map")
    parser.add_argument("--dtype")
    parser.add_argument("--cache-dir", type=Path, default=Path("/data/cache"))
    parser.add_argument("--offline", "--local-files-only", action="store_true")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--resume", action="store_true")
    output.add_argument("--overwrite", action="store_true")
    parser.add_argument("--timeout", type=float, default=3.0)
    return parser


def _run_directory(args: argparse.Namespace) -> Path:
    return (args.output_root / args.benchmark / args.model / args.artifact_label / args.run_id).expanduser().resolve()


def _prepare_directory(path: Path, args: argparse.Namespace) -> None:
    repository = REPOSITORY_ROOT.resolve()
    if path == Path(path.anchor) or path == repository or repository in path.parents:
        raise ValueError("Benchmark run directory must be outside the repository and filesystem root")
    if args.artifact_path is not None:
        artifact = args.artifact_path.expanduser().resolve()
        if path == artifact or path in artifact.parents or artifact in path.parents:
            raise ValueError("Benchmark run directory must not overlap the model artifact")
    if args.phase == "generate" and path.exists() and any(path.iterdir()):
        if args.overwrite:
            shutil.rmtree(path)
        elif not args.resume:
            raise ValueError("Run directory is non-empty; use --resume or --overwrite")
    if args.phase == "evaluate" and path.exists():
        evaluation_outputs = tuple(
            path / name for name in ("evaluation.json", "evaluation_manifest.json", "errors.jsonl")
        )
        existing_outputs = [output for output in evaluation_outputs if output.exists()]
        if existing_outputs and not args.overwrite:
            raise ValueError("Evaluation outputs already exist; use --overwrite")
        if args.overwrite:
            for output in evaluation_outputs:
                output.unlink(missing_ok=True)
    path.mkdir(parents=True, exist_ok=True)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")


def _versions() -> dict[str, str | None]:
    result = {}
    for package in ("torch", "transformers", "datasets"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def _git_commit() -> str | None:
    injected = os.environ.get("SRTP_PROJECT_GIT_COMMIT")
    if injected and injected != "unknown":
        return injected
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, check=True,
                              capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _source_dirty() -> bool | None:
    injected = os.environ.get("SRTP_SOURCE_DIRTY")
    if injected is not None:
        value = injected.strip().lower()
        if value in {"1", "true", "yes"}:
            return True
        if value in {"0", "false", "no"}:
            return False
        if value not in {"", "unknown"}:
            raise ValueError(f"Invalid SRTP_SOURCE_DIRTY value: {injected!r}")
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
            cwd=REPOSITORY_ROOT, check=True, capture_output=True, text=True,
        ).stdout
        return bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_provenance(args: argparse.Namespace) -> dict[str, Any]:
    if args.artifact_path is None:
        return {"kind": args.artifact_kind, "label": args.artifact_label, "path": None}
    path = args.artifact_path.resolve()
    result: dict[str, Any] = {"kind": args.artifact_kind, "label": args.artifact_label, "path": str(path)}
    sidecar = path / ".srtp_model_source.json"
    if sidecar.is_file():
        result["model_source"] = json.loads(sidecar.read_text(encoding="utf-8"))
    pruning = path / "pruning_manifest.json"
    if pruning.is_file():
        artifact_manifest = json.loads(pruning.read_text(encoding="utf-8"))
        result["pruning"] = artifact_manifest.get("pruning")
        result["source_model"] = artifact_manifest.get("model")
        result["pruning_manifest_sha256"] = _sha256(pruning)
    generation_config = path / "generation_config.json"
    if generation_config.is_file():
        result["generation_config_sha256"] = _sha256(generation_config)
    return result


def _selected_tasks(args: argparse.Namespace, benchmark: Any) -> tuple[list[Any], int]:
    tasks = benchmark.load_tasks()
    full_count = len(tasks)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        tasks = tasks[:args.limit]
    return tasks, full_count


def _record_key(record: dict[str, Any]) -> tuple[str, int]:
    return str(record["task_id"]), int(record["trial_index"])


def _validate_identity(record: dict[str, Any], args: argparse.Namespace, benchmark: Any, profile: Any) -> None:
    expected = {
        "benchmark": args.benchmark, "project_model_id": args.model,
        "artifact_kind": args.artifact_kind, "artifact_label": args.artifact_label,
        "prompt_protocol": benchmark.spec.prompt_protocol,
        "evaluation_profile_id": profile.profile_id,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"Conflicting generation field {key} for {_record_key(record)}")


def generate(args: argparse.Namespace, run_dir: Path, benchmark: Any, tasks: list[Any], full_count: int, profile: Any, overridden: bool) -> None:
    if args.artifact_path is None:
        raise ValueError("--artifact-path is required for generation")
    expected_keys = {(task.task_id, trial) for task in tasks for trial in range(profile.num_trials)}
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for record in (_read_jsonl(run_dir / "generations.jsonl") if args.resume else []):
        key = _record_key(record)
        if key in by_key:
            raise ValueError(f"Duplicate existing generation for {key}")
        if key not in expected_keys:
            raise ValueError(f"Existing generation {key} is outside this run")
        _validate_identity(record, args, benchmark, profile)
        by_key[key] = record
    spec = load_model_spec(args.model)
    loaded = load_model_artifact(
        spec, get_model_adapter(spec), args.artifact_path, kind=args.artifact_kind,
        options=LoadOptions(cache_dir=args.cache_dir, dtype=args.dtype, device=args.device,
                            device_map=args.device_map, local_files_only=args.offline),
    )
    ordered_keys = [(task.task_id, trial) for task in tasks for trial in range(profile.num_trials)]
    for task in tasks:
        for trial in range(profile.num_trials):
            key = (task.task_id, trial)
            if key in by_key and by_key[key].get("generation_success"):
                continue
            seed = trial
            try:
                generated = generate_one(
                    loaded.model, loaded.tokenizer, benchmark.build_prompt(task),
                    messages=benchmark.build_messages(task), profile=profile, seed=seed,
                )
                raw = generated.pop("raw_generation")
                record = {
                    "benchmark": args.benchmark, "task_id": task.task_id,
                    "trial_index": trial, "seed": seed, "project_model_id": args.model,
                    "artifact_kind": args.artifact_kind, "artifact_label": args.artifact_label,
                    "prompt_protocol": benchmark.spec.prompt_protocol,
                    "evaluation_profile_id": profile.profile_id,
                    **generated, "raw_generation": raw,
                    "processed_generation": benchmark.postprocess_generation(raw),
                    "generation_success": True, "error": None,
                }
            except Exception as error:
                prompt = benchmark.build_prompt(task)
                record = {
                    "benchmark": args.benchmark, "task_id": task.task_id,
                    "trial_index": trial, "seed": seed, "project_model_id": args.model,
                    "artifact_kind": args.artifact_kind, "artifact_label": args.artifact_label,
                    "prompt_protocol": benchmark.spec.prompt_protocol,
                    "evaluation_profile_id": profile.profile_id,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "raw_generation": "", "processed_generation": "",
                    "generation_success": False, "error": f"{type(error).__name__}: {error}",
                }
            by_key[key] = record
            _write_jsonl(run_dir / "generations.jsonl", [by_key[item] for item in ordered_keys if item in by_key])
    manifest = _manifest_base(args, benchmark, tasks, full_count, profile, overridden)
    artifact = {**_artifact_provenance(args), "structure": dict(loaded.structure)}
    manifest.update({"phase": "generation", "effective_evaluation_profile": profile.to_dict(),
                     "artifact": artifact,
                     "source_generation_config_sha256": artifact.get("generation_config_sha256")})
    (run_dir / "generation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def evaluate(args: argparse.Namespace, run_dir: Path, benchmark: Any, tasks: list[Any], full_count: int, profile: Any, overridden: bool) -> None:
    generations_path = run_dir / "generations.jsonl"
    records = _read_jsonl(generations_path)
    expected_keys = {(task.task_id, trial) for task in tasks for trial in range(profile.num_trials)}
    found = [_record_key(record) for record in records]
    if len(found) != len(set(found)) or set(found) != expected_keys:
        raise ValueError("Generation identities must exactly match every (task_id, trial_index)")
    required = {"benchmark", "task_id", "trial_index", "seed", "project_model_id",
                "artifact_kind", "artifact_label", "prompt_protocol", "evaluation_profile_id",
                "prompt_sha256", "raw_generation", "processed_generation", "generation_success", "error"}
    for record in records:
        if required.difference(record):
            raise ValueError(f"Generation record is missing fields: {sorted(required.difference(record))}")
        _validate_identity(record, args, benchmark, profile)
        if record["seed"] != record["trial_index"]:
            raise ValueError("Generation seed schedule must equal trial_index")
    by_key = {_record_key(record): record for record in records}
    outcomes: list[dict[str, Any]] = []
    for trial in range(profile.num_trials):
        for task in tasks:
            record = by_key[(task.task_id, trial)]
            if not record["generation_success"]:
                outcome = {"task_id": task.task_id, "passed": False, "status": "generation_error", "error": record["error"]}
            else:
                outcome = vars(benchmark.evaluate(task, record["processed_generation"], timeout=args.timeout))
            outcomes.append({"trial_index": trial, **outcome})
    trial_scores = []
    for trial in range(profile.num_trials):
        trial_rows = [row for row in outcomes if row["trial_index"] == trial]
        trial_scores.append(sum(row["passed"] for row in trial_rows) / len(tasks) if tasks else 0.0)
    passed = sum(row["passed"] for row in outcomes)
    summary = {
        "benchmark": args.benchmark,
        "benchmark_version": benchmark.spec.metadata.get("release_version", benchmark.spec.source_revision),
        "task_count": len(tasks), "num_trials": profile.num_trials,
        "attempted": len(outcomes), "passed": passed, "failed": len(outcomes) - passed,
        "timeouts": sum(row["status"] == "timeout" for row in outcomes),
        "execution_errors": sum(row["status"] == "execution_error" for row in outcomes),
        "per_trial_pass_at_1": trial_scores,
        "pass_at_1_mean": statistics.fmean(trial_scores) if trial_scores else 0.0,
        "pass_at_1_std": statistics.pstdev(trial_scores) if len(trial_scores) > 1 else 0.0,
    }
    (run_dir / "evaluation.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_jsonl(run_dir / "errors.jsonl", [row for row in outcomes if not row["passed"]])
    manifest = _manifest_base(args, benchmark, tasks, full_count, profile, overridden)
    generated_manifest_path = run_dir / "generation_manifest.json"
    generated_manifest = json.loads(generated_manifest_path.read_text()) if generated_manifest_path.is_file() else {}
    manifest.update({
        "phase": "evaluation", "effective_evaluation_profile": profile.to_dict(),
        "evaluator": {
            "per_test_timeout_seconds": args.timeout,
            "global_timeout_policy": "(timeout + 1) * test_count + 5"
            if args.benchmark == "livecodebench" else None,
        },
        "result_file_hashes": {name: _sha256(run_dir / name) for name in
                               ("generations.jsonl", "evaluation.json", "errors.jsonl")},
        "artifact": generated_manifest.get("artifact", _artifact_provenance(args)),
        "pruning_provenance": generated_manifest.get("artifact", {}).get("pruning"),
    })
    (run_dir / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _manifest_base(args: argparse.Namespace, benchmark: Any, tasks: list[Any], full_count: int, profile: Any, overridden: bool) -> dict[str, Any]:
    return {
        "schema_version": 2, "created_at": datetime.now(timezone.utc).isoformat(),
        "project_git_commit": _git_commit(), "source_dirty": _source_dirty(),
        "image_metadata": {"image": os.environ.get("SRTP_IMAGE"), "image_digest": os.environ.get("SRTP_IMAGE_DIGEST")},
        "versions": _versions(), "benchmark": args.benchmark,
        "benchmark_source_revision": benchmark.spec.source_revision,
        "benchmark_metadata": dict(benchmark.spec.metadata),
        "task_count": len(tasks), "task_ids": [task.task_id for task in tasks],
        "task_id_hash": task_id_hash(tasks), "full_task_count_before_limit": full_count,
        "limited_run": args.limit is not None, "prompt_protocol": benchmark.spec.prompt_protocol,
        "code_extraction_protocol": benchmark.spec.code_extraction_protocol,
        "metric": "pass@1", "project_model_id": args.model,
        "evaluation_profile_id": profile.profile_id, "evaluation_profile_version": profile.profile_version,
        "protocol_override": overridden, "num_trials": profile.num_trials,
        "seed_schedule": {"scope": "trial", "formula": "seed = trial_index", "values": list(range(profile.num_trials))},
        "artifact_kind": args.artifact_kind, "artifact_label": args.artifact_label,
        "pruning_provenance": _artifact_provenance(args).get("pruning"),
    }


def main() -> int:
    args = _parser().parse_args()
    run_dir = _run_directory(args)
    _prepare_directory(run_dir, args)
    benchmark = get_benchmark(args.benchmark)
    profile, overridden = load_evaluation_profile(args.model, args.benchmark).with_overrides(
        max_new_tokens=args.max_new_tokens, num_trials=args.num_trials,
    )
    tasks, full_count = _selected_tasks(args, benchmark)
    if args.phase == "generate":
        generate(args, run_dir, benchmark, tasks, full_count, profile, overridden)
    if args.phase == "evaluate":
        evaluate(args, run_dir, benchmark, tasks, full_count, profile, overridden)
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
