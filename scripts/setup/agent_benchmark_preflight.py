#!/usr/bin/env python3
"""Validate pinned benchmark checkout, environment, dataset path and optional Docker."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent_benchmarks import list_agent_benchmark_ids, load_agent_benchmark_spec  # noqa: E402
from src.agent_benchmarks.registry import get_agent_benchmark  # noqa: E402
from src.agent_benchmarks.swtbench_data import (  # noqa: E402
    SWTEvaluationDatasetSpec,
    dataset_payload,
    derive_evaluation_rows,
    read_filter_ids,
)


def _validate_snapshot(path: Path, expected: dict[str, object]) -> tuple[list[dict[str, object]], dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"Pinned dataset snapshot does not exist: {path}")
    sidecar_path = path.with_suffix(path.suffix + ".manifest.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    identity = {**expected, "sha256": digest}
    if sidecar != identity:
        raise ValueError(f"Dataset snapshot manifest mismatch: {sidecar_path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if len(rows) != expected["row_count"]:
        raise ValueError(f"Dataset snapshot row count mismatch: {path}")
    return rows, sidecar


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=list_agent_benchmark_ids())
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--task-repo", type=Path)
    parser.add_argument("--check-runtime", action="store_true")
    args = parser.parse_args()
    checks: dict[str, object] = {}
    try:
        spec = load_agent_benchmark_spec(args.benchmark)
        checkout = ROOT / spec.harness.source_checkout
        python = ROOT / spec.harness.runtime_python
        if not checkout.is_dir() or not python.is_file():
            raise FileNotFoundError("Pinned benchmark checkout/environment is missing; run setup_agent_benchmarks.py")
        revision = subprocess.run(("git", "-C", str(checkout), "rev-parse", "HEAD"), check=True, capture_output=True, text=True).stdout.strip()
        if revision != spec.harness.revision:
            raise ValueError(f"Harness checkout is {revision}, expected {spec.harness.revision}")
        checks.update({"config": "ok", "harness_revision": revision, "runtime_python": str(python)})
        dirty = subprocess.run(("git", "-C", str(checkout), "status", "--porcelain=v1", "--untracked-files=no"), check=True, capture_output=True, text=True).stdout.strip()
        if dirty:
            raise ValueError(f"Pinned harness checkout has tracked modifications: {checkout}")
        if args.benchmark.startswith("swebench_"):
            version = subprocess.run((str(python), "-c", "import swebench; print(swebench.__version__)"), check=True, capture_output=True, text=True).stdout.strip()
            if version != spec.harness.package_version:
                raise ValueError(f"SWE-bench package is {version}, expected {spec.harness.package_version}")
            checks["package_version"] = version
        else:
            version = subprocess.run((str(python), "-c", "import importlib.metadata as m; print(m.version('SWT-Bench'))"), check=True, capture_output=True, text=True).stdout.strip()
            if version != spec.harness.package_version:
                raise ValueError(f"SWT-Bench package is {version}, expected {spec.harness.package_version}")
            checks["package_version"] = version
        dataset_path = args.dataset_path or (ROOT / spec.dataset.local_path)
        rows, sidecar = _validate_snapshot(dataset_path, {"repo_id": spec.dataset.repo_id, "revision": spec.dataset.revision, "split": spec.dataset.split, "row_count": spec.dataset.expected_task_count})
        checks["dataset"] = {"path": str(dataset_path), **sidecar}
        if args.benchmark == "swtbench_verified":
            evaluation = SWTEvaluationDatasetSpec.from_mapping(spec.metadata["evaluation_dataset"])
            source_path = ROOT / evaluation.source_local_path
            source_rows, source_sidecar = _validate_snapshot(source_path, {
                "repo_id": evaluation.repo_id,
                "revision": evaluation.revision,
                "split": evaluation.split,
                "row_count": evaluation.source_expected_task_count,
            })
            filter_file = checkout / evaluation.filter_path
            filter_ids = read_filter_ids(filter_file, evaluation.filter_expected_count, evaluation.filter_sha256)
            derived_rows = derive_evaluation_rows(source_rows, rows, filter_ids, evaluation)
            derived_path = ROOT / evaluation.local_path
            if not derived_path.is_file():
                raise FileNotFoundError(f"Derived SWT evaluation snapshot does not exist: {derived_path}")
            if derived_path.read_bytes() != dataset_payload(derived_rows):
                raise ValueError("Derived SWT evaluation snapshot content/order mismatch")
            derived_manifest_path = derived_path.with_suffix(derived_path.suffix + ".manifest.json")
            derived_manifest = json.loads(derived_manifest_path.read_text(encoding="utf-8"))
            expected_manifest = {
                "derivation": "swebench_verified_minus_pinned_swt_filter",
                "source_repo_id": evaluation.repo_id,
                "source_revision": evaluation.revision,
                "source_split": evaluation.split,
                "source_row_count": len(source_rows),
                "source_sha256": source_sidecar["sha256"],
                "inference_repo_id": spec.dataset.repo_id,
                "inference_revision": spec.dataset.revision,
                "inference_split": spec.dataset.split,
                "inference_row_count": len(rows),
                "inference_sha256": sidecar["sha256"],
                "swt_harness_revision": spec.harness.revision,
                "filter_path": evaluation.filter_path,
                "filter_count": len(filter_ids),
                "filter_sha256": evaluation.filter_sha256,
                "row_count": len(derived_rows),
                "sha256": hashlib.sha256(derived_path.read_bytes()).hexdigest(),
            }
            if derived_manifest != expected_manifest:
                raise ValueError(f"Derived SWT evaluation manifest mismatch: {derived_manifest_path}")
            command = get_agent_benchmark(args.benchmark).evaluator_command(Path("predictions.jsonl"), "preflight", 1)
            command_dataset = Path(command[command.index("--dataset_name") + 1])
            if command_dataset != derived_path.resolve():
                raise ValueError("SWT evaluator command does not use the derived evaluation snapshot")
            if "--is_swt" in command or "--filter_swt" in command:
                raise ValueError("SWT evaluator command must use original SWE semantics without runtime filtering")
            checks["swt_evaluation"] = {
                "source": {"path": str(source_path), **source_sidecar},
                "filter_path": str(filter_file),
                "filter_count": len(filter_ids),
                "filter_sha256": evaluation.filter_sha256,
                "derived": {"path": str(derived_path), **derived_manifest},
                "inference_derived_id_sets_equal": True,
                "evaluator_command": list(command),
            }
        task_repo = args.task_repo
        if task_repo is None and spec.metadata.get("task_repo"):
            candidate = ROOT / ".agent_benchmarks" / Path(str(spec.metadata["task_repo"])).stem
            task_repo = candidate if candidate.is_dir() else None
        if task_repo is not None and not task_repo.is_dir():
            raise FileNotFoundError(f"Task repository does not exist: {task_repo}")
        if task_repo is not None and spec.metadata.get("task_repo_revision"):
            task_revision = subprocess.run(("git", "-C", str(task_repo), "rev-parse", "HEAD"), check=True, capture_output=True, text=True).stdout.strip()
            if task_revision != spec.metadata["task_repo_revision"]:
                raise ValueError(f"Task repository is {task_revision}, expected {spec.metadata['task_repo_revision']}")
        checks["task_repo"] = str(task_repo) if task_repo else "not_required_for_this_harness"
        if args.check_runtime:
            if shutil.which("docker") is None:
                raise FileNotFoundError("Docker executable is unavailable")
            docker = subprocess.run(("docker", "info"), check=False, capture_output=True, text=True, timeout=15)
            if docker.returncode:
                raise RuntimeError(f"Docker daemon check failed: {docker.stderr.strip()}")
            checks["docker"] = "ok"
        else:
            checks["docker"] = "not_checked"
        print(json.dumps({"status": "ok", "checks": checks}, indent=2))
        return 0
    except Exception as error:
        print(json.dumps({"status": "failed", "checks": checks, "error": f"{type(error).__name__}: {error}"}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
