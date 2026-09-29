#!/usr/bin/env python3
"""Install pinned SWE-bench and SWT-Bench in isolated environments."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "configs" / "agent_benchmarks"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent_benchmarks.swtbench_data import (  # noqa: E402
    SWTEvaluationDatasetSpec,
    dataset_payload,
    derive_evaluation_rows,
    read_filter_ids,
)


def _run(*args: str) -> None:
    subprocess.run(args, cwd=ROOT, check=True)


def _checkout(repository: str, revision: str, destination: Path) -> None:
    if not (destination / ".git").is_dir():
        destination.parent.mkdir(parents=True, exist_ok=True)
        _run("git", "clone", repository, str(destination))
    dirty = subprocess.run(("git", "-C", str(destination), "status", "--porcelain=v1", "--untracked-files=no"), check=True, capture_output=True, text=True).stdout.strip()
    if dirty:
        raise RuntimeError(f"Pinned checkout has tracked modifications: {destination}")
    _run("git", "-C", str(destination), "fetch", "--depth", "1", "origin", revision)
    _run("git", "-C", str(destination), "checkout", "--detach", revision)


def _snapshot_is_valid(path: Path, dataset: dict[str, object]) -> bool:
    sidecar_path = path.with_suffix(path.suffix + ".manifest.json")
    if not path.is_file() or not sidecar_path.is_file():
        return False
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    return sidecar == {
        "repo_id": dataset["repo_id"], "revision": dataset["revision"],
        "split": dataset["split"], "row_count": dataset["expected_task_count"],
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _materialize_dataset(venv_python: Path, dataset: dict[str, object]) -> Path:
    snapshot = ROOT / str(dataset["local_path"])
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if _snapshot_is_valid(snapshot, dataset):
        return snapshot
    materialize = """
import hashlib, json, sys
from pathlib import Path
from datasets import load_dataset
repo, revision, split, output = sys.argv[1:]
rows = load_dataset(repo, revision=revision, split=split).to_list()
payload = json.dumps(rows, ensure_ascii=False, sort_keys=True).encode('utf-8')
path = Path(output); path.write_bytes(payload)
manifest = {'repo_id': repo, 'revision': revision, 'split': split, 'row_count': len(rows), 'sha256': hashlib.sha256(payload).hexdigest()}
path.with_suffix(path.suffix + '.manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
"""
    _run(str(venv_python), "-c", materialize, str(dataset["repo_id"]), str(dataset["revision"]), str(dataset["split"]), str(snapshot))
    if not _snapshot_is_valid(snapshot, dataset):
        raise ValueError(f"Materialized dataset identity/count mismatch: {snapshot}")
    return snapshot


def _setup_swt_evaluation(config: dict[str, object], checkout: Path, venv_python: Path, inference_snapshot: Path) -> None:
    metadata = config["metadata"]
    assert isinstance(metadata, dict)
    spec = SWTEvaluationDatasetSpec.from_mapping(metadata["evaluation_dataset"])
    source_dataset = {
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "split": spec.split,
        "expected_task_count": spec.source_expected_task_count,
        "local_path": spec.source_local_path,
    }
    source_snapshot = _materialize_dataset(venv_python, source_dataset)
    source_rows = json.loads(source_snapshot.read_text(encoding="utf-8"))
    inference_rows = json.loads(inference_snapshot.read_text(encoding="utf-8"))
    filter_file = checkout / spec.filter_path
    filter_ids = read_filter_ids(filter_file, spec.filter_expected_count, spec.filter_sha256)
    derived = derive_evaluation_rows(source_rows, inference_rows, filter_ids, spec)
    payload = dataset_payload(derived)
    inference_manifest = json.loads(inference_snapshot.with_suffix(inference_snapshot.suffix + ".manifest.json").read_text(encoding="utf-8"))
    source_manifest = json.loads(source_snapshot.with_suffix(source_snapshot.suffix + ".manifest.json").read_text(encoding="utf-8"))
    manifest = {
        "derivation": "swebench_verified_minus_pinned_swt_filter",
        "source_repo_id": spec.repo_id,
        "source_revision": spec.revision,
        "source_split": spec.split,
        "source_row_count": len(source_rows),
        "source_sha256": source_manifest["sha256"],
        "inference_repo_id": config["dataset"]["repo_id"],
        "inference_revision": config["dataset"]["revision"],
        "inference_split": config["dataset"]["split"],
        "inference_row_count": len(inference_rows),
        "inference_sha256": inference_manifest["sha256"],
        "swt_harness_revision": config["harness"]["revision"],
        "filter_path": spec.filter_path,
        "filter_count": len(filter_ids),
        "filter_sha256": spec.filter_sha256,
        "row_count": len(derived),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    output = ROOT / spec.local_path
    sidecar = output.with_suffix(output.suffix + ".manifest.json")
    if output.is_file() and sidecar.is_file() and output.read_bytes() == payload and json.loads(sidecar.read_text(encoding="utf-8")) == manifest:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(output)
    sidecar.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _setup(benchmark: str, *, index_url: str | None) -> None:
    config = yaml.safe_load((CONFIG_ROOT / f"{benchmark}.yaml").read_text(encoding="utf-8"))
    harness = config["harness"]
    checkout = ROOT / harness["source_checkout"]
    venv = ROOT / Path(harness["runtime_python"]).parents[1]
    _checkout(harness["repository"], harness["revision"], checkout)
    metadata = config.get("metadata", {})
    if metadata.get("task_repo"):
        task_destination = ROOT / ".agent_benchmarks" / Path(metadata["task_repo"]).stem
        _checkout(metadata["task_repo"], metadata["task_repo_revision"], task_destination)
    if not (venv / "bin" / "python").is_file():
        _run(sys.executable, "-m", "venv", str(venv))
    pip_args = ("--index-url", index_url) if index_url else ()
    _run(str(venv / "bin" / "python"), "-m", "pip", "install", *pip_args, "-e", str(checkout))
    dataset = config["dataset"]
    snapshot = _materialize_dataset(venv / "bin" / "python", dataset)
    if benchmark == "swtbench_verified":
        _setup_swt_evaluation(config, checkout, venv / "bin" / "python", snapshot)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="append", choices=("swebench_verified", "swebench_multilingual", "swtbench_verified"))
    parser.add_argument("--index-url")
    args = parser.parse_args()
    requested = args.benchmark or ["swebench_verified", "swebench_multilingual", "swtbench_verified"]
    for item in dict.fromkeys(requested):
        _setup(item, index_url=args.index_url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
