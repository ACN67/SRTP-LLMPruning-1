#!/usr/bin/env python3
"""Prefetch pinned benchmark and calibration assets with provenance."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import urllib.request
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.evaluation import load_benchmark_spec, task_id_hash
from src.evaluation.base import BenchmarkTask
from src.evaluation.lcb_protocol import LCB_V6_SHA256
from src.calibration_assets import C4_ASSET, WIKITEXT_ASSET, calibration_asset_path
from src.downloads import verified_download


DEFAULT_HF_ENDPOINT = "https://hf-mirror.net"


def _download(url: str, target: Path) -> bytes:
    if target.is_file():
        return target.read_bytes()
    with urllib.request.urlopen(url) as response:
        payload = response.read()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return payload


def _human_eval(root: Path) -> dict:
    spec = load_benchmark_spec("humaneval")
    url = (
        "https://raw.githubusercontent.com/openai/human-eval/"
        f"{spec.source_revision}/data/HumanEval.jsonl.gz"
    )
    target = root / "humaneval" / "HumanEval.jsonl.gz"
    payload = _download(url, target)
    expected_hash = spec.metadata["dataset_sha256"]
    if _sha256(payload) != expected_hash:
        raise ValueError("Pinned HumanEval SHA256 mismatch")
    with gzip.open(target, "rt", encoding="utf-8") as handle:
        ids = [json.loads(line)["task_id"] for line in handle if line.strip()]
    if len(ids) != 164 or len(set(ids)) != 164:
        raise ValueError("Pinned HumanEval asset must contain 164 unique tasks")
    return {"source": url, "source_revision": spec.source_revision, "task_count": 164,
            "task_id_hash": _ids_hash(ids), "sha256": _sha256(payload)}


def _mbpp(root: Path) -> dict:
    spec = load_benchmark_spec("mbpp")
    url = (
        "https://raw.githubusercontent.com/google-research/google-research/"
        f"{spec.source_revision}/mbpp/mbpp.jsonl"
    )
    target = root / "mbpp" / "mbpp.jsonl"
    payload = _download(url, target)
    digest = _sha256(payload)
    if digest != spec.metadata["source_sha256"]:
        target.unlink(missing_ok=True)
        raise ValueError(f"MBPP SHA256 mismatch: {digest}")
    rows = [json.loads(line) for line in payload.decode().splitlines() if line.strip()]
    ids = [str(row["task_id"]) for row in rows if 11 <= int(row["task_id"]) <= 510]
    if len(ids) != 500:
        raise ValueError("Pinned MBPP test range must contain 500 tasks")
    return {"source": url, "source_revision": spec.source_revision, "sha256": digest,
            "task_count": 500, "task_id_hash": _ids_hash(ids)}


def _lcb(root: Path, endpoint: str = DEFAULT_HF_ENDPOINT) -> dict:
    spec = load_benchmark_spec("livecodebench")
    url = (
        f"{endpoint.rstrip('/')}/datasets/{spec.metadata['dataset_id']}/resolve/"
        f"{spec.metadata['dataset_revision']}/{spec.metadata['dataset_file']}"
    )
    target = root / "livecodebench" / "v6" / "test6.jsonl"
    payload = _download(url, target)
    digest = _sha256(payload)
    if digest != LCB_V6_SHA256:
        target.unlink(missing_ok=True)
        raise ValueError(f"LiveCodeBench v6 SHA256 mismatch: {digest}")
    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line]
    ids = [str(row["question_id"]) for row in rows]
    if len(ids) != 175 or len(set(ids)) != 175:
        raise ValueError("Pinned LiveCodeBench v6 must contain 175 unique tasks")
    return {"dataset_id": spec.metadata["dataset_id"],
            "dataset_revision": spec.metadata["dataset_revision"],
            "dataset_file": "test6.jsonl", "release": "v6",
            "sha256": digest, "download_endpoint": endpoint,
            "local_path": str(target), "task_count": 175,
            "task_id_hash": _ids_hash(ids)}


def _calibration(root: Path, endpoint: str = DEFAULT_HF_ENDPOINT) -> dict:
    result = {}
    for name, asset in (("c4", C4_ASSET), ("wikitext2", WIKITEXT_ASSET)):
        url = (
            f"{endpoint.rstrip('/')}/datasets/{asset['source']}/resolve/"
            f"{asset['revision']}/{asset['file']}"
        )
        target = calibration_asset_path(root, asset)
        verified_download(
            url, target, size=asset["size"], hash_type="sha256",
            expected_hash=asset["sha256"],
        )
        result[name] = {
            "source": asset["source"], "revision": asset["revision"],
            "file": asset["file"], "size": asset["size"], "sha256": asset["sha256"],
            "local_path": str(target), "download_endpoint": endpoint,
        }
    return result


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _ids_hash(ids: list[str]) -> str:
    tasks = [BenchmarkTask("", item, "", {}) for item in ids]
    return task_id_hash(tasks)


def prefetch(
    root: Path, *, benchmarks: bool, calibration: bool,
    hf_endpoint: str = DEFAULT_HF_ENDPOINT,
) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tool": "scripts/setup/prefetch_assets.py",
        "versions": {"datasets": version("datasets")},
        "benchmarks": {}, "calibration": {},
    }
    if benchmarks:
        manifest["benchmarks"] = {
            "humaneval": _human_eval(root), "mbpp": _mbpp(root),
            "livecodebench": _lcb(root, hf_endpoint),
        }
    if calibration:
        manifest["calibration"] = _calibration(root, hf_endpoint)
    (root / "srtp_assets_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--benchmarks", action="store_true")
    modes.add_argument("--calibration", action="store_true")
    modes.add_argument("--all", action="store_true")
    parser.add_argument("--root", type=Path, default=Path("/data/datasets"))
    parser.add_argument("--hf-endpoint", default=DEFAULT_HF_ENDPOINT)
    args = parser.parse_args()
    result = prefetch(
        args.root,
        benchmarks=args.benchmarks or args.all,
        calibration=args.calibration or args.all,
        hf_endpoint=args.hf_endpoint,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
