#!/usr/bin/env python3
"""Verify a canonical local model snapshot without loading model weights."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.models import (
    get_model_adapter, list_model_ids, load_model_spec, load_snapshot_manifest,
    snapshot_manifest_sha256, verify_runtime_snapshot,
)


def verify_snapshot(model_id: str, path: Path) -> dict:
    from transformers import AutoConfig, AutoTokenizer

    spec = load_model_spec(model_id)
    manifest = load_snapshot_manifest(model_id)
    path = path.resolve()
    verified = verify_runtime_snapshot(model_id, path)

    sidecar_path = path / ".srtp_model_source.json"
    if not sidecar_path.is_file():
        raise ValueError("Snapshot is missing .srtp_model_source.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    for key, expected in {
        "schema_version": 2,
        "project_model_id": model_id,
        "canonical_hf_repo": manifest["canonical_hf_repo"],
        "canonical_hf_revision": manifest["canonical_hf_revision"],
        "runtime_snapshot_manifest_sha256": snapshot_manifest_sha256(model_id),
        "verified_runtime_files": verified,
    }.items():
        if sidecar.get(key) != expected:
            raise ValueError(f"Snapshot provenance mismatch for {key}")
    if not sidecar.get("download_endpoint") or not sidecar.get("download_transport"):
        raise ValueError("Snapshot provenance is missing transport information")

    config = AutoConfig.from_pretrained(
        str(path), trust_remote_code=spec.trust_remote_code, local_files_only=True
    )
    get_model_adapter(spec).validate_config(config, spec)
    architectures = getattr(config, "architectures", None) or []
    if spec.expected_model_class not in architectures:
        raise ValueError(
            f"Snapshot config architectures {architectures!r} do not include {spec.expected_model_class!r}"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        str(path), trust_remote_code=spec.trust_remote_code, local_files_only=True
    )

    index_path = path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Invalid or empty model.safetensors.index.json weight_map")
    referenced = sorted(set(weight_map.values()))
    declared_shards = sorted(
        item["path"] for item in manifest["required_runtime_files"]
        if item["path"].endswith(".safetensors")
    )
    if referenced != declared_shards:
        raise ValueError("Weight index shards do not match the runtime snapshot manifest")
    return {
        "status": "pass", "project_model_id": model_id,
        "config_class": type(config).__name__, "tokenizer_class": type(tokenizer).__name__,
        "runtime_file_count": len(verified), "referenced_weight_shards": len(referenced),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=list_model_ids())
    parser.add_argument("--path", type=Path)
    parser.add_argument("--root", type=Path, default=Path("/data/models"))
    args = parser.parse_args()
    print(json.dumps(verify_snapshot(args.model, args.path or args.root / args.model), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
