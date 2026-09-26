#!/usr/bin/env python3
"""Download and verify canonical runtime model snapshots."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.downloads import verified_download
from src.models import (
    list_model_ids, load_snapshot_manifest, snapshot_manifest_sha256,
    verify_runtime_snapshot,
)
from src.models.snapshots import select_runtime_files


MODELSCOPE_ENDPOINT = "https://modelscope.cn/models"
OFFICIAL_HF_ENDPOINT = "https://huggingface.co"
SMOKE_FILES = ("config.json", "model.safetensors.index.json")


def runtime_file_url(
    manifest: dict, relative_path: str, *, download_source: str, endpoint: str | None = None
) -> str:
    quoted = urllib.parse.quote(relative_path, safe="/")
    if download_source == "domestic":
        base = (endpoint or MODELSCOPE_ENDPOINT).rstrip("/")
        return f"{base}/{manifest['domestic_modelscope_repo']}/resolve/master/{quoted}"
    if download_source == "official":
        base = (endpoint or OFFICIAL_HF_ENDPOINT).rstrip("/")
        return (
            f"{base}/{manifest['canonical_hf_repo']}/resolve/"
            f"{manifest['canonical_hf_revision']}/{quoted}"
        )
    raise ValueError(f"Unsupported download source: {download_source}")


def download_model(
    model_id: str,
    root: Path,
    *,
    download_source: str = "domestic",
    endpoint: str | None = None,
    include_files: Iterable[str] | None = None,
) -> Path:
    manifest = load_snapshot_manifest(model_id)
    selected = select_runtime_files(manifest, include_files)
    target = (root / model_id).resolve()
    target.mkdir(parents=True, exist_ok=True)
    complete_snapshot = include_files is None
    sidecar_path = target / ".srtp_model_source.json"
    if complete_snapshot:
        sidecar_path.unlink(missing_ok=True)

    for item in selected:
        url = runtime_file_url(
            manifest, item["path"], download_source=download_source, endpoint=endpoint,
        )
        verified_download(
            url, target / item["path"], size=int(item["size"]),
            hash_type=item["hash_type"], expected_hash=item["expected_hash"],
        )
    verified = verify_runtime_snapshot(
        model_id, target,
        paths=None if complete_snapshot else [item["path"] for item in selected],
    )

    if complete_snapshot:
        transport_endpoint = endpoint or (
            MODELSCOPE_ENDPOINT if download_source == "domestic" else OFFICIAL_HF_ENDPOINT
        )
        sidecar = {
            "schema_version": 2,
            "project_model_id": model_id,
            "canonical_hf_repo": manifest["canonical_hf_repo"],
            "canonical_hf_revision": manifest["canonical_hf_revision"],
            "download_source": download_source,
            "download_transport": "modelscope_resolve" if download_source == "domestic" else "huggingface_resolve",
            "download_endpoint": transport_endpoint,
            "modelscope_repo": manifest["domestic_modelscope_repo"] if download_source == "domestic" else None,
            "runtime_snapshot_manifest_sha256": snapshot_manifest_sha256(model_id),
            "verified_runtime_files": verified,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    return target


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--model", choices=list_model_ids())
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--root", type=Path, default=Path("/data/models"))
    parser.add_argument("--download-source", choices=("domestic", "official"), default="domestic")
    parser.add_argument("--endpoint", help="explicit source-compatible base endpoint")
    parser.add_argument(
        "--small-file-smoke", action="store_true",
        help="download only config and weight index; never write a success sidecar",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    ids = list_model_ids() if args.all else (args.model,)
    for model_id in ids:
        print(download_model(
            model_id, args.root, download_source=args.download_source, endpoint=args.endpoint,
            include_files=SMOKE_FILES if args.small_file_smoke else None,
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
