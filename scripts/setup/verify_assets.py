#!/usr/bin/env python3
"""Read-only verification for prefetched benchmark assets."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.evaluation.lcb_protocol import audit_v6
from src.calibration_assets import (
    C4_ASSET, WIKITEXT_ASSET, calibration_asset_path,
    load_local_c4, load_local_wikitext2,
)
from src.downloads import verify_file

HUMANEVAL_SHA256 = "b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef"
MBPP_SHA256 = "ccf64ceae9c5403bf50a044cb6d505bfd2a2963ee58338ba268fd65beab92a9f"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_calibration_assets(root: Path) -> dict:
    for asset in (C4_ASSET, WIKITEXT_ASSET):
        verify_file(
            calibration_asset_path(root, asset), size=asset["size"],
            hash_type="sha256", expected_hash=asset["sha256"],
        )
    c4 = load_local_c4(root)
    if len(c4) == 0 or not isinstance(c4[0].get("text"), str):
        raise ValueError("C4 first shard has no usable text row")
    wikitext = load_local_wikitext2(root)
    if len(wikitext) != 36718 or not isinstance(wikitext[0].get("text"), str):
        raise ValueError("WikiText-2 raw train structure mismatch")
    return {
        "c4": {key: C4_ASSET[key] for key in ("source", "revision", "file", "size", "sha256")},
        "wikitext2": {key: WIKITEXT_ASSET[key] for key in ("source", "revision", "file", "size", "sha256")},
        "c4_rows": len(c4), "wikitext2_rows": len(wikitext),
    }


def verify_assets(root: Path) -> dict:
    human = root / "humaneval" / "HumanEval.jsonl.gz"
    if _sha256(human) != HUMANEVAL_SHA256:
        raise ValueError("HumanEval asset SHA256 mismatch")
    with gzip.open(human, "rt", encoding="utf-8") as handle:
        human_ids = [json.loads(line)["task_id"] for line in handle if line.strip()]
    if len(human_ids) != 164 or len(set(human_ids)) != 164:
        raise ValueError("HumanEval must contain 164 unique tasks")

    mbpp = root / "mbpp" / "mbpp.jsonl"
    if _sha256(mbpp) != MBPP_SHA256:
        raise ValueError("MBPP asset SHA256 mismatch")
    mbpp_rows = [json.loads(line) for line in mbpp.read_text(encoding="utf-8").splitlines() if line]
    mbpp_ids = [int(row["task_id"]) for row in mbpp_rows if 11 <= int(row["task_id"]) <= 510]
    if mbpp_ids != list(range(11, 511)):
        raise ValueError("MBPP original test IDs must be exactly 11..510")

    lcb = audit_v6(root / "livecodebench" / "v6" / "test6.jsonl")
    calibration = verify_calibration_assets(root)
    return {
        "status": "pass",
        "humaneval": {"task_count": 164, "sha256": HUMANEVAL_SHA256},
        "mbpp": {"task_count": 500, "sha256": MBPP_SHA256},
        "livecodebench": lcb,
        "calibration": calibration,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/datasets"))
    args = parser.parse_args()
    print(json.dumps(verify_assets(args.root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
