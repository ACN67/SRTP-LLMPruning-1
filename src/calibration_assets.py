"""Pinned raw calibration asset identities and local dataset adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .downloads import verify_file


C4_ASSET = {
    "source": "allenai/c4",
    "revision": "1588ec454efa1a09f29cd18ddd04fe05fc8653a2",
    "file": "en/c4-train.00000-of-01024.json.gz",
    "size": 319308785,
    "sha256": "8ef8d75b0e045dec4aa5123a671b4564466b0707086a7ed1ba8721626dfffbc9",
    "relative_path": "calibration/c4/en/c4-train.00000-of-01024.json.gz",
}
WIKITEXT_ASSET = {
    "source": "Salesforce/wikitext",
    "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
    "file": "wikitext-2-raw-v1/train-00000-of-00001.parquet",
    "size": 6357543,
    "sha256": "e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7",
    "relative_path": "calibration/wikitext/wikitext-2-raw-v1/train-00000-of-00001.parquet",
}


def calibration_asset_path(root: Path, asset: dict[str, Any]) -> Path:
    return root / asset["relative_path"]


def _verify(root: Path, asset: dict[str, Any]) -> Path:
    path = calibration_asset_path(root, asset)
    verify_file(path, size=asset["size"], hash_type="sha256", expected_hash=asset["sha256"])
    return path


def load_local_c4(root: Path):
    from datasets import load_dataset
    path = _verify(root, C4_ASSET)
    return load_dataset("json", data_files={"train": str(path)}, split="train")


def load_local_wikitext2(root: Path):
    from datasets import load_dataset
    path = _verify(root, WIKITEXT_ASSET)
    return load_dataset("parquet", data_files={"train": str(path)}, split="train")
