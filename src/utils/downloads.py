"""Small dependency-free streaming download and content verification helpers."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path
from typing import Any


def file_hash(path: Path, hash_type: str, size: int | None = None) -> str:
    if hash_type == "sha256":
        digest = hashlib.sha256()
    elif hash_type == "git_blob_sha1":
        actual_size = path.stat().st_size if size is None else size
        digest = hashlib.sha1(f"blob {actual_size}\0".encode("ascii"))
    else:
        raise ValueError(f"Unsupported hash type: {hash_type}")
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, *, size: int, hash_type: str, expected_hash: str) -> None:
    if not path.is_file():
        raise ValueError(f"Required file is missing: {path}")
    actual_size = path.stat().st_size
    if actual_size != size:
        raise ValueError(f"File size mismatch for {path}: {actual_size} != {size}")
    actual_hash = file_hash(path, hash_type, size)
    if actual_hash != expected_hash:
        raise ValueError(f"File hash mismatch for {path}: {actual_hash} != {expected_hash}")


def verified_download(
    url: str,
    target: Path,
    *,
    size: int,
    hash_type: str,
    expected_hash: str,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """Stream to ``.part``, verify content, then atomically replace the target."""

    target = target.resolve()
    if target.is_file():
        try:
            verify_file(target, size=size, hash_type=hash_type, expected_hash=expected_hash)
            return {"path": str(target), "downloaded": False, "url": url}
        except ValueError:
            pass
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response, part.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        verify_file(part, size=size, hash_type=hash_type, expected_hash=expected_hash)
        part.replace(target)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    return {"path": str(target), "downloaded": True, "url": url}
