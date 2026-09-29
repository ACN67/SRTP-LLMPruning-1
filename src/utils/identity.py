"""Stable, path-independent identities for resumable experiment runs."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


def canonical_sha256(value: Any) -> str:
    """Hash a JSON-compatible value using one canonical serialization."""

    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_resume_identity(**fields: Any) -> dict[str, Any]:
    """Return the auditable identity fields together with their stable hash."""

    identity = dict(fields)
    return {"sha256": canonical_sha256(identity), "fields": identity}


def require_matching_resume_identity(
    previous: Mapping[str, Any], expected: Mapping[str, Any], *, scope: str,
) -> None:
    """Fail closed unless an existing run has exactly the expected identity."""

    actual = previous.get("resume_identity")
    if not isinstance(actual, Mapping):
        raise ValueError(
            f"Cannot resume {scope}: existing output has no resume_identity; "
            "use --overwrite to start a new run"
        )
    if actual.get("sha256") != expected.get("sha256") or actual.get("fields") != expected.get("fields"):
        raise ValueError(
            f"Cannot resume {scope}: artifact, task set, or protocol identity changed; "
            "use --overwrite to start a new run"
        )
