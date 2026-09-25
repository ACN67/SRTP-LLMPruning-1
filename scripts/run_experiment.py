#!/usr/bin/env python3
"""Validate an experiment request and emit a reproducible planned manifest."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.models import get_model_adapter, list_model_ids, load_model_spec  # noqa: E402
from src.pruning import PRUNER_REGISTRY  # noqa: E402


CONFIG_ROOT = REPOSITORY_ROOT / "configs"
EVAL_CONFIG_DIR = CONFIG_ROOT / "eval"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return data


def _available_benchmarks() -> tuple[str, ...]:
    return tuple(sorted(path.stem for path in EVAL_CONFIG_DIR.glob("*.yaml")))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=list_model_ids())
    parser.add_argument("--pruner", required=True, choices=tuple(PRUNER_REGISTRY))
    parser.add_argument("--sparsity", required=True, type=float)
    parser.add_argument("--benchmark", required=True, choices=_available_benchmarks())
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="write the planned manifest under experiments/generated",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="reserved for verified model, pruning, and evaluation implementations",
    )
    return parser


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    if not 0.0 <= args.sparsity < 1.0:
        raise ValueError("--sparsity must be in the half-open interval [0, 1)")

    model = load_model_spec(args.model)
    adapter = get_model_adapter(model)
    pruning = _load_yaml(CONFIG_ROOT / "pruning" / f"{args.pruner}.yaml")
    evaluation = _load_yaml(EVAL_CONFIG_DIR / f"{args.benchmark}.yaml")

    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "planned",
        "model": {
            "project_model_id": model.project_model_id,
            "display_name": model.display_name,
            "huggingface_repo_id": model.huggingface_repo_id,
            "architecture": model.architecture,
            "adapter": adapter.adapter_id,
            "requested_revision": model.revision,
            "resolved_revision": None,
        },
        "pruning": {
            "method": pruning["method"],
            "sparsity": args.sparsity,
            "implementation_status": pruning["implementation_status"],
        },
        "evaluation": {
            "benchmark": evaluation["benchmark"],
            "implementation_status": evaluation["implementation_status"],
        },
    }


def main() -> int:
    args = _parser().parse_args()
    try:
        manifest = build_manifest(args)
    except (KeyError, TypeError, ValueError) as error:
        print(f"configuration error: {error}", file=sys.stderr)
        return 2

    if args.execute:
        print(
            "execution is unavailable: model loading, pruning, and evaluation are placeholders",
            file=sys.stderr,
        )
        return 3

    rendered = json.dumps(manifest, indent=2, ensure_ascii=False)
    print(rendered)

    if args.write_manifest:
        output_dir = REPOSITORY_ROOT / "experiments" / "generated"
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        filename = (
            f"{timestamp}_{args.model}_{args.pruner}_{args.benchmark}_"
            f"s{args.sparsity:g}.json"
        )
        output_path = output_dir / filename
        output_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"manifest written to {output_path.relative_to(REPOSITORY_ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
