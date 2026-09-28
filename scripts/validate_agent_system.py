#!/usr/bin/env python3
"""Validate one complete executable Agent system configuration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.agent_evaluation import (  # noqa: E402
    list_agent_system_ids,
    load_agent_system_spec,
)
from src.agent_evaluation.runners import get_agent_runner  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", required=True, choices=list_agent_system_ids())
    parser.add_argument("--check-runtime", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    spec = load_agent_system_spec(args.system)
    if args.check_runtime:
        get_agent_runner(spec).validate_installation()
    output = {
        "status": "executable_config_validated",
        "canonical_config_hash": spec.canonical_config_hash,
        "runtime_checked": args.check_runtime,
        "system": spec.to_dict(),
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
