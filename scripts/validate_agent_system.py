#!/usr/bin/env python3
"""Validate one planned Agent system config without starting external software."""

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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", required=True, choices=list_agent_system_ids())
    return parser


def main() -> int:
    args = _parser().parse_args()
    spec = load_agent_system_spec(args.system)
    output = {
        "status": "config_validated",
        "execution_available": False,
        "server_started": False,
        "agent_started": False,
        "system": spec.to_dict(),
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
