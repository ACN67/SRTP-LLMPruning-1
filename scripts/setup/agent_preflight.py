#!/usr/bin/env python3
"""Check prerequisites for one Agent/vLLM system without launching it."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent_runner import get_agent_runner, list_agent_system_ids, load_agent_system_spec, normalize_endpoint, resolve_artifact, serving_provenance  # noqa: E402
from src.agent_runner.serving import port_is_available, resolve_granite_parser  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", required=True, choices=list_agent_system_ids())
    parser.add_argument("--artifact-path", required=True, type=Path)
    parser.add_argument("--endpoint", help="When provided, GPU/vLLM executable checks are skipped")
    parser.add_argument("--allow-unverified-external-endpoint", action="store_true")
    parser.add_argument("--allow-unverified-model", action="store_true")
    args = parser.parse_args()
    if args.endpoint and not args.allow_unverified_external_endpoint:
        parser.error("--endpoint requires --allow-unverified-external-endpoint")
    if args.allow_unverified_external_endpoint and not args.endpoint:
        parser.error("--allow-unverified-external-endpoint requires --endpoint")
    checks: dict[str, object] = {}
    try:
        system = load_agent_system_spec(args.system)
        checks["config"] = "ok"
        artifact = resolve_artifact(
            system, args.artifact_path,
            allow_unverified_model=args.allow_unverified_model,
        )
        checks["artifact"] = artifact.to_dict()
        checks["model_provenance_policy"] = (
            "explicit_unverified_opt_in"
            if args.allow_unverified_model else "verified_dense_or_canonical_artifact_required"
        )
        parser_path = resolve_granite_parser(system)
        checks["parser_plugin"] = str(parser_path) if parser_path else "not_required"
        get_agent_runner(system).validate_installation()
        checks["agent_runner"] = "ok"
        if args.endpoint:
            checks["endpoint"] = normalize_endpoint(args.endpoint)
            checks["serving_provenance"] = serving_provenance(
                artifact, external=True,
                allow_unverified_external_endpoint=True,
            )
            checks["serving"] = "external_endpoint"
        else:
            checks["serving_provenance"] = serving_provenance(
                artifact, external=False,
            )
            executable = (ROOT / system.serving.executable).resolve()
            if not executable.is_file():
                raise FileNotFoundError(f"vLLM executable is missing: {executable}")
            version = subprocess.run((str(executable), "--version"), check=True, capture_output=True, text=True).stdout
            if system.serving.version not in version:
                raise ValueError(f"Expected vLLM {system.serving.version}, got {version.strip()}")
            if shutil.which("nvidia-smi") is None:
                raise RuntimeError("nvidia-smi is unavailable; managed vLLM requires a CUDA GPU")
            if not port_is_available(system.serving.host, system.serving.port):
                raise RuntimeError(f"Port {system.serving.port} is already in use")
            checks["serving"] = "ok"
        if getattr(system.agent, "runtime", "local") == "docker" and shutil.which("docker") is None:
            raise RuntimeError("OpenHands docker runtime selected but Docker is unavailable")
        print(json.dumps({"status": "ok", "checks": checks}, indent=2, ensure_ascii=False))
        return 0
    except Exception as error:
        print(json.dumps({"status": "failed", "checks": checks, "error": f"{type(error).__name__}: {error}"}, indent=2, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
