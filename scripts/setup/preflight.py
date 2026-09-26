#!/usr/bin/env python3
"""Validate the software image, CUDA compatibility, and /data mounts."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.evaluation import (
    list_benchmarks, list_profile_model_ids, load_benchmark_spec,
    load_evaluation_profile,
)
from src.models import (
    get_model_adapter, list_model_ids, load_model_spec, load_snapshot_manifest,
)
from src.pruning import PRUNER_REGISTRY

DATA_DIRECTORIES = ("models", "cache", "datasets", "checkpoints", "results")
PACKAGES = (
    "torch", "transformers", "accelerate", "datasets", "PyYAML", "numpy",
    "tqdm", "huggingface_hub",
)
EXPECTED_VERSIONS = {
    "torch": "2.7.1",
    "transformers": "4.57.1",
    "accelerate": "1.15.0",
    "datasets": "5.0.1",
    "PyYAML": "6.0.3",
    "huggingface_hub": "0.36.2",
    "numpy": "2.4.6",
    "tqdm": "4.70.1",
}


def software_check() -> dict:
    package_versions = {}
    for package in PACKAGES:
        module_name = "yaml" if package == "PyYAML" else package
        try:
            import_module(module_name)
        except ImportError as error:
            raise RuntimeError(f"Required package cannot be imported: {package}") from error
        try:
            installed = version(package)
        except PackageNotFoundError as error:
            raise RuntimeError(f"Required package is missing: {package}") from error
        public = installed.split("+", 1)[0]
        if public != EXPECTED_VERSIONS[package]:
            raise RuntimeError(
                f"{package} version {installed} does not match pinned "
                f"{EXPECTED_VERSIONS[package]}"
            )
        package_versions[package] = installed
    models = list_model_ids()
    for model_id in models:
        spec = load_model_spec(model_id)
        get_model_adapter(spec)
        load_snapshot_manifest(model_id)
    benchmarks = list_benchmarks()
    for benchmark in benchmarks:
        load_benchmark_spec(benchmark)
    if set(list_profile_model_ids()) != set(models):
        raise RuntimeError("Evaluation profile registry does not match model registry")
    for model_id in models:
        for benchmark in benchmarks:
            load_evaluation_profile(model_id, benchmark)
    import yaml
    yaml_files = sorted((REPOSITORY_ROOT / "configs").glob("**/*.yaml"))
    for path in yaml_files:
        if not isinstance(yaml.safe_load(path.read_text(encoding="utf-8")), dict):
            raise RuntimeError(f"Configuration is not a mapping: {path}")
    for relative in ("src", "scripts", "configs", "tests", "third_party"):
        if not (REPOSITORY_ROOT / relative).is_dir():
            raise RuntimeError(f"Project directory is missing: {relative}")
    if set(PRUNER_REGISTRY) != {"magnitude", "wanda", "sparsegpt", "sleb"}:
        raise RuntimeError("Pruner registry does not match the server-ready contract")
    return {
        "status": "pass",
        "python": platform.python_version(),
        "packages": package_versions,
        "models": list(models),
        "pruners": sorted(PRUNER_REGISTRY),
        "benchmarks": list(benchmarks),
        "yaml_config_count": len(yaml_files),
        "data_contract": [f"/data/{name}" for name in DATA_DIRECTORIES],
    }


def gpu_check() -> dict:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    arch_list = torch.cuda.get_arch_list()
    devices = []
    for index in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(index)
        capability = f"sm_{major}{minor}"
        if capability not in arch_list:
            raise RuntimeError(
                f"GPU {index} capability {capability} is absent from torch arch list {arch_list}"
            )
        devices.append({
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "capability": capability,
        })
    return {
        "status": "pass", "torch_cuda": torch.version.cuda,
        "arch_list": arch_list, "device_count": len(devices), "devices": devices,
    }


def paths_check(root: Path = Path("/data")) -> dict:
    results = {}
    for name in DATA_DIRECTORIES:
        path = root / name
        if not path.is_dir():
            raise RuntimeError(f"Required mount directory is missing: {path}")
        probe = path / ".srtp_write_probe"
        try:
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError as error:
            raise RuntimeError(f"Mount directory is not writable: {path}") from error
        results[name] = str(path)
    return {"status": "pass", "paths": results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--software-only", action="store_true")
    modes.add_argument("--gpu", action="store_true")
    modes.add_argument("--paths", action="store_true")
    modes.add_argument("--all", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--data-root", type=Path, default=Path("/data"))
    args = parser.parse_args()
    try:
        result = {}
        if args.software_only or args.all:
            result["software"] = software_check()
        if args.gpu or args.all:
            result["gpu"] = gpu_check()
        if args.paths or args.all:
            result["paths"] = paths_check(args.data_root)
    except Exception as error:
        if args.json:
            print(json.dumps({"status": "fail", "error": str(error)}, indent=2))
        else:
            print(f"SERVER-READY CHECK: FAIL: {error}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps({"status": "pass", **result}, indent=2))
    else:
        print("SERVER-READY SOFTWARE CHECK: PASS" if args.software_only else "SERVER-READY CHECK: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
