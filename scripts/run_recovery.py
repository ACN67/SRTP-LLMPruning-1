#!/usr/bin/env python3
"""Run a configured recovery transformation on a canonical model artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from src.artifacts import load_model_artifact, resolve_model_artifact  # noqa: E402
from src.models import LoadOptions, get_model_adapter, list_model_ids, load_model_spec  # noqa: E402
from src.recovery import DatasetConfig, RECOVERY_METHODS, RecoveryConfig, get_recovery_method, prepare_dataset  # noqa: E402


def mapping(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return value


def _validate_output_dir(output_dir: Path, artifact_path: Path) -> Path:
    output = output_dir.expanduser().resolve()
    artifact = artifact_path.expanduser().resolve()
    repository = ROOT.resolve()
    if output == Path(output.anchor) or output == repository or repository in output.parents:
        raise ValueError("Recovery output directory must be outside the repository and filesystem root")
    if output == artifact or output in artifact.parents or artifact in output.parents:
        raise ValueError("Recovery output directory must not overlap the input artifact")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Recovery output directory is not empty: {output}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=RECOVERY_METHODS)
    parser.add_argument("--model", required=True, choices=list_model_ids())
    parser.add_argument("--artifact-path", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset-config", required=True, type=Path)
    parser.add_argument("--dataset-path", type=Path, help="Override local dataset path")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device"); parser.add_argument("--device-map"); parser.add_argument("--dtype")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    try:
        output_dir = _validate_output_dir(args.output_dir, args.artifact_path)
        spec = load_model_spec(args.model)
        adapter = get_model_adapter(spec)
        artifact = resolve_model_artifact(args.artifact_path, spec, adapter)
        config = RecoveryConfig.from_mapping(mapping(args.config))
        dataset_raw = mapping(args.dataset_config)
        if args.dataset_path is not None: dataset_raw["path"] = str(args.dataset_path)
        dataset_config = DatasetConfig.from_mapping(dataset_raw)
        options = LoadOptions(dtype=args.dtype or config.dtype, device=args.device, device_map=args.device_map, local_files_only=args.local_files_only)
        loaded = load_model_artifact(spec, adapter, args.artifact_path, options=options)
        dataset = prepare_dataset(dataset_config, loaded.tokenizer)
        result = get_recovery_method(args.method).recover(
            loaded=loaded,
            input_artifact=artifact,
            spec=spec,
            adapter=adapter,
            dataset=dataset,
            config=config,
            output_dir=output_dir,
            repository_root=ROOT,
        )
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False)); return 0
    except Exception as error:
        print(f"recovery failed: {type(error).__name__}: {error}", file=sys.stderr); return 2


if __name__ == "__main__": raise SystemExit(main())
