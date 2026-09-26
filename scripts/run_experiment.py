#!/usr/bin/env python3
"""Plan an experiment or execute verified Magnitude or Wanda pruning."""

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

from src.models import (  # noqa: E402
    LoadOptions,
    build_model_manifest,
    get_model_adapter,
    list_model_ids,
    load_dense_model,
    load_model_spec,
)
from src.pruning import (  # noqa: E402
    PRUNER_REGISTRY,
    CalibrationConfig,
    PruningRequest,
    WandaPruningContext,
    get_calibration_provider,
    get_pruner,
)

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


def _device_map(value: str | None) -> str | dict[str, Any] | None:
    if value is None:
        return None
    if value.lstrip().startswith("{"):
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise argparse.ArgumentTypeError("JSON device_map must be an object")
        return parsed
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=list_model_ids())
    parser.add_argument("--pruner", required=True, choices=tuple(PRUNER_REGISTRY))
    parser.add_argument("--sparsity", required=True, type=float)
    parser.add_argument("--benchmark", choices=_available_benchmarks())
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="write a planned manifest under experiments/generated",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="load, prune, and save; currently implemented for magnitude and wanda",
    )
    parser.add_argument("--local-path", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--dtype")
    parser.add_argument("--device")
    parser.add_argument(
        "--device-map",
        type=_device_map,
        help="Transformers device_map string such as 'auto', or a JSON object",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--calibration-source",
        choices=("c4",),
        help="Wanda calibration source (default from pruning config: c4)",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        help="Wanda calibration sample count (default: 128)",
    )
    parser.add_argument(
        "--calibration-seqlen",
        type=int,
        help="Wanda calibration token length (default: 2048)",
    )
    parser.add_argument(
        "--calibration-seed",
        type=int,
        help="Wanda calibration sampling seed (default: 0)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="external/persistent checkpoint directory; required with --execute",
    )
    parser.add_argument(
        "--overwrite-output-dir",
        action="store_true",
        help=(
            "allow Hugging Face save_pretrained to reuse a non-empty external "
            "output directory without deleting it"
        ),
    )
    return parser


def _load_options(args: argparse.Namespace) -> LoadOptions:
    return LoadOptions(
        local_path=args.local_path,
        cache_dir=args.cache_dir,
        dtype=args.dtype,
        device=args.device,
        device_map=args.device_map,
        local_files_only=args.local_files_only,
    )


def _resolve_calibration_config(
    args: argparse.Namespace,
    pruning: dict[str, Any],
) -> CalibrationConfig:
    return CalibrationConfig(
        source=args.calibration_source or pruning["calibration_source"],
        samples=(
            args.calibration_samples
            if args.calibration_samples is not None
            else int(pruning["calibration_samples"])
        ),
        sequence_length=(
            args.calibration_seqlen
            if args.calibration_seqlen is not None
            else int(pruning["calibration_sequence_length"])
        ),
        seed=(
            args.calibration_seed
            if args.calibration_seed is not None
            else int(pruning["calibration_seed"])
        ),
    )


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    request = PruningRequest(args.model, args.pruner, args.sparsity)
    spec = load_model_spec(args.model)
    adapter = get_model_adapter(spec)
    options = _load_options(args)
    model_record = build_model_manifest(
        spec,
        adapter,
        repository_root=REPOSITORY_ROOT,
        mode="planned_experiment",
        status="planned",
        options=options,
    )
    created_at = model_record.pop("timestamp")
    for key in ("schema_version", "mode", "status"):
        model_record.pop(key)
    pruning = _load_yaml(CONFIG_ROOT / "pruning" / f"{args.pruner}.yaml")
    calibration = None
    if args.pruner == "wanda":
        calibration = _resolve_calibration_config(args, pruning).to_dict()
    evaluation = None
    if args.benchmark:
        evaluation_config = _load_yaml(EVAL_CONFIG_DIR / f"{args.benchmark}.yaml")
        evaluation = {
            "benchmark": evaluation_config["benchmark"],
            "implementation_status": evaluation_config["implementation_status"],
        }
    return {
        "schema_version": 3,
        "created_at": created_at,
        "status": "planned",
        "model": model_record,
        "pruning": {
            "method": pruning["method"],
            "sparsity_ratio": request.sparsity,
            "implementation_status": pruning["implementation_status"],
            "implementation_version": pruning.get("implementation_version"),
            "pruning_type": pruning.get("structure"),
            "scope": pruning.get("scope"),
            "target_policy": pruning.get("target_policy"),
            "excluded_components": pruning.get("excluded_components"),
            "score": pruning.get("score"),
            "rounding": pruning.get("rounding"),
            "sequential_layerwise": pruning.get("sequential_layerwise"),
            "calibration": calibration,
        },
        "evaluation": evaluation,
    }


def _is_within(path: Path, directory: Path) -> bool:
    return path == directory or directory in path.parents


def _validate_output_directory(
    output_dir: Path,
    *,
    local_path: Path | None,
    overwrite: bool,
) -> Path:
    """Resolve and validate a checkpoint destination before model loading."""

    resolved_output = output_dir.expanduser().resolve()
    repository_root = REPOSITORY_ROOT.resolve()
    if _is_within(resolved_output, repository_root):
        raise ValueError(
            f"--output-dir must be outside the Git repository: {resolved_output}"
        )

    if local_path is not None:
        resolved_local = local_path.expanduser().resolve()
        if _is_within(resolved_output, resolved_local) or _is_within(
            resolved_local, resolved_output
        ):
            raise ValueError(
                "--output-dir and --local-path must not be equal, nested, or overlap: "
                f"output={resolved_output}, local={resolved_local}"
            )

    if resolved_output.exists():
        if not resolved_output.is_dir():
            raise ValueError(f"--output-dir exists and is not a directory: {resolved_output}")
        if next(resolved_output.iterdir(), None) is not None and not overwrite:
            raise ValueError(
                "--output-dir is non-empty; choose an empty directory or explicitly "
                "pass --overwrite-output-dir"
            )
    return resolved_output


def execute_experiment(args: argparse.Namespace) -> dict[str, Any]:
    """Execute a supported pruner and persist a standard HF checkpoint."""

    request = PruningRequest(args.model, args.pruner, args.sparsity)
    if args.pruner not in {"magnitude", "wanda"}:
        raise NotImplementedError(
            f"--execute is not implemented for pruner {args.pruner!r}"
        )
    if args.benchmark is not None:
        raise NotImplementedError(
            "benchmark execution is not implemented; omit --benchmark with --execute"
        )
    if args.output_dir is None:
        raise ValueError("--output-dir is required with --execute")

    output_dir = _validate_output_directory(
        args.output_dir,
        local_path=args.local_path,
        overwrite=args.overwrite_output_dir,
    )
    spec = load_model_spec(args.model)
    adapter = get_model_adapter(spec)
    options = _load_options(args)
    loaded = load_dense_model(spec, adapter, options)
    pruner = get_pruner(args.pruner)
    if args.pruner == "wanda":
        pruning_config = _load_yaml(CONFIG_ROOT / "pruning" / "wanda.yaml")
        calibration_config = _resolve_calibration_config(args, pruning_config)
        provider = get_calibration_provider(calibration_config.source)
        calibration_samples = provider.prepare(loaded.tokenizer, calibration_config)
        context = WandaPruningContext(calibration_config, calibration_samples)
        summary = pruner.prune(loaded.model, adapter, request, context)
    else:
        summary = pruner.prune(loaded.model, adapter, request)

    output_dir.mkdir(parents=True, exist_ok=True)
    loaded.model.save_pretrained(output_dir)
    loaded.tokenizer.save_pretrained(output_dir)

    model_record = build_model_manifest(
        spec,
        adapter,
        repository_root=REPOSITORY_ROOT,
        mode=f"{args.pruner}_pruning",
        status="completed",
        options=options,
        loaded=loaded,
    )
    manifest = {
        "schema_version": 3,
        "created_at": model_record.pop("timestamp"),
        "status": "completed",
        "model": model_record,
        "pruning": summary.to_dict(),
        "checkpoint": {
            "format": "huggingface_save_pretrained",
            "path": str(output_dir),
            "model_saved": True,
            "tokenizer_saved": True,
        },
        "evaluation": None,
    }
    (output_dir / "pruning_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def _write_planned_manifest(args: argparse.Namespace, rendered: str) -> Path:
    output_dir = REPOSITORY_ROOT / "experiments" / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    benchmark = f"_{args.benchmark}" if args.benchmark else ""
    output_path = output_dir / (
        f"{timestamp}_{args.model}_{args.pruner}{benchmark}_s{args.sparsity:g}.json"
    )
    output_path.write_text(rendered + "\n", encoding="utf-8")
    return output_path


def main() -> int:
    args = _parser().parse_args()
    try:
        manifest = execute_experiment(args) if args.execute else build_manifest(args)
    except NotImplementedError as error:
        print(f"execution unavailable: {error}", file=sys.stderr)
        return 3
    except Exception as error:
        print(f"experiment failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2

    rendered = json.dumps(manifest, indent=2, ensure_ascii=False)
    print(rendered)
    if args.write_manifest and not args.execute:
        output_path = _write_planned_manifest(args, rendered)
        print(f"manifest written to {output_path.relative_to(REPOSITORY_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
