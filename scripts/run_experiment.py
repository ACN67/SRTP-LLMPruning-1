#!/usr/bin/env python3
"""Plan or execute verified Magnitude, Wanda, SparseGPT, or SLEB pruning."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

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
from src.evaluation import load_evaluation_profile  # noqa: E402
from src.calibration_assets import load_local_c4, load_local_wikitext2  # noqa: E402
from src.pruning import (  # noqa: E402
    PRUNER_REGISTRY,
    CalibrationContext,
    CalibrationConfig,
    C4CalibrationProvider,
    PruningRequest,
    SLEBCalibrationConfig,
    SLEBCalibrationContext,
    SLEBPruner,
    SparseGPTPruner,
    SUPPORTED_TABP_MEASURES,
    TaBPCalibrationConfig,
    TaBPPruner,
    WikiText2SLEBCalibrationProvider,
    get_calibration_provider,
    get_pruner,
    get_sleb_calibration_provider,
    get_tabp_calibration_provider,
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
        help="load, prune, and save; implemented for all registered pruning methods",
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
        "--datasets-root", type=Path, default=Path("/data/datasets"),
        help="root containing pre-materialized calibration datasets",
    )
    parser.add_argument(
        "--calibration-source",
        choices=("c4", "wikitext2", "arc_easy", "wikitext"),
        help="method-specific calibration source",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        help=(
            "Wanda/SparseGPT independent sample count, or SLEB shuffled "
            "WikiText-2 source-row count (default: 128)"
        ),
    )
    parser.add_argument(
        "--calibration-seqlen",
        type=int,
        help="fixed sample length for Wanda/SparseGPT, or SLEB loss chunk length",
    )
    parser.add_argument(
        "--calibration-seed",
        type=int,
        help="calibration sampling/shuffle seed (default: 0)",
    )
    parser.add_argument(
        "--sparsegpt-percdamp",
        type=float,
        help="SparseGPT diagonal damping fraction (default: 0.01)",
    )
    parser.add_argument(
        "--sparsegpt-blocksize",
        type=int,
        help="SparseGPT adaptive input-column block size (default: 128)",
    )
    parser.add_argument(
        "--sleb-early-barrier",
        type=int,
        help="SLEB protected current blocks at the start (official default: 1)",
    )
    parser.add_argument(
        "--sleb-latter-barrier",
        type=int,
        help="SLEB protected current blocks at the end (official default: 1)",
    )
    parser.add_argument(
        "--tabp-ranking-strategy",
        choices=("ssn", "ddf"),
        help="TaBP internal ranking strategy (project default: ssn)",
    )
    parser.add_argument(
        "--tabp-mode",
        choices=("latter", "whole"),
        help="TaBP candidate scope (official default: latter)",
    )
    parser.add_argument(
        "--tabp-measure",
        choices=tuple(sorted(SUPPORTED_TABP_MEASURES)),
        help="TaBP block-output distribution statistic",
    )
    parser.add_argument(
        "--tabp-lm-head-type",
        choices=("frozen", "trained"),
        help="trained requires compatible upstream per-block checkpoints and is rejected here",
    )
    parser.add_argument(
        "--tabp-dataset-path",
        type=Path,
        help="optional local datasets save_to_disk path for ARC-Easy",
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


def _resolve_sparsegpt_options(
    args: argparse.Namespace,
    pruning: dict[str, Any],
) -> tuple[float, int]:
    percdamp_arg = getattr(args, "sparsegpt_percdamp", None)
    blocksize_arg = getattr(args, "sparsegpt_blocksize", None)
    percdamp = float(pruning["percdamp"] if percdamp_arg is None else percdamp_arg)
    blocksize = int(pruning["blocksize"] if blocksize_arg is None else blocksize_arg)
    if percdamp < 0:
        raise ValueError("SparseGPT percdamp must be non-negative")
    if blocksize <= 0:
        raise ValueError("SparseGPT blocksize must be positive")
    return percdamp, blocksize


def _resolve_sleb_config(
    args: argparse.Namespace,
    pruning: dict[str, Any],
) -> tuple[SLEBCalibrationConfig, int, int]:
    calibration = SLEBCalibrationConfig(
        source=getattr(args, "calibration_source", None)
        or pruning["calibration_source"],
        source_rows=(
            args.calibration_samples
            if getattr(args, "calibration_samples", None) is not None
            else int(pruning["calibration_source_rows"])
        ),
        sequence_length=(
            args.calibration_seqlen
            if getattr(args, "calibration_seqlen", None) is not None
            else int(pruning["calibration_sequence_length"])
        ),
        seed=(
            args.calibration_seed
            if getattr(args, "calibration_seed", None) is not None
            else int(pruning["calibration_seed"])
        ),
        sampling_semantics=pruning["calibration_sampling_semantics"],
        separator=pruning["calibration_separator"],
    )
    early_arg = getattr(args, "sleb_early_barrier", None)
    latter_arg = getattr(args, "sleb_latter_barrier", None)
    early = int(pruning["early_barrier"] if early_arg is None else early_arg)
    latter = int(pruning["latter_barrier"] if latter_arg is None else latter_arg)
    if early < 0 or latter < 0:
        raise ValueError("SLEB barriers must be non-negative")
    return calibration, early, latter


def _resolve_tabp_config(
    args: argparse.Namespace,
    pruning: dict[str, Any],
) -> tuple[TaBPCalibrationConfig, str, str, str, str]:
    source = getattr(args, "calibration_source", None) or pruning["calibration_source"]
    task_type = "text_generation" if source == "wikitext" else pruning["calibration_task_type"]
    sampling = (
        "concatenate_train_double_newline"
        if source == "wikitext"
        else pruning["calibration_sampling_semantics"]
    )
    calibration = TaBPCalibrationConfig(
        source=source,
        dataset_revision=(pruning["calibration_dataset_revision"] if source == "arc_easy" else None),
        task_type=task_type,
        samples=(
            args.calibration_samples
            if getattr(args, "calibration_samples", None) is not None
            else int(pruning["calibration_samples"])
        ),
        seed=(
            args.calibration_seed
            if getattr(args, "calibration_seed", None) is not None
            else int(pruning["calibration_seed"])
        ),
        sampling_semantics=sampling,
        n_windows=int(pruning["ddf_n_windows"]),
        n_steps=int(pruning["ddf_n_steps"]),
        window_size=int(pruning["ddf_window_size"]),
    )
    strategy = getattr(args, "tabp_ranking_strategy", None) or pruning["ranking_strategy"]
    mode = getattr(args, "tabp_mode", None) or pruning["mode"]
    measure = getattr(args, "tabp_measure", None) or pruning["measure"]
    lm_head_type = (
        getattr(args, "tabp_lm_head_type", None) or pruning["lm_head_type"]
    )
    if lm_head_type != "frozen":
        raise NotImplementedError(
            "TaBP trained LM-head execution is not available without audited "
            "per-block checkpoints; use --tabp-lm-head-type frozen"
        )
    if strategy == "ssn" and task_type != "qa":
        raise ValueError("This TaBP SSN adaptation requires ARC-Easy QA calibration")
    return calibration, strategy, mode, measure, lm_head_type


def _load_sleb_calibration_tokenizer(
    spec: Any,
    options: LoadOptions,
    loaded: Any,
) -> Any:
    """Load the official slow tokenizer solely for SLEB calibration."""

    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("SLEB calibration tokenizer requires Transformers") from error
    kwargs: dict[str, Any] = {
        "trust_remote_code": spec.trust_remote_code,
        "local_files_only": options.local_files_only,
        "use_fast": False,
    }
    if options.cache_dir is not None:
        kwargs["cache_dir"] = str(options.cache_dir.expanduser())
    if not loaded.is_local:
        kwargs["revision"] = spec.revision
    return AutoTokenizer.from_pretrained(loaded.source, **kwargs)


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
    if args.pruner in {"wanda", "sparsegpt"}:
        calibration = _resolve_calibration_config(args, pruning).to_dict()
    sleb_fields: dict[str, Any] = {}
    if args.pruner == "sleb":
        sleb_calibration, early, latter = _resolve_sleb_config(args, pruning)
        original_blocks = spec.expected_num_hidden_layers
        requested_remove = math.ceil(original_blocks * request.sparsity)
        capacity = original_blocks - early - latter
        if requested_remove > capacity:
            raise ValueError(
                f"SLEB requested_remove_count={requested_remove} exceeds "
                f"barrier-constrained capacity={max(capacity, 0)}"
            )
        calibration = sleb_calibration.to_dict()
        sleb_fields = {
            "official_core_input": "num_remove_blocks",
            "ratio_to_count_policy": "project_interface_ceil",
            "target_sparsity_ratio": request.sparsity,
            "expected_original_block_count": original_blocks,
            "requested_remove_count": requested_remove,
            "planned_achieved_block_sparsity": requested_remove / original_blocks,
            "early_barrier": early,
            "latter_barrier": latter,
            "selection_metric": pruning["selection_metric"],
            "selection_semantics": pruning["selection_semantics"],
            "candidate_tie_rule": pruning["candidate_tie_rule"],
            "greedy_iterative": pruning["greedy_iterative"],
        }
    tabp_fields: dict[str, Any] = {}
    if args.pruner == "tabp":
        tabp_calibration, strategy, mode, measure, lm_head_type = _resolve_tabp_config(
            args, pruning
        )
        original_blocks = spec.expected_num_hidden_layers
        requested_remove = math.ceil(original_blocks * request.sparsity)
        start = original_blocks // 2 if mode == "latter" else 0
        capacity = original_blocks - 1 if strategy == "ddf" else original_blocks - start - 1
        if requested_remove > capacity:
            raise ValueError(
                f"TaBP requested_remove_count={requested_remove} exceeds "
                f"mode={mode!r} capacity={capacity}"
            )
        calibration = tabp_calibration.to_dict()
        tabp_fields = {
            "official_core_input": "num_remove_blocks",
            "ratio_to_count_policy": "project_interface_ceil",
            "target_sparsity_ratio": request.sparsity,
            "expected_original_block_count": original_blocks,
            "requested_remove_count": requested_remove,
            "planned_achieved_block_sparsity": requested_remove / original_blocks,
            "planned_achieved_block_retention": (original_blocks - requested_remove) / original_blocks,
            "mode": mode,
            "measure": measure,
            "ssn_p": int(pruning["ssn_p"]),
            "lm_head_type": lm_head_type,
            "ranking_strategy": strategy,
            "trained_lm_head_support": pruning["trained_lm_head_support"],
        }
    sparsegpt_percdamp = pruning.get("percdamp")
    sparsegpt_blocksize = pruning.get("blocksize")
    if args.pruner == "sparsegpt":
        sparsegpt_percdamp, sparsegpt_blocksize = _resolve_sparsegpt_options(
            args, pruning
        )
    evaluation = None
    if args.benchmark:
        evaluation_config = _load_yaml(EVAL_CONFIG_DIR / f"{args.benchmark}.yaml")
        profile = load_evaluation_profile(args.model, args.benchmark)
        evaluation = {
            "benchmark": evaluation_config["benchmark"],
            "implementation_status": evaluation_config["implementation_status"],
            "source_revision": evaluation_config["source_revision"],
            "dataset_revision": evaluation_config.get("dataset_revision"),
            "release": evaluation_config.get("release_version"),
            "task_count": evaluation_config["expected_task_count"],
            "metric": evaluation_config["metric"],
            "prompt_protocol": evaluation_config["prompt_protocol"],
            "evaluation_profile": profile.to_dict(),
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
            "criterion": pruning.get("criterion"),
            "percdamp": sparsegpt_percdamp,
            "blocksize": sparsegpt_blocksize,
            "adaptive_mask_selection": pruning.get("adaptive_mask_selection"),
            "error_compensation": pruning.get("error_compensation"),
            "weight_update": pruning.get("weight_update"),
            "retraining": pruning.get("retraining"),
            "quantization": pruning.get("quantization"),
            "nm_sparsity": pruning.get("nm_sparsity"),
            "true_sequential": pruning.get("true_sequential"),
            "calibration": calibration,
            **sleb_fields,
            **tabp_fields,
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


class PreparedPruningMethod(NamedTuple):
    pruner: Any
    context: Any
    reduced_depth: bool = False


def _prepare_pruning_method(
    args: argparse.Namespace,
    pruning_config: dict[str, Any],
    *,
    request: PruningRequest,
    spec: Any,
    options: LoadOptions,
    loaded: Any,
) -> PreparedPruningMethod:
    """Build method-specific state while keeping runner load/save generic."""

    if args.pruner in {"wanda", "sparsegpt"}:
        calibration_config = _resolve_calibration_config(args, pruning_config)
        if args.local_files_only:
            provider = C4CalibrationProvider(lambda: load_local_c4(args.datasets_root))
        else:
            provider = get_calibration_provider(calibration_config.source)
        samples = provider.prepare(loaded.tokenizer, calibration_config)
        context = CalibrationContext(calibration_config, samples)
        if args.pruner == "sparsegpt":
            percdamp, blocksize = _resolve_sparsegpt_options(args, pruning_config)
            pruner = SparseGPTPruner(percdamp=percdamp, blocksize=blocksize)
        else:
            pruner = get_pruner(args.pruner)
        return PreparedPruningMethod(pruner, context)

    if args.pruner == "sleb":
        calibration, early, latter = _resolve_sleb_config(args, pruning_config)
        pruner = SLEBPruner(
            early_barrier=early,
            latter_barrier=latter,
            calibration_config=calibration,
        )
        context = None
        if request.sparsity != 0:
            calibration_tokenizer = _load_sleb_calibration_tokenizer(
                spec, options, loaded
            )
            if args.local_files_only:
                provider = WikiText2SLEBCalibrationProvider(
                    lambda: load_local_wikitext2(args.datasets_root)
                )
            else:
                provider = get_sleb_calibration_provider(calibration.source)
            context = provider.prepare(calibration_tokenizer, calibration)
        return PreparedPruningMethod(pruner, context, reduced_depth=True)

    if args.pruner == "tabp":
        calibration, strategy, mode, measure, lm_head_type = _resolve_tabp_config(
            args, pruning_config
        )
        pruner = TaBPPruner(
            ranking_strategy=strategy,
            mode=mode,
            measure=measure,
            lm_head_type=lm_head_type,
        )
        context = None
        if request.sparsity != 0:
            dataset_path = getattr(args, "tabp_dataset_path", None)
            if args.local_files_only and dataset_path is None:
                raise ValueError(
                    "Offline TaBP execution requires --tabp-dataset-path"
                )
            context = get_tabp_calibration_provider(calibration.source).prepare(
                loaded.tokenizer,
                calibration,
                dataset_path=dataset_path,
            )
        return PreparedPruningMethod(pruner, context, reduced_depth=True)

    return PreparedPruningMethod(get_pruner(args.pruner), None)


def execute_experiment(args: argparse.Namespace) -> dict[str, Any]:
    """Execute a supported pruner and persist a standard HF checkpoint."""

    request = PruningRequest(args.model, args.pruner, args.sparsity)
    if args.pruner not in PRUNER_REGISTRY:
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
    pruning_config = _load_yaml(CONFIG_ROOT / "pruning" / f"{args.pruner}.yaml")
    spec = load_model_spec(args.model)
    adapter = get_model_adapter(spec)
    options = _load_options(args)
    loaded = load_dense_model(spec, adapter, options)
    prepared = _prepare_pruning_method(
        args,
        pruning_config,
        request=request,
        spec=spec,
        options=options,
        loaded=loaded,
    )
    if prepared.context is not None:
        summary = prepared.pruner.prune(
            loaded.model, adapter, request, prepared.context
        )
    else:
        summary = prepared.pruner.prune(loaded.model, adapter, request)
    if prepared.reduced_depth:
        loaded = replace(loaded, structure=adapter.get_structure(loaded.model))

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
    if args.local_path is not None:
        sidecar_path = args.local_path.expanduser().resolve() / ".srtp_model_source.json"
        if sidecar_path.is_file():
            model_record["source_snapshot_provenance"] = json.loads(
                sidecar_path.read_text(encoding="utf-8")
            )
            model_record["source_snapshot_sidecar_sha256"] = hashlib.sha256(
                sidecar_path.read_bytes()
            ).hexdigest()
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
