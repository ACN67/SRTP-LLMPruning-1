#!/usr/bin/env python3
"""Validate pinned model metadata or perform a real dense-model smoke test."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--config-only",
        action="store_true",
        help="validate pinned project metadata without importing or downloading model weights",
    )
    mode.add_argument(
        "--load",
        action="store_true",
        help="load model and tokenizer, validate structure, and run a minimal forward pass",
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
        "--generate",
        action="store_true",
        help="after the required minimal forward pass, run a short generation",
    )
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="explicit manifest path; real --load runs otherwise write to experiments/generated",
    )
    return parser


def _manifest_path(args: argparse.Namespace) -> Path | None:
    if args.manifest is not None:
        return args.manifest.expanduser()
    if not args.load:
        return None
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (
        REPOSITORY_ROOT
        / "experiments"
        / "generated"
        / f"{timestamp}_{args.model}_dense_validation.json"
    )


def _write_manifest(path: Path | None, manifest: dict[str, Any]) -> None:
    rendered = json.dumps(manifest, indent=2, ensure_ascii=False)
    print(rendered)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
        try:
            shown = path.relative_to(REPOSITORY_ROOT)
        except ValueError:
            shown = path
        print(f"manifest written to {shown}")


def _move_inputs_to_embedding_device(
    inputs: Any,
    adapter: Any,
    model: Any,
) -> dict[str, Any]:
    embedding = adapter.get_embedding(model)
    device = embedding.weight.device
    return {name: tensor.to(device) for name, tensor in dict(inputs).items()}


def _runtime_smoke_test(
    loaded: Any,
    adapter: Any,
    *,
    prompt: str,
    generate: bool,
    max_new_tokens: int,
) -> tuple[dict[str, Any], str | None]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required for runtime validation") from error

    inputs = loaded.tokenizer(prompt, return_tensors="pt")
    model_inputs = _move_inputs_to_embedding_device(inputs, adapter, loaded.model)
    with torch.inference_mode():
        outputs = loaded.model(**model_inputs)
    logits = getattr(outputs, "logits", None)
    if logits is None or getattr(logits, "ndim", 0) != 3:
        raise RuntimeError("Minimal forward did not return rank-3 logits")

    checks: dict[str, Any] = {
        "model_loaded": True,
        "tokenizer_loaded": True,
        "adapter_structure_validated": True,
        "minimal_forward": True,
        "generation": None,
    }
    generated_text = None
    if generate:
        if max_new_tokens <= 0:
            raise ValueError("--max-new-tokens must be positive")
        with torch.inference_mode():
            generated = loaded.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        generated_text = loaded.tokenizer.decode(generated[0], skip_special_tokens=True)
        checks["generation"] = True
    return checks, generated_text


def main() -> int:
    args = _parser().parse_args()
    if args.generate and not args.load:
        print("configuration error: --generate requires --load", file=sys.stderr)
        return 2

    spec = load_model_spec(args.model)
    adapter = get_model_adapter(spec)
    options = LoadOptions(
        local_path=args.local_path,
        cache_dir=args.cache_dir,
        dtype=args.dtype,
        device=args.device,
        device_map=args.device_map,
        local_files_only=args.local_files_only,
    )
    manifest_path = _manifest_path(args)

    if args.config_only:
        manifest = build_model_manifest(
            spec,
            adapter,
            repository_root=REPOSITORY_ROOT,
            mode="config_only",
            status="config_validated",
            options=options,
        )
        manifest["checks"] = {
            "project_config_schema": True,
            "immutable_revision": True,
            "adapter_registered": True,
            "weights_loaded": False,
            "gpu_runtime_validation": "pending",
        }
        _write_manifest(manifest_path, manifest)
        return 0

    try:
        loaded = load_dense_model(spec, adapter, options)
        checks, generated_text = _runtime_smoke_test(
            loaded,
            adapter,
            prompt=args.prompt,
            generate=args.generate,
            max_new_tokens=args.max_new_tokens,
        )
        manifest = build_model_manifest(
            spec,
            adapter,
            repository_root=REPOSITORY_ROOT,
            mode="dense_load",
            status="validated",
            options=options,
            loaded=loaded,
        )
        manifest["checks"] = checks
        if generated_text is not None:
            manifest["generation"] = {
                "prompt": args.prompt,
                "max_new_tokens": args.max_new_tokens,
                "text": generated_text,
            }
        _write_manifest(manifest_path, manifest)
        return 0
    except Exception as error:  # preserve a failure manifest for real load attempts
        manifest = build_model_manifest(
            spec,
            adapter,
            repository_root=REPOSITORY_ROOT,
            mode="dense_load",
            status="failed",
            options=options,
        )
        manifest["error"] = f"{type(error).__name__}: {error}"
        _write_manifest(manifest_path, manifest)
        print(f"dense validation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
