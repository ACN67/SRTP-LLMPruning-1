"""PEFT-backed LoRA recovery with explicit merge and sparsity safety."""

from __future__ import annotations

import gc
import json
import math
import os
import platform
import random
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.artifacts import LineageOperation, ModelArtifact, artifact_inventory, read_artifact_manifest, write_artifact_manifest
from src.models.base import BaseModelAdapter, ModelSpec
from src.models.loader import LoadedModel

from .base import RecoveryConfig, RecoveryResult
from .data import PreparedDataset, tokenize_dataset


PEFT_VERSION = "0.18.1"
PEFT_REVISION = "e3398fc05c556f64a8e8dc248628bae657107488"


def matched_target_modules(model: Any, targets: tuple[str, ...]) -> tuple[str, ...]:
    names = tuple(sorted(name for name, module in model.named_modules() if name and any(name == target or name.endswith("." + target) for target in targets) and hasattr(module, "weight")))
    missing = [target for target in targets if not any(name == target or name.endswith("." + target) for name in names)]
    if not names:
        raise ValueError("LoRA target_modules matched zero model modules")
    if missing:
        raise ValueError(f"LoRA target_modules do not exist in this architecture: {missing}")
    return names


def sparsity_stats(model: Any, names: tuple[str, ...]) -> dict[str, Any]:
    modules = dict(model.named_modules()); zeros = total = 0
    for name in names:
        weight = modules[name].weight.detach()
        zeros += int((weight == 0).sum().item()); total += weight.numel()
    return {"zero_count": zeros, "element_count": total, "sparsity": zeros / total if total else 0.0}


def _collate(items: list[dict[str, list[int]]], pad_id: int, torch: Any) -> dict[str, Any]:
    length = max(len(item["input_ids"]) for item in items)
    def padded(key: str, fill: int) -> Any:
        return torch.tensor([item[key] + [fill] * (length - len(item[key])) for item in items], dtype=torch.long)
    return {"input_ids": padded("input_ids", pad_id), "attention_mask": padded("attention_mask", 0), "labels": padded("labels", -100)}


def _git_identity(root: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(("git", "-C", str(root), *args), check=True, capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain=v1", "--untracked-files=all"))}


class LoRARecovery:
    method = "lora"

    def recover(self, *, loaded: LoadedModel, input_artifact: ModelArtifact, spec: ModelSpec, adapter: BaseModelAdapter, dataset: PreparedDataset, config: RecoveryConfig, output_dir: Path, repository_root: Path) -> RecoveryResult:
        import peft
        import torch
        from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM

        if peft.__version__ != PEFT_VERSION:
            raise RuntimeError(f"Expected peft {PEFT_VERSION}, found {peft.__version__}")
        if input_artifact.representation != "full_checkpoint":
            raise ValueError("LoRA recovery input must resolve to a full checkpoint")
        if input_artifact.is_weight_sparse and config.merge_policy == "standard_merge" and not config.allow_sparse_merge:
            raise ValueError("standard_merge on a weight-sparse artifact requires allow_sparse_merge=true because LoRA deltas can refill zeros")
        root = output_dir.expanduser().resolve()
        repository = repository_root.expanduser().resolve()
        input_root = Path(input_artifact.path).expanduser().resolve()
        if root == Path(root.anchor) or root == repository or repository in root.parents:
            raise ValueError(
                "Recovery output directory must be outside the repository and filesystem root"
            )
        if root == input_root or root in input_root.parents or input_root in root.parents:
            raise ValueError("Recovery output directory must not overlap the input artifact")
        if root.exists() and any(root.iterdir()):
            raise ValueError(f"Recovery output directory is not empty: {root}")
        root.mkdir(parents=True, exist_ok=True)
        adapter_dir = root / "adapter"
        merged_dir = root / "merged_model"
        random.seed(config.seed); torch.manual_seed(config.seed)
        targets = matched_target_modules(loaded.model, config.target_modules)
        depth_before = adapter.get_num_blocks(loaded.model)
        sparse_before = sparsity_stats(loaded.model, targets)
        tokenized = tokenize_dataset(dataset, loaded.tokenizer, config.max_sequence_length)
        lora_config = LoraConfig(r=config.rank, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout, target_modules=list(config.target_modules), bias=config.bias, task_type="CAUSAL_LM")
        model = get_peft_model(loaded.model, lora_config)
        if config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            model.config.use_cache = False
        trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
        before = {name: parameter.detach().cpu().clone() for name, parameter in trainable.items()}
        optimizer = torch.optim.AdamW(trainable.values(), lr=config.learning_rate, weight_decay=config.weight_decay)
        device = next(model.parameters()).device; pad_id = loaded.tokenizer.pad_token_id
        if pad_id is None: pad_id = loaded.tokenizer.eos_token_id or 0
        started_at = datetime.now(timezone.utc).isoformat(); started = time.monotonic(); model.train(); optimizer.zero_grad()
        target_steps = config.max_steps if config.max_steps > 0 else math.ceil(len(tokenized) * config.num_train_epochs / config.batch_size / config.gradient_accumulation_steps)
        losses: list[float] = []; optimizer_steps = 0; micro_step = 0
        while optimizer_steps < target_steps:
            start = (micro_step * config.batch_size) % len(tokenized); batch_items = [tokenized[(start + index) % len(tokenized)] for index in range(config.batch_size)]
            batch = {key: value.to(device) for key, value in _collate(batch_items, pad_id, torch).items()}
            loss = model(**batch, use_cache=False).loss / config.gradient_accumulation_steps; loss.backward(); losses.append(float(loss.detach().cpu()) * config.gradient_accumulation_steps)
            micro_step += 1
            if micro_step % config.gradient_accumulation_steps == 0:
                optimizer.step(); optimizer.zero_grad(); optimizer_steps += 1
        updated = tuple(name for name, parameter in trainable.items() if not torch.equal(before[name], parameter.detach().cpu()))
        if not updated: raise RuntimeError("LoRA optimizer completed without updating trainable parameters")
        model.save_pretrained(adapter_dir, safe_serialization=True)
        if hasattr(loaded.tokenizer, "save_pretrained"): loaded.tokenizer.save_pretrained(adapter_dir)
        PeftConfig.from_pretrained(adapter_dir)
        _entries, adapter_hash = artifact_inventory(adapter_dir)
        recovery_op = LineageOperation("recovery", "lora", config.canonical_hash, input_artifact.content_sha256, {"merge_policy": config.merge_policy, "rank": config.rank, "alpha": config.lora_alpha}, {"peft_version": PEFT_VERSION, "peft_revision": PEFT_REVISION, "protocol_status": config.protocol_status})
        adapter_artifact = ModelArtifact(str(adapter_dir), "recovered", "peft_adapter", False, spec.project_model_id, spec.architecture, spec.model_type, adapter.adapter_id, depth_before, adapter_hash, input_artifact.lineage + (recovery_op,), {"direct_evaluation": True, "vllm_serving": False}, {"base_artifact_path": input_artifact.path, "base_artifact_hash": input_artifact.content_sha256, "base_artifact_manifest_provenance_sha256": input_artifact.manifest_provenance_sha256})
        write_artifact_manifest(adapter_dir, adapter_artifact)
        adapter_artifact = read_artifact_manifest(adapter_dir)
        merge_decision = {"policy": config.merge_policy, "weight_sparse_input": input_artifact.is_weight_sparse, "explicit_sparsity_change_opt_in": config.allow_sparse_merge}
        merged_artifact = None; sparse_after = sparse_before; reload_status = "adapter_config_reloaded"
        if config.merge_policy == "adapter_only":
            base = model.unload()
            if hasattr(base, "peft_config"): delattr(base, "peft_config")
            reloaded = PeftModel.from_pretrained(base, adapter_dir); reloaded.eval()
            with torch.inference_mode(): reloaded(input_ids=torch.tensor([tokenized[0]["input_ids"]], device=device), use_cache=False)
            reload_status = "adapter_reloaded_and_forward_validated"
        else:
            merged = model.merge_and_unload(); sparse_after = sparsity_stats(merged, targets)
            if adapter.get_num_blocks(merged) != depth_before: raise RuntimeError("LoRA merge changed model layer count")
            merged.save_pretrained(merged_dir, safe_serialization=True)
            if hasattr(loaded.tokenizer, "save_pretrained"): loaded.tokenizer.save_pretrained(merged_dir)
            _entries, merged_hash = artifact_inventory(merged_dir)
            merged_artifact = ModelArtifact(str(merged_dir), "recovered", "full_checkpoint", True, spec.project_model_id, spec.architecture, spec.model_type, adapter.adapter_id, depth_before, merged_hash, input_artifact.lineage + (recovery_op,), {"direct_evaluation": True, "vllm_serving": True}, {"adapter_artifact_hash": adapter_hash, "input_artifact_manifest_provenance_sha256": input_artifact.manifest_provenance_sha256, "sparsity_preserved": sparse_before == sparse_after})
            write_artifact_manifest(merged_dir, merged_artifact)
            merged_artifact = read_artifact_manifest(merged_dir)
            del merged; gc.collect()
            reloaded = AutoModelForCausalLM.from_pretrained(merged_dir, local_files_only=True, dtype=getattr(torch, config.dtype)); reloaded.eval()
            if adapter.get_num_blocks(reloaded) != depth_before: raise RuntimeError("Reloaded merged model changed layer count")
            with torch.inference_mode(): reloaded(input_ids=torch.tensor([tokenized[0]["input_ids"]]), use_cache=False)
            reload_status = "merged_checkpoint_reloaded_and_forward_validated"
        ended_at = datetime.now(timezone.utc).isoformat(); runtime = time.monotonic() - started
        manifest = {
            "schema_version": 1, "status": "completed", "started_at": started_at, "ended_at": ended_at, "runtime_seconds": runtime,
            "project": _git_identity(repository_root), "method": "lora", "peft": {"version": PEFT_VERSION, "revision": PEFT_REVISION},
            "input_artifact": input_artifact.to_dict(), "model": {"project_model_id": spec.project_model_id, "architecture": spec.architecture, "layer_count": depth_before, "dtype": loaded.dtype},
            "recovery_config": asdict(config), "recovery_config_hash": config.canonical_hash,
            "matched_target_modules": {"count": len(targets), "names": list(targets)}, "dataset": dict(dataset.provenance),
            "training_metrics": {"optimizer_steps": optimizer_steps, "micro_steps": micro_step, "losses": losses, "updated_parameter_count": len(updated), "updated_parameters": list(updated)},
            "runtime": {"python": platform.python_version(), "torch": torch.__version__, "device": str(device), "cuda_available": torch.cuda.is_available()},
            "adapter_artifact": adapter_artifact.to_dict(), "merged_artifact": merged_artifact.to_dict() if merged_artifact else {"status": "not_requested"},
            "merge_safety": {**merge_decision, "sparsity_before": sparse_before, "sparsity_after": sparse_after, "sparsity_preserved": sparse_before == sparse_after},
            "reload_validation": reload_status,
        }
        manifest_path = root / "recovery_manifest.json"; manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return RecoveryResult("success", str(adapter_dir), str(merged_dir) if merged_artifact else None, str(manifest_path), manifest["training_metrics"], targets)
