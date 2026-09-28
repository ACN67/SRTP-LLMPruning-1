"""Configuration and result contracts for post-pruning recovery."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class RecoveryConfig:
    rank: int
    lora_alpha: int
    lora_dropout: float
    target_modules: tuple[str, ...]
    bias: str
    learning_rate: float
    weight_decay: float
    max_steps: int
    num_train_epochs: float
    batch_size: int
    gradient_accumulation_steps: int
    max_sequence_length: int
    seed: int
    dtype: str
    gradient_checkpointing: bool
    save_interval: int
    log_interval: int
    merge_policy: str
    allow_sparse_merge: bool
    protocol_status: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "RecoveryConfig":
        values = dict(raw)
        if values.pop("schema_version", None) != 1:
            raise ValueError("Unsupported recovery config schema_version")
        target_modules = values.get("target_modules")
        if not isinstance(target_modules, list) or not target_modules:
            raise ValueError("LoRA target_modules must be a non-empty list")
        if any(not isinstance(item, str) or not item for item in target_modules):
            raise ValueError("LoRA target_modules must contain non-empty strings")
        if len(set(target_modules)) != len(target_modules):
            raise ValueError("LoRA target_modules must not contain duplicates")
        values["target_modules"] = tuple(target_modules)
        config = cls(**values)
        if min(config.rank, config.lora_alpha, config.batch_size, config.gradient_accumulation_steps, config.max_sequence_length, config.save_interval, config.log_interval) <= 0:
            raise ValueError("Positive LoRA/training limits are required")
        if not 0 <= config.lora_dropout < 1 or config.learning_rate <= 0 or config.weight_decay < 0:
            raise ValueError("Invalid LoRA optimizer parameters")
        if config.max_steps < 0 or config.num_train_epochs < 0:
            raise ValueError("Training step/epoch budgets cannot be negative")
        if (config.max_steps > 0) == (config.num_train_epochs > 0):
            raise ValueError("Exactly one of max_steps or num_train_epochs must be positive")
        if config.dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("Unsupported recovery dtype")
        if config.bias not in {"none", "all", "lora_only"}:
            raise ValueError("Unsupported LoRA bias mode")
        if config.merge_policy not in {"adapter_only", "standard_merge"}:
            raise ValueError("Unsupported merge policy")
        if config.protocol_status != "implementation_baseline_not_final_experiment_protocol":
            raise ValueError("Recovery config must state its non-final protocol status")
        return config

    @property
    def canonical_hash(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class RecoveryResult:
    status: str
    adapter_path: str
    merged_model_path: str | None
    manifest_path: str
    training_metrics: Mapping[str, Any]
    matched_target_modules: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
