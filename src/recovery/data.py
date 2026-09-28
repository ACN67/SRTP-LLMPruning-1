"""Dataset acquisition, formatting, provenance, and causal-LM tokenization."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.artifacts.manifest import sha256_file


FORMATTER_VERSION = "1.0"


@dataclass(frozen=True)
class DatasetConfig:
    source: str
    schema: str
    path: str = ""
    repo_id: str = ""
    subset: str = ""
    revision: str = ""
    split: str = "train"
    sample_limit: int = 0
    seed: int = 0
    text_field: str = "text"
    prompt_field: str = "prompt"
    completion_field: str = "completion"
    messages_field: str = "messages"
    separator: str = "\n"
    contamination_audit_status: str = "not_run_for_implementation_smoke"

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "DatasetConfig":
        value = cls(**dict(raw))
        if value.source not in {"local_json", "local_jsonl", "local_parquet", "huggingface"}:
            raise ValueError(f"Unsupported dataset source: {value.source}")
        if value.schema not in {"text", "prompt_completion", "messages"}:
            raise ValueError(f"Unsupported recovery schema: {value.schema}")
        if value.sample_limit < 0:
            raise ValueError("sample_limit cannot be negative")
        if value.source.startswith("local_") and not value.path:
            raise ValueError("Local dataset requires path")
        if value.source == "huggingface" and (not value.repo_id or not value.revision):
            raise ValueError("Hugging Face dataset requires repo_id and exact revision")
        return value


@dataclass(frozen=True)
class PreparedDataset:
    texts: tuple[str, ...]
    provenance: Mapping[str, Any]


def _records(config: DatasetConfig) -> tuple[Sequence[Mapping[str, Any]], dict[str, Any]]:
    if config.source in {"local_json", "local_jsonl"}:
        path = Path(config.path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Recovery dataset is missing: {path}")
        if config.source == "local_jsonl":
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            raw = json.loads(path.read_text(encoding="utf-8")); rows = raw if isinstance(raw, list) else raw.get("data", [])
        return rows, {"path": str(path), "sha256": sha256_file(path)}
    from datasets import load_dataset
    if config.source == "local_parquet":
        path = Path(config.path).expanduser().resolve()
        dataset = load_dataset("parquet", data_files={config.split: str(path)}, split=config.split)
        return dataset, {"path": str(path), "sha256": sha256_file(path)}
    dataset = load_dataset(
        config.repo_id,
        config.subset or None,
        revision=config.revision,
        split=config.split,
    )
    return dataset, {
        "repo_id": config.repo_id,
        "subset": config.subset,
        "revision": config.revision,
    }


def _format(row: Mapping[str, Any], config: DatasetConfig, tokenizer: Any) -> str:
    if config.schema == "text":
        value = row.get(config.text_field)
        if not isinstance(value, str) or not value.strip(): raise ValueError("Text row lacks non-empty text")
        return value
    if config.schema == "prompt_completion":
        prompt, completion = row.get(config.prompt_field), row.get(config.completion_field)
        if not isinstance(prompt, str) or not isinstance(completion, str): raise ValueError("Prompt/completion row has invalid fields")
        return prompt + config.separator + completion
    messages = row.get(config.messages_field)
    if not isinstance(messages, list) or not messages: raise ValueError("Messages row lacks a non-empty list")
    for message in messages:
        if not isinstance(message, Mapping) or not isinstance(message.get("role"), str) or not isinstance(message.get("content"), str):
            raise ValueError("Messages must contain role/content strings")
    if not hasattr(tokenizer, "apply_chat_template"): raise ValueError("Tokenizer has no chat template support")
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def prepare_dataset(config: DatasetConfig, tokenizer: Any) -> PreparedDataset:
    rows, identity = _records(config)
    row_count = len(rows)
    if row_count == 0:
        raise ValueError("Recovery dataset is empty")
    rng = random.Random(config.seed)
    if config.sample_limit and config.sample_limit < row_count:
        order = rng.sample(range(row_count), config.sample_limit)
        sampling_strategy = "seeded_without_replacement"
    else:
        order = list(range(row_count))
        rng.shuffle(order)
        sampling_strategy = "seeded_full_permutation"
    texts = tuple(_format(rows[index], config, tokenizer) for index in order)
    provenance = {
        **identity,
        **asdict(config),
        "source_row_count": row_count,
        "sample_count": len(texts),
        "sampling_strategy": sampling_strategy,
        "formatter_identity": "srtp_recovery_formatter",
        "formatter_version": FORMATTER_VERSION,
    }
    return PreparedDataset(texts, provenance)


def tokenize_dataset(dataset: PreparedDataset, tokenizer: Any, max_length: int) -> list[dict[str, list[int]]]:
    items = []
    for text in dataset.texts:
        encoded = tokenizer(text, truncation=True, max_length=max_length, add_special_tokens=True)
        ids = list(encoded["input_ids"])
        if not ids: raise ValueError("Tokenization produced an empty example")
        items.append({"input_ids": ids, "attention_mask": list(encoded.get("attention_mask", [1] * len(ids))), "labels": ids.copy()})
    return items
