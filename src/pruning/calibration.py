"""Reusable calibration protocols and C4 sample preparation."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class CalibrationConfig:
    source: str = "c4"
    samples: int = 128
    sequence_length: int = 2048
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("calibration source must be non-empty")
        if self.samples <= 0:
            raise ValueError("calibration samples must be positive")
        if self.sequence_length <= 0:
            raise ValueError("calibration sequence length must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CalibrationSample:
    input_ids: Any
    attention_mask: Any | None = None


@dataclass(frozen=True)
class WandaPruningContext:
    config: CalibrationConfig
    samples: tuple[CalibrationSample, ...]

    def __post_init__(self) -> None:
        if len(self.samples) != self.config.samples:
            raise ValueError(
                f"Expected {self.config.samples} calibration samples, "
                f"received {len(self.samples)}"
            )
        for index, sample in enumerate(self.samples):
            shape = getattr(sample.input_ids, "shape", ())
            if len(shape) != 2 or shape[0] != 1 or shape[1] != self.config.sequence_length:
                raise ValueError(
                    f"Calibration sample {index} must have shape "
                    f"[1, {self.config.sequence_length}], received {tuple(shape)}"
                )
            if sample.attention_mask is not None and tuple(
                sample.attention_mask.shape
            ) != tuple(shape):
                raise ValueError(
                    f"Calibration sample {index} attention mask shape does not "
                    "match input_ids"
                )


def _default_c4_loader() -> Sequence[Any]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("C4 calibration requires the datasets package") from error
    return load_dataset(
        "allenai/c4",
        "en",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
    )


class C4CalibrationProvider:
    """Official-style random document and contiguous token-span sampler."""

    def __init__(
        self,
        dataset_loader: Callable[[], Sequence[Any]] | None = None,
    ) -> None:
        self._dataset_loader = dataset_loader or _default_c4_loader

    def prepare(
        self,
        tokenizer: Any,
        config: CalibrationConfig,
    ) -> tuple[CalibrationSample, ...]:
        if config.source != "c4":
            raise ValueError(f"C4 provider cannot prepare source {config.source!r}")
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("Calibration preparation requires PyTorch") from error

        dataset = self._dataset_loader()
        if len(dataset) == 0:
            raise ValueError("C4 training dataset is empty")
        rng = random.Random(config.seed)
        prepared: list[CalibrationSample] = []
        attempts = 0
        max_attempts = max(config.samples * 100, len(dataset) * 10)
        while len(prepared) < config.samples and attempts < max_attempts:
            attempts += 1
            document = dataset[rng.randint(0, len(dataset) - 1)]
            text = document["text"]
            encoded = tokenizer(text, return_tensors="pt")
            input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
            token_count = input_ids.shape[1]
            if token_count <= config.sequence_length:
                continue
            start = rng.randint(0, token_count - config.sequence_length - 1)
            stop = start + config.sequence_length
            span = input_ids[:, start:stop].clone()
            prepared.append(
                CalibrationSample(
                    input_ids=span,
                    attention_mask=torch.ones_like(span),
                )
            )
        if len(prepared) != config.samples:
            raise ValueError(
                "Could not sample enough C4 documents longer than the requested "
                f"token length {config.sequence_length}"
            )
        return tuple(prepared)


def get_calibration_provider(source: str) -> C4CalibrationProvider:
    if source == "c4":
        return C4CalibrationProvider()
    raise KeyError(f"Unsupported calibration source: {source!r}")
