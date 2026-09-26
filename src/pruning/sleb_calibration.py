"""Official fixed-commit WikiText-2 calibration construction for SLEB."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class SLEBCalibrationConfig:
    source: str = "wikitext2"
    source_rows: int = 128
    sequence_length: int = 2048
    seed: int = 0
    sampling_semantics: str = "shuffled_first_rows_concat"
    separator: str = "\n\n"

    def __post_init__(self) -> None:
        if self.source != "wikitext2":
            raise ValueError("SLEB canonical calibration source must be 'wikitext2'")
        if self.source_rows <= 0:
            raise ValueError("SLEB calibration source rows must be positive")
        if self.sequence_length <= 0:
            raise ValueError("SLEB loss sequence length must be positive")
        if self.sampling_semantics != "shuffled_first_rows_concat":
            raise ValueError("Unsupported SLEB calibration sampling semantics")
        if self.separator != "\n\n":
            raise ValueError("SLEB canonical calibration separator must be '\\n\\n'")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SLEBCalibrationContext:
    config: SLEBCalibrationConfig
    input_ids: Any
    token_stream_length: int
    tokenizer_class: str | None = None
    tokenizer_is_fast: bool | None = None

    def __post_init__(self) -> None:
        shape = getattr(self.input_ids, "shape", ())
        if len(shape) != 2 or shape[0] != 1:
            raise ValueError(
                f"SLEB calibration input_ids must have shape [1, tokens], got {shape}"
            )
        if shape[1] != self.token_stream_length:
            raise ValueError("SLEB token stream length does not match input_ids")


def _default_wikitext2_loader() -> Any:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("SLEB WikiText-2 calibration requires datasets") from error
    return load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split="train",
    )


class WikiText2SLEBCalibrationProvider:
    """Build the official shuffled-row, concatenated WikiText-2 token stream."""

    def __init__(self, dataset_loader: Callable[[], Any] | None = None) -> None:
        self._dataset_loader = dataset_loader or _default_wikitext2_loader

    def prepare(
        self,
        tokenizer: Any,
        config: SLEBCalibrationConfig,
    ) -> SLEBCalibrationContext:
        tokenizer_is_fast = getattr(tokenizer, "is_fast", None)
        if tokenizer_is_fast is True:
            raise ValueError(
                "SLEB canonical calibration requires use_fast=False tokenizer"
            )
        dataset = self._dataset_loader()
        shuffled = dataset.shuffle(seed=config.seed)
        selected = shuffled[: config.source_rows]
        texts = selected["text"]
        if len(texts) != config.source_rows:
            raise ValueError(
                f"WikiText-2 returned {len(texts)} rows; expected {config.source_rows}"
            )
        encoded = tokenizer(config.separator.join(texts), return_tensors="pt")
        input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        return SLEBCalibrationContext(
            config=config,
            input_ids=input_ids,
            token_stream_length=int(input_ids.shape[1]),
            tokenizer_class=type(tokenizer).__name__,
            tokenizer_is_fast=tokenizer_is_fast,
        )


def get_sleb_calibration_provider(source: str) -> WikiText2SLEBCalibrationProvider:
    if source == "wikitext2":
        return WikiText2SLEBCalibrationProvider()
    raise KeyError(f"Unsupported SLEB calibration source: {source!r}")
