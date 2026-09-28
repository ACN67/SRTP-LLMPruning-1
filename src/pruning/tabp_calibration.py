"""Task-aware calibration preparation for TaBP."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

ARC_EASY_REVISION = "210d026faf9955653af8916fad021475a3f00453"
ARC_PREFIX = "Answer the following multiple choice question by giving the most appropriate response.\n"
ARC_POSTFIX = "Answer: "


@dataclass(frozen=True)
class TaBPCalibrationConfig:
    source: str = "arc_easy"
    dataset_revision: str | None = ARC_EASY_REVISION
    task_type: str = "qa"
    samples: int = 1024
    seed: int = 0
    sampling_semantics: str = "first_n_valid_rows"
    n_windows: int = 32
    n_steps: int = 32
    window_size: int = 32

    def __post_init__(self) -> None:
        if self.source == "arc_easy":
            if self.task_type != "qa":
                raise ValueError("ARC-Easy TaBP calibration requires task_type='qa'")
            if self.dataset_revision != ARC_EASY_REVISION:
                raise ValueError("TaBP ARC-Easy revision must remain pinned")
            if self.sampling_semantics != "first_n_valid_rows":
                raise ValueError("Unsupported ARC-Easy sampling semantics")
        elif self.source == "wikitext":
            if self.task_type != "text_generation":
                raise ValueError("WikiText TaBP calibration requires task_type='text_generation'")
            if self.sampling_semantics != "concatenate_train_double_newline":
                raise ValueError("Unsupported WikiText sampling semantics")
        else:
            raise ValueError(f"Unsupported TaBP calibration source: {self.source!r}")
        if self.samples <= 0 or self.n_windows <= 0 or self.n_steps <= 0 or self.window_size <= 0:
            raise ValueError("TaBP calibration sizes must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaBPCalibrationSample:
    input_ids: Any
    attention_mask: Any | None
    allowed_token_ids: tuple[int, ...]
    key_token_id: int


@dataclass(frozen=True)
class TaBPCalibrationContext:
    config: TaBPCalibrationConfig
    samples: tuple[TaBPCalibrationSample, ...]
    tokenizer_class: str
    tokenizer_is_fast: bool | None
    text_token_ids: Any | None = None


def _answer_token_id(tokenizer: Any, label: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(label)
    unknown = getattr(tokenizer, "unk_token_id", None)
    if token_id is not None and token_id != unknown:
        return int(token_id)
    encoded = tokenizer(label, add_special_tokens=False)["input_ids"]
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    if len(encoded) != 1:
        raise ValueError(f"ARC answer label is not one tokenizer token: {label!r}")
    return int(encoded[0])


def format_arc_easy(example: dict[str, Any]) -> tuple[str, tuple[str, ...], str]:
    choices = example["choices"]
    labels = tuple(str(value) for value in choices["label"])
    texts = tuple(str(value) for value in choices["text"])
    if len(labels) != len(texts) or not labels:
        raise ValueError("ARC-Easy choices must contain aligned labels and text")
    options = "".join(f"{label}. {text}\n" for label, text in zip(labels, texts))
    prompt = ARC_PREFIX + "Question: " + str(example["question"]) + "\n" + options + ARC_POSTFIX
    answer = str(example["answerKey"])
    if answer not in labels:
        raise ValueError(f"ARC-Easy answer key {answer!r} is not a choice label")
    return prompt, labels, answer


class ARCEasyTaBPCalibrationProvider:
    """Prepare the official ordered ARC-Easy training subset."""

    def __init__(self, loader: Callable[[], Iterable[dict[str, Any]]] | None = None) -> None:
        self.loader = loader

    def _rows(self, config: TaBPCalibrationConfig, dataset_path: Path | None) -> Iterable[dict[str, Any]]:
        if self.loader is not None:
            return self.loader()
        try:
            from datasets import load_dataset, load_from_disk
        except ImportError as error:
            raise RuntimeError("TaBP calibration requires the datasets package") from error
        if dataset_path is not None:
            loaded = load_from_disk(str(dataset_path.expanduser().resolve()))
            return loaded["train"] if hasattr(loaded, "keys") and "train" in loaded else loaded
        return load_dataset("allenai/ai2_arc", "ARC-Easy", split="train", revision=config.dataset_revision)

    def prepare(self, tokenizer: Any, config: TaBPCalibrationConfig, *, dataset_path: Path | None = None) -> TaBPCalibrationContext:
        if config.source != "arc_easy":
            raise ValueError("ARC-Easy provider received a non-ARC configuration")
        prepared: list[TaBPCalibrationSample] = []
        for example in self._rows(config, dataset_path):
            prompt, labels, answer = format_arc_easy(example)
            allowed = tuple(_answer_token_id(tokenizer, label) for label in labels)
            if len(set(allowed)) != len(allowed):
                raise ValueError("ARC-Easy answer labels map to duplicate token IDs")
            encoded = tokenizer(prompt, return_tensors="pt", truncation=True)
            prepared.append(TaBPCalibrationSample(
                input_ids=encoded["input_ids"].cpu(),
                attention_mask=(encoded.get("attention_mask").cpu() if encoded.get("attention_mask") is not None else None),
                allowed_token_ids=allowed,
                key_token_id=_answer_token_id(tokenizer, answer),
            ))
            if len(prepared) >= config.samples:
                break
        if len(prepared) != config.samples:
            raise ValueError(f"TaBP requested {config.samples} calibration rows, got {len(prepared)}")
        return TaBPCalibrationContext(config, tuple(prepared), type(tokenizer).__name__, getattr(tokenizer, "is_fast", None))


class WikiTextTaBPCalibrationProvider:
    """Prepare the pinned code's concatenated WikiText token stream."""

    def __init__(self, loader: Callable[[], Iterable[dict[str, Any]]] | None = None) -> None:
        self.loader = loader

    def _rows(self, config: TaBPCalibrationConfig, dataset_path: Path | None) -> Iterable[dict[str, Any]]:
        if self.loader is not None:
            return self.loader()
        try:
            from datasets import load_dataset, load_from_disk
        except ImportError as error:
            raise RuntimeError("TaBP calibration requires the datasets package") from error
        if dataset_path is not None:
            loaded = load_from_disk(str(dataset_path.expanduser().resolve()))
            return loaded["train"] if hasattr(loaded, "keys") and "train" in loaded else loaded
        return load_dataset("wikitext", "wikitext-2-raw-v1", split="train", revision=config.dataset_revision)

    def prepare(self, tokenizer: Any, config: TaBPCalibrationConfig, *, dataset_path: Path | None = None) -> TaBPCalibrationContext:
        if config.source != "wikitext":
            raise ValueError("WikiText provider received a non-WikiText configuration")
        joined = "\n\n".join(str(row["text"]) for row in self._rows(config, dataset_path))
        token_ids = tokenizer(joined, return_tensors="pt")["input_ids"].cpu()
        required = config.n_windows * config.window_size
        if token_ids.shape[1] < required:
            raise ValueError(f"TaBP WikiText requires at least {required} tokens, got {token_ids.shape[1]}")
        return TaBPCalibrationContext(config, (), type(tokenizer).__name__, getattr(tokenizer, "is_fast", None), token_ids)


def get_tabp_calibration_provider(source: str) -> Any:
    if source == "arc_easy":
        return ARCEasyTaBPCalibrationProvider()
    if source == "wikitext":
        return WikiTextTaBPCalibrationProvider()
    raise KeyError(f"Unknown TaBP calibration source: {source!r}")
