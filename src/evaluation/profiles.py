"""Model-by-benchmark generation profiles for fair dense/pruned comparison."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import yaml


DEFAULT_PROFILE_DIR = Path(__file__).resolve().parents[2] / "configs" / "evaluation_profiles"


@dataclass(frozen=True)
class EvaluationProfile:
    project_model_id: str
    benchmark: str
    profile_id: str
    profile_version: int
    use_chat_template: bool
    chat_template_kwargs: Mapping[str, Any]
    do_sample: bool
    temperature: float | None
    top_p: float | None
    top_k: int | None
    max_new_tokens: int
    num_trials: int
    source: str
    max_new_tokens_source: str | None = None

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0 or self.num_trials <= 0:
            raise ValueError("Generation limits and trial count must be positive")
        if not self.do_sample and any(
            value is not None for value in (self.temperature, self.top_p, self.top_k)
        ):
            raise ValueError("Greedy profiles must not set sampling parameters")

    def with_overrides(
        self, *, max_new_tokens: int | None = None, num_trials: int | None = None
    ) -> tuple["EvaluationProfile", bool]:
        updated = replace(
            self,
            max_new_tokens=(max_new_tokens if max_new_tokens is not None else self.max_new_tokens),
            num_trials=(num_trials if num_trials is not None else self.num_trials),
        )
        return updated, updated != self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def list_profile_model_ids(config_dir: Path = DEFAULT_PROFILE_DIR) -> tuple[str, ...]:
    return tuple(sorted(path.stem for path in config_dir.glob("*.yaml")))


def load_evaluation_profile(
    project_model_id: str,
    benchmark: str,
    config_dir: Path = DEFAULT_PROFILE_DIR,
) -> EvaluationProfile:
    path = config_dir / f"{project_model_id}.yaml"
    if not path.is_file():
        raise KeyError(f"No evaluation profile for model {project_model_id!r}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw.get("project_model_id") != project_model_id:
        raise ValueError(f"Evaluation profile identity mismatch: {path}")
    try:
        values = dict(raw["benchmarks"][benchmark])
    except KeyError as error:
        raise KeyError(
            f"No {benchmark!r} profile for model {project_model_id!r}"
        ) from error
    return EvaluationProfile(
        project_model_id=project_model_id,
        benchmark=benchmark,
        profile_version=int(raw["profile_version"]),
        source=str(raw["source"]),
        **values,
    )
