"""Paired dense/pruned task-level statistics."""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import asdict, dataclass
from typing import Mapping


@dataclass(frozen=True)
class PairedBootstrapResult:
    task_count: int
    dense_mean: float
    pruned_mean: float
    absolute_delta: float
    relative_retention_rate: float
    confidence_level: float
    ci_low: float
    ci_high: float
    resamples: int
    seed: int

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def absolute_delta(dense: float, pruned: float) -> float:
    """Return ``pruned - dense`` so degradation is negative."""

    dense_value = float(dense)
    pruned_value = float(pruned)
    if not math.isfinite(dense_value) or not math.isfinite(pruned_value):
        raise ValueError("Dense and pruned values must be finite")
    return pruned_value - dense_value


def relative_retention_rate(dense: float, pruned: float) -> float:
    """Return ``pruned / dense``; a zero dense baseline is undefined."""

    dense_value = float(dense)
    pruned_value = float(pruned)
    if not math.isfinite(dense_value) or not math.isfinite(pruned_value):
        raise ValueError("Dense and pruned values must be finite")
    if dense_value == 0:
        raise ZeroDivisionError("Relative retention is undefined for a zero dense baseline")
    return pruned_value / dense_value


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _aligned(
    dense_by_task: Mapping[str, float],
    pruned_by_task: Mapping[str, float],
) -> tuple[list[float], list[float]]:
    dense_keys = set(dense_by_task)
    pruned_keys = set(pruned_by_task)
    if dense_keys != pruned_keys:
        missing = sorted(dense_keys - pruned_keys)
        extra = sorted(pruned_keys - dense_keys)
        raise ValueError(
            "Dense/pruned task identities must match exactly; "
            f"missing_in_pruned={missing}, extra_in_pruned={extra}"
        )
    if not dense_keys:
        raise ValueError("Paired analysis requires at least one task")
    identities = sorted(dense_keys)
    dense = [float(dense_by_task[key]) for key in identities]
    pruned = [float(pruned_by_task[key]) for key in identities]
    if not all(math.isfinite(value) for value in dense + pruned):
        raise ValueError("Paired task values must be finite")
    return dense, pruned


def paired_bootstrap_ci(
    dense_by_task: Mapping[str, float],
    pruned_by_task: Mapping[str, float],
    *,
    resamples: int = 10_000,
    seed: int = 0,
    confidence_level: float = 0.95,
) -> PairedBootstrapResult:
    """Percentile CI for the paired mean delta, resampling task identities."""

    if resamples <= 0:
        raise ValueError("Bootstrap resamples must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be between zero and one")
    dense, pruned = _aligned(dense_by_task, pruned_by_task)
    deltas = [pruned_value - dense_value for dense_value, pruned_value in zip(dense, pruned)]
    rng = random.Random(seed)
    count = len(deltas)
    sampled_means = [
        statistics.fmean(deltas[rng.randrange(count)] for _ in range(count))
        for _ in range(resamples)
    ]
    alpha = (1 - confidence_level) / 2
    dense_mean = statistics.fmean(dense)
    pruned_mean = statistics.fmean(pruned)
    return PairedBootstrapResult(
        task_count=count,
        dense_mean=dense_mean,
        pruned_mean=pruned_mean,
        absolute_delta=absolute_delta(dense_mean, pruned_mean),
        relative_retention_rate=relative_retention_rate(dense_mean, pruned_mean),
        confidence_level=confidence_level,
        ci_low=_quantile(sampled_means, alpha),
        ci_high=_quantile(sampled_means, 1 - alpha),
        resamples=resamples,
        seed=seed,
    )
