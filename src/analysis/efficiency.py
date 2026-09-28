"""Measured efficiency utilities without serving-only proxy metrics."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence


def tokens_per_second(generated_tokens: int, wall_time_seconds: float) -> float:
    if generated_tokens < 0:
        raise ValueError("Generated token count must be non-negative")
    if wall_time_seconds <= 0:
        raise ValueError("Wall time must be positive")
    return generated_tokens / wall_time_seconds


def checkpoint_size_bytes(
    path: Path,
    *,
    exclude_names: Sequence[str] = (),
) -> int:
    """Sum ordinary files without following symlinks or named exclusions."""

    root = path.expanduser().resolve()
    excluded = set(exclude_names)
    if root.is_file():
        if root.name in excluded:
            return 0
        return root.stat().st_size
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint path does not exist: {root}")
    return sum(
        item.stat().st_size
        for item in root.rglob("*")
        if item.is_file() and not item.is_symlink() and item.name not in excluded
    )


def pareto_efficient_indices(
    points: Sequence[Mapping[str, Any]],
    objectives: Mapping[str, str],
) -> tuple[int, ...]:
    """Return input indices not dominated across explicit min/max objectives."""

    if not objectives:
        raise ValueError("At least one Pareto objective is required")
    invalid = {direction for direction in objectives.values() if direction not in {"min", "max"}}
    if invalid:
        raise ValueError(f"Pareto directions must be min or max: {sorted(invalid)}")
    values: list[dict[str, float]] = []
    for index, point in enumerate(points):
        missing = [name for name in objectives if name not in point]
        if missing:
            raise ValueError(f"Point {index} is missing objectives: {missing}")
        numeric = {name: float(point[name]) for name in objectives}
        if not all(math.isfinite(value) for value in numeric.values()):
            raise ValueError(f"Point {index} has a non-finite objective")
        values.append(numeric)

    efficient: list[int] = []
    for candidate_index, candidate in enumerate(values):
        dominated = False
        for other_index, other in enumerate(values):
            if candidate_index == other_index:
                continue
            no_worse = all(
                other[name] >= candidate[name]
                if direction == "max" else other[name] <= candidate[name]
                for name, direction in objectives.items()
            )
            strictly_better = any(
                other[name] > candidate[name]
                if direction == "max" else other[name] < candidate[name]
                for name, direction in objectives.items()
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            efficient.append(candidate_index)
    return tuple(efficient)
