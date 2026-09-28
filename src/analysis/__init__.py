"""Benchmark-agnostic paired statistics and efficiency helpers."""

from .efficiency import checkpoint_size_bytes, pareto_efficient_indices, tokens_per_second
from .statistics import (
    PairedBootstrapResult,
    absolute_delta,
    paired_bootstrap_ci,
    relative_retention_rate,
)

__all__ = [
    "PairedBootstrapResult",
    "absolute_delta",
    "checkpoint_size_bytes",
    "paired_bootstrap_ci",
    "pareto_efficient_indices",
    "relative_retention_rate",
    "tokens_per_second",
]
