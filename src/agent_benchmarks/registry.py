"""Configuration-backed registry for implemented Agent benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .base import AgentBenchmarkSpec, repository_root
from .swebench_multilingual import SWEbenchMultilingualAdapter
from .swebench_verified import SWEbenchVerifiedAdapter
from .swtbench_verified import SWTbenchVerifiedAdapter


CONFIG_ROOT = repository_root() / "configs" / "agent_benchmarks"
ADAPTERS: dict[str, type[Any]] = {"swebench_verified": SWEbenchVerifiedAdapter, "swebench_multilingual": SWEbenchMultilingualAdapter, "swtbench_verified": SWTbenchVerifiedAdapter}


def list_agent_benchmark_ids() -> tuple[str, ...]:
    return tuple(sorted(ADAPTERS))


def load_agent_benchmark_spec(benchmark_id: str, config_root: Path = CONFIG_ROOT) -> AgentBenchmarkSpec:
    if benchmark_id not in ADAPTERS:
        raise KeyError(f"Unknown Agent benchmark {benchmark_id!r}")
    path = config_root / f"{benchmark_id}.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    spec = AgentBenchmarkSpec.from_mapping(raw)
    if spec.benchmark_id != benchmark_id:
        raise ValueError(f"Benchmark filename/config identity mismatch: {path}")
    return spec


def get_agent_benchmark(benchmark_id: str, config_root: Path = CONFIG_ROOT) -> Any:
    return ADAPTERS[benchmark_id](load_agent_benchmark_spec(benchmark_id, config_root))
