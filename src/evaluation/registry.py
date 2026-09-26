"""Configuration-backed benchmark registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .base import BenchmarkSpec
from .benchmarks import HumanEvalBenchmark, LiveCodeBenchBenchmark, MBPPBenchmark


DEFAULT_EVAL_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "eval"
_BENCHMARKS = {
    "humaneval": HumanEvalBenchmark,
    "mbpp": MBPPBenchmark,
    "livecodebench": LiveCodeBenchBenchmark,
}


def list_benchmarks() -> tuple[str, ...]:
    return tuple(sorted(_BENCHMARKS))


def load_benchmark_spec(name: str, config_dir: Path = DEFAULT_EVAL_CONFIG_DIR) -> BenchmarkSpec:
    if name not in _BENCHMARKS:
        raise KeyError(f"Unknown benchmark {name!r}")
    path = config_dir / f"{name}.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = (
        "benchmark", "display_name", "source_revision", "expected_task_count",
        "metric", "prompt_protocol", "code_extraction_protocol",
        "dataset_path",
    )
    missing = [key for key in required if raw.get(key) in (None, "")]
    if missing:
        raise ValueError(f"Benchmark config missing: {', '.join(missing)}")
    if raw["benchmark"] != name or raw["metric"] != "pass@1":
        raise ValueError(f"Invalid benchmark identity or metric in {path}")
    values = {key: raw[key] for key in required}
    metadata = {key: value for key, value in raw.items() if key not in required}
    return BenchmarkSpec(**values, metadata=metadata)


def get_benchmark(name: str, **kwargs: Any):
    spec = load_benchmark_spec(name)
    return _BENCHMARKS[name](spec, **kwargs) if kwargs else _BENCHMARKS[name](spec)
