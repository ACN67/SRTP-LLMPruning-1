"""Paired statistics, Pareto, and result-analysis CLI tests."""

from __future__ import annotations

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from src.analysis import (
    absolute_delta,
    checkpoint_size_bytes,
    paired_bootstrap_ci,
    pareto_efficient_indices,
    relative_retention_rate,
    tokens_per_second,
)


ROOT = Path(__file__).resolve().parents[2]


def _outcome(
    task_id: str,
    trial_index: int,
    passed: bool,
    *,
    benchmark: str = "humaneval",
    project_model_id: str = "klear_agentforge_8b",
    evaluation_profile_id: str = "klear_agentforge_8b",
    prompt_sha256: str | None = None,
) -> dict[str, object]:
    return {
        "task_id": task_id,
        "trial_index": trial_index,
        "passed": passed,
        "benchmark": benchmark,
        "project_model_id": project_model_id,
        "evaluation_profile_id": evaluation_profile_id,
        "prompt_sha256": prompt_sha256 or f"sha-{task_id}",
    }


class StatisticsTests(unittest.TestCase):
    def test_deltas_retention_and_deterministic_paired_bootstrap(self):
        dense = {"a": 1.0, "b": 1.0, "c": 0.0, "d": 0.0}
        pruned = {"a": 1.0, "b": 0.0, "c": 0.0, "d": 0.0}
        first = paired_bootstrap_ci(dense, pruned, resamples=500, seed=17)
        second = paired_bootstrap_ci(dense, pruned, resamples=500, seed=17)
        self.assertEqual(first, second)
        self.assertEqual(first.task_count, 4)
        self.assertEqual(first.absolute_delta, -0.25)
        self.assertEqual(first.relative_retention_rate, 0.5)
        self.assertAlmostEqual(absolute_delta(0.8, 0.6), -0.2)
        self.assertAlmostEqual(relative_retention_rate(0.8, 0.6), 0.75)

    def test_identity_mismatch_and_zero_baseline_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "identities"):
            paired_bootstrap_ci({"a": 1}, {"b": 1}, resamples=10)
        with self.assertRaises(ZeroDivisionError):
            relative_retention_rate(0, 0)

    def test_empty_and_non_finite_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one task"):
            paired_bootstrap_ci({}, {}, resamples=10)
        with self.assertRaisesRegex(ValueError, "finite"):
            paired_bootstrap_ci({"a": math.nan}, {"a": 1.0}, resamples=10)
        with self.assertRaisesRegex(ValueError, "finite"):
            absolute_delta(math.inf, 1.0)


class EfficiencyTests(unittest.TestCase):
    def test_pareto_handles_mixed_objective_directions(self):
        points = [
            {"quality": 0.9, "latency": 10, "vram": 8},
            {"quality": 0.8, "latency": 8, "vram": 7},
            {"quality": 0.7, "latency": 12, "vram": 9},
        ]
        self.assertEqual(
            pareto_efficient_indices(
                points, {"quality": "max", "latency": "min", "vram": "min"}
            ),
            (0, 1),
        )

    def test_measured_rate_and_checkpoint_bytes(self):
        self.assertEqual(tokens_per_second(50, 2.0), 25.0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a").write_bytes(b"123")
            (root / "b").write_bytes(b"45")
            (root / "artifact_manifest.json").write_bytes(b"manifest")
            self.assertEqual(checkpoint_size_bytes(root), 13)
            self.assertEqual(
                checkpoint_size_bytes(root, exclude_names=("artifact_manifest.json",)),
                5,
            )

    def test_invalid_measured_rate_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            tokens_per_second(1, 0)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            tokens_per_second(-1, 1)

    def test_pruning_peak_vram_helper_is_cpu_safe(self):
        spec = importlib.util.spec_from_file_location(
            "run_experiment_cpu_metrics", ROOT / "scripts" / "run_pruning.py"
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        self.assertIsNone(module._start_cuda_peak(torch.nn.Linear(2, 2)))


class AnalyzeResultsCliTests(unittest.TestCase):
    def test_exact_trial_alignment_and_task_aggregation(self):
        spec = importlib.util.spec_from_file_location(
            "analyze_results_test", ROOT / "scripts" / "analyze_results.py"
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dense = root / "dense.jsonl"
            pruned = root / "pruned.jsonl"
            dense.write_text(
                "".join(json.dumps(row) + "\n" for row in (
                    _outcome("a", 0, True),
                    _outcome("b", 0, True),
                )), encoding="utf-8"
            )
            pruned.write_text(
                "".join(json.dumps(row) + "\n" for row in (
                    _outcome("a", 0, True),
                    _outcome("b", 0, False),
                )), encoding="utf-8"
            )
            args = module._parser().parse_args([
                "--dense-outcomes", str(dense), "--pruned-outcomes", str(pruned),
                "--resamples", "100", "--seed", "3",
            ])
            result = module.analyze(args)
        self.assertEqual(result["task_count"], 2)
        self.assertEqual(result["absolute_delta"], -0.5)
        self.assertEqual(result["trial_identity_alignment"], "exact")
        self.assertEqual(result["provenance_alignment"], "exact")

    def test_cli_rejects_provenance_mismatch(self):
        spec = importlib.util.spec_from_file_location(
            "analyze_results_provenance", ROOT / "scripts" / "analyze_results.py"
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        mismatches = {
            "benchmark": "mbpp",
            "project_model_id": "granite_4_2_8b",
            "evaluation_profile_id": "different_profile",
            "prompt_sha256": "different_prompt",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dense = root / "dense.jsonl"
            pruned = root / "pruned.jsonl"
            dense.write_text(json.dumps(_outcome("a", 0, True)) + "\n")
            args = module._parser().parse_args([
                "--dense-outcomes", str(dense), "--pruned-outcomes", str(pruned),
                "--resamples", "10",
            ])
            for field, value in mismatches.items():
                with self.subTest(field=field):
                    row = _outcome("a", 0, False)
                    row[field] = value
                    pruned.write_text(json.dumps(row) + "\n")
                    with self.assertRaisesRegex(ValueError, "provenance"):
                        module.analyze(args)

    def test_cli_rejects_trial_mismatch_and_zero_dense_mean(self):
        spec = importlib.util.spec_from_file_location(
            "analyze_results_rejections", ROOT / "scripts" / "analyze_results.py"
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dense = root / "dense.jsonl"
            pruned = root / "pruned.jsonl"
            dense.write_text(json.dumps(_outcome("a", 0, True)) + "\n")
            pruned.write_text(json.dumps(_outcome("a", 1, True)) + "\n")
            args = module._parser().parse_args([
                "--dense-outcomes", str(dense), "--pruned-outcomes", str(pruned),
                "--resamples", "10",
            ])
            with self.assertRaisesRegex(ValueError, "identities"):
                module.analyze(args)
            zero = json.dumps(_outcome("a", 0, False)) + "\n"
            dense.write_text(zero)
            pruned.write_text(zero)
            with self.assertRaises(ZeroDivisionError):
                module.analyze(args)


if __name__ == "__main__":
    unittest.main()
