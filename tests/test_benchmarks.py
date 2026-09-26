"""Pinned benchmark assets, task-set validation, and result workflow tests."""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import inspect
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import src.evaluation as evaluation
from src.evaluation import BenchmarkSpec, BenchmarkTask, TaskEvaluation, load_benchmark_spec
from src.evaluation.benchmarks import HumanEvalBenchmark, LiveCodeBenchBenchmark, MBPPBenchmark

ROOT = Path(__file__).resolve().parents[1]


def altered(spec, **changes):
    values = vars(spec).copy()
    values.update(changes)
    return BenchmarkSpec(**values)


class BenchmarkAssetTests(unittest.TestCase):
    def test_benchmark_contract_has_one_authority_per_concept(self):
        fields = BenchmarkSpec.__dataclass_fields__
        self.assertNotIn("max_new_tokens", fields)
        self.assertNotIn("generation_protocol", fields)
        self.assertFalse(hasattr(evaluation, "extract_python_code"))
        for benchmark_type in (HumanEvalBenchmark, MBPPBenchmark, LiveCodeBenchBenchmark):
            self.assertNotIn("offline", inspect.signature(benchmark_type.load_tasks).parameters)

    def test_pinned_configs(self):
        human = load_benchmark_spec("humaneval")
        mbpp = load_benchmark_spec("mbpp")
        lcb = load_benchmark_spec("livecodebench")
        self.assertEqual((human.source_revision, human.expected_task_count),
            ("6d43fb980f9fee3c892a914eda09951f772ad10d", 164))
        self.assertEqual(mbpp.metadata["source_sha256"],
            "ccf64ceae9c5403bf50a044cb6d505bfd2a2963ee58338ba268fd65beab92a9f")
        self.assertEqual(mbpp.metadata["task_id_range"], [11, 510])
        self.assertEqual((lcb.metadata["dataset_revision"], lcb.metadata["release_version"],
                          lcb.metadata["dataset_file"], lcb.metadata["dataset_sha256"],
                          lcb.expected_task_count),
            ("0fe84c3912ea0c4d4a78037083943e8f0c4dd505", "v6", "test6.jsonl",
             "bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5", 175))

    def test_humaneval_requires_exactly_164_unique_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "HumanEval.jsonl.gz")
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                for index in range(164):
                    handle.write(json.dumps({"task_id": f"HumanEval/{index}", "prompt": "", "test": "", "entry_point": "f"}) + "\n")
            benchmark = HumanEvalBenchmark(altered(load_benchmark_spec("humaneval"), dataset_path=str(path)))
            self.assertEqual(len(benchmark.load_tasks()), 164)

    def test_mbpp_uses_only_11_through_510(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "mbpp.jsonl")
            rows = [{"task_id": index, "text": "x", "test_list": ["assert True"]} for index in range(1, 511)]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            benchmark = MBPPBenchmark(altered(load_benchmark_spec("mbpp"), dataset_path=str(path)))
            tasks = benchmark.load_tasks()
            self.assertEqual((tasks[0].task_id, tasks[-1].task_id, len(tasks)), ("11", "510", 500))

    def test_lcb_local_verified_jsonl_requires_175_unique_tasks(self):
        rows = [{"question_id": str(index), "question_content": "q", "starter_code": "",
                 "metadata": "{}", "public_test_cases": "[]", "private_test_cases": "[]"}
                for index in range(175)]
        payload = "".join(json.dumps(row) + "\n" for row in rows).encode()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "test6.jsonl")
            path.write_bytes(payload)
            original = load_benchmark_spec("livecodebench")
            metadata = dict(original.metadata)
            metadata["dataset_sha256"] = hashlib.sha256(payload).hexdigest()
            benchmark = LiveCodeBenchBenchmark(altered(original, dataset_path=str(path), metadata=metadata))
            self.assertEqual(len(benchmark.load_tasks()), 175)


class BenchmarkCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT / "scripts" / "run_benchmark.py"
        spec = importlib.util.spec_from_file_location("run_benchmark_test", path)
        cls.module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(cls.module)

    def args(self, root):
        return SimpleNamespace(
            phase="evaluate", benchmark="humaneval", model="tiny",
            artifact_path=None, artifact_kind="dense", artifact_label="dense",
            output_root=Path(root), run_id="run", limit=None,
            max_new_tokens=None, num_trials=None, device=None,
            device_map=None, dtype=None, cache_dir=Path(root), offline=True,
            resume=False, overwrite=False, timeout=1.0,
        )

    def test_evaluation_rejects_missing_and_duplicate_tasks(self):
        benchmark = SimpleNamespace(
            spec=SimpleNamespace(source_revision="a" * 40, metadata={}, prompt_protocol="p",
                code_extraction_protocol="e"),
        )
        tasks = [BenchmarkTask("humaneval", "a", "", {}), BenchmarkTask("humaneval", "b", "", {})]
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(directory)
            run_dir = self.module._run_directory(args)
            run_dir.mkdir(parents=True)
            Path(run_dir, "generations.jsonl").write_text(
                json.dumps({"task_id": "a", "trial_index": 0}) + "\n", encoding="utf-8"
            )
            profile = SimpleNamespace(profile_id="p1", profile_version=2, num_trials=1,
                                      to_dict=lambda: {})
            with self.assertRaisesRegex(ValueError, "exactly match"):
                self.module.evaluate(args, run_dir, benchmark, tasks, 2, profile, False)
            Path(run_dir, "generations.jsonl").write_text(
                (json.dumps({"task_id": "a", "trial_index": 0}) + "\n") * 2, encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "exactly match"):
                self.module.evaluate(args, run_dir, benchmark, tasks, 2, profile, False)

    def test_multi_trial_schema_reports_repeated_pass_at_1(self):
        task = BenchmarkTask("humaneval", "a", "", {})
        benchmark = SimpleNamespace(
            spec=SimpleNamespace(source_revision="a" * 40, metadata={}, prompt_protocol="p",
                code_extraction_protocol="e"),
            evaluate=lambda task, code, timeout: TaskEvaluation(
                task.task_id, code == "good", "passed" if code == "good" else "wrong"
            ),
        )
        profile = SimpleNamespace(profile_id="p2", profile_version=2, num_trials=2,
                                  to_dict=lambda: {"profile_id": "p2", "num_trials": 2})
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(directory)
            args.model = "klear_agentforge_8b"
            run_dir = self.module._run_directory(args)
            run_dir.mkdir(parents=True)
            common = {"benchmark": "humaneval", "task_id": "a",
                      "project_model_id": args.model, "artifact_kind": "dense",
                      "artifact_label": "dense", "prompt_protocol": "p",
                      "evaluation_profile_id": "p2", "prompt_sha256": "x",
                      "raw_generation": "x", "generation_success": True, "error": None}
            records = [
                {**common, "trial_index": 0, "seed": 0, "processed_generation": "good"},
                {**common, "trial_index": 1, "seed": 1, "processed_generation": "bad"},
            ]
            self.module._write_jsonl(run_dir / "generations.jsonl", records)
            self.module.evaluate(args, run_dir, benchmark, [task], 1, profile, False)
            summary = json.loads((run_dir / "evaluation.json").read_text())
            manifest = json.loads((run_dir / "evaluation_manifest.json").read_text())
        self.assertEqual(summary["per_trial_pass_at_1"], [1.0, 0.0])
        self.assertEqual((summary["pass_at_1_mean"], summary["pass_at_1_std"]), (0.5, 0.5))
        self.assertNotIn("pass_at_8", summary)
        self.assertEqual(manifest["evaluator"]["per_test_timeout_seconds"], 1.0)

    def test_pruning_plan_records_full_benchmark_provenance(self):
        path = ROOT / "scripts" / "run_experiment.py"
        spec = importlib.util.spec_from_file_location("run_experiment_benchmark_plan", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        args = module._parser().parse_args([
            "--model", "klear_agentforge_8b", "--pruner", "magnitude",
            "--sparsity", "0.2", "--benchmark", "livecodebench",
        ])
        evaluation = module.build_manifest(args)["evaluation"]
        self.assertEqual(evaluation["source_revision"], "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24")
        self.assertEqual(evaluation["dataset_revision"], "0fe84c3912ea0c4d4a78037083943e8f0c4dd505")
        self.assertEqual((evaluation["release"], evaluation["task_count"], evaluation["metric"]),
            ("v6", 175, "pass@1"))
        self.assertEqual(evaluation["evaluation_profile"]["profile_id"], "klear_lcb_v6_sampling_chat_v2")

    def test_output_directory_refuses_repository_and_nonempty_by_default(self):
        args = self.args(ROOT)
        args.output_root = ROOT
        with self.assertRaisesRegex(ValueError, "outside"):
            self.module._prepare_directory(ROOT / "inside", args)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "run")
            path.mkdir()
            Path(path, "existing").write_text("x")
            args.output_root = Path(directory)
            args.phase = "generate"
            with self.assertRaisesRegex(ValueError, "non-empty"):
                self.module._prepare_directory(path, args)

    def test_phase_specific_overwrite_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "run")
            path.mkdir()
            generations = path / "generations.jsonl"
            generation_manifest = path / "generation_manifest.json"
            generations.write_text("generation-input", encoding="utf-8")
            generation_manifest.write_text("manifest-input", encoding="utf-8")
            for name in ("evaluation.json", "evaluation_manifest.json", "errors.jsonl"):
                (path / name).write_text("old", encoding="utf-8")

            args = self.args(directory)
            with self.assertRaisesRegex(ValueError, "Evaluation outputs already exist"):
                self.module._prepare_directory(path, args)

            args.overwrite = True
            self.module._prepare_directory(path, args)
            self.assertEqual(generations.read_text(), "generation-input")
            self.assertEqual(generation_manifest.read_text(), "manifest-input")
            for name in ("evaluation.json", "evaluation_manifest.json", "errors.jsonl"):
                self.assertFalse((path / name).exists())

            args.phase = "generate"
            self.module._prepare_directory(path, args)
            self.assertTrue(path.is_dir())
            self.assertEqual(list(path.iterdir()), [])

    def test_timeout_value_changes_evaluator_manifest(self):
        task = BenchmarkTask("humaneval", "a", "", {})
        benchmark = SimpleNamespace(
            spec=SimpleNamespace(source_revision="a" * 40, metadata={}, prompt_protocol="p",
                code_extraction_protocol="e"),
            evaluate=lambda task, code, timeout: TaskEvaluation(task.task_id, True, "passed"),
        )
        profile = SimpleNamespace(profile_id="p", profile_version=2, num_trials=1,
                                  to_dict=lambda: {"profile_id": "p"})
        recorded = []
        with tempfile.TemporaryDirectory() as directory:
            for index, timeout in enumerate((1.5, 7.0)):
                args = self.args(directory)
                args.model = "klear_agentforge_8b"
                args.run_id = f"run-{index}"
                args.timeout = timeout
                run_dir = self.module._run_directory(args)
                run_dir.mkdir(parents=True)
                record = {
                    "benchmark": "humaneval", "task_id": "a", "trial_index": 0, "seed": 0,
                    "project_model_id": args.model, "artifact_kind": "dense", "artifact_label": "dense",
                    "prompt_protocol": "p", "evaluation_profile_id": "p", "prompt_sha256": "x",
                    "raw_generation": "x", "processed_generation": "x",
                    "generation_success": True, "error": None,
                }
                self.module._write_jsonl(run_dir / "generations.jsonl", [record])
                self.module.evaluate(args, run_dir, benchmark, [task], 1, profile, False)
                manifest = json.loads((run_dir / "evaluation_manifest.json").read_text())
                recorded.append(manifest["evaluator"]["per_test_timeout_seconds"])
        self.assertEqual(recorded, [1.5, 7.0])

    def test_output_directory_must_not_overlap_model_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory, "model")
            artifact.mkdir()
            args = self.args(directory)
            args.artifact_path = artifact
            with self.assertRaisesRegex(ValueError, "overlap"):
                self.module._prepare_directory(artifact / "results" / "run", args)

    def test_output_path_components_reject_traversal(self):
        with self.assertRaisesRegex(Exception, "safe path component"):
            self.module._safe_component("../../models")

    def test_removed_partition_interface_is_absent(self):
        self.assertFalse((ROOT / "src" / "evaluation" / "splits.py").exists())
        destinations = {action.dest for action in self.module._parser()._actions}
        self.assertNotIn("split", destinations)
        self.assertNotIn("--use-chat-template", self.module._parser().format_help())

    def test_combined_run_phase_is_rejected(self):
        phase = next(action for action in self.module._parser()._actions if action.dest == "phase")
        self.assertEqual(tuple(phase.choices), ("generate", "evaluate"))
        with self.assertRaises(SystemExit):
            self.module._parser().parse_args([
                "--phase", "run", "--benchmark", "humaneval",
                "--model", "klear_agentforge_8b", "--run-id", "x",
            ])

    def test_manifest_uses_injected_build_source_provenance(self):
        benchmark = SimpleNamespace(
            spec=SimpleNamespace(
                source_revision="a" * 40, metadata={}, prompt_protocol="p",
                code_extraction_protocol="e",
            )
        )
        profile = SimpleNamespace(profile_id="p", profile_version=2, num_trials=1)
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(directory)
            args.model = "klear_agentforge_8b"
            with patch.dict(os.environ, {
                "SRTP_PROJECT_GIT_COMMIT": "b" * 40,
                "SRTP_SOURCE_DIRTY": "true",
            }, clear=False), patch.object(
                self.module.subprocess, "run",
                side_effect=AssertionError("injected build provenance should avoid git subprocesses"),
            ):
                manifest = self.module._manifest_base(
                    args, benchmark, [], 0, profile, False,
                )
        self.assertEqual(manifest["project_git_commit"], "b" * 40)
        self.assertTrue(manifest["source_dirty"])


if __name__ == "__main__":
    unittest.main()
