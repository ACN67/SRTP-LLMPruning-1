import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.agent_benchmarks import get_agent_benchmark
from src.agent_benchmarks.swtbench_data import SWTEvaluationDatasetSpec, dataset_payload, derive_evaluation_rows


class SWTbenchVerifiedTests(unittest.TestCase):
    def test_test_generation_framing_and_exact_command(self):
        adapter = get_agent_benchmark("swtbench_verified")
        row = {"instance_id": "a__b-1", "repo": "a/b", "base_commit": "c" * 40, "problem_statement": "broken behavior", "patch": "gold test", "test_patch": "gold fix"}
        instance = adapter.load_instance(row)
        self.assertIn("add or modify tests", instance.problem_statement)
        self.assertIn("Do not fix", instance.problem_statement)
        self.assertNotIn("gold test", json.dumps(instance.to_dict()))
        command = adapter.evaluator_command(Path("p.jsonl"), "run", 2, instance_ids=("a__b-1",))
        self.assertTrue(command[0].endswith("/.venv-swtbench/bin/python"))
        self.assertEqual(command[1:4], ("-m", "src.main", "--dataset_name"))
        self.assertTrue(command[command.index("--dataset_name") + 1].endswith("/.agent_benchmarks/datasets/swtbench_verified_eval/test.json"))
        self.assertNotIn("--is_swt", command)
        self.assertNotIn("--filter_swt", command)
        with self.assertRaisesRegex(ValueError, "derived original-SWE"):
            adapter.evaluator_command(Path("p.jsonl"), "run", 1, dataset_path=Path("wrong.json"))
        self.assertEqual(command[command.index("--exec_mode") + 1], "unit_test")

    def test_inference_and_evaluation_identities_and_counts(self):
        adapter = get_agent_benchmark("swtbench_verified")
        inference = adapter.spec.dataset
        evaluation = adapter.evaluation_dataset
        self.assertNotEqual(inference.repo_id, evaluation.repo_id)
        self.assertEqual(inference.expected_task_count, 433)
        self.assertEqual((evaluation.source_expected_task_count, evaluation.filter_expected_count, evaluation.expected_task_count), (500, 67, 433))
        self.assertEqual(adapter.spec.metadata["protocol"], "project_raw_swt_harness_protocol")
        self.assertEqual(adapter.spec.metadata["generation_task_field"], "problem_statement")
        self.assertEqual(set(adapter.spec.metadata["excluded_prompt_fields"]), {"text", "hits", "patch", "test_patch"})

    def test_gold_uses_original_swe_derived_snapshot(self):
        adapter = get_agent_benchmark("swtbench_verified")
        command = adapter.evaluator_command(Path("ignored.jsonl"), "gold", 1, gold=True)
        self.assertEqual(command[command.index("--predictions_path") + 1], "gold")
        self.assertTrue(command[command.index("--dataset_name") + 1].endswith("/swtbench_verified_eval/test.json"))
        self.assertNotIn("--is_swt", command)
        self.assertNotIn("--filter_swt", command)

    def test_original_swe_field_semantics_are_preserved_for_evaluation(self):
        original = {"instance_id": "org__repo-1", "patch": "diff --git a/src.py b/src.py\n--- a/src.py\n+++ b/src.py\nCODE_FIX", "test_patch": "diff --git a/test.py b/test.py\nREPRO_TEST"}
        converted = {"instance_id": original["instance_id"], "patch": f"<patch>\n{original['test_patch']}\n</patch>", "test_patch": "--- a/src.py\n+++ b/src.py\nCODE_FIX"}
        output = dataset_payload([original])
        spec = SWTEvaluationDatasetSpec(
            repo_id="SWE-bench/SWE-bench_Verified", revision="a" * 40, split="test",
            source_expected_task_count=1, expected_task_count=1,
            source_local_path="source.json", local_path="derived.json",
            filter_path="filter.txt", filter_expected_count=0,
            filter_sha256=hashlib.sha256(b"").hexdigest(),
            output_sha256=hashlib.sha256(output).hexdigest(),
        )
        derived = derive_evaluation_rows([original], [converted], set(), spec)
        self.assertEqual(converted["patch"][len("<patch>\n"):-len("\n</patch>")], original["test_patch"])
        self.assertEqual(converted["test_patch"], original["patch"].split("\n", 1)[1])
        self.assertEqual(derived[0]["patch"], original["patch"])
        self.assertEqual(derived[0]["test_patch"], original["test_patch"])

    def test_swt_success_is_not_swe_resolved_semantics(self):
        adapter = get_agent_benchmark("swtbench_verified")
        report = {"per_instance": {"a": {"success": True, "applicable": True, "resolved": False}, "b": {"success": False, "applicable": True}}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "report.json"); path.write_text(json.dumps(report))
            result = adapter.parse_report(path)
        self.assertEqual((result["successful"], result["applicable"], result["success_rate"]), (1, 2, .5))
        self.assertNotIn("resolved_rate", result)

    def test_pinned_official_aggregate_schema(self):
        adapter = get_agent_benchmark("swtbench_verified")
        report = {"total_instances": 2, "completed_ids": ["a"], "resolved_ids": ["a"], "unresolved_ids": [], "error_ids": ["b"]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "report.json"); path.write_text(json.dumps(report))
            result = adapter.parse_report(path)
        self.assertEqual((result["successful"], result["applicable"], result["attempted"]), (1, 1, 2))
        self.assertTrue(result["per_instance"]["a"]["success"])
        self.assertFalse(result["per_instance"]["b"]["applicable"])


if __name__ == "__main__": unittest.main()
