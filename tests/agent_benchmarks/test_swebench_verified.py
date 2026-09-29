import json
import tempfile
import unittest
from pathlib import Path

from src.agent_benchmarks import get_agent_benchmark


ROW = {"instance_id": "owner__repo-1", "repo": "owner/repo", "base_commit": "a" * 40, "problem_statement": "Fix the public bug", "patch": "SECRET GOLD", "test_patch": "SECRET TEST", "FAIL_TO_PASS": "SECRET NAMES", "version": "1.0"}


class SWEbenchVerifiedTests(unittest.TestCase):
    def test_schema_prompt_no_gold_leakage_and_command(self):
        adapter = get_agent_benchmark("swebench_verified")
        instance = adapter.load_instance(ROW)
        rendered = json.dumps(instance.to_dict())
        self.assertIn("Fix the public bug", rendered)
        for secret in ("SECRET GOLD", "SECRET TEST", "SECRET NAMES"):
            self.assertNotIn(secret, rendered)
        command = adapter.evaluator_command(Path("predictions.jsonl"), "run-1", 3, instance_ids=(instance.instance_id,), task_repo=Path("tasks"), report_dir=Path("reports"))
        self.assertTrue(command[0].endswith("/.venv-swebench/bin/swebench"))
        self.assertEqual(command[1], "eval")
        self.assertTrue(command[2].endswith("/.agent_benchmarks/datasets/swebench_verified/test.json"))
        self.assertIn("--predictions", command)
        self.assertIn("--task-repo", command)
        self.assertEqual(command[-2:], ("--instance", instance.instance_id))

    def test_v5_report_resolved_unresolved_mapping(self):
        adapter = get_agent_benchmark("swebench_verified")
        report = {"submitted_ids": ["a", "b"], "completed_ids": ["a", "b"], "resolved_ids": ["a"], "unresolved_ids": ["b"], "error_ids": [], "incomplete_ids": []}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "report.json"); path.write_text(json.dumps(report))
            parsed = adapter.parse_report(path)
        self.assertEqual((parsed["resolved"], parsed["resolved_rate"]), (1, .5))
        self.assertTrue(parsed["per_instance"]["a"]["resolved"])
        self.assertFalse(parsed["per_instance"]["b"]["resolved"])


if __name__ == "__main__": unittest.main()
