import unittest
from pathlib import Path

from src.agent_benchmarks import get_agent_benchmark


class SWEbenchMultilingualTests(unittest.TestCase):
    def test_independent_adapter_languages_command_and_aggregate(self):
        adapter = get_agent_benchmark("swebench_multilingual")
        rows = [
            {"instance_id": "go-1", "repo": "o/go", "base_commit": "a" * 40, "problem_statement": "go issue", "language": "Go"},
            {"instance_id": "rust-1", "repo": "o/rust", "base_commit": "b" * 40, "problem_statement": "rust issue", "language": "Rust"},
        ]
        instances = {item.instance_id: item for item in map(adapter.load_instance, rows)}
        command = adapter.evaluator_command(Path("p.jsonl"), "run", 2)
        self.assertTrue(command[2].endswith("/.agent_benchmarks/datasets/swebench_multilingual/test.json"))
        aggregate = adapter.aggregate({"attempted": 2, "resolved": 1, "per_instance": {"go-1": {"resolved": True}, "rust-1": {"resolved": False}}}, instances)
        self.assertEqual(aggregate["per_language"]["Go"], {"attempted": 1, "resolved": 1, "rate": 1.0})
        self.assertEqual(aggregate["per_language"]["Rust"]["rate"], 0.0)


if __name__ == "__main__": unittest.main()
