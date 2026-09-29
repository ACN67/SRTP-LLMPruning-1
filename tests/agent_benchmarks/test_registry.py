import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.agent_benchmarks import get_agent_benchmark, list_agent_benchmark_ids, load_agent_benchmark_spec
from src.agent_benchmarks.common import write_predictions
from tests.agent_runner.test_agent_runner import ROOT, make_artifact


class AgentBenchmarkRegistryTests(unittest.TestCase):
    def test_three_configs_and_exact_pins(self):
        self.assertEqual(set(list_agent_benchmark_ids()), {"swebench_verified", "swebench_multilingual", "swtbench_verified"})
        expected = {
            "swebench_verified": (500, "87ab1f6ced28f75ba73ca899dc759b019310944a"),
            "swebench_multilingual": (300, "87ab1f6ced28f75ba73ca899dc759b019310944a"),
            "swtbench_verified": (433, "443e03b385089bf8bd787fb8b79fb9a49b7fb2a0"),
        }
        for benchmark_id, values in expected.items():
            spec = load_agent_benchmark_spec(benchmark_id)
            self.assertEqual((spec.dataset.expected_task_count, spec.harness.revision), values)
            self.assertEqual(get_agent_benchmark(benchmark_id).benchmark_id, benchmark_id)
            self.assertNotIn("planned", str(spec.to_dict()).lower())
        multilingual = load_agent_benchmark_spec("swebench_multilingual")
        self.assertEqual(multilingual.metadata["expected_repo_count"], 41)
        self.assertNotIn("observed_unique_repo_ids_at_pinned_revision", multilingual.metadata)

    def test_three_cli_generation_and_official_harness_dry_runs(self):
        rows = {
            "swebench_verified": {"instance_id": "a__b-1", "repo": "a/b", "base_commit": "a" * 40, "problem_statement": "fix"},
            "swebench_multilingual": {"instance_id": "apache__druid-1", "repo": "apache/druid", "base_commit": "b" * 40, "problem_statement": "fix"},
            "swtbench_verified": {"instance_id": "a__b-2", "repo": "a/b", "base_commit": "c" * 40, "problem_statement": "test"},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); artifact = make_artifact(root / "model")
            for benchmark, row in rows.items():
                dataset = root / f"{benchmark}.jsonl"; dataset.write_text(json.dumps(row) + "\n")
                common = [sys.executable, str(ROOT / "scripts/run_agent_benchmark.py"), "--benchmark", benchmark, "--system", "klear_agentforge_8b", "--artifact-path", str(artifact), "--output-root", str(root / "results"), "--run-id", "dry", "--repo-cache-root", str(root / "repos"), "--workspace-root", str(root / "work"), "--allow-unverified-model"]
                subprocess.run((*common, "--phase", "generate", "--dataset-path", str(dataset), "--offline", "--allow-incomplete-dataset", "--dry-run"), cwd=ROOT, check=True, capture_output=True, text=True)
                prediction_path = root / "results" / benchmark / "klear_agentforge_8b" / "dry" / "predictions.jsonl"
                write_predictions(prediction_path, [{"instance_id": row["instance_id"], "model_name_or_path": "klear_agentforge_8b", "model_patch": "diff --git a/a b/a\n"}])
                subprocess.run((*common, "--phase", "evaluate", "--dry-run"), cwd=ROOT, check=True, capture_output=True, text=True)
                command = json.loads((prediction_path.parent / "evaluation/evaluator_command.json").read_text())["command"]
                self.assertTrue("--run-id" in command or "--run_id" in command)


if __name__ == "__main__": unittest.main()
