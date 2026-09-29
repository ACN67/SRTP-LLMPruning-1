import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.agent_benchmarks.common import provision_repository, read_predictions, run_harness, write_predictions
from src.agent_benchmarks.execution import load_dataset_records, select_records
from src.agent_benchmarks.registry import load_agent_benchmark_spec


def git(repo, *args):
    return subprocess.run(("git", "-C", str(repo), *args), check=True, capture_output=True, text=True).stdout.strip()


class DatasetAndPersistenceTests(unittest.TestCase):
    def test_local_dataset_schema_selection_and_offline_failure(self):
        spec = load_agent_benchmark_spec("swebench_verified").dataset
        rows = [{"instance_id": "b"}, {"instance_id": "a"}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "tiny.jsonl")
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            loaded = load_dataset_records(spec, offline=True, local_path=path, validate_expected_count=False)
            self.assertEqual([row["instance_id"] for row in select_records(loaded, instance_ids=("a", "b"), limit=1)], ["a"])
            with self.assertRaises(KeyError):
                select_records(loaded, instance_ids=("missing",))
            with self.assertRaises(FileNotFoundError):
                load_dataset_records(spec, offline=True, local_path=Path(directory, "missing"))

    def test_prediction_order_hash_and_duplicate_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "predictions.jsonl")
            records = [
                {"instance_id": "z", "model_name_or_path": "m", "model_patch": "z"},
                {"instance_id": "a", "model_name_or_path": "m", "model_patch": "a"},
            ]
            first = write_predictions(path, records)
            self.assertEqual([item["instance_id"] for item in read_predictions(path)], ["a", "z"])
            self.assertEqual(first, write_predictions(path, reversed(records)))
            with self.assertRaisesRegex(ValueError, "exactly once"):
                write_predictions(path, [records[0], records[0]])

    def test_fake_evaluator_subprocess_and_dry_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command = (sys.executable, "-c", "print('official-stub')")
            dry = run_harness(command, cwd=root, output_dir=root / "dry", timeout=5, dry_run=True)
            self.assertIsNone(dry.exit_code)
            self.assertFalse((root / "dry/evaluator.stdout.log").exists())
            actual = run_harness(command, cwd=root, output_dir=root / "actual", timeout=5, dry_run=False)
            self.assertEqual(actual.exit_code, 0)
            self.assertIn("official-stub", (root / "actual/evaluator.stdout.log").read_text())


class RepositoryProvisioningTests(unittest.TestCase):
    def test_detached_clean_isolated_worktrees_and_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            git(source, "init", "-q")
            git(source, "config", "user.email", "test@example.com")
            git(source, "config", "user.name", "Test")
            (source / "file.txt").write_text("base\n")
            git(source, "add", ".")
            git(source, "commit", "-qm", "base")
            commit = git(source, "rev-parse", "HEAD")
            one = provision_repository(str(source), commit, "one", repo_cache_root=root / "cache", workspace_root=root / "work", offline=True)
            two = provision_repository(str(source), commit, "two", repo_cache_root=root / "cache", workspace_root=root / "work", offline=True)
            self.assertNotEqual(one.worktree_path, two.worktree_path)
            (one.worktree_path / "file.txt").write_text("changed\n")
            self.assertEqual((two.worktree_path / "file.txt").read_text(), "base\n")
            self.assertEqual((source / "file.txt").read_text(), "base\n")
            self.assertEqual(one.base_commit, commit)
            one.cleanup(); two.cleanup()
            self.assertFalse(one.worktree_path.exists())


if __name__ == "__main__":
    unittest.main()
