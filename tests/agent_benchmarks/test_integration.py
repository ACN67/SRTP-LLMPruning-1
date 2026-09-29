import json
import subprocess
import tempfile
import threading
import unittest
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer
from pathlib import Path

from src.agent_benchmarks import AgentBenchmarkResult, get_agent_benchmark
from src.agent_benchmarks.common import prediction_sha256, provision_repository, write_predictions
from src.agent_runner import VLLMServer, get_agent_runner, load_agent_system_spec, resolve_artifact
from tests.agent_runner.test_agent_runner import OpenAIHandler, ROOT, make_artifact


@unittest.skipUnless((ROOT / ".venv-miniswe/bin/python").is_file(), "mini-swe runtime not installed")
class BenchmarkEndToEndTests(unittest.TestCase):
    def test_tiny_instance_real_agent_runner_prediction_and_fake_evaluator(self):
        OpenAIHandler.calls = []
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), OpenAIHandler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True); thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory); source = root / "source"; source.mkdir()
                subprocess.run(("git", "init", "-q", str(source)), check=True)
                subprocess.run(("git", "-C", str(source), "config", "user.email", "a@b.c"), check=True)
                subprocess.run(("git", "-C", str(source), "config", "user.name", "T"), check=True)
                (source / "README.md").write_text("base\n")
                subprocess.run(("git", "-C", str(source), "add", "."), check=True)
                subprocess.run(("git", "-C", str(source), "commit", "-qm", "base"), check=True)
                commit = subprocess.run(("git", "-C", str(source), "rev-parse", "HEAD"), check=True, capture_output=True, text=True).stdout.strip()
                adapter = get_agent_benchmark("swebench_verified")
                instance = adapter.load_instance({"instance_id": "tiny__repo-1", "repo": str(source), "base_commit": commit, "problem_statement": "Create result.txt containing fixed"})
                provisioned = provision_repository(instance.repo, instance.base_commit, instance.instance_id, repo_cache_root=root / "cache", workspace_root=root / "work", offline=True)
                system = load_agent_system_spec("klear_agentforge_8b")
                artifact = resolve_artifact(
                    system, make_artifact(root / "model"), allow_unverified_model=True
                )
                endpoint = f"http://127.0.0.1:{httpd.server_address[1]}"
                with VLLMServer(
                    system, artifact, root / "server", endpoint=endpoint,
                    allow_unverified_external_endpoint=True, startup_timeout=2,
                ) as server:
                    result = get_agent_runner(system).run(adapter.prepare_task(instance, provisioned.worktree_path), server.endpoint, root / "agent", timeout=60, artifact_provenance=artifact.to_dict())
                prediction = adapter.build_prediction(result)
                digest = write_predictions(root / "predictions.jsonl", [prediction])
                official = {"submitted_ids": [instance.instance_id], "completed_ids": [instance.instance_id], "resolved_ids": [instance.instance_id], "unresolved_ids": [], "error_ids": [], "incomplete_ids": []}
                report_path = root / "official-report.json"; report_path.write_text(json.dumps(official))
                parsed = adapter.parse_report(report_path)
                now = datetime.now(timezone.utc).isoformat()
                benchmark_result = AgentBenchmarkResult(adapter.benchmark_id, instance.instance_id, system.system_id, artifact.to_dict(), str(root / "agent/agent_result.json"), str(root / "predictions.jsonl"), prediction_sha256(prediction), "completed", parsed["per_instance"][instance.instance_id]["resolved"], str(report_path), {"harness_revision": adapter.spec.harness.revision}, now, now, result.runtime_seconds)
                self.assertEqual(result.status, "patch_generated")
                self.assertIn("result.txt", result.changed_files)
                self.assertEqual(len(digest), 64)
                self.assertTrue(benchmark_result.success)
                provisioned.cleanup()
        finally:
            httpd.shutdown(); httpd.server_close(); thread.join(timeout=2)


if __name__ == "__main__": unittest.main()
