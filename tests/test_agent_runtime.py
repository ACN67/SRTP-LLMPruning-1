import json
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from src.agent_evaluation import AgentResult, RepositoryTask, VLLMServer, get_agent_runner, load_agent_system_spec, resolve_artifact
from src.models import load_model_spec
from src.agent_evaluation.serving import ServingStartupError, build_vllm_command, port_is_available, resolve_granite_parser
from src.agent_evaluation.task import extract_patch


def make_artifact(root, model_type="qwen3", depth=36, pruning=None):
    root.mkdir()
    model_id = "granite_4_2_8b" if model_type == "granite" else "klear_agentforge_8b"
    spec = load_model_spec(model_id)
    (root / "config.json").write_text(json.dumps({
        "model_type": model_type, "num_hidden_layers": depth,
        "hidden_size": spec.expected_hidden_size,
        "intermediate_size": spec.expected_intermediate_size,
        "num_attention_heads": spec.expected_num_attention_heads,
        "num_key_value_heads": spec.expected_num_key_value_heads,
    }))
    (root / "model.safetensors").write_bytes(b"fake")
    if pruning:
        (root / "pruning_manifest.json").write_text(json.dumps({
            "schema_version": 3,
            "model": {"project_model_id": spec.project_model_id, "architecture": spec.architecture, "model_type": spec.model_type, "adapter": spec.adapter, "actual_block_count": depth},
            "pruning": {**pruning, "post_pruning_actual_block_count": depth},
        }))
    return root


class FakeProcess:
    def __init__(self, returncode=None, terminate_stops=True):
        self.returncode, self.pid = returncode, 1234
        self.terminate_stops, self.terminated, self.killed = terminate_stops, False, False
    def poll(self): return self.returncode
    def terminate(self):
        self.terminated = True
        if self.terminate_stops: self.returncode = 0
    def wait(self, timeout=None):
        if self.returncode is None: raise subprocess.TimeoutExpired("fake", timeout)
        return self.returncode
    def kill(self): self.killed = True; self.returncode = -9


class OpenAIHandler(BaseHTTPRequestHandler):
    calls = []
    def log_message(self, *args): pass
    def _send(self, payload):
        data = json.dumps(payload).encode(); self.send_response(200)
        self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(data)))
        self.end_headers(); self.wfile.write(data)
    def do_GET(self):
        self._send({"object": "list", "data": [{"id": "srtp-klear-agentforge-8b", "object": "model"}]})
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0)); request = json.loads(self.rfile.read(length))
        type(self).calls.append((self.path, request["model"]))
        if len(type(self).calls) == 1:
            content = "THOUGHT: make the requested change\n```bash\nprintf 'fixed\\n' > result.txt\n```"
        else:
            content = "THOUGHT: submit the completed change\n```bash\necho MINI_SWE_AGENT_FINAL_OUTPUT && git add -A && git diff --cached\n```"
        self._send({"id": "chatcmpl-test", "object": "chat.completion", "created": 1, "model": request["model"], "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}})


class AgentRuntimeTests(unittest.TestCase):
    def test_artifact_dense_pruned_and_granite_parser(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            dense = resolve_artifact(load_agent_system_spec("klear_agentforge_8b"), make_artifact(base / "dense"))
            self.assertEqual(dense.kind, "dense")
            for method in ("magnitude", "wanda", "sparsegpt"):
                pruned = resolve_artifact(load_agent_system_spec("klear_agentforge_8b"), make_artifact(base / method, pruning={"pruner": method}))
                self.assertEqual((pruned.kind, pruned.num_hidden_layers), ("pruned", 36))
            for method in ("sleb", "tabp"):
                pruned = resolve_artifact(load_agent_system_spec("klear_agentforge_8b"), make_artifact(base / method, depth=30, pruning={"pruner": method, "condition": "ssn" if method == "tabp" else "canonical"}))
                self.assertEqual((pruned.kind, pruned.num_hidden_layers), ("pruned", 30))
        self.assertTrue(resolve_granite_parser(load_agent_system_spec("granite_4_2_8b")).is_file())

    def test_exact_command_and_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            system = load_agent_system_spec("klear_agentforge_8b")
            item = resolve_artifact(system, make_artifact(Path(tmp) / "model"))
            serving = system.serving.with_overrides({"port": 8123, "tensor_parallel_size": 2})
            command = build_vllm_command(serving, item)
            self.assertEqual(command[1:3], ("serve", item.path))
            self.assertEqual(command[command.index("--port") + 1], "8123")
            self.assertEqual(command[command.index("--tensor-parallel-size") + 1], "2")

    def test_port_conflict(self):
        sock = socket.socket(); sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]
        try: self.assertFalse(port_is_available("0.0.0.0", port))
        finally: sock.close()

    def server(self, tmp, process, query, timeout=.03):
        system = load_agent_system_spec("klear_agentforge_8b")
        item = resolve_artifact(system, make_artifact(Path(tmp) / "model"))
        return VLLMServer(system, item, Path(tmp) / "out", startup_timeout=timeout, poll_interval=.001, terminate_timeout=.001, popen_factory=lambda *a, **k: process, models_query=query)

    def test_lifecycle_cleanup_and_forced_kill(self):
        with tempfile.TemporaryDirectory() as tmp:
            process = FakeProcess()
            with self.server(tmp, process, lambda *a: ("srtp-klear-agentforge-8b",)) as server:
                self.assertIsNotNone(server.ready_at)
            self.assertTrue(process.terminated)
        with tempfile.TemporaryDirectory() as tmp:
            process = FakeProcess(terminate_stops=False); server = self.server(tmp, process, lambda *a: ("srtp-klear-agentforge-8b",))
            server.start(); server.stop(); self.assertTrue(process.killed)

    def test_timeout_early_death_and_mismatch(self):
        for kind in ("timeout", "death", "mismatch"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                process = FakeProcess(7 if kind == "death" else None)
                query = ((lambda *a: (_ for _ in ()).throw(OSError("wait"))) if kind == "timeout" else (lambda *a: ("wrong",)))
                with self.assertRaises(ServingStartupError): self.server(tmp, process, query).start()

    def test_repository_dirty_safety_patch_and_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp); subprocess.run(("git", "init", "-q", str(repo)), check=True)
            subprocess.run(("git", "-C", str(repo), "config", "user.email", "a@b.c"), check=True); subprocess.run(("git", "-C", str(repo), "config", "user.name", "T"), check=True)
            (repo / "a.txt").write_text("a\n"); subprocess.run(("git", "-C", str(repo), "add", "."), check=True); subprocess.run(("git", "-C", str(repo), "commit", "-qm", "base"), check=True)
            task = RepositoryTask("x", repo, "change"); task.validate(); (repo / "a.txt").write_text("b\n")
            (repo / "new.txt").write_text("new\n")
            with self.assertRaises(ValueError): task.validate()
            patch, names = extract_patch(repo)
            self.assertIn("+b", patch)
            self.assertIn("+new", patch)
            self.assertEqual(names, ("a.txt", "new.txt"))
        required = dict(system_id="s", project_model_id="m", artifact_provenance={}, task_id="t", status="success", started_at="a", ended_at="b", runtime_seconds=1, framework="f", framework_version="1", framework_revision="0"*40, effective_agent_config={}, endpoint="e", serving_manifest_path="s", trajectory_path="t", stdout_path="o", stderr_path="e", patch_path="p", patch_sha256="h", changed_files=("a",), repository_before={}, repository_after={})
        self.assertEqual(AgentResult(**required).to_dict()["changed_files"], ["a"])

    def test_three_cli_dry_runs(self):
        models = {"klear_agentforge_8b": ("qwen3", 36), "swe_lego_qwen3_8b": ("qwen3", 36), "granite_4_2_8b": ("granite", 40)}
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); repo = base / "repo"; subprocess.run(("git", "init", "-q", str(repo)), check=True)
            subprocess.run(("git", "-C", str(repo), "config", "user.email", "a@b.c"), check=True); subprocess.run(("git", "-C", str(repo), "config", "user.name", "T"), check=True)
            (repo / "a").write_text("a"); subprocess.run(("git", "-C", str(repo), "add", "."), check=True); subprocess.run(("git", "-C", str(repo), "commit", "-qm", "base"), check=True)
            task = base / "task.md"; task.write_text("change a")
            for name, (kind, depth) in models.items():
                model = make_artifact(base / name, kind, depth); output = base / (name + "-out")
                subprocess.run((sys.executable, str(ROOT / "scripts/run_agent_system.py"), "--system", name, "--artifact-path", str(model), "--repo-path", str(repo), "--task-file", str(task), "--output-dir", str(output), "--dry-run"), cwd=ROOT, check=True, capture_output=True, text=True)
                manifest = json.loads((output / "run_manifest.json").read_text())
                self.assertEqual(manifest["result"]["status"], "dry_run_validated")
                self.assertNotIn("api_key", manifest["effective_serving_config"])

    @unittest.skipUnless((ROOT / ".venv-miniswe/bin/python").is_file(), "mini-swe runtime not installed")
    def test_real_miniswe_agent_loop_against_fake_openai(self):
        OpenAIHandler.calls = []
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), OpenAIHandler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True); thread.start()
        endpoint = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp); repo = base / "repo"; repo.mkdir()
                subprocess.run(("git", "init", "-q", str(repo)), check=True)
                subprocess.run(("git", "-C", str(repo), "config", "user.email", "a@b.c"), check=True); subprocess.run(("git", "-C", str(repo), "config", "user.name", "T"), check=True)
                (repo / "README.md").write_text("base\n"); subprocess.run(("git", "-C", str(repo), "add", "."), check=True); subprocess.run(("git", "-C", str(repo), "commit", "-qm", "base"), check=True)
                system = load_agent_system_spec("klear_agentforge_8b")
                item = resolve_artifact(system, make_artifact(base / "model"))
                with VLLMServer(system, item, base / "server", endpoint=endpoint, startup_timeout=2) as server:
                    result = get_agent_runner(system).run(RepositoryTask("real-loop", repo, "Create result.txt containing fixed"), server.endpoint, base / "out", timeout=60, artifact_provenance=item.to_dict())
                self.assertEqual(result.status, "success", (base / "out/agent.stderr.log").read_text())
                self.assertIn("result.txt", result.changed_files)
                self.assertTrue((base / "out/trajectory.json").is_file())
                self.assertEqual(OpenAIHandler.calls, [("/v1/chat/completions", "srtp-klear-agentforge-8b")] * 2)
        finally:
            httpd.shutdown(); httpd.server_close(); thread.join(timeout=2)


if __name__ == "__main__": unittest.main()
