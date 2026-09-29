import hashlib
import json
import importlib.util
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
from src.agent_runner import AgentResult, RepositoryTask, VLLMServer, get_agent_runner, load_agent_system_spec, resolve_artifact
from src.artifacts import LineageOperation, ModelArtifact, artifact_inventory, write_artifact_manifest
from src.models import load_model_spec
from src.agent_runner.serving import ServingStartupError, build_vllm_command, port_is_available, resolve_granite_parser
from src.agent_runner.task import extract_patch


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
        _entries, digest = artifact_inventory(root)
        method = pruning["pruner"]
        operation = LineageOperation(
            "pruning",
            method,
            "b" * 64,
            "c" * 64,
            {**pruning, "post_pruning_actual_block_count": depth},
            {
                "source": "test",
                "structure_effect": (
                    "reduced_depth"
                    if depth < spec.expected_num_hidden_layers
                    else "weight_sparse"
                ),
            },
        )
        artifact = ModelArtifact(
            str(root.resolve()),
            "pruned",
            "full_checkpoint",
            True,
            spec.project_model_id,
            spec.architecture,
            spec.model_type,
            spec.adapter,
            depth,
            digest,
            (operation,),
            {"direct_evaluation": True, "vllm_serving": True},
            {},
        )
        write_artifact_manifest(root, artifact)
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


class AgentRunnerTests(unittest.TestCase):
    def test_generation_with_task_failure_finishes_manifest_for_evaluation_audit(self):
        path = ROOT / "scripts" / "run_agent_benchmark.py"
        spec = importlib.util.spec_from_file_location("run_agent_benchmark_completion_test", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        class FakeProvisioned:
            worktree_path = Path("/tmp/fake-worktree")
            def manifest(self): return {"status": "provisioned"}
            def cleanup(self): pass

        class FakeServer:
            endpoint = "http://127.0.0.1:8000"
            def __init__(self, *_args, **_kwargs): pass
            def start(self): return self
            def stop(self): pass
            def manifest(self):
                return {
                    "mode": "managed_subprocess",
                    "serving_provenance": {
                        "checkpoint_identity_status": "verified_managed_local_artifact"
                    },
                }

        class FakeRunner:
            def run(self, task, endpoint, output, **_kwargs):
                return AgentResult(
                    "klear_agentforge_8b", "klear_agentforge_8b", {}, task.task_id,
                    "generation_failed", "a", "b", 0.1, "fake", "1", "0" * 40,
                    {}, endpoint, "", "", "", "", "", "", (), {}, {},
                    error_type="RuntimeError", error_message="expected task failure",
                )

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            dataset = base / "tasks.json"
            dataset.write_text(json.dumps([{
                "instance_id": "owner__repo-1", "repo": "owner/repo",
                "base_commit": "a" * 40, "problem_statement": "fix",
            }]))
            model = make_artifact(base / "model")
            output_root = base / "results"
            argv = [
                str(path), "--phase", "generate", "--benchmark", "swebench_verified",
                "--system", "klear_agentforge_8b", "--artifact-path", str(model),
                "--output-root", str(output_root), "--run-id", "failed-task",
                "--repo-cache-root", str(base / "cache"),
                "--workspace-root", str(base / "work"),
                "--dataset-path", str(dataset), "--offline", "--allow-incomplete-dataset",
                "--allow-unverified-model",
            ]
            with patch.object(sys, "argv", argv), patch.object(
                module, "VLLMServer", FakeServer,
            ), patch.object(
                module, "get_agent_runner", return_value=FakeRunner(),
            ), patch.object(
                module, "provision_repository", return_value=FakeProvisioned(),
            ):
                self.assertEqual(module.main(), 4)
            run_dir = (
                output_root / "swebench_verified" / "klear_agentforge_8b" / "failed-task"
            )
            manifest = json.loads((run_dir / "generate_run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "completed")
            self.assertIn("ended_at", manifest)
            self.assertEqual(
                manifest["prediction_file_sha256"],
                hashlib.sha256((run_dir / "predictions.jsonl").read_bytes()).hexdigest(),
            )
            self.assertEqual(
                manifest["outcomes"]["owner__repo-1"]["status"],
                "generation_failed",
            )

    def test_generation_upstream_validator_rejects_manifest_and_identity_changes(self):
        path = ROOT / "scripts" / "run_agent_benchmark.py"
        spec = importlib.util.spec_from_file_location("run_agent_benchmark_provenance_test", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions = root / "predictions.jsonl"
            predictions.write_text('{"instance_id":"task-1"}\n', encoding="utf-8")
            contract = module.build_resume_identity(artifact="a", tasks="t")
            expected = {
                "schema_version": module.RUN_MANIFEST_SCHEMA_VERSION,
                "phase": "generate",
                "status": "completed",
                "run_id": "run",
                "benchmark": {"id": "benchmark"},
                "system_id": "system",
                "system_config_hash": "s" * 64,
                "selection": {"instance_ids": [], "limit": 1},
                "generation_contract_identity": contract,
            }
            manifest_path = root / "generate_run_manifest.json"
            manifest = {
                **expected,
                "ended_at": "2026-09-30T00:00:00+00:00",
                "prediction_file_sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
            }

            def write(value):
                manifest_path.write_text(json.dumps(value), encoding="utf-8")

            write(manifest)
            reference = module._validate_generation_upstream(
                manifest_path, predictions, expected=expected,
            )
            self.assertEqual(reference["mode"], "generated_predictions")
            self.assertEqual(len(reference["generation_manifest_sha256"]), 64)

            predictions.write_text('{"instance_id":"tampered"}\n', encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "predictions file hash"):
                module._validate_generation_upstream(manifest_path, predictions, expected=expected)
            predictions.write_text('{"instance_id":"task-1"}\n', encoding="utf-8")

            manifest_path.unlink()
            with self.assertRaisesRegex(SystemExit, "is missing"):
                module._validate_generation_upstream(manifest_path, predictions, expected=expected)
            manifest_path.write_text("{broken", encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "invalid JSON"):
                module._validate_generation_upstream(manifest_path, predictions, expected=expected)

            for key, value in (
                ("phase", "evaluate"),
                ("status", "running"),
                ("run_id", "other"),
                ("benchmark", {"id": "changed"}),
                ("system_id", "changed"),
                ("system_config_hash", "x" * 64),
                ("selection", {"instance_ids": [], "limit": 2}),
                ("generation_contract_identity", module.build_resume_identity(artifact="b", tasks="t")),
            ):
                with self.subTest(key=key):
                    write({**manifest, key: value})
                    with self.assertRaisesRegex(SystemExit, key):
                        module._validate_generation_upstream(
                            manifest_path, predictions, expected=expected,
                        )

    def test_normal_generate_manifest_to_evaluate_and_gold_path(self):
        path = ROOT / "scripts" / "run_agent_benchmark.py"
        spec = importlib.util.spec_from_file_location("run_agent_benchmark_chain_test", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            dataset = base / "tasks.json"
            rows = [
                {"instance_id": f"owner__repo-{index}", "repo": "owner/repo",
                 "base_commit": "a" * 40, "problem_statement": f"fix {index}"}
                for index in (1, 2)
            ]
            dataset.write_text(json.dumps(rows), encoding="utf-8")
            model = make_artifact(base / "model")
            output_root = base / "results"
            run_dir = output_root / "swebench_verified" / "klear_agentforge_8b" / "chain"
            run_dir.mkdir(parents=True)
            predictions_path = run_dir / "predictions.jsonl"
            module.write_predictions(predictions_path, [{
                "instance_id": "owner__repo-1",
                "model_name_or_path": "klear_agentforge_8b",
                "model_patch": "diff --git a/a b/a\n",
            }])
            adapter = module.get_agent_benchmark("swebench_verified")
            system = module.load_agent_system_spec("klear_agentforge_8b")
            artifact = module.resolve_artifact(system, model, allow_unverified_model=True)
            records = module.select_records(rows, instance_ids=(), limit=1)
            contract = module._generation_contract(
                artifact=artifact, adapter=adapter, system=system, records=records,
            )
            generation = {
                "schema_version": module.RUN_MANIFEST_SCHEMA_VERSION,
                "phase": "generate", "status": "completed", "run_id": "chain",
                "benchmark": adapter.spec.to_dict(), "system_id": system.system_id,
                "system_config_hash": system.canonical_config_hash,
                "artifact": artifact.to_dict(),
                "selection": {"instance_ids": [], "limit": 1},
                "generation_contract_identity": contract,
                "prediction_file_sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
                "ended_at": "2026-09-30T00:00:00+00:00",
            }
            generation_path = run_dir / "generate_run_manifest.json"
            generation_path.write_text(json.dumps(generation), encoding="utf-8")
            command = [
                sys.executable, str(path), "--phase", "evaluate",
                "--benchmark", "swebench_verified", "--system", "klear_agentforge_8b",
                "--artifact-path", str(model), "--output-root", str(output_root),
                "--run-id", "chain", "--repo-cache-root", str(base / "cache"),
                "--workspace-root", str(base / "work"), "--dataset-path", str(dataset),
                "--limit", "1", "--offline", "--allow-incomplete-dataset", "--dry-run",
                "--allow-unverified-model",
            ]
            subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
            evaluated = json.loads((run_dir / "evaluate_run_manifest.json").read_text())
            self.assertEqual(evaluated["evaluation_input"]["mode"], "generated_predictions")
            self.assertEqual(
                evaluated["evaluation_input"]["generation_manifest_sha256"],
                hashlib.sha256(generation_path.read_bytes()).hexdigest(),
            )

            (predictions_path).write_text("tampered\n", encoding="utf-8")
            rejected = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("predictions file hash", rejected.stderr)
            module.write_predictions(predictions_path, [{
                "instance_id": "owner__repo-1",
                "model_name_or_path": "klear_agentforge_8b",
                "model_patch": "diff --git a/a b/a\n",
            }])

            other_model = make_artifact(base / "other-model")
            (other_model / "model.safetensors").write_bytes(b"other")
            artifact_rejected = subprocess.run(
                [str(other_model) if item == str(model) else item for item in command],
                cwd=ROOT, capture_output=True, text=True,
            )
            self.assertNotEqual(artifact_rejected.returncode, 0)
            self.assertIn("generation_contract_identity", artifact_rejected.stderr)

            selection_changed = command.copy()
            selection_changed[selection_changed.index("--limit") + 1] = "2"
            selection_rejected = subprocess.run(
                selection_changed, cwd=ROOT, capture_output=True, text=True,
            )
            self.assertNotEqual(selection_rejected.returncode, 0)
            self.assertIn("selection", selection_rejected.stderr)

            gold_command = command.copy()
            gold_command[gold_command.index("chain")] = "gold-only"
            gold_command.extend(("--gold",))
            subprocess.run(gold_command, cwd=ROOT, check=True, capture_output=True, text=True)
            gold_manifest = json.loads((
                output_root / "swebench_verified" / "klear_agentforge_8b"
                / "gold-only" / "evaluate_run_manifest.json"
            ).read_text())
            self.assertEqual(gold_manifest["evaluation_input"]["mode"], "gold_evaluator_only")
    def test_single_agent_resume_and_overwrite_are_mutually_exclusive(self):
        path = ROOT / "scripts" / "run_agent_system.py"
        spec = importlib.util.spec_from_file_location("run_agent_system_cli_test", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        with self.assertRaises(SystemExit):
            module._parser().parse_args([
                "--system", "klear_agentforge_8b", "--artifact-path", "model",
                "--repo-path", "repo", "--task-file", "task.md",
                "--output-dir", "out", "--resume", "--overwrite",
            ])
        with self.assertRaisesRegex(SystemExit, "allow-dirty"):
            with patch.object(sys, "argv", [
                str(path), "--system", "klear_agentforge_8b", "--artifact-path", "model",
                "--repo-path", "repo", "--task-file", "task.md", "--output-dir", "out",
                "--resume", "--allow-dirty",
            ]):
                module.main()
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(
            SystemExit, "run_manifest.json is missing"
        ):
            with patch.object(sys, "argv", [
                str(path), "--system", "klear_agentforge_8b", "--artifact-path", "model",
                "--repo-path", "repo", "--task-file", "task.md",
                "--output-dir", str(Path(tmp) / "out"), "--resume",
            ]):
                module.main()

    def test_artifact_dense_pruned_and_granite_parser(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            dense = resolve_artifact(load_agent_system_spec("klear_agentforge_8b"), make_artifact(base / "dense"), allow_unverified_model=True)
            self.assertEqual(dense.kind, "dense")
            for method in ("magnitude", "wanda", "sparsegpt"):
                pruned = resolve_artifact(load_agent_system_spec("klear_agentforge_8b"), make_artifact(base / method, pruning={"pruner": method}))
                self.assertEqual((pruned.kind, pruned.num_hidden_layers), ("pruned", 36))
            for method in ("sleb", "tabp"):
                pruned = resolve_artifact(load_agent_system_spec("klear_agentforge_8b"), make_artifact(base / method, depth=30, pruning={"pruner": method, "condition": "ssn" if method == "tabp" else "canonical"}))
                self.assertEqual((pruned.kind, pruned.num_hidden_layers), ("pruned", 30))
            recovered_path = make_artifact(base / "recovered", pruning={"pruner": "magnitude"})
            manifest_path = recovered_path / "artifact_manifest.json"
            raw = json.loads(manifest_path.read_text())
            raw["artifact"]["kind"] = "recovered"
            raw["artifact"]["lineage"].append({
                "operation": "recovery",
                "method": "lora",
                "config_hash": "d" * 64,
                "input_artifact_hash": raw["artifact"]["content_sha256"],
                "parameters": {"merge_policy": "standard_merge"},
                "provenance": {"peft_version": "0.18.1"},
            })
            write_artifact_manifest(
                recovered_path, ModelArtifact.from_dict(raw["artifact"]),
            )
            recovered = resolve_artifact(
                load_agent_system_spec("klear_agentforge_8b"), recovered_path
            )
            self.assertEqual(recovered.kind, "recovered")
            self.assertEqual(
                recovered.canonical_artifact["lineage"][-1]["operation"], "recovery"
            )
        self.assertTrue(resolve_granite_parser(load_agent_system_spec("granite_4_2_8b")).is_file())

    def test_exact_command_and_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            system = load_agent_system_spec("klear_agentforge_8b")
            item = resolve_artifact(system, make_artifact(Path(tmp) / "model"), allow_unverified_model=True)
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
        item = resolve_artifact(
            system, make_artifact(Path(tmp) / "model"), allow_unverified_model=True
        )
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
        required = dict(system_id="s", project_model_id="m", artifact_provenance={}, task_id="t", status="patch_generated", started_at="a", ended_at="b", runtime_seconds=1, framework="f", framework_version="1", framework_revision="0"*40, effective_agent_config={}, endpoint="e", serving_manifest_path="s", trajectory_path="t", stdout_path="o", stderr_path="e", patch_path="p", patch_sha256="h", changed_files=("a",), repository_before={}, repository_after={})
        self.assertEqual(AgentResult(**required).to_dict()["changed_files"], ["a"])
        with self.assertRaisesRegex(ValueError, "patch_generated"):
            AgentResult(**{**required, "status": "success"})

    def test_three_cli_dry_runs(self):
        models = {"klear_agentforge_8b": ("qwen3", 36), "swe_lego_qwen3_8b": ("qwen3", 36), "granite_4_2_8b": ("granite", 40)}
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); repo = base / "repo"; subprocess.run(("git", "init", "-q", str(repo)), check=True)
            subprocess.run(("git", "-C", str(repo), "config", "user.email", "a@b.c"), check=True); subprocess.run(("git", "-C", str(repo), "config", "user.name", "T"), check=True)
            (repo / "a").write_text("a"); subprocess.run(("git", "-C", str(repo), "add", "."), check=True); subprocess.run(("git", "-C", str(repo), "commit", "-qm", "base"), check=True)
            task = base / "task.md"; task.write_text("change a")
            for name, (kind, depth) in models.items():
                model = make_artifact(base / name, kind, depth); output = base / (name + "-out")
                subprocess.run((sys.executable, str(ROOT / "scripts/run_agent_system.py"), "--system", name, "--artifact-path", str(model), "--repo-path", str(repo), "--task-file", str(task), "--output-dir", str(output), "--dry-run", "--allow-unverified-model"), cwd=ROOT, check=True, capture_output=True, text=True)
                manifest = json.loads((output / "run_manifest.json").read_text())
                self.assertEqual(manifest["result"]["status"], "dry_run_validated")
                self.assertNotIn("api_key", manifest["effective_serving_config"])
                self.assertEqual(manifest["model_provenance_policy"], "explicit_unverified_opt_in")
                self.assertEqual(
                    manifest["serving_provenance"]["checkpoint_identity_status"],
                    "verified_managed_local_artifact",
                )

    def test_single_external_endpoint_requires_opt_in_and_binds_normalized_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            subprocess.run(("git", "init", "-q", str(repo)), check=True)
            subprocess.run(("git", "-C", str(repo), "config", "user.email", "a@b.c"), check=True)
            subprocess.run(("git", "-C", str(repo), "config", "user.name", "T"), check=True)
            (repo / "a").write_text("a")
            subprocess.run(("git", "-C", str(repo), "add", "."), check=True)
            subprocess.run(("git", "-C", str(repo), "commit", "-qm", "base"), check=True)
            task = base / "task.md"
            task.write_text("change a")
            model = make_artifact(base / "model")
            output = base / "out"
            system = load_agent_system_spec("klear_agentforge_8b")
            item = resolve_artifact(system, model, allow_unverified_model=True)
            with self.assertRaisesRegex(ValueError, "cannot be verified"):
                VLLMServer(system, item, base / "server-rejected", endpoint="http://example.test")
            server = VLLMServer(
                system, item, base / "server-accepted", endpoint="HTTP://Example.TEST:80/",
                allow_unverified_external_endpoint=True,
            )
            self.assertEqual(server.endpoint, "http://example.test")
            self.assertEqual(
                server.manifest()["serving_provenance"]["checkpoint_identity_status"],
                "unverified_external_endpoint",
            )
            command = [
                sys.executable, str(ROOT / "scripts/run_agent_system.py"),
                "--system", "klear_agentforge_8b", "--artifact-path", str(model),
                "--repo-path", str(repo), "--task-file", str(task),
                "--output-dir", str(output), "--dry-run", "--allow-unverified-model",
                "--endpoint", "HTTP://Example.TEST:80/",
            ]
            rejected = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("--allow-unverified-external-endpoint", rejected.stderr)
            accepted = [*command, "--allow-unverified-external-endpoint"]
            subprocess.run(accepted, cwd=ROOT, check=True, capture_output=True, text=True)
            manifest = json.loads((output / "run_manifest.json").read_text())
            self.assertEqual(manifest["endpoint"], "http://example.test")
            self.assertEqual(
                manifest["serving_provenance"]["checkpoint_identity_status"],
                "unverified_external_endpoint",
            )
            self.assertTrue(
                manifest["serving_provenance"]["explicit_unverified_external_opt_in"]
            )
            changed = accepted.copy()
            changed[changed.index("--endpoint") + 1] = "http://example.test:81"
            resumed = subprocess.run(
                [*changed, "--resume"], cwd=ROOT, capture_output=True, text=True,
            )
            self.assertNotEqual(resumed.returncode, 0)
            self.assertIn("identity changed", resumed.stderr)

    def test_single_agent_resume_rejects_changed_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            subprocess.run(("git", "init", "-q", str(repo)), check=True)
            subprocess.run(("git", "-C", str(repo), "config", "user.email", "a@b.c"), check=True)
            subprocess.run(("git", "-C", str(repo), "config", "user.name", "T"), check=True)
            (repo / "a").write_text("a")
            subprocess.run(("git", "-C", str(repo), "add", "."), check=True)
            subprocess.run(("git", "-C", str(repo), "commit", "-qm", "base"), check=True)
            task = base / "task.md"
            task.write_text("change a")
            model = make_artifact(base / "model")
            output = base / "out"
            command = [
                sys.executable, str(ROOT / "scripts/run_agent_system.py"),
                "--system", "klear_agentforge_8b", "--artifact-path", str(model),
                "--repo-path", str(repo), "--task-file", str(task),
                "--output-dir", str(output), "--dry-run", "--allow-unverified-model",
            ]
            subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
            (model / "model.safetensors").write_bytes(b"different")
            resumed = subprocess.run(
                [*command, "--resume"], cwd=ROOT, check=False,
                capture_output=True, text=True,
            )
            self.assertEqual(resumed.returncode, 2)
            self.assertIn("identity changed", resumed.stderr)

    def test_batch_dry_run_granite_parser_and_resume_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            dataset = base / "tasks.json"
            dataset.write_text(json.dumps([{
                "instance_id": "owner__repo-1", "repo": "owner/repo",
                "base_commit": "a" * 40, "problem_statement": "fix it",
            }]))
            commands = {}
            for system_id, model_type, depth in (
                ("granite_4_2_8b", "granite", 40),
                ("klear_agentforge_8b", "qwen3", 36),
            ):
                model = make_artifact(base / system_id, model_type, depth)
                output_root = base / f"results-{system_id}"
                command = [
                    sys.executable, str(ROOT / "scripts/run_agent_benchmark.py"),
                    "--phase", "generate", "--benchmark", "swebench_verified",
                    "--system", system_id, "--artifact-path", str(model),
                    "--output-root", str(output_root), "--run-id", "same",
                    "--repo-cache-root", str(base / "cache"),
                    "--workspace-root", str(base / "work"),
                    "--dataset-path", str(dataset), "--offline",
                    "--allow-incomplete-dataset", "--dry-run",
                    "--allow-unverified-model",
                ]
                subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
                manifest_path = output_root / "swebench_verified" / system_id / "same" / "run_manifest.json"
                manifest = json.loads(manifest_path.read_text())
                commands[system_id] = manifest["vllm_command"]
                if system_id == "klear_agentforge_8b":
                    (model / "model.safetensors").write_bytes(b"different")
                    resumed = subprocess.run(
                        [*command, "--resume"], cwd=ROOT, check=False,
                        capture_output=True, text=True,
                    )
                    self.assertNotEqual(resumed.returncode, 0)
                    self.assertIn("identity changed", resumed.stderr)
            self.assertIn("--reasoning-parser-plugin", commands["granite_4_2_8b"])
            self.assertNotIn("--reasoning-parser-plugin", commands["klear_agentforge_8b"])

            external_model = make_artifact(base / "external-model")
            external_root = base / "external-results"
            external = [
                sys.executable, str(ROOT / "scripts/run_agent_benchmark.py"),
                "--phase", "generate", "--benchmark", "swebench_verified",
                "--system", "klear_agentforge_8b", "--artifact-path", str(external_model),
                "--output-root", str(external_root), "--run-id", "external",
                "--repo-cache-root", str(base / "cache"),
                "--workspace-root", str(base / "work"),
                "--dataset-path", str(dataset), "--offline",
                "--allow-incomplete-dataset", "--dry-run", "--allow-unverified-model",
                "--endpoint", "HTTP://Example.TEST:80/",
            ]
            rejected = subprocess.run(external, cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("--allow-unverified-external-endpoint", rejected.stderr)
            accepted = [*external, "--allow-unverified-external-endpoint"]
            subprocess.run(accepted, cwd=ROOT, check=True, capture_output=True, text=True)
            external_manifest_path = (
                external_root / "swebench_verified" / "klear_agentforge_8b"
                / "external" / "generate_run_manifest.json"
            )
            external_manifest = json.loads(external_manifest_path.read_text())
            self.assertEqual(
                external_manifest["execution_options"]["external_endpoint"],
                "http://example.test",
            )
            self.assertEqual(
                external_manifest["serving_provenance"]["checkpoint_identity_status"],
                "unverified_external_endpoint",
            )
            changed = accepted.copy()
            changed[changed.index("--endpoint") + 1] = "http://example.test:81"
            resumed = subprocess.run(
                [*changed, "--resume"], cwd=ROOT, capture_output=True, text=True,
            )
            self.assertNotEqual(resumed.returncode, 0)
            self.assertIn("identity changed", resumed.stderr)

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
                item = resolve_artifact(system, make_artifact(base / "model"), allow_unverified_model=True)
                with VLLMServer(
                    system, item, base / "server", endpoint=endpoint,
                    allow_unverified_external_endpoint=True, startup_timeout=2,
                ) as server:
                    result = get_agent_runner(system).run(RepositoryTask("real-loop", repo, "Create result.txt containing fixed"), server.endpoint, base / "out", timeout=60, artifact_provenance=item.to_dict())
                self.assertEqual(result.status, "patch_generated", (base / "out/agent.stderr.log").read_text())
                self.assertIn("result.txt", result.changed_files)
                self.assertTrue((base / "out/trajectory.json").is_file())
                self.assertEqual(OpenAIHandler.calls, [("/v1/chat/completions", "srtp-klear-agentforge-8b")] * 2)
        finally:
            httpd.shutdown(); httpd.server_close(); thread.join(timeout=2)


if __name__ == "__main__": unittest.main()
