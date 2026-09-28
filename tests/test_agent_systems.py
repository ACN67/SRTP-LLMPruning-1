import json
import subprocess
import sys
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]

from src.agent_evaluation import list_agent_system_ids, load_agent_system_spec
from src.evaluation.registry import list_benchmarks
from src.models import list_model_ids


class AgentSystemRegistryTests(unittest.TestCase):
    def test_three_systems_load_and_match_model_registry(self):
        expected = {
            "granite_4_2_8b", "klear_agentforge_8b", "swe_lego_qwen3_8b"
        }
        self.assertEqual(set(list_agent_system_ids()), expected)
        self.assertTrue(expected.issubset(set(list_model_ids())))
        for system_id in expected:
            with self.subTest(system_id=system_id):
                spec = load_agent_system_spec(system_id)
                self.assertEqual(spec.system_id, system_id)
                self.assertEqual(spec.project_model_id, system_id)
                self.assertEqual(spec.implementation_status, "planned")

    def test_swe_lego_official_recipe_is_exact(self):
        spec = load_agent_system_spec("swe_lego_qwen3_8b")
        self.assertEqual(spec.agent.framework, "openhands")
        self.assertEqual(spec.agent.version, "0.53.0")
        self.assertEqual(spec.agent.agent_class, "CodeActAgent")
        self.assertEqual(spec.agent.max_iterations, 100)
        self.assertEqual(spec.agent.runs, 1)
        self.assertEqual(spec.agent.mode, "swe")
        self.assertEqual(spec.agent.worker_count, 24)
        self.assertFalse(spec.agent.use_hint_text)
        self.assertFalse(spec.agent.enable_plan_mode)
        self.assertFalse(spec.agent.add_in_context_learning_example)
        self.assertEqual(spec.agent.instruction_template_name, "swe_default.j2")
        self.assertEqual(spec.serving.backend, "vllm")
        self.assertEqual(spec.serving.max_model_len, 163840)
        self.assertEqual(spec.serving.tensor_parallel_size, 8)
        self.assertEqual(spec.serving.gpu_memory_utilization, 0.9)
        self.assertEqual(spec.serving.max_num_seqs, 24)
        self.assertEqual(spec.generation.temperature, 0.0)
        self.assertEqual(spec.generation.max_input_tokens, 147456)
        self.assertEqual(spec.generation.max_output_tokens, 16384)
        self.assertEqual(spec.score_identity.recipe_status, "official_score_recipe")

    def test_klear_unknown_score_overrides_remain_explicit(self):
        spec = load_agent_system_spec("klear_agentforge_8b")
        self.assertEqual(spec.agent.framework, "mini_swe_agent_plus")
        self.assertEqual(spec.agent.step_limit, 200)
        self.assertEqual(spec.agent.environment_cwd, "/testbed")
        self.assertEqual(spec.agent.command_timeout_seconds, 60)
        self.assertIsNone(spec.serving.max_model_len)
        self.assertIsNone(spec.generation.max_output_tokens)
        self.assertEqual(spec.agent.provenance.status, "official_agent_config")
        self.assertEqual(spec.score_identity.recipe_status, "score_recipe_undisclosed")
        self.assertTrue(spec.score_identity.unknown_fields)

    def test_granite_unknown_agent_recipe_and_official_serving(self):
        spec = load_agent_system_spec("granite_4_2_8b")
        self.assertEqual(spec.agent.framework, "openhands")
        self.assertIsNone(spec.agent.version)
        self.assertIsNone(spec.agent.max_iterations)
        self.assertIsNone(spec.agent.enable_plan_mode)
        self.assertEqual(spec.score_identity.recipe_status, "score_recipe_undisclosed")
        self.assertEqual(spec.serving.max_model_len, 131072)
        self.assertEqual(spec.serving.reasoning_parser, "granite_thinking_parser")
        self.assertEqual(spec.serving.tool_call_parser, "qwen3_coder")
        self.assertTrue(spec.serving.enable_auto_tool_choice)
        self.assertEqual(spec.generation.temperature, 1.0)
        self.assertEqual(spec.generation.top_p, 0.95)
        self.assertTrue(spec.generation.enable_thinking)

    def test_swebench_config_contains_no_system_runtime_parameters(self):
        path = ROOT / "configs" / "agent_benchmarks" / "swebench_verified.yaml"
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertEqual(raw["dataset"], "princeton-nlp/SWE-bench_Verified")
        self.assertEqual(raw["expected_task_count"], 500)
        self.assertEqual(raw["harness"]["release"], "v4.0.4")
        rendered = json.dumps(raw)
        for forbidden in (
            "max_iterations", "step_limit", "agent_class", "temperature",
            "tensor_parallel_size", "max_model_len",
        ):
            self.assertNotIn(forbidden, rendered)

    def test_direct_registry_is_unchanged(self):
        self.assertEqual(
            set(list_benchmarks()), {"humaneval", "mbpp", "livecodebench"}
        )

    def test_validation_script_is_config_only(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "validate_agent_system.py"),
             "--system", "swe_lego_qwen3_8b"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        output = json.loads(result.stdout)
        self.assertEqual(output["status"], "config_validated")
        self.assertFalse(output["execution_available"])
        self.assertFalse(output["server_started"])
        self.assertFalse(output["agent_started"])


if __name__ == "__main__":
    unittest.main()
