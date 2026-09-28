import json
import subprocess
import sys
import unittest
from dataclasses import asdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
from src.agent_runtime import list_agent_system_ids, load_agent_system_spec
from src.direct_evaluation.registry import list_benchmarks
from src.models import list_model_ids


def incomplete(value):
    if value is None:
        return True
    if isinstance(value, str):
        return value.lower() in {"unknown", "planned", "todo", "placeholder"}
    if isinstance(value, dict):
        return any(incomplete(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(incomplete(item) for item in value)
    return False


class AgentSystemRegistryTests(unittest.TestCase):
    def test_three_ready_systems_have_complete_active_configs(self):
        expected = {"granite_4_2_8b", "klear_agentforge_8b", "swe_lego_qwen3_8b"}
        self.assertEqual(set(list_agent_system_ids()), expected)
        self.assertTrue(expected.issubset(set(list_model_ids())))
        hashes = set()
        for system_id in expected:
            spec = load_agent_system_spec(system_id)
            self.assertEqual(spec.implementation_status, "ready")
            active = {"serving": asdict(spec.serving), "generation": asdict(spec.generation), "agent": asdict(spec.agent)}
            self.assertFalse(incomplete(active))
            self.assertEqual(len(spec.canonical_config_hash), 64)
            hashes.add(spec.canonical_config_hash)
        self.assertEqual(len(hashes), 3)

    def test_canonical_values(self):
        klear = load_agent_system_spec("klear_agentforge_8b")
        self.assertEqual((klear.agent.step_limit, klear.agent.cost_limit), (200, 3.0))
        self.assertEqual(klear.agent.tools, ("edit_via_str_replace",))
        swe = load_agent_system_spec("swe_lego_qwen3_8b")
        self.assertEqual((swe.agent.worker_count, swe.agent.benchmark_worker_count), (1, 24))
        self.assertEqual((swe.serving.tensor_parallel_size, swe.serving.max_num_seqs), (8, 24))
        self.assertEqual((swe.generation.max_input_tokens, swe.generation.max_output_tokens), (147456, 16384))
        granite = load_agent_system_spec("granite_4_2_8b")
        self.assertIn("granite_thinking_parser", granite.serving.extra_args)
        self.assertEqual(granite.generation.request_parameters, {"temperature": 1.0, "top_p": 0.95, "do_sample": True})
        self.assertTrue(granite.agent.native_tool_calling)

    def test_swebench_has_no_system_runtime_parameters(self):
        raw = yaml.safe_load((ROOT / "configs/agent_benchmarks/swebench_verified.yaml").read_text())
        self.assertEqual(raw["expected_task_count"], 500)
        rendered = json.dumps(raw)
        for forbidden in ("max_iterations", "step_limit", "temperature", "tensor_parallel_size"):
            self.assertNotIn(forbidden, rendered)

    def test_direct_registry_unchanged(self):
        self.assertEqual(set(list_benchmarks()), {"humaneval", "mbpp", "livecodebench"})

    def test_validation_script(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/validate_agent_system.py"), "--system", "swe_lego_qwen3_8b"],
            cwd=ROOT, check=True, capture_output=True, text=True,
        )
        output = json.loads(result.stdout)
        self.assertEqual(output["status"], "executable_config_validated")
        self.assertFalse(output["runtime_checked"])


if __name__ == "__main__":
    unittest.main()
