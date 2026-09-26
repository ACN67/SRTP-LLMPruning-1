"""Basic configuration and registry smoke tests."""

import unittest
from pathlib import Path

import yaml

from src.models import get_model_adapter, list_model_ids, load_model_spec
from src.pruning import PRUNER_REGISTRY, PruningRequest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class ModelRegistryTests(unittest.TestCase):
    def test_expected_model_ids_are_stable(self) -> None:
        self.assertEqual(
            set(list_model_ids()),
            {"klear_agentforge_8b", "granite_4_2_8b"},
        )

    def test_full_upstream_ids_and_architectures(self) -> None:
        expected = {
            "klear_agentforge_8b": (
                "Kwai-Klear/Klear-AgentForge-8B",
                "Qwen3",
                "qwen3",
            ),
            "granite_4_2_8b": (
                "ibm-granite/granite-4.2-8b",
                "Granite",
                "granite",
            ),
        }
        for project_model_id, values in expected.items():
            with self.subTest(project_model_id=project_model_id):
                spec = load_model_spec(project_model_id)
                adapter = get_model_adapter(spec)
                self.assertEqual(
                    (spec.huggingface_repo_id, spec.architecture, adapter.adapter_id),
                    values,
                )


class PrunerRegistryTests(unittest.TestCase):
    def test_expected_pruners_are_registered(self) -> None:
        self.assertEqual(
            set(PRUNER_REGISTRY),
            {"magnitude", "wanda", "sparsegpt", "sleb"},
        )

    def test_invalid_sparsity_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PruningRequest("klear_agentforge_8b", "wanda", 1.0)

    def test_pruning_configs_match_registry(self) -> None:
        config_dir = REPOSITORY_ROOT / "configs" / "pruning"
        for method_id in PRUNER_REGISTRY:
            with self.subTest(method_id=method_id):
                with (config_dir / f"{method_id}.yaml").open(encoding="utf-8") as handle:
                    config = yaml.safe_load(handle)
                self.assertEqual(config["method"], method_id)
                expected = (
                    "implemented"
                    if method_id in {"magnitude", "wanda", "sparsegpt", "sleb"}
                    else "placeholder"
                )
                self.assertEqual(config["implementation_status"], expected)


class EvaluationConfigTests(unittest.TestCase):
    def test_expected_benchmark_configs(self) -> None:
        config_dir = REPOSITORY_ROOT / "configs" / "eval"
        expected = {"humaneval", "mbpp", "livecodebench"}
        self.assertEqual({path.stem for path in config_dir.glob("*.yaml")}, expected)
        for benchmark_id in expected:
            with self.subTest(benchmark_id=benchmark_id):
                with (config_dir / f"{benchmark_id}.yaml").open(encoding="utf-8") as handle:
                    config = yaml.safe_load(handle)
                self.assertEqual(config["benchmark"], benchmark_id)
                self.assertEqual(config["implementation_status"], "implemented")
                self.assertEqual(config["metric"], "pass@1")
                self.assertGreater(config["expected_task_count"], 0)
                self.assertNotIn("max_new_tokens", config)
                self.assertNotIn("generation_protocol", config)

if __name__ == "__main__":
    unittest.main()
