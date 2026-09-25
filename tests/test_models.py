"""Dense model registry, adapter, loader, manifest, and CLI tests."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.models import (
    LoadOptions,
    build_model_manifest,
    get_model_adapter,
    load_dense_model,
    load_model_spec,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class FakeLinear:
    pass


class FakeOtherModule:
    pass


class FakeBlock:
    def named_modules(self):
        return (
            ("", self),
            ("self_attn.q_proj", FakeLinear()),
            ("mlp.down_proj", FakeLinear()),
            ("input_layernorm", FakeOtherModule()),
        )


def fake_model(spec):
    config = SimpleNamespace(model_type=spec.model_type, **spec.expected_structure())
    backbone = SimpleNamespace(
        layers=[FakeBlock() for _ in range(spec.expected_num_hidden_layers)],
        embed_tokens=object(),
        norm=object(),
    )
    model_type = type(spec.expected_model_class, (), {})
    model = model_type()
    model.config = config
    model.model = backbone
    model.lm_head = object()
    model.to = lambda device: model
    model.eval = lambda: model
    return model


class ModelConfigTests(unittest.TestCase):
    def test_pinned_revisions_and_expected_dimensions(self) -> None:
        expected = {
            "klear_agentforge_8b": (
                "fa3d41e92e9ce7a5b4a52a3e7439aa00521f40c9",
                36,
                12288,
            ),
            "granite_4_2_8b": (
                "f8de16cdcdbc6c779ca517604e050d82cc119e44",
                40,
                12800,
            ),
        }
        for model_id, values in expected.items():
            with self.subTest(model_id=model_id):
                spec = load_model_spec(model_id)
                self.assertRegex(spec.revision, r"^[0-9a-f]{40}$")
                self.assertEqual(
                    (
                        spec.revision,
                        spec.expected_num_hidden_layers,
                        spec.expected_intermediate_size,
                    ),
                    values,
                )
                self.assertEqual(spec.default_dtype, "bfloat16")
                self.assertFalse(spec.trust_remote_code)


class AdapterTraversalTests(unittest.TestCase):
    def test_both_adapters_expose_native_structure(self) -> None:
        for model_id in ("klear_agentforge_8b", "granite_4_2_8b"):
            with self.subTest(model_id=model_id):
                spec = load_model_spec(model_id)
                adapter = get_model_adapter(spec)
                model = fake_model(spec)
                structure = adapter.validate_loaded_model(model, spec)
                self.assertEqual(
                    structure["actual_block_count"],
                    spec.expected_num_hidden_layers,
                )
                self.assertIs(adapter.get_embedding(model), model.model.embed_tokens)
                self.assertIs(adapter.get_final_norm(model), model.model.norm)
                self.assertIs(adapter.get_lm_head(model), model.lm_head)
                linear = adapter.get_linear_modules(
                    adapter.get_blocks(model)[0],
                    linear_type=FakeLinear,
                )
                self.assertEqual(
                    set(linear),
                    {"self_attn.q_proj", "mlp.down_proj"},
                )


class LoaderTests(unittest.TestCase):
    def test_remote_loader_uses_one_pinned_revision(self) -> None:
        spec = load_model_spec("klear_agentforge_8b")
        adapter = get_model_adapter(spec)
        config = fake_model(spec).config
        config._commit_hash = spec.revision
        calls: dict[str, tuple[str, dict]] = {}

        class AutoConfig:
            @staticmethod
            def from_pretrained(source, **kwargs):
                calls["config"] = (source, kwargs)
                return config

        class AutoTokenizer:
            @staticmethod
            def from_pretrained(source, **kwargs):
                calls["tokenizer"] = (source, kwargs)
                return object()

        model = fake_model(spec)

        class AutoModel:
            @staticmethod
            def from_pretrained(source, **kwargs):
                calls["model"] = (source, kwargs)
                return model

        torch = SimpleNamespace(
            bfloat16="bfloat16",
            float16="float16",
            float32="float32",
        )
        with patch(
            "src.models.loader._runtime_imports",
            return_value=(torch, AutoConfig, AutoModel, AutoTokenizer),
        ):
            loaded = load_dense_model(spec, adapter, LoadOptions(device_map="auto"))

        self.assertEqual(loaded.resolved_revision, spec.revision)
        for component in ("config", "tokenizer", "model"):
            self.assertEqual(calls[component][0], spec.huggingface_repo_id)
            self.assertEqual(calls[component][1]["revision"], spec.revision)
            self.assertFalse(calls[component][1]["trust_remote_code"])
        self.assertEqual(calls["model"][1]["device_map"], "auto")

    def test_local_override_does_not_claim_upstream_revision(self) -> None:
        spec = load_model_spec("granite_4_2_8b")
        adapter = get_model_adapter(spec)
        config = fake_model(spec).config
        calls: list[dict] = []

        class Factory:
            @staticmethod
            def from_pretrained(source, **kwargs):
                calls.append(kwargs)
                if "dtype" in kwargs:
                    return fake_model(spec)
                if len(calls) == 1:
                    return config
                return object()

        torch = SimpleNamespace(
            bfloat16="bfloat16",
            float16="float16",
            float32="float32",
        )
        with tempfile.TemporaryDirectory() as directory, patch(
            "src.models.loader._runtime_imports",
            return_value=(torch, Factory, Factory, Factory),
        ):
            loaded = load_dense_model(
                spec,
                adapter,
                LoadOptions(local_path=Path(directory)),
            )
        self.assertIsNone(loaded.resolved_revision)
        self.assertTrue(all("revision" not in kwargs for kwargs in calls))


class ManifestAndCliTests(unittest.TestCase):
    def test_config_manifest_contains_reproducibility_fields(self) -> None:
        spec = load_model_spec("klear_agentforge_8b")
        adapter = get_model_adapter(spec)
        manifest = build_model_manifest(
            spec,
            adapter,
            repository_root=REPOSITORY_ROOT,
            mode="config_only",
            status="config_validated",
        )
        required = {
            "project_model_id",
            "display_name",
            "hf_repo",
            "requested_revision",
            "resolved_revision",
            "architecture",
            "adapter",
            "dtype",
            "device",
            "device_map",
            "torch_version",
            "transformers_version",
            "cuda_version",
            "python_version",
            "tokenizer_class",
            "model_class",
            "num_hidden_layers",
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "num_key_value_heads",
            "timestamp",
            "git_commit",
        }
        self.assertTrue(required.issubset(manifest))
        self.assertEqual(manifest["resolved_revision"], spec.revision)
        self.assertIsNone(manifest["model_class"])

    def test_config_only_cli_needs_no_heavy_runtime(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "scripts/validate_model.py",
                "--model",
                "granite_4_2_8b",
                "--config-only",
            ],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        manifest = json.loads(result.stdout)
        self.assertEqual(manifest["status"], "config_validated")
        self.assertEqual(manifest["actual_block_count"], None)
        self.assertEqual(manifest["checks"]["gpu_runtime_validation"], "pending")


if __name__ == "__main__":
    unittest.main()
