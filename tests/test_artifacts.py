"""Manifest-aware dense/same-depth/reduced artifact loading tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    GraniteConfig,
    GraniteForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from src.models import LoadOptions, ModelSpec, load_model_artifact
from src.models.adapters.granite import GraniteAdapter
from src.models.adapters.qwen3 import Qwen3Adapter


class FakeAutoTokenizer:
    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        return cls()


def tiny_pair(kind):
    if kind == "qwen":
        config = Qwen3Config(
            vocab_size=16, hidden_size=8, intermediate_size=16,
            num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1,
            head_dim=4, max_position_embeddings=16,
        )
        model = Qwen3ForCausalLM(config).eval()
        adapter = Qwen3Adapter()
        architecture, model_type = "Qwen3", "qwen3"
    else:
        config = GraniteConfig(
            vocab_size=16, hidden_size=8, intermediate_size=16,
            num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1,
            max_position_embeddings=16,
        )
        model = GraniteForCausalLM(config).eval()
        adapter = GraniteAdapter()
        architecture, model_type = "Granite", "granite"
    spec = ModelSpec(
        project_model_id=f"tiny_{kind}", display_name=f"Tiny {kind}",
        huggingface_repo_id=f"example/{kind}", architecture=architecture,
        adapter=adapter.adapter_id, model_type=model_type,
        expected_model_class=type(model).__name__, revision="a" * 40,
        default_dtype="float32", local_path=None, trust_remote_code=False,
        expected_num_hidden_layers=3, expected_hidden_size=8,
        expected_intermediate_size=16, expected_num_attention_heads=2,
        expected_num_key_value_heads=1,
    )
    return model, adapter, spec


def manifest(spec, adapter, pruner, depth):
    return {
        "schema_version": 3,
        "model": {
            "project_model_id": spec.project_model_id,
            "architecture": spec.architecture,
            "model_type": spec.model_type,
            "adapter": adapter.adapter_id,
            "requested_revision": spec.revision,
            "resolved_revision": spec.revision,
            "hidden_size": spec.expected_hidden_size,
            "intermediate_size": spec.expected_intermediate_size,
            "num_attention_heads": spec.expected_num_attention_heads,
            "num_key_value_heads": spec.expected_num_key_value_heads,
        },
        "pruning": {
            "pruner": pruner,
            "implementation_version": "1.0",
            "post_pruning_actual_block_count": depth,
        },
        "checkpoint": {"format": "huggingface_save_pretrained"},
    }


class ArtifactLoaderTests(unittest.TestCase):
    def _load(self, spec, adapter, path, kind="pruned"):
        with patch(
            "src.models.artifacts._runtime_imports",
            return_value=(torch, AutoConfig, AutoModelForCausalLM, FakeAutoTokenizer),
        ):
            return load_model_artifact(
                spec, adapter, path, kind=kind, options=LoadOptions(dtype="float32")
            )

    def test_dense_tiny_hf_artifact_uses_strict_dense_validation(self):
        model, adapter, spec = tiny_pair("qwen")
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            with patch("src.models.loader._runtime_imports", return_value=(
                torch, AutoConfig, AutoModelForCausalLM, FakeAutoTokenizer
            )):
                loaded = load_model_artifact(spec, adapter, Path(directory), kind="dense",
                    options=LoadOptions(dtype="float32"))
        self.assertEqual(loaded.structure["actual_block_count"], 3)

    def test_same_depth_pruned_artifacts_for_all_unstructured_methods(self):
        for pruner in ("magnitude", "wanda", "sparsegpt"):
            model, adapter, spec = tiny_pair("qwen")
            with tempfile.TemporaryDirectory() as directory:
                model.save_pretrained(directory)
                Path(directory, "pruning_manifest.json").write_text(
                    json.dumps(manifest(spec, adapter, pruner, 3)), encoding="utf-8"
                )
                loaded = self._load(spec, adapter, Path(directory))
                self.assertEqual(loaded.structure["actual_block_count"], 3)

    def test_sleb_reduced_qwen_and_granite_load_forward_and_generate(self):
        for kind in ("qwen", "granite"):
            model, adapter, spec = tiny_pair(kind)
            metadata = adapter.capture_block_removal_metadata(model)
            blocks = list(adapter.get_blocks(model))
            adapter.replace_blocks(model, [blocks[0], blocks[2]])
            adapter.finalize_block_removal(model, (0, 2), metadata)
            with tempfile.TemporaryDirectory() as directory:
                model.save_pretrained(directory)
                Path(directory, "pruning_manifest.json").write_text(
                    json.dumps(manifest(spec, adapter, "sleb", 2)), encoding="utf-8"
                )
                loaded = self._load(spec, adapter, Path(directory))
                ids = torch.tensor([[1, 2, 3]])
                with torch.inference_mode():
                    logits = loaded.model(input_ids=ids, use_cache=False).logits
                    generated = loaded.model.generate(
                        input_ids=ids, max_new_tokens=1, do_sample=False,
                        pad_token_id=0, use_cache=True,
                    )
            self.assertEqual(loaded.structure["actual_block_count"], 2)
            self.assertEqual(logits.shape[:2], ids.shape)
            self.assertEqual(generated.shape, (1, 4))

    def test_missing_manifest_is_rejected(self):
        model, adapter, spec = tiny_pair("qwen")
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            with self.assertRaisesRegex(ValueError, "requires"):
                self._load(spec, adapter, Path(directory))

    def test_wrong_identity_is_rejected_before_weight_load(self):
        model, adapter, spec = tiny_pair("qwen")
        bad = manifest(spec, adapter, "magnitude", 3)
        bad["model"]["project_model_id"] = "wrong"
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            Path(directory, "pruning_manifest.json").write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "project_model_id"):
                self._load(spec, adapter, Path(directory))

    def test_wrong_architecture_is_rejected_before_weight_load(self):
        model, adapter, spec = tiny_pair("qwen")
        bad = manifest(spec, adapter, "magnitude", 3)
        bad["model"]["architecture"] = "Granite"
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            Path(directory, "pruning_manifest.json").write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "architecture"):
                self._load(spec, adapter, Path(directory))

    def test_unstructured_reduced_and_sleb_depth_mismatch_are_rejected(self):
        for pruner, recorded_depth in (("magnitude", 2), ("sleb", 1)):
            model, adapter, spec = tiny_pair("qwen")
            metadata = adapter.capture_block_removal_metadata(model)
            blocks = list(adapter.get_blocks(model))
            adapter.replace_blocks(model, blocks[:2])
            adapter.finalize_block_removal(model, (0, 1), metadata)
            with tempfile.TemporaryDirectory() as directory:
                model.save_pretrained(directory)
                Path(directory, "pruning_manifest.json").write_text(
                    json.dumps(manifest(spec, adapter, pruner, recorded_depth)), encoding="utf-8"
                )
                with self.assertRaisesRegex(ValueError, "depth"):
                    self._load(spec, adapter, Path(directory))


if __name__ == "__main__":
    unittest.main()
