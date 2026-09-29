"""Manifest-aware dense/same-depth/reduced artifact loading tests."""

from __future__ import annotations

import json
import os
import shutil
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

from src.models import LoadOptions, ModelSpec
from src.artifacts import (
    LineageOperation, ModelArtifact, artifact_inventory, load_model_artifact,
    resolve_model_artifact, write_artifact_manifest,
)
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


def write_pruned_artifact(path, spec, adapter, pruner, depth, structure_effect=None):
    _entries, digest = artifact_inventory(Path(path))
    operation = LineageOperation(
        operation="pruning",
        method=pruner,
        config_hash="b" * 64,
        input_artifact_hash="c" * 64,
        parameters={"post_pruning_actual_block_count": depth},
        provenance={
            "source": "test",
            "structure_effect": structure_effect or (
                "reduced_depth"
                if depth < spec.expected_num_hidden_layers
                else "weight_sparse"
            ),
        },
    )
    artifact = ModelArtifact(
        path=str(Path(path).resolve()),
        kind="pruned",
        representation="full_checkpoint",
        standalone=True,
        project_model_id=spec.project_model_id,
        architecture=spec.architecture,
        model_type=spec.model_type,
        adapter_id=adapter.adapter_id,
        num_hidden_layers=depth,
        content_sha256=digest,
        lineage=(operation,),
        capabilities={"direct_evaluation": True, "vllm_serving": True},
        metadata={},
    )
    write_artifact_manifest(Path(path), artifact)
    return artifact



class ArtifactLoaderTests(unittest.TestCase):
    def test_canonical_manifest_lineage_tamper_is_rejected(self):
        model, adapter, spec = tiny_pair("qwen")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model.save_pretrained(root)
            write_pruned_artifact(root, spec, adapter, "magnitude", 3)
            manifest_path = root / "artifact_manifest.json"
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(raw["schema_version"], 2)
            raw["artifact"]["lineage"][0]["method"] = "wanda"
            manifest_path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "provenance hash mismatch"):
                resolve_model_artifact(root, spec, adapter)

    def test_canonical_artifact_relocation_preserves_manifest_identity(self):
        model, adapter, spec = tiny_pair("qwen")
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            original = parent / "original"
            original.mkdir()
            model.save_pretrained(original)
            write_pruned_artifact(original, spec, adapter, "magnitude", 3)
            manifest_path = original / "artifact_manifest.json"
            reformatted = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_path.write_text(
                json.dumps(reformatted, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            before = resolve_model_artifact(original, spec, adapter)
            moved = parent / "moved"
            shutil.move(str(original), moved)
            after = resolve_model_artifact(moved, spec, adapter)
            self.assertEqual(after.path, str(moved.resolve()))
            self.assertEqual(
                after.manifest_provenance_sha256,
                before.manifest_provenance_sha256,
            )
            (moved / "model.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "content hash"):
                resolve_model_artifact(moved, spec, adapter)

    def test_legacy_canonical_manifest_fails_with_migration_message(self):
        model, adapter, spec = tiny_pair("qwen")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model.save_pretrained(root)
            write_pruned_artifact(root, spec, adapter, "magnitude", 3)
            manifest_path = root / "artifact_manifest.json"
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            raw["schema_version"] = 1
            manifest_path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "lack manifest provenance protection"):
                resolve_model_artifact(root, spec, adapter)

    def test_inventory_rejects_symlinks_instead_of_silently_skipping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "real.bin").write_bytes(b"x")
            try:
                os.symlink(root / "real.bin", root / "linked.bin")
            except OSError as error:
                self.skipTest(f"symlink creation unavailable: {error}")
            with self.assertRaisesRegex(ValueError, "must not contain symlinks"):
                artifact_inventory(root)

    def test_dense_verified_provenance_is_an_execution_boundary(self):
        model, adapter, spec = tiny_pair("qwen")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model.save_pretrained(root)
            with self.assertRaisesRegex(ValueError, "lacks verified provenance"):
                resolve_model_artifact(
                    root, spec, adapter, require_verified_dense=True,
                )
            unverified = resolve_model_artifact(root, spec, adapter)
            self.assertEqual(unverified.metadata["manifest_status"], "raw_dense_checkpoint")
            entries, digest = artifact_inventory(root)
            runtime_files = [
                {"path": item["path"], "size": item["size"],
                 "hash_type": "sha256", "expected_hash": item["sha256"]}
                for item in entries
            ]
            sidecar = {
                "schema_version": 3,
                "verification_status": "verified",
                "project_model_id": spec.project_model_id,
                "canonical_hf_repo": spec.huggingface_repo_id,
                "canonical_hf_revision": spec.revision,
                "runtime_snapshot_manifest_sha256": "m" * 64,
                "verified_runtime_files": runtime_files,
                "verified_artifact_content_sha256": digest,
            }
            (root / ".srtp_model_source.json").write_text(json.dumps(sidecar))
            snapshot = {
                "canonical_hf_repo": spec.huggingface_repo_id,
                "canonical_hf_revision": spec.revision,
                "required_runtime_files": runtime_files,
            }
            legacy_sidecar = dict(sidecar, schema_version=2)
            (root / ".srtp_model_source.json").write_text(json.dumps(legacy_sidecar))
            with patch("src.artifacts.validation.load_snapshot_manifest", return_value=snapshot), patch(
                "src.artifacts.validation.snapshot_manifest_sha256", return_value="m" * 64
            ), patch("src.artifacts.validation.artifact_inventory") as inventory_mock, self.assertRaisesRegex(
                ValueError, "schema_version"
            ):
                resolve_model_artifact(root, spec, adapter, require_verified_dense=True)
            inventory_mock.assert_not_called()
            (root / ".srtp_model_source.json").write_text(json.dumps(sidecar))
            with patch("src.artifacts.validation.load_snapshot_manifest", return_value=snapshot), patch(
                "src.artifacts.validation.snapshot_manifest_sha256", return_value="m" * 64
            ):
                artifact = resolve_model_artifact(
                    root, spec, adapter, require_verified_dense=True,
                )
            self.assertEqual(artifact.metadata["manifest_status"], "verified_dense_snapshot")
            (root / "tamper.bin").write_bytes(b"changed")
            with patch("src.artifacts.validation.load_snapshot_manifest", return_value=snapshot), patch(
                "src.artifacts.validation.snapshot_manifest_sha256", return_value="m" * 64
            ), self.assertRaisesRegex(ValueError, "verified provenance mismatch"):
                resolve_model_artifact(root, spec, adapter, require_verified_dense=True)

    def _load(self, spec, adapter, path, kind="pruned"):
        with patch(
            "src.models.loader._runtime_imports",
            return_value=(torch, AutoConfig, AutoModelForCausalLM, FakeAutoTokenizer),
        ), patch("transformers.AutoTokenizer.from_pretrained", return_value=FakeAutoTokenizer()):
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
                write_pruned_artifact(directory, spec, adapter, pruner, 3)
                loaded = self._load(spec, adapter, Path(directory))
                self.assertEqual(loaded.structure["actual_block_count"], 3)

    def test_reduced_qwen_and_granite_load_forward_and_generate(self):
        for pruner in ("sleb", "tabp"):
            for kind in ("qwen", "granite"):
                with self.subTest(pruner=pruner, kind=kind):
                    model, adapter, spec = tiny_pair(kind)
                    metadata = adapter.capture_block_removal_metadata(model)
                    blocks = list(adapter.get_blocks(model))
                    adapter.replace_blocks(model, [blocks[0], blocks[2]])
                    adapter.finalize_block_removal(model, (0, 2), metadata)
                    with tempfile.TemporaryDirectory() as directory:
                        model.save_pretrained(directory)
                        write_pruned_artifact(directory, spec, adapter, pruner, 2)
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
            with self.assertRaisesRegex(ValueError, "does not match"):
                self._load(spec, adapter, Path(directory))

    def test_corrupt_canonical_manifest_is_rejected(self):
        model, adapter, spec = tiny_pair("qwen")
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            Path(directory, "artifact_manifest.json").write_text("{broken", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid artifact manifest JSON"):
                self._load(spec, adapter, Path(directory))

    def test_structure_effect_is_method_agnostic(self):
        operation = LineageOperation(
            "pruning",
            "future_pruner",
            "b" * 64,
            "c" * 64,
            provenance={"structure_effect": "weight_sparse"},
        )
        artifact = ModelArtifact(
            "/tmp/future", "pruned", "full_checkpoint", True,
            "model", "Qwen3", "qwen3", "qwen3", 3, "a" * 64,
            (operation,),
        )
        self.assertTrue(artifact.is_weight_sparse)
        self.assertEqual(artifact.pruning_structure_effect, "weight_sparse")

    def test_artifact_kind_requires_matching_lineage(self):
        with self.assertRaisesRegex(ValueError, "Pruned artifacts require"):
            ModelArtifact(
                "/tmp/pruned", "pruned", "full_checkpoint", True,
                "model", "Qwen3", "qwen3", "qwen3", 3, "a" * 64,
            )
        pruning = LineageOperation("pruning", "magnitude", "b" * 64, "c" * 64, provenance={"structure_effect": "weight_sparse"})
        with self.assertRaisesRegex(ValueError, "Recovered artifacts require"):
            ModelArtifact(
                "/tmp/recovered", "recovered", "full_checkpoint", True,
                "model", "Qwen3", "qwen3", "qwen3", 3, "a" * 64,
                (pruning,),
            )

    def test_wrong_identity_is_rejected_before_weight_load(self):
        model, adapter, spec = tiny_pair("qwen")
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            write_pruned_artifact(directory, spec, adapter, "magnitude", 3)
            manifest_path = Path(directory, "artifact_manifest.json")
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            raw["artifact"]["project_model_id"] = "wrong"
            manifest_path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identity"):
                self._load(spec, adapter, Path(directory))

    def test_wrong_architecture_is_rejected_before_weight_load(self):
        model, adapter, spec = tiny_pair("qwen")
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            write_pruned_artifact(directory, spec, adapter, "magnitude", 3)
            manifest_path = Path(directory, "artifact_manifest.json")
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            raw["artifact"]["architecture"] = "Granite"
            manifest_path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identity"):
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
                write_pruned_artifact(
                    directory,
                    spec,
                    adapter,
                    pruner,
                    recorded_depth,
                    structure_effect=("weight_sparse" if pruner == "magnitude" else "reduced_depth"),
                )
                with self.assertRaisesRegex(ValueError, "depth"):
                    self._load(spec, adapter, Path(directory))


if __name__ == "__main__":
    unittest.main()
