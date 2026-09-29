"""Real PEFT LoRA and recovery dataset integration tests on tiny local models."""

import json
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GraniteConfig, GraniteForCausalLM, PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from src.artifacts import LineageOperation, ModelArtifact, artifact_inventory, load_model_artifact, resolve_model_artifact, write_artifact_manifest
from src.models import LoadOptions, ModelSpec
from src.models.adapters.granite import GraniteAdapter
from src.models.adapters.qwen3 import Qwen3Adapter
from src.recovery.base import RecoveryConfig
from src.recovery.data import DatasetConfig, prepare_dataset, tokenize_dataset
from src.recovery.lora import LoRARecovery, matched_target_modules

ROOT = Path(__file__).resolve().parents[2]


def tokenizer():
    backend = Tokenizer(WordLevel({"<pad>": 0, "<eos>": 1, "<unk>": 2, "hello": 3, "world": 4, "fix": 5, "code": 6}, unk_token="<unk>")); backend.pre_tokenizer = Whitespace()
    value = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
    value.chat_template = "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}"
    return value


def tiny(kind="qwen", depth=2):
    if kind == "qwen":
        config = Qwen3Config(vocab_size=7, hidden_size=8, intermediate_size=16, num_hidden_layers=depth, num_attention_heads=2, num_key_value_heads=1, head_dim=4, max_position_embeddings=32)
        model, adapter, architecture, model_type = Qwen3ForCausalLM(config), Qwen3Adapter(), "Qwen3", "qwen3"
    else:
        config = GraniteConfig(vocab_size=7, hidden_size=8, intermediate_size=16, num_hidden_layers=depth, num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=32)
        model, adapter, architecture, model_type = GraniteForCausalLM(config), GraniteAdapter(), "Granite", "granite"
    spec = ModelSpec(f"tiny_{kind}", f"Tiny {kind}", f"local/{kind}", architecture, adapter.adapter_id, model_type, type(model).__name__, "a"*40, "float32", None, False, 2, 8, 16, 2, 1)
    return model, adapter, spec


def save_artifact(path, kind="qwen", method=None, depth=2, sparse=False):
    model, adapter, spec = tiny(kind, depth)
    if sparse:
        with torch.no_grad():
            for name, module in model.named_modules():
                if name.endswith("q_proj"): module.weight.view(-1)[::2] = 0
    model.save_pretrained(path); tokenizer().save_pretrained(path)
    _entries, digest = artifact_inventory(path)
    lineage = ()
    artifact_kind = "dense"
    if method:
        artifact_kind = "pruned"
        structure_effect = (
            "reduced_depth"
            if depth < spec.expected_num_hidden_layers
            else "weight_sparse"
        )
        lineage = (
            LineageOperation(
                "pruning",
                method,
                "b" * 64,
                "c" * 64,
                {"condition": "ssn" if method == "tabp" else "canonical"},
                {"structure_effect": structure_effect},
            ),
        )
    artifact = ModelArtifact(str(path), artifact_kind, "full_checkpoint", True, spec.project_model_id, spec.architecture, spec.model_type, adapter.adapter_id, depth, digest, lineage, {"direct_evaluation": True, "vllm_serving": True}, {})
    write_artifact_manifest(path, artifact)
    return adapter, spec, artifact


def config(merge="adapter_only", allow=False):
    return RecoveryConfig(2, 4, 0.0, ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"), "none", 1e-2, 0.0, 1, 0, 1, 1, 16, 7, "float32", False, 1, 1, merge, allow, "implementation_baseline_not_final_experiment_protocol")


def dataset(path, tok):
    path.write_text('{"text":"hello world"}\n{"text":"fix code"}\n', encoding="utf-8")
    return prepare_dataset(DatasetConfig("local_jsonl", "text", path=str(path), sample_limit=2, seed=3), tok)


class RecoveryTests(unittest.TestCase):
    def run_recovery(self, base, adapter, spec, artifact, merge="adapter_only", allow=False):
        loaded = load_model_artifact(spec, adapter, base, options=LoadOptions(dtype="float32"))
        data = dataset(base.parent / "data.jsonl", loaded.tokenizer); out = base.parent / f"out-{merge}-{allow}"
        result = LoRARecovery().recover(loaded=loaded, input_artifact=artifact, spec=spec, adapter=adapter, dataset=data, config=config(merge, allow), output_dir=out, repository_root=ROOT)
        return result, out

    def test_dense_qwen_and_granite_real_lora_adapter_reload_forward_generate(self):
        for kind in ("qwen", "granite"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp) / "base"; base.mkdir(); adapter, spec, artifact = save_artifact(base, kind)
                self.assertEqual(len(matched_target_modules(tiny(kind)[0], config().target_modules)), 14)
                result, out = self.run_recovery(base, adapter, spec, artifact)
                self.assertEqual(result.status, "success"); self.assertTrue((out / "adapter/artifact_manifest.json").is_file())
                loaded = load_model_artifact(spec, adapter, out / "adapter", options=LoadOptions(dtype="float32"))
                ids = torch.tensor([[3, 4]]); self.assertEqual(loaded.model(input_ids=ids).logits.shape[:2], ids.shape)
                self.assertEqual(loaded.model.generate(ids, max_new_tokens=1, pad_token_id=0).shape, (1, 3))

    def test_reduced_depth_standard_merge_preserves_depth_and_lineage(self):
        for kind in ("qwen", "granite"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp) / "base"; base.mkdir(); adapter, spec, artifact = save_artifact(base, kind, "tabp", depth=1)
                result, out = self.run_recovery(base, adapter, spec, artifact, "standard_merge")
                merged = resolve_model_artifact(out / "merged_model", spec, adapter)
                self.assertEqual(merged.num_hidden_layers, 1); self.assertEqual([x.operation for x in merged.lineage], ["pruning", "recovery"])
                self.assertIsNotNone(result.merged_model_path)

    def test_sparse_merge_guard_adapter_only_and_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base"; base.mkdir(); adapter, spec, artifact = save_artifact(base, "qwen", "magnitude", sparse=True)
            result, out = self.run_recovery(base, adapter, spec, artifact)
            self.assertTrue((out / "adapter/adapter_model.safetensors").is_file()); self.assertIsNone(result.merged_model_path)
            loaded = load_model_artifact(spec, adapter, base, options=LoadOptions(dtype="float32")); data = dataset(Path(tmp) / "guard.jsonl", loaded.tokenizer)
            with self.assertRaisesRegex(ValueError, "allow_sparse_merge"):
                LoRARecovery().recover(loaded=loaded, input_artifact=artifact, spec=spec, adapter=adapter, dataset=data, config=config("standard_merge"), output_dir=Path(tmp)/"rejected", repository_root=ROOT)
            result, out = self.run_recovery(base, adapter, spec, artifact, "standard_merge", True)
            manifest = json.loads((out / "recovery_manifest.json").read_text())
            self.assertTrue(manifest["merge_safety"]["explicit_sparsity_change_opt_in"]); self.assertIn("sparsity_before", manifest["merge_safety"])

    def test_adapter_reload_rejects_changed_base_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base"
            base.mkdir()
            adapter, spec, artifact = save_artifact(base, "qwen")
            result, out = self.run_recovery(base, adapter, spec, artifact)
            self.assertEqual(result.status, "success")
            (base / "artifact_manifest.json").unlink()
            (base / "tamper.txt").write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "base artifact hash mismatch"):
                load_model_artifact(
                    spec,
                    adapter,
                    out / "adapter",
                    options=LoadOptions(dtype="float32"),
                )

    def test_adapter_and_base_can_move_and_relocate_by_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            base.mkdir()
            adapter, spec, artifact = save_artifact(base, "qwen")
            result, out = self.run_recovery(base, adapter, spec, artifact)
            self.assertEqual(result.status, "success")
            moved_base = root / "moved" / "base"
            moved_adapter = root / "moved" / "adapter"
            moved_base.parent.mkdir()
            shutil.move(str(base), moved_base)
            shutil.move(str(out / "adapter"), moved_adapter)
            loaded = load_model_artifact(
                spec, adapter, moved_adapter,
                options=LoadOptions(dtype="float32"),
                base_artifact_path=moved_base,
            )
            ids = torch.tensor([[3, 4]])
            self.assertEqual(loaded.model(input_ids=ids).logits.shape[:2], ids.shape)

    def test_gradient_checkpointing_path_updates_lora_parameters(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base"
            base.mkdir()
            adapter, spec, artifact = save_artifact(base, "qwen")
            loaded = load_model_artifact(
                spec, adapter, base, options=LoadOptions(dtype="float32")
            )
            data = dataset(Path(tmp) / "data.jsonl", loaded.tokenizer)
            result = LoRARecovery().recover(
                loaded=loaded,
                input_artifact=artifact,
                spec=spec,
                adapter=adapter,
                dataset=data,
                config=replace(config(), gradient_checkpointing=True),
                output_dir=Path(tmp) / "gc-out",
                repository_root=ROOT,
            )
            self.assertEqual(result.status, "success")
            self.assertGreater(result.training_metrics["updated_parameter_count"], 0)

    def test_recovery_output_cannot_overlap_input_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base"
            base.mkdir()
            adapter, spec, artifact = save_artifact(base, "qwen")
            loaded = load_model_artifact(
                spec, adapter, base, options=LoadOptions(dtype="float32")
            )
            data = dataset(Path(tmp) / "data.jsonl", loaded.tokenizer)
            with self.assertRaisesRegex(ValueError, "must not overlap"):
                LoRARecovery().recover(
                    loaded=loaded,
                    input_artifact=artifact,
                    spec=spec,
                    adapter=adapter,
                    dataset=data,
                    config=config(),
                    output_dir=base / "nested-recovery",
                    repository_root=ROOT,
                )

    def test_project_recovery_configs_validate_as_nonfinal_baseline(self):
        import yaml

        recovery_raw = yaml.safe_load(
            (ROOT / "configs/recovery/lora.yaml").read_text(encoding="utf-8")
        )
        recovery = RecoveryConfig.from_mapping(recovery_raw)
        self.assertEqual(
            recovery.protocol_status,
            "implementation_baseline_not_final_experiment_protocol",
        )
        self.assertEqual(recovery.merge_policy, "adapter_only")
        dataset_raw = yaml.safe_load(
            (ROOT / "configs/recovery/dataset_implementation_smoke.yaml").read_text(
                encoding="utf-8"
            )
        )
        dataset_config = DatasetConfig.from_mapping(dataset_raw)
        self.assertEqual(
            dataset_config.contamination_audit_status,
            "not_run_for_implementation_smoke",
        )

    def test_recovery_output_safety(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base"
            base.mkdir()
            adapter, spec, artifact = save_artifact(base, "qwen")
            loaded = load_model_artifact(
                spec, adapter, base, options=LoadOptions(dtype="float32")
            )
            data = dataset(Path(tmp) / "safety.jsonl", loaded.tokenizer)
            with self.assertRaisesRegex(ValueError, "outside the repository"):
                LoRARecovery().recover(
                    loaded=loaded,
                    input_artifact=artifact,
                    spec=spec,
                    adapter=adapter,
                    dataset=data,
                    config=config(),
                    output_dir=ROOT / "forbidden-recovery-output",
                    repository_root=ROOT,
                )
            with self.assertRaisesRegex(ValueError, "must not overlap"):
                LoRARecovery().recover(
                    loaded=loaded,
                    input_artifact=artifact,
                    spec=spec,
                    adapter=adapter,
                    dataset=data,
                    config=config(),
                    output_dir=base / "nested-output",
                    repository_root=ROOT,
                )

    def test_dataset_schemas_sampling_truncation_and_failures(self):
        tok = tokenizer()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = [("text", [{"text":"hello world"}]), ("prompt_completion", [{"prompt":"hello", "completion":"world"}]), ("messages", [{"messages":[{"role":"user", "content":"hello"}]}])]
            for schema, rows in cases:
                path=root/f"{schema}.json"; path.write_text(json.dumps(rows)); prepared=prepare_dataset(DatasetConfig("local_json", schema, path=str(path), seed=9), tok)
                self.assertEqual(prepared.provenance["sha256"], __import__("hashlib").sha256(path.read_bytes()).hexdigest()); self.assertLessEqual(len(tokenize_dataset(prepared,tok,1)[0]["input_ids"]),1)
            many=root/"many.json"; many.write_text(json.dumps([{"text":str(i)} for i in range(10)]))
            cfg=DatasetConfig("local_json","text",path=str(many),sample_limit=3,seed=5); self.assertEqual(prepare_dataset(cfg,tok).texts,prepare_dataset(cfg,tok).texts)
            empty=root/"empty.json"; empty.write_text("[]")
            with self.assertRaisesRegex(ValueError,"empty"): prepare_dataset(DatasetConfig("local_json","text",path=str(empty)),tok)
            bad=root/"bad.json"; bad.write_text('[{"wrong":"x"}]')
            with self.assertRaises(ValueError): prepare_dataset(DatasetConfig("local_json","text",path=str(bad)),tok)


if __name__ == "__main__": unittest.main()
