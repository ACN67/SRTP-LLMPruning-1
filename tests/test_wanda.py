"""Calibration, native replay, Wanda algorithm, integration, and CLI tests."""

from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from transformers import GraniteConfig, GraniteForCausalLM, Qwen3Config, Qwen3ForCausalLM

from src.models.adapters.granite import GraniteAdapter
from src.models.adapters.qwen3 import Qwen3Adapter
from src.models.replay import capture_native_calibration, replay_captured_block
from src.pruning import (
    C4CalibrationProvider,
    CalibrationConfig,
    CalibrationSample,
    PruningRequest,
    WandaActivationStats,
    WandaPruner,
    WandaPruningContext,
    wanda_mask,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def qwen_tiny():
    return Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=64,
            use_cache=False,
        )
    )


def granite_tiny():
    return GraniteForCausalLM(
        GraniteConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            use_cache=False,
        )
    )


def injected_context(input_ids):
    samples = tuple(
        CalibrationSample(ids, torch.ones_like(ids)) for ids in input_ids
    )
    return WandaPruningContext(
        CalibrationConfig(
            source="injected",
            samples=len(samples),
            sequence_length=input_ids[0].shape[1],
            seed=0,
        ),
        samples,
    )


class CalibrationProviderTests(unittest.TestCase):
    class FakeTokenizer:
        def __call__(self, text, return_tensors):
            length = int(text.split(":")[1])
            return {"input_ids": torch.arange(length).reshape(1, -1)}

    def test_c4_sampling_is_seeded_fixed_length_contiguous_and_recorded(self) -> None:
        dataset = [{"text": "tokens:3"}, {"text": "tokens:20"}, {"text": "tokens:30"}]
        config = CalibrationConfig(source="c4", samples=4, sequence_length=8, seed=7)
        first = C4CalibrationProvider(lambda: dataset).prepare(self.FakeTokenizer(), config)
        second = C4CalibrationProvider(lambda: dataset).prepare(self.FakeTokenizer(), config)
        self.assertEqual(config.to_dict(), {
            "source": "c4", "samples": 4, "sequence_length": 8, "seed": 7
        })
        self.assertEqual(len(first), 4)
        for left, right in zip(first, second):
            self.assertTrue(torch.equal(left.input_ids, right.input_ids))
            self.assertEqual(tuple(left.input_ids.shape), (1, 8))
            self.assertTrue(torch.all(torch.diff(left.input_ids[0]) == 1))
            self.assertTrue(torch.equal(left.attention_mask, torch.ones_like(left.input_ids)))


class WandaAlgorithmTests(unittest.TestCase):
    def test_activation_statistic_matches_direct_l2_reference(self) -> None:
        activations = [
            torch.tensor([[[3.0, 4.0, 0.0], [0.0, 0.0, 5.0]]]),
            torch.tensor([[[4.0, 3.0, 12.0]]]),
        ]
        stats = WandaActivationStats(3, torch.device("cpu"))
        for value in activations:
            stats.add(value)
        direct = torch.cat(activations, dim=1).reshape(-1, 3).square().sum(0).sqrt()
        self.assertTrue(torch.equal(stats.input_l2(), direct))

    def test_mask_matches_official_wrappedgpt_reference(self) -> None:
        torch.manual_seed(23)
        activations = [torch.randn(1, 6, 5) for _ in range(4)]
        stats = WandaActivationStats(5, torch.device("cpu"))
        scaler_row = torch.zeros(5)
        nsamples = 0

        for inputs in activations:
            stats.add(inputs)
            inp = inputs
            tmp = inp.shape[0]
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
            scaler_row *= nsamples / (nsamples + tmp)
            nsamples += tmp
            scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / nsamples

        project_l2 = stats.input_l2()
        reference_l2 = scaler_row.sqrt()
        weight = torch.randn(4, 5)
        self.assertTrue(
            torch.equal(
                wanda_mask(weight, project_l2, 0.4),
                wanda_mask(weight, reference_l2, 0.4),
            )
        )
        ratio = project_l2 / reference_l2
        self.assertTrue(torch.allclose(ratio, torch.full_like(ratio, ratio[0])))

    def test_wanda_differs_from_magnitude_and_matches_manual_score(self) -> None:
        weight = torch.tensor([[0.1, 0.2, 1.0, 2.0], [0.1, 0.3, 1.5, 3.0]])
        input_l2 = torch.tensor([100.0, 1.0, 1.0, 1.0])
        mask = wanda_mask(weight, input_l2, 0.25)
        manual_score = weight.abs() * input_l2.reshape(1, -1)
        expected = torch.zeros_like(weight, dtype=torch.bool)
        expected.scatter_(1, torch.argsort(manual_score, dim=1, stable=True)[:, :1], True)
        magnitude = torch.zeros_like(weight, dtype=torch.bool)
        magnitude.scatter_(1, torch.argsort(weight.abs(), dim=1, stable=True)[:, :1], True)
        self.assertTrue(torch.equal(mask, expected))
        self.assertFalse(torch.equal(mask, magnitude))
        self.assertTrue(torch.all(mask[:, 1]))

    def test_per_row_floor_exact_k_and_tie_stability(self) -> None:
        weight = torch.ones(3, 4)
        mask = wanda_mask(weight, torch.ones(4), 0.5)
        self.assertEqual(mask.sum(dim=1).tolist(), [2, 2, 2])
        self.assertTrue(torch.all(mask[:, :2]))
        self.assertFalse(bool(mask[:, 2:].any()))
        floor_mask = wanda_mask(torch.ones(2, 5), torch.ones(5), 0.3)
        self.assertEqual(floor_mask.sum(dim=1).tolist(), [1, 1])


class NativeReplayEquivalenceTests(unittest.TestCase):
    def test_qwen3_all_block_native_replay_equivalence(self) -> None:
        self._assert_equivalent(qwen_tiny(), Qwen3Adapter())

    def test_granite_all_block_native_replay_equivalence(self) -> None:
        self._assert_equivalent(granite_tiny(), GraniteAdapter())

    def _assert_equivalent(self, model, adapter) -> None:
        torch.manual_seed(5)
        model.eval()
        sample = CalibrationSample(
            torch.randint(0, model.config.vocab_size, (1, 8)),
            torch.ones(1, 8, dtype=torch.long),
        )
        captured = capture_native_calibration(
            model, adapter, [sample], capture_outputs=True
        ).samples[0]
        blocks = adapter.get_blocks(model)
        self.assertEqual(len(blocks), len(captured.native_block_outputs))
        for block_index, block in enumerate(blocks):
            self.assertTrue(
                {
                    "attention_mask",
                    "position_ids",
                    "past_key_values",
                    "use_cache",
                    "cache_position",
                    "position_embeddings",
                }.issubset(captured.block_contexts[block_index].keyword_args)
            )
            block_input = (
                captured.first_block_input
                if block_index == 0
                else captured.native_block_outputs[block_index - 1]
            )
            with torch.no_grad():
                replayed = replay_captured_block(
                    adapter,
                    block,
                    block_input,
                    captured.block_contexts[block_index],
                )
            self.assertTrue(
                torch.allclose(
                    replayed,
                    captured.native_block_outputs[block_index],
                    atol=1e-6,
                    rtol=1e-5,
                ),
                f"native replay mismatch at block {block_index}",
            )


class SequentialBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2, bias=True)

    def forward(self, hidden_states):
        return self.proj(hidden_states)


class SequentialBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(4, 2)
        self.layers = nn.ModuleList([SequentialBlock(), SequentialBlock()])
        self.norm = nn.Identity()

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        hidden = self.embed_tokens(input_ids)
        for block in self.layers:
            hidden = block(hidden)
        return hidden


class SequentialModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SequentialBackbone()
        self.lm_head = nn.Linear(2, 4, bias=False)


class SequentialPropagationTests(unittest.TestCase):
    def test_next_block_statistics_receive_pruned_previous_output(self) -> None:
        model = SequentialModel().eval()
        with torch.no_grad():
            model.model.embed_tokens.weight.fill_(1.0)
            model.model.layers[0].proj.weight.copy_(
                torch.tensor([[0.1, 10.0], [0.2, 20.0]])
            )
            model.model.layers[1].proj.weight.copy_(
                torch.tensor([[1.0, 2.0], [3.0, 4.0]])
            )
            for block in model.model.layers:
                block.proj.bias.zero_()
        ids = torch.tensor([[0]])
        embedding_before = model.model.embed_tokens.weight.detach().clone()
        lm_head_before = model.lm_head.weight.detach().clone()
        biases_before = [block.proj.bias.detach().clone() for block in model.model.layers]
        embedded = model.model.embed_tokens(ids)
        dense_block0 = model.model.layers[0](embedded).detach().clone()
        seen = []
        handle = model.model.layers[1].proj.register_forward_pre_hook(
            lambda _module, inputs: seen.append(inputs[0].detach().clone())
        )
        try:
            summary = WandaPruner().prune(
                model,
                Qwen3Adapter(),
                PruningRequest("toy", "wanda", 0.5),
                injected_context([ids]),
            )
        finally:
            handle.remove()
        pruned_block0 = model.model.layers[0](embedded).detach()
        self.assertGreaterEqual(len(seen), 3)
        self.assertFalse(torch.equal(dense_block0, pruned_block0))
        self.assertTrue(torch.equal(seen[1], pruned_block0))
        self.assertEqual(summary.requested_mask_count, 4)
        self.assertTrue(torch.equal(model.model.embed_tokens.weight, embedding_before))
        self.assertTrue(torch.equal(model.lm_head.weight, lm_head_before))
        for block, expected_bias in zip(model.model.layers, biases_before):
            self.assertTrue(torch.equal(block.proj.bias, expected_bias))

    def test_preexisting_zero_accounting_is_separate_from_requested_mask(self) -> None:
        model = SequentialModel().eval()
        with torch.no_grad():
            model.model.embed_tokens.weight.fill_(1.0)
            for block in model.model.layers:
                block.proj.weight.copy_(torch.tensor([[0.0, 2.0], [3.0, 4.0]]))
                block.proj.bias.zero_()
        summary = WandaPruner().prune(
            model,
            Qwen3Adapter(),
            PruningRequest("toy", "wanda", 0.5),
            injected_context([torch.tensor([[0]])]),
        )
        self.assertEqual(summary.targeted_weights, 8)
        self.assertEqual(summary.requested_mask_count, 4)
        self.assertEqual(summary.preexisting_zeros, 2)
        self.assertEqual(summary.newly_zeroed_weights, 2)
        self.assertEqual(summary.post_pruning_zeros, 4)
        self.assertEqual(summary.achieved_mask_sparsity, 0.5)
        self.assertEqual(summary.achieved_zero_sparsity, 0.5)


class TinyWandaIntegrationTests(unittest.TestCase):
    def test_tiny_qwen3_wanda_forward_save_reload(self) -> None:
        self._assert_wanda_round_trip(qwen_tiny(), Qwen3ForCausalLM, Qwen3Adapter())

    def test_tiny_granite_wanda_forward_save_reload(self) -> None:
        self._assert_wanda_round_trip(
            granite_tiny(), GraniteForCausalLM, GraniteAdapter()
        )

    def _assert_wanda_round_trip(self, model, model_class, adapter) -> None:
        torch.manual_seed(11)
        model.eval()
        ids = [torch.randint(0, model.config.vocab_size, (1, 8)) for _ in range(2)]
        embedding_before = adapter.get_embedding(model).weight.detach().clone()
        final_norm_before = {
            name: value.detach().clone()
            for name, value in adapter.get_final_norm(model).state_dict().items()
        }
        lm_head_before = adapter.get_lm_head(model).weight.detach().clone()
        target_weights_before = {
            f"blocks.{block_index}.{name}": module.weight.detach().clone()
            for block_index, block in enumerate(adapter.get_blocks(model))
            for name, module in adapter.get_linear_modules(block).items()
        }
        target_state_names = {
            f"model.layers.{block_index}.{name}.weight"
            for block_index, block in enumerate(adapter.get_blocks(model))
            for name in adapter.get_linear_modules(block)
        }
        excluded_state_before = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
            if name not in target_state_names
        }
        summary = WandaPruner().prune(
            model,
            adapter,
            PruningRequest("tiny", "wanda", 0.25),
            injected_context(ids),
        )
        self.assertEqual(summary.scope, "per_output_row")
        self.assertTrue(summary.sequential_layerwise)
        self.assertGreater(summary.number_of_target_modules, 0)
        self.assertTrue(
            all(
                item.requested_mask_count
                == item.shape[0] * int(item.shape[1] * 0.25)
                for item in summary.per_module
            )
        )
        names = {item.module for item in summary.per_module}
        self.assertTrue(any("self_attn" in name for name in names))
        self.assertTrue(any("mlp" in name for name in names))
        masks = {
            item.module: (
                adapter.get_linear_modules(
                    adapter.get_blocks(model)[int(item.module.split(".")[1])]
                )[item.module.split(".", 2)[2]].weight
                == 0
            ).cpu().clone()
            for item in summary.per_module
        }
        self.assertTrue(torch.equal(adapter.get_embedding(model).weight, embedding_before))
        self.assertEqual(
            adapter.get_final_norm(model).state_dict().keys(), final_norm_before.keys()
        )
        for name, value in adapter.get_final_norm(model).state_dict().items():
            self.assertTrue(torch.equal(value, final_norm_before[name]))
        self.assertTrue(torch.equal(adapter.get_lm_head(model).weight, lm_head_before))
        for name, expected in excluded_state_before.items():
            self.assertTrue(torch.equal(model.state_dict()[name], expected), name)
        for name, mask in masks.items():
            parts = name.split(".", 2)
            weight = adapter.get_linear_modules(
                adapter.get_blocks(model)[int(parts[1])]
            )[parts[2]].weight.detach().cpu()
            before = target_weights_before[name].cpu()
            self.assertTrue(torch.equal(weight[~mask], before[~mask]), name)
        with torch.inference_mode():
            logits = model(input_ids=ids[0], use_cache=False).logits
        self.assertEqual(tuple(logits.shape), (1, 8, model.config.vocab_size))
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            reloaded = model_class.from_pretrained(directory).eval()
            blocks = adapter.get_blocks(reloaded)
            for name, expected in masks.items():
                parts = name.split(".", 2)
                actual = adapter.get_linear_modules(blocks[int(parts[1])])[
                    parts[2]
                ].weight == 0
                self.assertTrue(torch.equal(actual.cpu(), expected), name)
            with torch.inference_mode():
                reloaded_logits = reloaded(input_ids=ids[0], use_cache=False).logits
            self.assertEqual(reloaded_logits.shape, logits.shape)


class WandaCliTests(unittest.TestCase):
    @staticmethod
    def _load_script_module(name):
        script_path = REPOSITORY_ROOT / "scripts" / "run_experiment.py"
        spec = importlib.util.spec_from_file_location(name, script_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    def test_wanda_planned_manifest_records_canonical_calibration(self) -> None:
        module = self._load_script_module("run_experiment_wanda_plan")
        args = module._parser().parse_args([
            "--model", "klear_agentforge_8b",
            "--pruner", "wanda",
            "--sparsity", "0.2",
        ])
        manifest = module.build_manifest(args)
        self.assertEqual(manifest["pruning"]["scope"], "per_output_row")
        self.assertEqual(manifest["pruning"]["calibration"], {
            "source": "c4", "samples": 128, "sequence_length": 2048, "seed": 0
        })

    def test_wanda_invalid_output_is_rejected_before_loading_or_calibration(self) -> None:
        module = self._load_script_module("run_experiment_wanda_unsafe")
        args = module._parser().parse_args([
            "--model", "klear_agentforge_8b",
            "--pruner", "wanda",
            "--sparsity", "0.2",
            "--execute",
            "--output-dir", str(REPOSITORY_ROOT / "unsafe-wanda"),
        ])
        with patch.object(
            module, "load_dense_model", side_effect=AssertionError("must not load")
        ), patch.object(
            module, "get_calibration_provider", side_effect=AssertionError("must not calibrate")
        ):
            with self.assertRaisesRegex(ValueError, "outside the Git repository"):
                module.execute_experiment(args)

    def test_wanda_execute_mock_resolves_calibration_and_writes_manifest(self) -> None:
        module = self._load_script_module("run_experiment_wanda")
        model = qwen_tiny().eval()
        tokenizer = SimpleNamespace(
            save_pretrained=lambda path: Path(path, "tokenizer.ok").touch()
        )
        loaded = SimpleNamespace(model=model, tokenizer=tokenizer)
        sample = CalibrationSample(torch.tensor([[1, 2, 3, 4]]), torch.ones(1, 4, dtype=torch.long))
        provider = SimpleNamespace(prepare=lambda _tokenizer, _config: (sample,))
        args = argparse.Namespace(
            model="klear_agentforge_8b", pruner="wanda", sparsity=0.25,
            benchmark=None, output_dir=None, local_path=None, cache_dir=None,
            dtype=None, device=None, device_map=None, local_files_only=False,
            overwrite_output_dir=False, calibration_source="c4",
            calibration_samples=1, calibration_seqlen=4, calibration_seed=9,
        )
        fake_model_manifest = {
            "schema_version": 2, "timestamp": "2026-01-01T00:00:00+00:00",
            "mode": "wanda_pruning", "status": "completed",
            "project_model_id": args.model,
        }
        with tempfile.TemporaryDirectory() as directory:
            args.output_dir = Path(directory)
            with patch.object(module, "load_dense_model", return_value=loaded), patch.object(
                module, "get_calibration_provider", return_value=provider
            ), patch.object(
                module, "build_model_manifest", return_value=fake_model_manifest
            ), patch.object(
                module,
                "_load_sleb_calibration_tokenizer",
                side_effect=AssertionError("Wanda must not load SLEB tokenizer"),
            ):
                manifest = module.execute_experiment(args)
            self.assertEqual(manifest["pruning"]["pruner"], "wanda")
            self.assertEqual(manifest["pruning"]["calibration_sample_count"], 1)
            self.assertEqual(manifest["pruning"]["calibration_sequence_length"], 4)
            persisted = json.loads(Path(directory, "pruning_manifest.json").read_text())
            self.assertTrue(persisted["pruning"]["sequential_layerwise"])


if __name__ == "__main__":
    unittest.main()
