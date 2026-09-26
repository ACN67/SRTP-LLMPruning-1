"""SparseGPT official-core fidelity, orchestration, architecture, and CLI tests."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from transformers import GraniteConfig, GraniteForCausalLM, Qwen3Config, Qwen3ForCausalLM

from src.models.adapters.granite import GraniteAdapter
from src.models.adapters.qwen3 import Qwen3Adapter
from src.pruning import (
    CalibrationConfig,
    CalibrationContext,
    CalibrationSample,
    PruningRequest,
    SparseGPTHessian,
    SparseGPTPruner,
    sparsegpt_reconstruct,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def official_hessian_reference(activations):
    width = activations[0].shape[-1]
    H = torch.zeros(width, width, dtype=torch.float32)
    nsamples = 0
    for inputs in activations:
        inp = inputs
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        H *= nsamples / (nsamples + tmp)
        nsamples += tmp
        inp = math.sqrt(2 / nsamples) * inp.float()
        H += inp.matmul(inp.t())
    return H


def official_fasterprune_reference(
    weight, H, sparsity, *, percdamp=0.01, blocksize=128
):
    W = weight.clone().float()
    mask = torch.zeros_like(W, dtype=torch.bool)
    if sparsity == 0:
        return W.to(weight.dtype), mask, 0.0
    H = H.clone().float()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    damp = percdamp * torch.mean(torch.diag(H))
    diag = torch.arange(W.shape[1])
    H[diag, diag] += damp
    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    Hinv = torch.linalg.cholesky(H, upper=True)
    for i1 in range(0, W.shape[1], blocksize):
        i2 = min(i1 + blocksize, W.shape[1])
        count = i2 - i1
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        tmp = W1 ** 2 / torch.diag(Hinv1).reshape((1, -1)) ** 2
        thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
        mask1 = tmp <= thresh
        for i in range(count):
            w = W1[:, i]
            d = Hinv1[i, i]
            q = w.clone()
            q[mask1[:, i]] = 0
            Q1[:, i] = q
            err1 = (w - q) / d
            W1[:, i:] -= err1.unsqueeze(1).matmul(
                Hinv1[i, i:].unsqueeze(0)
            )
            Err1[:, i] = err1
        W[:, i1:i2] = Q1
        mask[:, i1:i2] = mask1
        W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
    return W.reshape(weight.shape).to(weight.dtype), mask, float(damp.item())


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


def calibration_context(input_ids):
    samples = tuple(
        CalibrationSample(ids, torch.ones_like(ids)) for ids in input_ids
    )
    return CalibrationContext(
        CalibrationConfig(
            source="injected",
            samples=len(samples),
            sequence_length=input_ids[0].shape[1],
            seed=0,
        ),
        samples,
    )


class SparseGPTCoreTests(unittest.TestCase):
    def test_hessian_accumulator_matches_complete_official_reference(self) -> None:
        torch.manual_seed(31)
        activations = [torch.randn(1, 7, 5) for _ in range(4)]
        accumulator = SparseGPTHessian(5, torch.device("cpu"))
        for inputs in activations:
            accumulator.add(inputs)
        expected = official_hessian_reference(activations)
        self.assertEqual(accumulator.H.dtype, torch.float32)
        self.assertTrue(torch.allclose(accumulator.H, expected, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(accumulator.H, accumulator.H.t()))

    def test_dead_input_channel_is_zeroed(self) -> None:
        weight = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
        )
        H = torch.diag(torch.tensor([2.0, 0.0, 3.0, 4.0]))
        result = sparsegpt_reconstruct(weight, H, 0.25, blocksize=4)
        self.assertTrue(torch.equal(result.weight[:, 1], torch.zeros(2)))

    def test_default_damping_is_one_percent_of_mean_diagonal(self) -> None:
        weight = torch.arange(1.0, 13.0).reshape(3, 4)
        H = torch.diag(torch.tensor([2.0, 4.0, 6.0, 8.0]))
        result = sparsegpt_reconstruct(weight, H, 0.25, blocksize=4)
        self.assertAlmostEqual(result.damping, 0.01 * 5.0, places=7)

    def test_threshold_ties_are_not_rewritten_as_exact_k(self) -> None:
        weight = torch.ones(2, 4)
        result = sparsegpt_reconstruct(weight, torch.eye(4), 0.5, blocksize=4)
        self.assertEqual(int(result.mask.sum().item()), 8)
        self.assertNotEqual(int(result.mask.sum().item()), int(weight.numel() * 0.5))

    def test_zero_sparsity_is_an_explicit_no_op(self) -> None:
        weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = sparsegpt_reconstruct(weight, torch.zeros(2, 2), 0.0)
        self.assertTrue(torch.equal(result.weight, weight))
        self.assertFalse(bool(result.mask.any()))

    def test_official_fasterprune_output_mask_and_reconstruction_match(self) -> None:
        torch.manual_seed(37)
        weight = torch.randn(5, 8)
        inputs = torch.randn(24, 8)
        H = 2 * inputs.t().matmul(inputs)
        expected_weight, expected_mask, expected_damp = official_fasterprune_reference(
            weight, H, 0.35, percdamp=0.01, blocksize=4
        )
        result = sparsegpt_reconstruct(
            weight, H, 0.35, percdamp=0.01, blocksize=4
        )
        self.assertTrue(torch.equal(result.mask, expected_mask))
        self.assertTrue(
            torch.allclose(result.weight, expected_weight, atol=1e-6, rtol=1e-5)
        )
        self.assertAlmostEqual(result.damping, expected_damp, places=7)
        self.assertTrue(torch.all(result.weight[result.mask] == 0))
        self.assertTrue(torch.any((~result.mask) & (result.weight != weight)))

    def test_adaptive_blocks_select_from_reconstructed_current_weights(self) -> None:
        torch.manual_seed(0)
        weight = torch.randn(4, 6)
        inputs = torch.randn(20, 6)
        H = 2 * inputs.t().matmul(inputs)
        result = sparsegpt_reconstruct(weight, H, 0.4, blocksize=2)

        damp = 0.01 * torch.diag(H).mean()
        damped = H.clone()
        diag = torch.arange(6)
        damped[diag, diag] += damp
        Hinv = torch.linalg.cholesky(
            torch.cholesky_inverse(torch.linalg.cholesky(damped)), upper=True
        )
        static_mask = torch.zeros_like(weight, dtype=torch.bool)
        for i1 in range(0, 6, 2):
            i2 = i1 + 2
            original_block = weight[:, i1:i2]
            Hinv1 = Hinv[i1:i2, i1:i2]
            score = original_block ** 2 / torch.diag(Hinv1).reshape(1, -1) ** 2
            threshold = torch.sort(score.flatten())[0][int(score.numel() * 0.4)]
            static_mask[:, i1:i2] = score <= threshold
        self.assertFalse(torch.equal(result.mask, static_mask))


class SequentialBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=True)

    def forward(self, hidden_states):
        return self.proj(hidden_states)


class SequentialBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 4)
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
        self.lm_head = nn.Linear(4, 8, bias=False)


class SparseGPTSequentialTests(unittest.TestCase):
    def test_next_block_receives_reconstructed_previous_block_output(self) -> None:
        torch.manual_seed(41)
        model = SequentialModel().eval()
        ids = torch.tensor([[0, 1, 2, 3]])
        embedded = model.model.embed_tokens(ids).detach()
        before = model.model.layers[0].proj.weight.detach().clone()
        seen = []
        handle = model.model.layers[1].proj.register_forward_pre_hook(
            lambda _module, inputs: seen.append(inputs[0].detach().clone())
        )
        try:
            SparseGPTPruner(blocksize=2).prune(
                model,
                Qwen3Adapter(),
                PruningRequest("toy", "sparsegpt", 0.4),
                calibration_context([ids]),
            )
        finally:
            handle.remove()

        reconstructed_weight = model.model.layers[0].proj.weight.detach()
        reconstructed_output = F.linear(
            embedded, reconstructed_weight, model.model.layers[0].proj.bias
        )
        zero_only_weight = before.masked_fill(reconstructed_weight == 0, 0)
        zero_only_output = F.linear(
            embedded, zero_only_weight, model.model.layers[0].proj.bias
        )
        self.assertGreaterEqual(len(seen), 3)
        self.assertTrue(torch.allclose(seen[1], reconstructed_output))
        self.assertFalse(torch.allclose(seen[1], zero_only_output))


class SparseGPTStatisticsAndErrorsTests(unittest.TestCase):
    def test_nominal_requested_and_actual_threshold_counts_are_distinct(self) -> None:
        model = SequentialModel().eval()
        model.model.layers = nn.ModuleList([model.model.layers[0]])
        ids = torch.tensor([[0, 1, 2, 3]])
        with torch.no_grad():
            model.model.embed_tokens.weight[:4].copy_(torch.tensor([
                [1.0, 0.2, 0.1, 0.3],
                [0.4, 1.0, 0.2, 0.1],
                [0.1, 0.3, 1.0, 0.2],
                [0.2, 0.1, 0.4, 1.0],
            ]))
            model.model.layers[0].proj.weight.copy_(
                torch.arange(1.0, 17.0).reshape(4, 4) / 10
            )
            model.model.layers[0].proj.bias.zero_()

        inputs = model.model.embed_tokens(ids).detach()
        weight = model.model.layers[0].proj.weight.detach().clone()
        H = official_hessian_reference([inputs])
        _, official_mask, _ = official_fasterprune_reference(
            weight, H, 0.25, blocksize=2
        )
        official_actual = int(torch.count_nonzero(official_mask).item())
        nominal_requested = sum(
            int(4 * block_width * 0.25) for block_width in (2, 2)
        )
        self.assertEqual(official_actual, 6)
        self.assertEqual(nominal_requested, 4)

        summary = SparseGPTPruner(blocksize=2).prune(
            model,
            Qwen3Adapter(),
            PruningRequest("toy", "sparsegpt", 0.25),
            calibration_context([ids]),
        )
        module_stats = summary.per_module[0]
        self.assertEqual(summary.requested_mask_count, nominal_requested)
        self.assertEqual(module_stats.requested_mask_count, nominal_requested)
        self.assertEqual(summary.actual_mask_count, official_actual)
        self.assertEqual(
            round(module_stats.achieved_mask_sparsity * module_stats.targeted_weights),
            official_actual,
        )
        self.assertEqual(
            summary.achieved_mask_sparsity,
            summary.actual_mask_sparsity,
        )
        self.assertNotEqual(summary.actual_mask_count, summary.requested_mask_count)

    def test_cholesky_failure_reports_orchestration_context(self) -> None:
        class IndefiniteHessian:
            def __init__(self, in_features, device):
                self.H = torch.eye(in_features, dtype=torch.float32, device=device)
                self.H[0, 1] = 2
                self.H[1, 0] = 2

            def add(self, _inputs):
                return None

            def clear(self):
                self.H = None

        model = SequentialModel().eval()
        model.model.layers = nn.ModuleList([model.model.layers[0]])
        ids = torch.tensor([[0, 1, 2, 3]])
        with patch("src.pruning.sparsegpt.SparseGPTHessian", IndefiniteHessian):
            with self.assertRaises(RuntimeError) as raised:
                SparseGPTPruner(percdamp=0.01, blocksize=2).prune(
                    model,
                    Qwen3Adapter(),
                    PruningRequest("toy", "sparsegpt", 0.25),
                    calibration_context([ids]),
                )
        message = str(raised.exception)
        for fragment in ("block=0", "module=proj", "input_width=4", "percdamp=0.01"):
            self.assertIn(fragment, message)


class TinySparseGPTIntegrationTests(unittest.TestCase):
    def test_tiny_qwen3_sparsegpt_forward_save_reload(self) -> None:
        self._assert_round_trip(qwen_tiny(), Qwen3ForCausalLM, Qwen3Adapter())

    def test_tiny_granite_sparsegpt_forward_save_reload(self) -> None:
        self._assert_round_trip(
            granite_tiny(), GraniteForCausalLM, GraniteAdapter()
        )

    def _assert_round_trip(self, model, model_class, adapter) -> None:
        torch.manual_seed(43)
        model.eval()
        ids = [torch.randint(0, model.config.vocab_size, (1, 8)) for _ in range(2)]
        target_names = {
            f"model.layers.{block_index}.{name}.weight"
            for block_index, block in enumerate(adapter.get_blocks(model))
            for name in adapter.get_linear_modules(block)
        }
        target_before = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
            if name in target_names
        }
        excluded_before = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
            if name not in target_names
        }
        summary = SparseGPTPruner().prune(
            model,
            adapter,
            PruningRequest("tiny", "sparsegpt", 0.25),
            calibration_context(ids),
        )
        self.assertEqual(summary.number_of_target_modules, 14)
        self.assertEqual(summary.percdamp, 0.01)
        self.assertEqual(summary.blocksize, 128)
        self.assertTrue(summary.reconstruction)
        self.assertTrue(summary.weight_update)
        self.assertFalse(summary.true_sequential)
        names = {item.module for item in summary.per_module}
        self.assertTrue(any("self_attn" in name for name in names))
        self.assertTrue(any("mlp" in name for name in names))
        changed_unmasked = False
        for name, before in target_before.items():
            after = model.state_dict()[name]
            changed_unmasked |= bool(torch.any((after != 0) & (after != before)))
        self.assertTrue(changed_unmasked)
        for name, before in excluded_before.items():
            self.assertTrue(torch.equal(model.state_dict()[name], before), name)
        with torch.inference_mode():
            logits = model(input_ids=ids[0], use_cache=False).logits
        self.assertEqual(tuple(logits.shape), (1, 8, model.config.vocab_size))

        zero_masks = {
            name: (model.state_dict()[name] == 0).cpu().clone()
            for name in target_names
        }
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            reloaded = model_class.from_pretrained(directory).eval()
            for name, expected in zero_masks.items():
                self.assertTrue(
                    torch.equal((reloaded.state_dict()[name] == 0).cpu(), expected), name
                )
            with torch.inference_mode():
                reloaded_logits = reloaded(input_ids=ids[0], use_cache=False).logits
            self.assertEqual(reloaded_logits.shape, logits.shape)


class SparseGPTCliTests(unittest.TestCase):
    @staticmethod
    def _load_script_module(name):
        script_path = REPOSITORY_ROOT / "scripts" / "run_experiment.py"
        spec = importlib.util.spec_from_file_location(name, script_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    def test_planned_manifest_records_official_defaults_and_overrides(self) -> None:
        module = self._load_script_module("run_experiment_sparsegpt_plan")
        args = module._parser().parse_args([
            "--model", "klear_agentforge_8b",
            "--pruner", "sparsegpt",
            "--sparsity", "0.3",
            "--sparsegpt-percdamp", "0.02",
            "--sparsegpt-blocksize", "64",
        ])
        manifest = module.build_manifest(args)
        pruning = manifest["pruning"]
        self.assertEqual(pruning["percdamp"], 0.02)
        self.assertEqual(pruning["blocksize"], 64)
        self.assertTrue(pruning["error_compensation"])
        self.assertTrue(pruning["weight_update"])
        self.assertFalse(pruning["true_sequential"])
        self.assertEqual(pruning["calibration"], {
            "source": "c4", "samples": 128, "sequence_length": 2048, "seed": 0
        })

    def test_invalid_output_is_rejected_before_loading_or_calibration(self) -> None:
        module = self._load_script_module("run_experiment_sparsegpt_unsafe")
        args = module._parser().parse_args([
            "--model", "klear_agentforge_8b",
            "--pruner", "sparsegpt",
            "--sparsity", "0.2",
            "--execute",
            "--output-dir", str(REPOSITORY_ROOT / "unsafe-sparsegpt"),
        ])
        with patch.object(
            module, "load_dense_model", side_effect=AssertionError("must not load")
        ), patch.object(
            module, "get_calibration_provider", side_effect=AssertionError("must not calibrate")
        ):
            with self.assertRaisesRegex(ValueError, "outside the Git repository"):
                module.execute_experiment(args)

    def test_execute_mock_resolves_calibration_and_writes_manifest(self) -> None:
        module = self._load_script_module("run_experiment_sparsegpt")
        model = qwen_tiny().eval()
        tokenizer = SimpleNamespace(
            save_pretrained=lambda path: Path(path, "tokenizer.ok").touch()
        )
        loaded = SimpleNamespace(model=model, tokenizer=tokenizer)
        sample = CalibrationSample(
            torch.tensor([[1, 2, 3, 4]]), torch.ones(1, 4, dtype=torch.long)
        )
        provider = SimpleNamespace(prepare=lambda _tokenizer, _config: (sample,))
        args = argparse.Namespace(
            model="klear_agentforge_8b", pruner="sparsegpt", sparsity=0.25,
            benchmark=None, output_dir=None, local_path=None, cache_dir=None,
            dtype=None, device=None, device_map=None, local_files_only=False,
            overwrite_output_dir=False, calibration_source="c4",
            calibration_samples=1, calibration_seqlen=4, calibration_seed=9,
            sparsegpt_percdamp=0.01, sparsegpt_blocksize=16,
        )
        fake_model_manifest = {
            "schema_version": 3, "timestamp": "2026-01-01T00:00:00+00:00",
            "mode": "sparsegpt_pruning", "status": "completed",
            "project_model_id": args.model,
        }
        with tempfile.TemporaryDirectory() as directory:
            args.output_dir = Path(directory)
            with patch.object(module, "load_dense_model", return_value=loaded), patch.object(
                module, "get_calibration_provider", return_value=provider
            ), patch.object(
                module, "build_model_manifest", return_value=fake_model_manifest
            ):
                manifest = module.execute_experiment(args)
            pruning = manifest["pruning"]
            self.assertEqual(pruning["pruner"], "sparsegpt")
            self.assertEqual(pruning["method"], "sparsegpt")
            self.assertEqual(pruning["structure"], "unstructured")
            self.assertEqual(pruning["calibration_sample_count"], 1)
            self.assertEqual(pruning["percdamp"], 0.01)
            self.assertEqual(pruning["blocksize"], 16)
            persisted = json.loads(Path(directory, "pruning_manifest.json").read_text())
            self.assertTrue(persisted["pruning"]["reconstruction"])


if __name__ == "__main__":
    unittest.main()
