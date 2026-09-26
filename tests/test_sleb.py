"""SLEB fixed-commit fidelity, greedy search, architecture, and CLI tests."""

from __future__ import annotations

import importlib.util
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from transformers import GraniteConfig, GraniteForCausalLM, Qwen3Config, Qwen3ForCausalLM

from src.models import LoadOptions, LoadedModel, load_model_spec
from src.models.adapters.granite import GraniteAdapter
from src.models.adapters.qwen3 import Qwen3Adapter
from src.pruning import (
    PruningRequest,
    SLEBCalibrationConfig,
    SLEBCalibrationContext,
    SLEBPruner,
    WikiText2SLEBCalibrationProvider,
    greedy_block_search,
    ratio_to_remove_count,
    sleb_get_loss,
    temporary_block_removal,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class FakeDataset:
    def __init__(self, texts):
        self.texts = list(texts)

    def shuffle(self, seed):
        texts = list(self.texts)
        random.Random(seed).shuffle(texts)
        return FakeDataset(texts)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return {"text": self.texts[item]}
        return {"text": self.texts[item]}


class RecordingTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, text, return_tensors):
        self.calls.append((text, return_tensors))
        return {"input_ids": torch.tensor([[ord(char) % 31 for char in text]])}


class SlowCalibrationTokenizer(RecordingTokenizer):
    is_fast = False


class FastCalibrationTokenizer(RecordingTokenizer):
    is_fast = True


def official_calibration_reference(dataset, tokenizer, config):
    shuffled = dataset.shuffle(seed=config.seed)
    return tokenizer(
        "\n\n".join(shuffled[: config.source_rows]["text"]),
        return_tensors="pt",
    )["input_ids"]


class SLEBCalibrationTests(unittest.TestCase):
    def test_wikitext2_stream_matches_official_construction(self) -> None:
        texts = ["zero", "one", "two", "three", "four"]
        config = SLEBCalibrationConfig(source_rows=3, sequence_length=4, seed=17)
        tokenizer = RecordingTokenizer()
        context = WikiText2SLEBCalibrationProvider(
            lambda: FakeDataset(texts)
        ).prepare(tokenizer, config)
        reference_tokenizer = RecordingTokenizer()
        expected = official_calibration_reference(
            FakeDataset(texts), reference_tokenizer, config
        )
        self.assertTrue(torch.equal(context.input_ids, expected))
        self.assertEqual(tokenizer.calls, reference_tokenizer.calls)
        self.assertEqual(len(tokenizer.calls), 1)
        self.assertEqual(tokenizer.calls[0][0].count("\n\n"), 2)
        self.assertEqual(context.token_stream_length, expected.shape[1])

    def test_same_seed_is_deterministic_and_source_rows_are_exact(self) -> None:
        texts = [str(index) for index in range(10)]
        config = SLEBCalibrationConfig(source_rows=4, sequence_length=2, seed=9)
        contexts = []
        calls = []
        for _ in range(2):
            tokenizer = RecordingTokenizer()
            contexts.append(
                WikiText2SLEBCalibrationProvider(
                    lambda: FakeDataset(texts)
                ).prepare(tokenizer, config)
            )
            calls.append(tokenizer.calls[0][0])
        self.assertTrue(torch.equal(contexts[0].input_ids, contexts[1].input_ids))
        self.assertEqual(calls[0], calls[1])
        self.assertEqual(len(calls[0].split("\n\n")), 4)

    def test_raw_empty_and_whitespace_rows_are_not_filtered(self) -> None:
        texts = ["alpha", "", "   ", "omega"]
        config = SLEBCalibrationConfig(source_rows=4, sequence_length=2, seed=0)
        tokenizer = RecordingTokenizer()
        context = WikiText2SLEBCalibrationProvider(
            lambda: FakeDataset(texts)
        ).prepare(tokenizer, config)
        reference_tokenizer = RecordingTokenizer()
        expected = official_calibration_reference(
            FakeDataset(texts), reference_tokenizer, config
        )
        self.assertTrue(torch.equal(context.input_ids, expected))
        self.assertEqual(tokenizer.calls[0][0], reference_tokenizer.calls[0][0])

    def test_fast_tokenizer_is_rejected(self) -> None:
        provider = WikiText2SLEBCalibrationProvider(
            lambda: FakeDataset(["one", "two"])
        )
        with self.assertRaisesRegex(ValueError, "use_fast=False"):
            provider.prepare(
                FastCalibrationTokenizer(),
                SLEBCalibrationConfig(source_rows=2, sequence_length=2),
            )


class DeterministicLM(nn.Module):
    def __init__(self, vocab_size=13):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.vocab_size = vocab_size
        self.config = SimpleNamespace(use_cache=True)
        self.calls = []

    def forward(self, input_ids, use_cache=None):
        self.calls.append((input_ids.detach().clone(), use_cache))
        vocab = torch.arange(self.vocab_size, dtype=torch.float32)
        logits = input_ids.float().unsqueeze(-1) * 0.07 + vocab * 0.11
        logits = logits + (input_ids.unsqueeze(-1) == vocab).float() * 0.3
        return SimpleNamespace(logits=logits + self.anchor)


class DeterministicBF16LM(nn.Module):
    def __init__(self, vocab_size=13):
        super().__init__()
        generator = torch.Generator().manual_seed(67)
        self.register_buffer(
            "table",
            (torch.randn(32, vocab_size, generator=generator) * 3).to(torch.bfloat16),
        )
        self.config = SimpleNamespace(use_cache=True)

    def forward(self, input_ids, use_cache=None):
        return SimpleNamespace(logits=self.table[input_ids])


def official_get_loss_reference(model, input_ids, sequence_length, batch_size=1):
    nsamples = input_ids.numel() // sequence_length
    losses = []
    for i in range(0, nsamples, batch_size):
        j = min(i + batch_size, nsamples)
        inputs = input_ids[:, i * sequence_length : j * sequence_length]
        inputs = inputs.reshape(j - i, sequence_length)
        lm_logits = model(inputs).logits
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]
        loss = nn.CrossEntropyLoss()(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
        )
        losses.append(loss.float() * sequence_length * (j - i))
    return torch.stack(losses).sum().item()


class SLEBLossTests(unittest.TestCase):
    def test_loss_matches_official_chunk_scale_shift_and_sum(self) -> None:
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        expected = official_get_loss_reference(
            DeterministicLM(), input_ids, sequence_length=4
        )
        model = DeterministicLM()
        actual = sleb_get_loss(model, input_ids, sequence_length=4)
        self.assertAlmostEqual(actual, expected, places=6)
        self.assertEqual(len(model.calls), 2)
        self.assertTrue(all(call[1] is False for call in model.calls))

    def test_tail_tokens_are_ignored(self) -> None:
        full = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        with_tail = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 12, 12, 12]])
        self.assertEqual(
            sleb_get_loss(DeterministicLM(), full, sequence_length=4),
            sleb_get_loss(DeterministicLM(), with_tail, sequence_length=4),
        )

    def test_too_short_stream_has_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "shorter than one full"):
            sleb_get_loss(
                DeterministicLM(), torch.tensor([[1, 2, 3]]), sequence_length=4
            )

    def test_bfloat16_loss_matches_official_dtype_order(self) -> None:
        input_ids = torch.tensor([[1, 7, 3, 9, 2, 8, 4, 10, 6]])
        expected = official_get_loss_reference(
            DeterministicBF16LM(), input_ids, sequence_length=4
        )
        actual = sleb_get_loss(
            DeterministicBF16LM(), input_ids, sequence_length=4
        )
        self.assertEqual(actual, expected)

        model = DeterministicBF16LM()
        wrong_losses = []
        for offset in (0, 4):
            inputs = input_ids[:, offset : offset + 4]
            logits = model(inputs).logits
            wrong_losses.append(
                nn.CrossEntropyLoss()(
                    logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
                    inputs[:, 1:].reshape(-1),
                ).float()
                * 4
            )
        wrong_fp32_before_ce = torch.stack(wrong_losses).sum().item()
        self.assertNotEqual(expected, wrong_fp32_before_ce)


class TaggedBlock(nn.Module):
    def __init__(self, original_index):
        super().__init__()
        self.original_index = original_index
        self.value = nn.Parameter(torch.tensor(float(original_index)))
        self.self_attn = nn.Identity()
        self.self_attn.layer_idx = original_index


class SearchBackbone(nn.Module):
    def __init__(self, count):
        super().__init__()
        self.layers = nn.ModuleList([TaggedBlock(index) for index in range(count)])
        self.embed_tokens = nn.Embedding(4, 2)
        self.norm = nn.Identity()


class SearchModel(nn.Module):
    def __init__(self, count=5):
        super().__init__()
        self.model = SearchBackbone(count)
        self.lm_head = nn.Linear(2, 4, bias=False)
        self.config = SimpleNamespace(num_hidden_layers=count, use_cache=True)


def tags(model):
    return tuple(block.original_index for block in model.model.layers)


def minimal_official_search(count, removals, scorer, early=1, latter=1):
    alive = list(range(count))
    order = []
    for _ in range(removals):
        best_score = float("inf")
        best_position = -1
        for position in range(early, len(alive) - latter):
            candidate = tuple(alive[:position] + alive[position + 1 :])
            score = scorer(candidate)
            if score < best_score:
                best_score = score
                best_position = position
        order.append(alive[best_position])
        del alive[best_position]
    return order, alive


class SLEBSearchTests(unittest.TestCase):
    def test_dynamic_greedy_matches_reference_and_rescores_after_removal(self) -> None:
        def score(state):
            values = {
                (0, 2, 3, 4): 0.0,
                (0, 1, 3, 4): 1.0,
                (0, 1, 2, 4): 2.0,
                (0, 3, 4): 10.0,
                (0, 2, 4): -1.0,
            }
            return values[state]

        expected_order, expected_alive = minimal_official_search(5, 2, score)
        model = SearchModel()
        result = greedy_block_search(
            model, Qwen3Adapter(), 2, lambda current: score(tags(current))
        )
        self.assertEqual(list(result.removal_order_original_indices), expected_order)
        self.assertEqual(list(result.retained_original_indices), expected_alive)
        self.assertEqual(expected_order, [1, 3])
        self.assertEqual(tags(model), tuple(expected_alive))

    def test_strict_tie_keeps_first_candidate(self) -> None:
        model = SearchModel()
        result = greedy_block_search(
            model, Qwen3Adapter(), 1, lambda _current: 7.0
        )
        self.assertEqual(result.removal_order_original_indices, (1,))

    def test_default_barriers_and_zero_barrier_override(self) -> None:
        default_seen = []
        greedy_block_search(
            SearchModel(),
            Qwen3Adapter(),
            1,
            lambda current: default_seen.append(tags(current)) or 0.0,
        )
        removed_in_trials = [{0, 1, 2, 3, 4}.difference(state).pop() for state in default_seen]
        self.assertEqual(removed_in_trials, [1, 2, 3])

        override_seen = []
        result = greedy_block_search(
            SearchModel(),
            Qwen3Adapter(),
            1,
            lambda current: override_seen.append(tags(current)) or 0.0,
            early_barrier=0,
            latter_barrier=0,
        )
        removed_override = [
            {0, 1, 2, 3, 4}.difference(state).pop() for state in override_seen
        ]
        self.assertEqual(removed_override, [0, 1, 2, 3, 4])
        self.assertEqual(result.removal_order_original_indices, (0,))

    def test_capacity_validation_happens_before_scoring(self) -> None:
        called = []
        with self.assertRaisesRegex(ValueError, "capacity"):
            greedy_block_search(
                SearchModel(4),
                Qwen3Adapter(),
                3,
                lambda _model: called.append(True) or 0.0,
            )
        self.assertEqual(called, [])

    def test_temporary_removal_restores_exact_container_on_exception(self) -> None:
        model = SearchModel()
        adapter = Qwen3Adapter()
        original = adapter.get_blocks(model)
        original_objects = list(original)

        def deliberate_failure(current):
            self.assertEqual(tags(current), (0, 2, 3, 4))
            raise RuntimeError("deliberate")

        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            greedy_block_search(model, adapter, 1, deliberate_failure)
        self.assertIs(adapter.get_blocks(model), original)
        self.assertEqual(list(adapter.get_blocks(model)), original_objects)

    def test_candidate_count_and_trace_integrity(self) -> None:
        calls = []
        result = greedy_block_search(
            SearchModel(),
            Qwen3Adapter(),
            2,
            lambda current: calls.append(tags(current)) or float(sum(tags(current))),
        )
        self.assertEqual(result.candidate_evaluation_count, 3 + 2)
        self.assertEqual(len(calls), 5)
        self.assertEqual(len(result.search_trace), 2)
        for round_trace in result.search_trace:
            self.assertEqual(
                len(round_trace["candidate_scores"]),
                len(round_trace["eligible_original_indices"]),
            )
            self.assertIn(
                round_trace["selected_original_index"],
                round_trace["eligible_original_indices"],
            )


def qwen_tiny(num_hidden_layers=4):
    return Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            max_position_embeddings=32,
            use_cache=True,
        )
    )


def granite_tiny(num_hidden_layers=4):
    return GraniteForCausalLM(
        GraniteConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
            use_cache=True,
        )
    )


def small_context(tokenizer_class=None, tokenizer_is_fast=None):
    config = SLEBCalibrationConfig(source_rows=3, sequence_length=4, seed=0)
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
    return SLEBCalibrationContext(
        config,
        input_ids,
        input_ids.shape[1],
        tokenizer_class=tokenizer_class,
        tokenizer_is_fast=tokenizer_is_fast,
    )


class SLEBAdapterTests(unittest.TestCase):
    def test_qwen_metadata_subsets_original_layer_types(self) -> None:
        model = SearchModel(4)
        model.config.layer_types = ["first", "second", "third", "fourth"]
        adapter = Qwen3Adapter()
        metadata = adapter.capture_block_removal_metadata(model)
        adapter.replace_blocks(model, [model.model.layers[i] for i in (0, 2, 3)])
        adapter.finalize_block_removal(model, (0, 2, 3), metadata)
        self.assertEqual(model.config.layer_types, ["first", "third", "fourth"])
        self.assertEqual(
            [block.self_attn.layer_idx for block in model.model.layers], [0, 1, 2]
        )

    def test_qwen_temporary_removal_matches_final_physical_removal(self) -> None:
        self._assert_temporary_equivalence(qwen_tiny(), Qwen3Adapter())

    def test_granite_temporary_removal_matches_final_physical_removal(self) -> None:
        self._assert_temporary_equivalence(granite_tiny(), GraniteAdapter())

    def test_qwen_multi_block_search_state_removal_equivalence(self) -> None:
        direct = qwen_tiny(5).eval()
        sequential = qwen_tiny(5).eval()
        sequential.load_state_dict(direct.state_dict())
        adapter = Qwen3Adapter()
        direct_blocks = list(adapter.get_blocks(direct))
        adapter.replace_blocks(direct, [direct_blocks[index] for index in (0, 2, 4)])
        sequential_blocks = list(adapter.get_blocks(sequential))
        adapter.replace_blocks(
            sequential, sequential_blocks[:1] + sequential_blocks[2:]
        )
        sequential_blocks = list(adapter.get_blocks(sequential))
        adapter.replace_blocks(
            sequential, sequential_blocks[:2] + sequential_blocks[3:]
        )
        self.assertEqual(
            [block.self_attn.layer_idx for block in adapter.get_blocks(direct)],
            [0, 2, 4],
        )
        self.assertEqual(
            [block.self_attn.layer_idx for block in adapter.get_blocks(sequential)],
            [0, 2, 4],
        )
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        with torch.inference_mode():
            direct_logits = direct(input_ids=ids, use_cache=False).logits
            sequential_logits = sequential(input_ids=ids, use_cache=False).logits
        self.assertTrue(
            torch.allclose(direct_logits, sequential_logits, atol=1e-6, rtol=1e-5)
        )

    def _assert_temporary_equivalence(self, model, adapter):
        torch.manual_seed(51)
        model.eval()
        physical = type(model)(model.config)
        physical.load_state_dict(model.state_dict())
        physical.eval()
        ids = torch.randint(0, model.config.vocab_size, (1, 6))
        with temporary_block_removal(model, adapter, 1):
            with torch.inference_mode():
                temporary_logits = model(input_ids=ids, use_cache=False).logits
        metadata = adapter.capture_block_removal_metadata(physical)
        blocks = list(adapter.get_blocks(physical))
        adapter.replace_blocks(physical, blocks[:1] + blocks[2:])
        adapter.finalize_block_removal(physical, (0, 2, 3), metadata)
        with torch.inference_mode():
            physical_logits = physical(input_ids=ids, use_cache=False).logits
        self.assertTrue(
            torch.allclose(temporary_logits, physical_logits, atol=1e-6, rtol=1e-5)
        )


class TinySLEBIntegrationTests(unittest.TestCase):
    def test_tiny_qwen3_full_sleb_round_trip_cache_and_generate(self) -> None:
        self._assert_round_trip(qwen_tiny(), Qwen3ForCausalLM, Qwen3Adapter())

    def test_tiny_granite_full_sleb_round_trip_cache_and_generate(self) -> None:
        self._assert_round_trip(
            granite_tiny(), GraniteForCausalLM, GraniteAdapter()
        )

    def test_qwen3_multi_round_preserves_search_metadata_then_finalizes(self) -> None:
        self._assert_multi_round(qwen_tiny(5), Qwen3Adapter(), check_cleanup=True)

    def test_granite_multi_round_preserves_search_metadata_then_finalizes(self) -> None:
        self._assert_multi_round(granite_tiny(5), GraniteAdapter())

    def _assert_multi_round(self, model, adapter, check_cleanup=False):
        model.eval()
        original_blocks = [
            {name: tensor.detach().clone() for name, tensor in block.state_dict().items()}
            for block in adapter.get_blocks(model)
        ]
        original_layer_types = None
        if isinstance(adapter, Qwen3Adapter):
            model.config.layer_types = [f"original_type_{index}" for index in range(5)]
            original_layer_types = tuple(model.config.layer_types)
        trial_layer_indices = []

        def inspect_search_state(current_model, *_args, **_kwargs):
            self.assertEqual(current_model.config.num_hidden_layers, 5)
            if isinstance(adapter, Qwen3Adapter):
                self.assertEqual(
                    tuple(current_model.config.layer_types), original_layer_types
                )
            indices = [
                block.self_attn.layer_idx for block in adapter.get_blocks(current_model)
            ]
            trial_layer_indices.append(indices)
            return float(sum(indices))

        empty_cache = patch("torch.cuda.empty_cache") if check_cleanup else None
        with patch("src.pruning.sleb.sleb_get_loss", side_effect=inspect_search_state):
            if empty_cache is None:
                summary = SLEBPruner(
                    calibration_config=small_context().config
                ).prune(
                    model,
                    adapter,
                    PruningRequest("tiny", "sleb", 0.4),
                    small_context(),
                )
            else:
                with empty_cache as cleanup:
                    summary = SLEBPruner(
                        calibration_config=small_context().config
                    ).prune(
                        model,
                        adapter,
                        PruningRequest("tiny", "sleb", 0.4),
                        small_context(),
                    )
                self.assertEqual(cleanup.call_count, summary.candidate_evaluation_count)

        self.assertEqual(summary.removal_order_original_indices, (3, 2))
        self.assertEqual(summary.retained_original_indices, (0, 1, 4))
        self.assertEqual(summary.candidate_evaluation_count, 5)
        self.assertEqual(trial_layer_indices, [
            [0, 2, 3, 4],
            [0, 1, 3, 4],
            [0, 1, 2, 4],
            [0, 2, 4],
            [0, 1, 4],
        ])
        self.assertEqual(model.config.num_hidden_layers, 3)
        self.assertEqual(
            [block.self_attn.layer_idx for block in adapter.get_blocks(model)],
            [0, 1, 2],
        )
        if isinstance(adapter, Qwen3Adapter):
            self.assertEqual(
                model.config.layer_types,
                [original_layer_types[index] for index in (0, 1, 4)],
            )
        for block, original_index in zip(
            adapter.get_blocks(model), summary.retained_original_indices
        ):
            for name, tensor in block.state_dict().items():
                self.assertTrue(
                    torch.equal(tensor, original_blocks[original_index][name]), name
                )

    def _assert_round_trip(self, model, model_class, adapter):
        torch.manual_seed(53)
        model.eval()
        original_blocks = [
            {name: tensor.detach().clone() for name, tensor in block.state_dict().items()}
            for block in adapter.get_blocks(model)
        ]
        outside_before = {
            "embedding": adapter.get_embedding(model).weight.detach().clone(),
            "lm_head": adapter.get_lm_head(model).weight.detach().clone(),
            "norm": {
                name: tensor.detach().clone()
                for name, tensor in adapter.get_final_norm(model).state_dict().items()
            },
        }
        summary = SLEBPruner(
            calibration_config=small_context().config
        ).prune(
            model,
            adapter,
            PruningRequest("tiny", "sleb", 0.25),
            small_context(),
        )
        self.assertEqual(summary.original_block_count, 4)
        self.assertEqual(summary.requested_remove_count, 1)
        self.assertEqual(summary.removed_block_count, 1)
        self.assertEqual(summary.achieved_block_sparsity, 0.25)
        self.assertEqual(summary.candidate_evaluation_count, 2)
        self.assertEqual(summary.scored_full_chunks, 2)
        self.assertEqual(summary.dropped_tail_tokens, 1)
        self.assertLess(summary.final_parameter_count, summary.original_parameter_count)
        self.assertEqual(len(adapter.get_blocks(model)), 3)
        self.assertEqual(model.config.num_hidden_layers, 3)
        self.assertTrue(model.config.use_cache)
        if isinstance(adapter, Qwen3Adapter):
            self.assertEqual(len(model.config.layer_types), 3)
        self.assertEqual(
            [block.self_attn.layer_idx for block in adapter.get_blocks(model)],
            [0, 1, 2],
        )
        for block, original_index in zip(
            adapter.get_blocks(model), summary.retained_original_indices
        ):
            for name, tensor in block.state_dict().items():
                self.assertTrue(
                    torch.equal(tensor, original_blocks[original_index][name]), name
                )
        self.assertTrue(
            torch.equal(adapter.get_embedding(model).weight, outside_before["embedding"])
        )
        self.assertTrue(
            torch.equal(adapter.get_lm_head(model).weight, outside_before["lm_head"])
        )
        for name, tensor in adapter.get_final_norm(model).state_dict().items():
            self.assertTrue(torch.equal(tensor, outside_before["norm"][name]))

        ids = torch.tensor([[1, 2, 3, 4]])
        mask = torch.ones_like(ids)
        with torch.inference_mode():
            before_save = model(
                input_ids=ids, attention_mask=mask, use_cache=False
            ).logits
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            reloaded = model_class.from_pretrained(directory).eval()
            self.assertEqual(len(adapter.get_blocks(reloaded)), 3)
            self.assertEqual(reloaded.config.num_hidden_layers, 3)
            if isinstance(adapter, Qwen3Adapter):
                self.assertEqual(len(reloaded.config.layer_types), 3)
            self.assertEqual(
                [block.self_attn.layer_idx for block in adapter.get_blocks(reloaded)],
                [0, 1, 2],
            )
            with torch.inference_mode():
                after_reload = reloaded(
                    input_ids=ids, attention_mask=mask, use_cache=False
                ).logits
                cached = reloaded(
                    input_ids=ids, attention_mask=mask, use_cache=True
                ).logits
                generated = reloaded.generate(
                    input_ids=ids,
                    attention_mask=mask,
                    max_new_tokens=2,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=0,
                )
            self.assertTrue(
                torch.allclose(before_save, after_reload, atol=1e-6, rtol=1e-5)
            )
            self.assertEqual(cached.shape, before_save.shape)
            self.assertEqual(generated.shape, (1, ids.shape[1] + 2))

    def test_zero_sparsity_is_no_op_without_calibration(self) -> None:
        model = qwen_tiny().eval()
        before = {name: value.detach().clone() for name, value in model.state_dict().items()}
        summary = SLEBPruner().prune(
            model,
            Qwen3Adapter(),
            PruningRequest("tiny", "sleb", 0.0),
        )
        self.assertEqual(summary.requested_remove_count, 0)
        self.assertEqual(summary.removal_order_original_indices, ())
        self.assertEqual(summary.candidate_evaluation_count, 0)
        self.assertEqual(len(model.model.layers), 4)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)

    def test_ratio_uses_project_ceil_mapping(self) -> None:
        self.assertEqual(ratio_to_remove_count(36, 0.20), 8)
        self.assertEqual(ratio_to_remove_count(4, 0.01), 1)
        self.assertEqual(ratio_to_remove_count(4, 0.0), 0)


class SLEBCliTests(unittest.TestCase):
    @staticmethod
    def _load_script_module(name):
        path = REPOSITORY_ROOT / "scripts" / "run_experiment.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    def test_planned_manifest_records_official_and_interface_semantics(self) -> None:
        module = self._load_script_module("run_experiment_sleb_plan")
        args = module._parser().parse_args([
            "--model", "klear_agentforge_8b",
            "--pruner", "sleb",
            "--sparsity", "0.2",
        ])
        pruning = module.build_manifest(args)["pruning"]
        self.assertEqual(pruning["requested_remove_count"], 8)
        self.assertEqual(pruning["planned_achieved_block_sparsity"], 8 / 36)
        self.assertEqual(pruning["early_barrier"], 1)
        self.assertEqual(pruning["latter_barrier"], 1)
        self.assertEqual(pruning["calibration"]["source"], "wikitext2")
        self.assertEqual(pruning["calibration"]["source_rows"], 128)
        self.assertEqual(pruning["calibration"]["separator"], "\n\n")

    def test_planned_manifest_accepts_debug_barrier_override(self) -> None:
        module = self._load_script_module("run_experiment_sleb_barriers")
        args = module._parser().parse_args([
            "--model", "klear_agentforge_8b", "--pruner", "sleb",
            "--sparsity", "0.2", "--sleb-early-barrier", "0",
            "--sleb-latter-barrier", "0",
        ])
        pruning = module.build_manifest(args)["pruning"]
        self.assertEqual(pruning["early_barrier"], 0)
        self.assertEqual(pruning["latter_barrier"], 0)

    def test_remote_slow_tokenizer_reuses_dense_source_revision_and_options(self) -> None:
        module = self._load_script_module("run_experiment_sleb_remote_tokenizer")
        spec = load_model_spec("klear_agentforge_8b")
        options = LoadOptions(
            cache_dir=Path("/tmp/shared-model-cache"),
            local_files_only=True,
        )
        loaded = SimpleNamespace(source=spec.huggingface_repo_id, is_local=False)
        sentinel = object()
        with patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=sentinel
        ) as load_tokenizer:
            actual = module._load_sleb_calibration_tokenizer(spec, options, loaded)
        self.assertIs(actual, sentinel)
        load_tokenizer.assert_called_once_with(
            loaded.source,
            trust_remote_code=spec.trust_remote_code,
            local_files_only=True,
            use_fast=False,
            cache_dir="/tmp/shared-model-cache",
            revision=spec.revision,
        )

    def test_local_slow_tokenizer_reuses_local_source_without_revision(self) -> None:
        module = self._load_script_module("run_experiment_sleb_local_tokenizer")
        spec = load_model_spec("klear_agentforge_8b")
        with tempfile.TemporaryDirectory() as directory:
            source = str(Path(directory).resolve())
            options = LoadOptions(
                local_path=Path(directory),
                cache_dir=Path("/tmp/shared-model-cache"),
                local_files_only=True,
            )
            loaded = SimpleNamespace(source=source, is_local=True)
            sentinel = object()
            with patch(
                "transformers.AutoTokenizer.from_pretrained", return_value=sentinel
            ) as load_tokenizer:
                actual = module._load_sleb_calibration_tokenizer(spec, options, loaded)
        self.assertIs(actual, sentinel)
        load_tokenizer.assert_called_once_with(
            source,
            trust_remote_code=spec.trust_remote_code,
            local_files_only=True,
            use_fast=False,
            cache_dir="/tmp/shared-model-cache",
        )

    def test_invalid_output_is_rejected_before_load_or_dataset(self) -> None:
        module = self._load_script_module("run_experiment_sleb_unsafe")
        args = module._parser().parse_args([
            "--model", "klear_agentforge_8b", "--pruner", "sleb",
            "--sparsity", "0.2", "--execute",
            "--output-dir", str(REPOSITORY_ROOT / "unsafe-sleb"),
        ])
        with patch.object(
            module, "load_dense_model", side_effect=AssertionError("must not load")
        ), patch.object(
            module,
            "get_sleb_calibration_provider",
            side_effect=AssertionError("must not prepare dataset"),
        ), patch.object(
            module,
            "_load_sleb_calibration_tokenizer",
            side_effect=AssertionError("must not load tokenizer"),
        ):
            with self.assertRaisesRegex(ValueError, "outside the Git repository"):
                module.execute_experiment(args)

    def test_zero_ratio_execute_skips_provider_and_refreshes_structure(self) -> None:
        module = self._load_script_module("run_experiment_sleb_zero")
        model = qwen_tiny().eval()
        adapter = Qwen3Adapter()
        tokenizer = SimpleNamespace(
            save_pretrained=lambda path: Path(path, "tokenizer.ok").touch()
        )
        loaded = LoadedModel(
            model=model,
            tokenizer=tokenizer,
            config=model.config,
            source="tiny",
            is_local=True,
            requested_revision="local",
            resolved_revision=None,
            dtype="float32",
            structure=adapter.get_structure(model),
        )
        args = module._parser().parse_args([
            "--model", "klear_agentforge_8b", "--pruner", "sleb",
            "--sparsity", "0", "--execute",
        ])
        with tempfile.TemporaryDirectory() as directory:
            args.output_dir = Path(directory)
            with patch.object(module, "load_dense_model", return_value=loaded), patch.object(
                module,
                "get_sleb_calibration_provider",
                side_effect=AssertionError("zero ratio must not create provider"),
            ), patch.object(
                module,
                "_load_sleb_calibration_tokenizer",
                side_effect=AssertionError("zero ratio must not load tokenizer"),
            ):
                manifest = module.execute_experiment(args)
        self.assertEqual(manifest["pruning"]["requested_remove_count"], 0)
        self.assertIsNone(manifest["pruning"]["calibration_tokenizer_class"])
        self.assertIsNone(manifest["pruning"]["calibration_tokenizer_is_fast"])
        self.assertEqual(manifest["model"]["num_hidden_layers"], 4)
        self.assertEqual(manifest["model"]["actual_block_count"], 4)

    def test_mock_execute_records_fresh_reduced_structure(self) -> None:
        module = self._load_script_module("run_experiment_sleb_execute")
        model = qwen_tiny().eval()
        adapter = Qwen3Adapter()
        tokenizer = SimpleNamespace(
            save_pretrained=lambda path: Path(path, "tokenizer.ok").touch()
        )
        loaded = LoadedModel(
            model=model,
            tokenizer=tokenizer,
            config=model.config,
            source="tiny",
            is_local=True,
            requested_revision="local",
            resolved_revision=None,
            dtype="float32",
            structure=adapter.get_structure(model),
        )
        calibration_tokenizer = SlowCalibrationTokenizer()
        provider = SimpleNamespace(
            prepare=lambda current_tokenizer, _config: small_context(
                type(current_tokenizer).__name__, current_tokenizer.is_fast
            )
        )
        args = module._parser().parse_args([
            "--model", "klear_agentforge_8b", "--pruner", "sleb",
            "--sparsity", "0.25", "--execute",
            "--calibration-samples", "3", "--calibration-seqlen", "4",
        ])
        with tempfile.TemporaryDirectory() as directory:
            args.output_dir = Path(directory)
            with patch.object(module, "load_dense_model", return_value=loaded), patch.object(
                module, "get_sleb_calibration_provider", return_value=provider
            ), patch.object(
                module,
                "_load_sleb_calibration_tokenizer",
                return_value=calibration_tokenizer,
            ):
                manifest = module.execute_experiment(args)
            persisted = json.loads(Path(directory, "pruning_manifest.json").read_text())
        self.assertEqual(manifest["model"]["num_hidden_layers"], 3)
        self.assertEqual(manifest["model"]["actual_block_count"], 3)
        self.assertEqual(persisted["pruning"]["removed_block_count"], 1)
        self.assertEqual(persisted["pruning"]["post_pruning_actual_block_count"], 3)
        self.assertEqual(
            persisted["pruning"]["calibration_tokenizer_class"],
            "SlowCalibrationTokenizer",
        )
        self.assertIs(persisted["pruning"]["calibration_tokenizer_is_fast"], False)


if __name__ == "__main__":
    unittest.main()
