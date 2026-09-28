"""TaBP official-semantics, tiny-model, and planning tests."""

from __future__ import annotations

import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from transformers import GraniteConfig, GraniteForCausalLM, Qwen3Config, Qwen3ForCausalLM

from src.models.adapters.granite import GraniteAdapter
from src.models.adapters.qwen3 import Qwen3Adapter
from src.pruning import (
    ARC_EASY_REVISION,
    PRUNER_REGISTRY,
    PruningRequest,
    TaBPCalibrationConfig,
    TaBPCalibrationContext,
    TaBPCalibrationSample,
    TaBPPruner,
    aggregate_ddf_scores,
    aggregate_ssn_scores,
    format_arc_easy,
    rank_blocks_ssn,
)

ROOT = Path(__file__).resolve().parents[2]


def tiny_pair(kind: str):
    common = dict(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=32, bos_token_id=1, eos_token_id=2, pad_token_id=0,
    )
    if kind == "qwen":
        model = Qwen3ForCausalLM(Qwen3Config(**common, head_dim=4))
        return model.eval(), Qwen3Adapter(), Qwen3ForCausalLM
    model = GraniteForCausalLM(GraniteConfig(**common))
    return model.eval(), GraniteAdapter(), GraniteForCausalLM


def context() -> TaBPCalibrationContext:
    sample = TaBPCalibrationSample(
        input_ids=torch.tensor([[1, 8, 9]]),
        attention_mask=torch.ones((1, 3), dtype=torch.long),
        allowed_token_ids=(4, 5, 6, 7), key_token_id=4,
    )
    return TaBPCalibrationContext(
        TaBPCalibrationConfig(samples=1), (sample,), "TinyTokenizer", True,
    )


class OfficialSemanticsTests(unittest.TestCase):
    def test_ssn_p1_mean_absolute_shift_and_numpy_tie_order(self):
        scores = aggregate_ssn_scores((
            {0: 1.0, 1: 1.5, 2: 1.25},
            {0: 3.0, 1: 2.0, 2: 4.0},
        ), (1, 2))
        self.assertEqual(scores, {1: 0.75, 2: 1.125})
        tied = np.array([0.0, 0.0, 0.0])
        self.assertEqual(tuple(np.argsort(tied)), (0, 1, 2))

    def test_ddf_qa_counts_undesirable_and_sorts_descending(self):
        scores = aggregate_ddf_scores(
            ({0: 1.0, 1: 2.0, 2: 1.0},), 3, "entropy", "qa"
        )
        self.assertEqual(scores, {0: 0.0, 1: 1.0, 2: 0.0})
        self.assertEqual(tuple(np.argsort(-np.array(list(scores.values())))), (1, 0, 2))

    def test_ddf_text_counts_desirable_and_sorts_descending(self):
        scores = aggregate_ddf_scores(
            ({0: 1.0, 1: 2.0, 2: 1.0},), 3, "entropy", "text_generation"
        )
        self.assertEqual(scores, {0: 0.0, 1: 0.0, 2: 1.0})
        self.assertEqual(tuple(np.argsort(-np.array(list(scores.values())))), (2, 0, 1))

    def test_arc_prompt_pin_and_allowed_labels(self):
        prompt, labels, answer = format_arc_easy({
            "question": "Which?", "choices": {"label": ["A", "B"], "text": ["one", "two"]},
            "answerKey": "B",
        })
        self.assertEqual(prompt, "Answer the following multiple choice question by giving the most appropriate response.\nQuestion: Which?\nA. one\nB. two\nAnswer: ")
        self.assertEqual((labels, answer), (("A", "B"), "B"))
        self.assertEqual(TaBPCalibrationConfig().dataset_revision, ARC_EASY_REVISION)


class TinyRealModelTests(unittest.TestCase):
    def test_qwen_and_granite_rank_remove_forward_generate_save_reload(self):
        for kind in ("qwen", "granite"):
            with self.subTest(kind=kind):
                torch.manual_seed(0)
                model, adapter, model_class = tiny_pair(kind)
                ranking = rank_blocks_ssn(model, adapter, context(), mode="whole", measure="entropy")
                self.assertEqual(set(ranking.ranking_original_indices), {1, 2, 3})
                self.assertTrue(all(math.isfinite(value) for value in ranking.scores_by_original_index.values()))
                summary = TaBPPruner(mode="whole").prune(
                    model, adapter, PruningRequest("tiny", "tabp", 0.25), context()
                )
                self.assertEqual((summary.removed_block_count, summary.post_pruning_actual_block_count), (1, 3))
                with torch.inference_mode():
                    self.assertEqual(tuple(model(torch.tensor([[1, 8, 9]])).logits.shape), (1, 3, 32))
                    generated = model.generate(torch.tensor([[1, 8, 9]]), max_new_tokens=2, use_cache=True, pad_token_id=0)
                self.assertEqual(generated.shape[1], 5)
                with tempfile.TemporaryDirectory() as directory:
                    model.save_pretrained(directory)
                    reloaded = model_class.from_pretrained(directory).eval()
                    self.assertEqual(len(adapter.get_blocks(reloaded)), 3)
                    with torch.inference_mode():
                        self.assertEqual(reloaded.generate(torch.tensor([[1, 8]]), max_new_tokens=1, use_cache=True, pad_token_id=0).shape[1], 3)

    def test_latter_mode_observes_boundary_but_ranks_after_it(self):
        model, adapter, _ = tiny_pair("qwen")
        ranking = rank_blocks_ssn(model, adapter, context(), mode="latter", measure="entropy")
        self.assertEqual(ranking.eligible_original_indices, (3,))

    def test_ddf_qa_qwen_and_granite_end_to_end(self):
        for kind in ("qwen", "granite"):
            with self.subTest(kind=kind):
                torch.manual_seed(23)
                model, adapter, model_class = tiny_pair(kind)
                summary = TaBPPruner(
                    ranking_strategy="ddf", mode="whole", measure="entropy"
                ).prune(
                    model, adapter, PruningRequest("tiny", "tabp", 0.25), context()
                )
                self.assertEqual(summary.ranking_strategy, "ddf")
                self.assertEqual(set(summary.ranked_block_indices), set(range(4)))
                self.assertTrue(
                    all(math.isfinite(value) for value in summary.scores_by_original_index.values())
                )
                self.assertEqual(summary.removed_block_count, 1)
                self.assertEqual(adapter.get_num_blocks(model), 3)
                with torch.inference_mode():
                    logits = model(torch.tensor([[1, 8, 9]])).logits
                    generated = model.generate(
                        torch.tensor([[1, 8, 9]]), max_new_tokens=2,
                        use_cache=True, pad_token_id=0,
                    )
                self.assertEqual(tuple(logits.shape), (1, 3, 32))
                self.assertEqual(generated.shape[1], 5)
                with tempfile.TemporaryDirectory() as directory:
                    model.save_pretrained(directory)
                    reloaded = model_class.from_pretrained(directory).eval()
                    self.assertEqual(len(adapter.get_blocks(reloaded)), 3)
                    with torch.inference_mode():
                        regenerated = reloaded.generate(
                            torch.tensor([[1, 8]]), max_new_tokens=1,
                            use_cache=True, pad_token_id=0,
                        )
                    self.assertEqual(regenerated.shape[1], 3)

    def test_trained_head_is_not_silently_randomized(self):
        model, adapter, _ = tiny_pair("qwen")
        with self.assertRaisesRegex(NotImplementedError, "explicit compatible"):
            rank_blocks_ssn(model, adapter, context(), mode="whole", measure="entropy", lm_head_type="trained")


class PlanningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("run_experiment_tabp_plan", ROOT / "scripts" / "run_pruning.py")
        cls.module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(cls.module)

    def test_registry_has_one_top_level_tabp_method(self):
        self.assertIn("tabp", PRUNER_REGISTRY)
        self.assertNotIn("tabp_ssn", PRUNER_REGISTRY)

    def test_all_three_project_models_plan_with_canonical_config(self):
        for model_id in ("klear_agentforge_8b", "swe_lego_qwen3_8b", "granite_4_2_8b"):
            with self.subTest(model=model_id):
                args = self.module._parser().parse_args(["--model", model_id, "--pruner", "tabp", "--sparsity", "0.2"])
                pruning = self.module.build_manifest(args)["pruning"]
                self.assertEqual(pruning["ranking_strategy"], "ssn")
                self.assertEqual((pruning["mode"], pruning["measure"]), ("latter", "entropy"))
                self.assertEqual(pruning["calibration"]["samples"], 1024)
                self.assertIn("planned_achieved_block_retention", pruning)


if __name__ == "__main__":
    unittest.main()
