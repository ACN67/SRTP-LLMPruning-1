"""Benchmark protocol, evaluator-fidelity, profile, and generation tests."""

from __future__ import annotations

import base64
import json
import pickle
import tempfile
import unittest
import zlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from src.evaluation import BenchmarkSpec, BenchmarkTask, list_benchmarks, load_evaluation_profile
from src.evaluation.benchmarks import HumanEvalBenchmark, LiveCodeBenchBenchmark, MBPPBenchmark
from src.evaluation.generation import generate_one
from src.evaluation.lcb_protocol import (
    FORMAT_WITHOUT_STARTER, SYSTEM_MESSAGE_GENERIC, chat_messages, decode_test_cases,
    extract_lcb_code, generic_question_prompt, load_verified_v6,
)


def spec(name):
    return BenchmarkSpec(name, name, "a" * 40, 1, "pass@1", f"{name}_prompt_v1",
                         "code_extraction_v1", "/missing", {})


class EvaluatorTests(unittest.TestCase):
    def test_registry_has_only_three_full_benchmarks(self):
        self.assertEqual(list_benchmarks(), ("humaneval", "livecodebench", "mbpp"))

    def test_humaneval_official_suffix_and_full_solution(self):
        benchmark = HumanEvalBenchmark(spec("humaneval"))
        task = BenchmarkTask("humaneval", "HumanEval/0", "def add(a, b):\n    ", {
            "test": "def check(candidate):\n    assert candidate(1, 2) == 3", "entry_point": "add",
        })
        self.assertTrue(benchmark.evaluate(task, "return a + b").passed)
        self.assertTrue(benchmark.evaluate(task, "def add(a,b): return a+b").passed)
        self.assertFalse(benchmark.evaluate(task, "return a - b").passed)

    def test_humaneval_chat_prompt_and_extraction(self):
        benchmark = HumanEvalBenchmark(spec("humaneval"))
        task = BenchmarkTask("humaneval", "x", "def f():\n    ", {"entry_point": "f"})
        self.assertIn("complete runnable solution", benchmark.build_prompt(task))
        self.assertEqual(benchmark.postprocess_generation("thinking\n```python\ndef f(): return 1\n```"), "def f(): return 1")
        self.assertEqual(benchmark.postprocess_generation("def f(): return 1"), "def f(): return 1")

    def test_mbpp_uses_import_setup_and_multiple_tests(self):
        benchmark = MBPPBenchmark(spec("mbpp"))
        task = BenchmarkTask("mbpp", "11", "root", {
            "text": "integer square root", "test_imports": ["import math"],
            "test_setup_code": "EXPECTED = 3", "test_list": [
                "assert root(10) == EXPECTED", "assert root(16) == 4"],
        })
        self.assertTrue(benchmark.evaluate(task, "def root(x): return math.isqrt(x)").passed)
        self.assertFalse(benchmark.evaluate(task, "def root(x): return x").passed)
        self.assertEqual(benchmark.build_prompt(task), '"""\ninteger square root\nassert root(10) == EXPECTED\n"""\n')

    def test_mbpp_setup_can_instantiate_candidate_class(self):
        benchmark = MBPPBenchmark(spec("mbpp"))
        task = BenchmarkTask("mbpp", "367", "tree", {
            "text": "tree value", "test_imports": [],
            "test_setup_code": "root = Node(7)",
            "test_list": ["assert value(root) == 7"],
        })
        code = "class Node:\n    def __init__(self, x): self.x=x\ndef value(node): return node.x"
        self.assertTrue(benchmark.evaluate(task, code).passed)

    def test_lcb_multiline_function_solution_and_tuple_list(self):
        benchmark = LiveCodeBenchBenchmark(spec("livecodebench"))
        task = BenchmarkTask("livecodebench", "f", "", {
            "metadata": {"func_name": "pair"},
            "public_test_cases": [{"input": "2\n3", "output": "[2, 3]"}],
            "private_test_cases": [], "starter_code": "",
        })
        code = "class Solution:\n    def pair(self, a, b): return (a, b)"
        project = benchmark.evaluate(task, code)
        sample = {"input_output": json.dumps({
            "inputs": ["2\n3"], "outputs": ["[2, 3]"], "fn_name": "pair",
        })}
        from src.evaluation.vendor.livecodebench import testing_util
        with patch.object(testing_util, "reliability_guard", return_value=None):
            reference_results, _ = testing_util.run_test(sample, code)
        self.assertEqual(project.passed, all(item is True for item in reference_results))

    def test_lcb_stdin_decimal_normalization_wrong_and_error(self):
        benchmark = LiveCodeBenchBenchmark(spec("livecodebench"))
        task = BenchmarkTask("livecodebench", "s", "", {
            "metadata": {}, "public_test_cases": [{"input": "", "output": "1.00  2"}],
            "private_test_cases": [], "starter_code": "",
        })
        self.assertTrue(benchmark.evaluate(task, "print('1.0 2.0')").passed)
        self.assertEqual(benchmark.evaluate(task, "print('3')").status, "wrong")
        self.assertEqual(benchmark.evaluate(task, "raise RuntimeError('x')").status, "execution_error")


class LCBProtocolTests(unittest.TestCase):
    def test_official_prompt_and_messages_are_exact(self):
        expected = ("### Question:\nQ\n\n### Format: " + FORMAT_WITHOUT_STARTER +
                    "\n```python\n# YOUR CODE HERE\n```\n\n### Answer: (use the provided format with backticks)\n\n")
        self.assertEqual(generic_question_prompt("Q", ""), expected)
        self.assertEqual(chat_messages("Q", ""), [
            {"role": "system", "content": SYSTEM_MESSAGE_GENERIC},
            {"role": "user", "content": expected},
        ])

    def test_official_last_fence_extraction(self):
        self.assertEqual(extract_lcb_code("plain code"), "")
        self.assertEqual(extract_lcb_code("```python\na=1\n```\ntext\n```python\nb=2\n```"), "b=2")
        self.assertEqual(extract_lcb_code("<think>x</think>\n```python\nprint(1)\n```"), "print(1)")

    def test_compressed_private_decode_requires_verified_hash(self):
        value = base64.b64encode(zlib.compress(pickle.dumps(json.dumps([{"input": "1", "output": "1"}])))).decode()
        with self.assertRaisesRegex(ValueError, "verified pinned asset"):
            decode_test_cases(value, verified_asset=False)
        self.assertEqual(decode_test_cases(value, verified_asset=True)[0]["input"], "1")

    def test_sha_is_checked_before_pickle(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "test6.jsonl")
            path.write_text(json.dumps({"question_id": "1", "question_content": "q",
                                        "public_test_cases": "[]", "private_test_cases": "not-safe"}) + "\n")
            with patch("src.evaluation.lcb_protocol.pickle.loads") as loads:
                with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                    load_verified_v6(path)
                loads.assert_not_called()


class ProfileTests(unittest.TestCase):
    def test_humaneval_and_mbpp_are_greedy_single_trial(self):
        for model in ("klear_agentforge_8b", "granite_4_2_8b"):
            for benchmark in ("humaneval", "mbpp"):
                profile = load_evaluation_profile(model, benchmark)
                self.assertFalse(profile.do_sample)
                self.assertTrue(profile.use_chat_template)
                self.assertEqual(profile.num_trials, 1)

    def test_klear_lcb_sampling_profile(self):
        profile = load_evaluation_profile("klear_agentforge_8b", "livecodebench")
        self.assertEqual((profile.temperature, profile.top_p, profile.top_k, profile.max_new_tokens),
                         (0.6, 0.95, 20, 4096))
        self.assertEqual(dict(profile.chat_template_kwargs), {})

    def test_granite_lcb_thinking_profile(self):
        profile = load_evaluation_profile("granite_4_2_8b", "livecodebench")
        self.assertEqual((profile.temperature, profile.top_p, profile.top_k, profile.max_new_tokens),
                         (1.0, 0.95, 50, 8192))
        self.assertEqual(profile.chat_template_kwargs, {"enable_thinking": True})


class TinyTokenizer:
    eos_token_id = 0
    pad_token_id = None
    def __call__(self, text, return_tensors, truncation=False):
        return {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)}
    def decode(self, tokens, skip_special_tokens=True):
        return "generated"
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
        return "CHAT:" + messages[-1]["content"] + json.dumps(kwargs, sort_keys=True)


class RecordingModel:
    def __init__(self, *, eos_token_id, pad_token_id, top_k=50):
        self.config = SimpleNamespace(max_position_embeddings=10000)
        self.generation_config = SimpleNamespace(
            eos_token_id=eos_token_id, pad_token_id=pad_token_id,
            temperature=0.8, top_p=0.9, top_k=top_k,
            num_beams=1, repetition_penalty=1.0,
        )
        self.device = None
        self.kwargs = None

    def generate(self, input_ids, attention_mask, **kwargs):
        self.kwargs = kwargs
        return torch.cat((input_ids, torch.tensor([[4]])), dim=1)


class TinyGenerationTests(unittest.TestCase):
    def test_profile_generation_preserves_model_eos_and_records_seed(self):
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=16, hidden_size=8, intermediate_size=16,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            head_dim=4, max_position_embeddings=16)).eval()
        profile = load_evaluation_profile("klear_agentforge_8b", "humaneval")
        profile, _ = profile.with_overrides(max_new_tokens=2)
        result = generate_one(model, TinyTokenizer(), "prompt", messages=[{"role": "user", "content": "prompt"}],
                              profile=profile, seed=7)
        self.assertEqual(result["raw_generation"], "generated")
        self.assertEqual(result["seed"], 7)
        self.assertNotIn("eos_token_id", model.generation_config.to_diff_dict() if hasattr(model.generation_config, "to_diff_dict") else {})
        self.assertTrue(result["chat_template_used"])
        effective = result["effective_generation_config"]
        self.assertEqual((effective["temperature"], effective["top_p"], effective["top_k"]),
                         (None, None, None))

    def test_sampling_effective_config_records_profile_and_model_stop_policy(self):
        cases = (
            ("klear_agentforge_8b", [151645, 151643], 151643, 20),
            ("granite_4_2_8b", 49152, 0, 50),
        )
        for model_id, eos_token_id, pad_token_id, expected_top_k in cases:
            with self.subTest(model=model_id):
                model = RecordingModel(eos_token_id=eos_token_id, pad_token_id=pad_token_id)
                profile = load_evaluation_profile(model_id, "livecodebench")
                profile, _ = profile.with_overrides(max_new_tokens=2)
                result = generate_one(
                    model, TinyTokenizer(), "prompt", messages=[{"role": "user", "content": "prompt"}],
                    profile=profile, seed=0,
                )
                effective = result["effective_generation_config"]
                self.assertEqual(effective["top_k"], expected_top_k)
                self.assertEqual(model.kwargs["top_k"], expected_top_k)
                self.assertEqual(effective["eos_token_id"], eos_token_id)
                self.assertEqual(effective["pad_token_id"], pad_token_id)
                self.assertTrue(effective["do_sample"])

    def test_prompt_truncation_is_refused(self):
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=16, hidden_size=8, intermediate_size=16,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            head_dim=4, max_position_embeddings=4))
        profile = load_evaluation_profile("klear_agentforge_8b", "humaneval")
        profile, _ = profile.with_overrides(max_new_tokens=2)
        with self.assertRaisesRegex(ValueError, "truncation refused"):
            generate_one(model, TinyTokenizer(), "x", messages=[{"role": "user", "content": "x"}],
                         profile=profile, seed=0)


if __name__ == "__main__":
    unittest.main()
