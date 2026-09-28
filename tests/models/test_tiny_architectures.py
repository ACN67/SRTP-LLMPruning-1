"""Real Transformers Qwen3 and Granite tiny-model integration tests."""

from __future__ import annotations

import tempfile
import unittest

try:
    import torch
    from transformers import (
        GraniteConfig,
        GraniteForCausalLM,
        PretrainedConfig,
        PreTrainedModel,
        Qwen3Config,
        Qwen3ForCausalLM,
    )
    from transformers.modeling_outputs import CausalLMOutput
except ImportError:
    torch = None

from src.models.adapters.granite import GraniteAdapter
from src.models.adapters.qwen3 import Qwen3Adapter
from src.pruning import MagnitudePruner, PruningRequest


if torch is not None:
    class ToyConfig(PretrainedConfig):
        model_type = "magnitude_test_toy"

        def __init__(self, vocab_size=32, hidden_size=16, **kwargs):
            kwargs.setdefault("tie_word_embeddings", False)
            super().__init__(**kwargs)
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size


    class ToyBlock(torch.nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.proj = torch.nn.Linear(hidden_size, hidden_size)

        def forward(self, hidden_states):
            return self.proj(hidden_states)


    class ToyBackbone(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size)
            self.layers = torch.nn.ModuleList(
                [ToyBlock(config.hidden_size), ToyBlock(config.hidden_size)]
            )
            self.norm = torch.nn.LayerNorm(config.hidden_size)


    class ToyHFCausalLM(PreTrainedModel):
        config_class = ToyConfig

        def __init__(self, config):
            super().__init__(config)
            self.model = ToyBackbone(config)
            self.lm_head = torch.nn.Linear(
                config.hidden_size, config.vocab_size, bias=False
            )
            self.post_init()

        def forward(self, input_ids, **kwargs):
            hidden_states = self.model.embed_tokens(input_ids)
            for block in self.model.layers:
                hidden_states = block(hidden_states)
            return CausalLMOutput(logits=self.lm_head(self.model.norm(hidden_states)))


@unittest.skipIf(torch is None, "PyTorch and Transformers are required")
class TinyArchitectureIntegrationTests(unittest.TestCase):
    def _assert_round_trip(self, model, model_class, adapter) -> None:
        model.eval()
        inputs = torch.randint(0, model.config.vocab_size, (1, 8))
        blocks = adapter.get_blocks(model)
        self.assertEqual(len(blocks), 2)
        self.assertTrue(adapter.get_linear_modules(blocks[0]))

        summary = MagnitudePruner().prune(
            model,
            adapter,
            PruningRequest("tiny", "magnitude", 0.25),
        )
        self.assertGreater(summary.number_of_target_modules, 0)
        self.assertGreater(summary.newly_zeroed_weights, 0)
        self.assertEqual(summary.scope, "per_module_flattened")
        self.assertTrue(
            all(
                stats.requested_mask_count
                == int(stats.targeted_weights * summary.sparsity_ratio)
                for stats in summary.per_module
            )
        )
        self.assertEqual(
            summary.requested_mask_count,
            sum(stats.requested_mask_count for stats in summary.per_module),
        )
        before_masks = {
            stats.module: (
                adapter.get_linear_modules(blocks[int(stats.module.split(".")[1])])[
                    stats.module.split(".", 2)[2]
                ].weight
                == 0
            ).cpu().clone()
            for stats in summary.per_module
        }
        with torch.inference_mode():
            logits = model(input_ids=inputs, use_cache=False).logits
        self.assertEqual(tuple(logits.shape), (1, 8, model.config.vocab_size))

        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            reloaded = model_class.from_pretrained(directory).eval()
            reload_blocks = adapter.get_blocks(reloaded)
            for module_name, expected_mask in before_masks.items():
                parts = module_name.split(".", 2)
                actual = adapter.get_linear_modules(reload_blocks[int(parts[1])])[
                    parts[2]
                ].weight == 0
                self.assertTrue(torch.equal(actual.cpu(), expected_mask), module_name)
            with torch.inference_mode():
                reloaded_logits = reloaded(input_ids=inputs, use_cache=False).logits
            self.assertEqual(reloaded_logits.shape, logits.shape)

    def test_tiny_qwen3_prune_forward_save_reload(self) -> None:
        config = Qwen3Config(
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
        self._assert_round_trip(Qwen3ForCausalLM(config), Qwen3ForCausalLM, Qwen3Adapter())

    def test_toy_huggingface_checkpoint_round_trip(self) -> None:
        self._assert_round_trip(
            ToyHFCausalLM(ToyConfig()), ToyHFCausalLM, Qwen3Adapter()
        )

    def test_tiny_granite_prune_forward_save_reload(self) -> None:
        config = GraniteConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            use_cache=False,
        )
        self._assert_round_trip(
            GraniteForCausalLM(config), GraniteForCausalLM, GraniteAdapter()
        )


if __name__ == "__main__":
    unittest.main()
