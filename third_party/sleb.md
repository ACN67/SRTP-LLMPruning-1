# SLEB upstream audit

## Paper and official implementation

- Jiwon Song, Kyungseok Oh, Taesu Kim, Hyungjun Kim, Yulhwa Kim, and Jae-Joon Kim, **SLEB: Streamlining LLMs through Redundancy Verification and Elimination of Transformer Blocks**, Proceedings of the 41st International Conference on Machine Learning, PMLR 235:46136-46155, 2024.
- [Official PMLR paper page](https://proceedings.mlr.press/v235/song24f.html)
- [Official repository: jiwonsong-dev/SLEB](https://github.com/jiwonsong-dev/SLEB)
- License: MIT for SLEB. The vendored EleutherAI lm-evaluation-harness also carries its own MIT license and attribution.
- Suggested upstream pin: [`d07129af60520e751087b8abb04a268a3c7ec861`](https://github.com/jiwonsong-dev/SLEB/tree/d07129af60520e751087b8abb04a268a3c7ec861) (audited repository HEAD on 2026-09-25; commit date 2025-02-04).

The official implementation supports only OPT and LLaMA through explicit name-string dispatch. It does not support Qwen3 or Granite.

## Core source and algorithm flow

Core files and symbols:

- [`sleb.py:sleb`](https://github.com/jiwonsong-dev/SLEB/blob/d07129af60520e751087b8abb04a268a3c7ec861/sleb.py#L61-L208): greedy block search, evaluation, and text result logging.
- [`sleb.py:get_loss`](https://github.com/jiwonsong-dev/SLEB/blob/d07129af60520e751087b8abb04a268a3c7ec861/sleb.py#L16-L58): summed next-token negative log likelihood on calibration text.
- [`utils/onoff_utils/onoff.py`](https://github.com/jiwonsong-dev/SLEB/blob/d07129af60520e751087b8abb04a268a3c7ec861/utils/onoff_utils/onoff.py): dispatch based on whether the model name contains `opt` or `llama`.
- `utils/onoff_utils/onoff_llama.py` and `onoff_opt.py`: copied decoder-layer wrappers with a `pass_layer` bypass.
- [`utils/block_remove.py`](https://github.com/jiwonsong-dev/SLEB/blob/d07129af60520e751087b8abb04a268a3c7ec861/utils/block_remove.py): physical deletion for OPT or LLaMA and LLaMA attention `layer_idx` reindexing.
- `utils/data_utils.py`: WikiText2/C4 loading.
- `eval.py` and `latency.py`: evaluation of a supplied removal list and basic latency comparison.

Execution flow:

1. Load the full model and replace every OPT/LLaMA decoder block with an on/off wrapper.
2. For every removal step, temporarily bypass each eligible remaining block in turn.
3. Run the whole calibration token stream and compute language-model loss.
4. Permanently bypass the candidate producing the lowest loss, then repeat greedily.
5. Convert the bypass list into physical block deletion for evaluation.
6. Append the removal order and metrics to a text file.

Granularity is a complete transformer decoder block. It does not prune individual weights, heads, MLP neurons, or hidden dimensions.

## Information and resource requirements

- Calibration: required; WikiText2 or C4.
- Activation/Hessian/gradient: no stored activation statistics, Hessian, or gradients; the criterion is full-model next-token loss.
- Training: none.
- Compute: high. Removing `R` blocks from `L` candidates requires roughly `R(2L-R-1)/2` candidate evaluations, excluding barriers. For 20% removal this is about 252 candidate evaluations for 36 Qwen3 blocks and 284 for 40 Granite blocks, each over the selected calibration text.
- Memory: full 8B model plus ordinary forward activations; less auxiliary matrix memory than SparseGPT, but substantially more repeated inference.

## Architecture and API assumptions

- Dispatch is based on `model.name.lower()` containing exactly `opt` or `llama`.
- Non-OPT/non-LLaMA models are returned unchanged by `block_replace`; `turn_off`/`turn_on` do nothing; `block_remove` returns `None`.
- `num_blocks` is a manual CLI value (default 32), not derived safely from config.
- The LLaMA wrapper copies an old decoder forward signature and implementation, including old `past_key_value` tuple conventions.
- The wrapper does not handle current shared `position_embeddings`, `cache_position`, or modern `Cache` semantics.
- Physical deletion writes `self_attn.layer_idx` but does not update `config.num_hidden_layers`.
- The conditions `if '30b' or '66b' or '70b' in model_name` in `sleb.py` and `eval.py` are always true in Python, so the zero-shot parallelization decision is buggy.
- Dependencies are pinned to PyTorch 2.2.0, Transformers 4.37.2, datasets 2.16.1, and accelerate 0.26.1.

Copying these old wrapper forwards into Qwen3 or Granite would be fragile and risks changing rotary embeddings, cache behavior, GQA handling, residual scaling, and future Transformers behavior.

## Saving and reload

The official SLEB flow saves only text results/removal lists. It never calls `save_pretrained` for the block-reduced model.

Naively saving after deleting modules is unsafe because the config still advertises the original number of layers. Standard reload can instantiate the old depth and report missing weights or create newly initialized blocks. A supported artifact needs, at minimum:

- physical deletion from the native `ModuleList` after selection;
- updated `config.num_hidden_layers`;
- attention `layer_idx` reindexing where the architecture/cache uses it;
- removal manifest mapping original to retained block indices;
- tokenizer/config/model save;
- clean-process `from_pretrained` and generation tests with and without cache.

Whether a depth-reduced checkpoint is fully reloadable through unmodified upstream Qwen3/Granite classes is plausible but **not verified**.

## Target-model adaptation

### `Kwai-Klear/Klear-AgentForge-8B` / Qwen3

Difficulty: **High**.

- The model name contains neither `llama` nor `opt`, so upstream selection is a no-op and physical removal returns `None`.
- Qwen3’s current native block forward carries shared rotary embeddings and cache positions; the old LLaMA wrapper is incompatible.
- GQA itself is preserved if the native block remains intact, but copied forward/cache logic could break it.
- The configured 36 blocks must be discovered rather than manually passed.
- Use a Qwen3 adapter with an identity/bypass wrapper that preserves the expected return contract under `use_cache=False`, rather than copying the Qwen3 decoder implementation.

### `ibm-granite/granite-4.2-8b` / Granite

Difficulty: **High**.

- The Granite name also misses both upstream dispatch branches.
- Granite’s native decoder includes architecture-specific attention and residual multipliers; an old LLaMA forward wrapper would silently lose those semantics.
- Current rotary/cache requirements and 40-block depth must be handled by the Granite adapter.
- Physical deletion, config mutation, reindexing, saving, and clean reload all require new validation.

## Recommendation

Do not copy the upstream decoder wrappers. Retain the paper’s greedy loss-based selection and removal-list semantics, but build project-owned architecture adapters that:

1. discover native blocks and depth;
2. temporarily bypass a block without reimplementing its normal forward;
3. score with `use_cache=False` using a bounded, versioned calibration corpus;
4. physically delete selected blocks only after search;
5. update config and layer indices;
6. verify native Hugging Face save/reload and generation.

This is a reasonable wrapper/adapter adaptation of the algorithm, but it is the highest-risk method in the first implementation batch because it changes model structure and has a large search cost.

