# Wanda upstream audit

## Paper and official implementation

- Mingjie Sun, Zhuang Liu, Anna Bair, and J. Zico Kolter, **A Simple and Effective Pruning Approach for Large Language Models**, ICLR 2024.
- [Paper, arXiv:2306.11695](https://arxiv.org/abs/2306.11695)
- [Official project page](https://eric-mingjie.github.io/wanda/home.html)
- [Official repository: locuslab/wanda](https://github.com/locuslab/wanda)
- License: MIT.
- Suggested upstream pin: [`8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6`](https://github.com/locuslab/wanda/tree/8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6) (audited repository HEAD on 2026-09-25; commit date 2023-11-03).

The official README reports LLaMA/LLaMA-2 support and a separate OPT path. It also carries Magnitude and SparseGPT baselines. It does not claim Qwen3 or Granite support.

## Core source and algorithm flow

Core files and symbols:

- [`main.py:get_llm/main`](https://github.com/locuslab/wanda/blob/8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6/main.py): model loading, CLI dispatch, evaluation, and saving.
- [`lib/prune.py:prepare_calibration_input`](https://github.com/locuslab/wanda/blob/8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6/lib/prune.py#L58-L95): replaces the first block with a catcher to capture hidden states and forward kwargs.
- [`lib/prune.py:prune_wanda`](https://github.com/locuslab/wanda/blob/8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6/lib/prune.py#L127-L210): layer-by-layer calibration replay, mask creation, and pruning.
- [`lib/layerwrapper.py:WrappedGPT`](https://github.com/locuslab/wanda/blob/8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6/lib/layerwrapper.py): per-input-channel activation norm accumulation.
- [`lib/data.py`](https://github.com/locuslab/wanda/blob/8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6/lib/data.py): WikiText2/C4 loading and sampling.
- `lib/prune_opt.py`: duplicated OPT-specific traversal.

Execution flow:

1. Load a Hugging Face causal LM in FP16 with `device_map="auto"` and set `model.seqlen` to `config.max_position_embeddings`.
2. Sample 128 C4 sequences by default.
3. Capture first-block inputs, `attention_mask`, and `position_ids` in fixed tensors.
4. For each transformer block, attach hooks to every exact `nn.Linear`, replay calibration inputs, and accumulate input-channel squared L2 norms.
5. Score each weight as `abs(weight) * sqrt(input_channel_statistic)`.
6. Select the lowest scores per output row for unstructured pruning, or per N:M group, then set those weights to zero.
7. Replay the pruned block so the next block is calibrated on propagated pruned activations.

Wanda is weight-level pruning. It does not alter head count, MLP width, hidden size, or block count.

## Information and resource requirements

- Calibration: required; official path hardcodes C4 in `prune_wanda`.
- Activation information: required, via forward hooks.
- Hessian: not used.
- Gradients/retraining/weight update: not used by the main Wanda method.
- Default calibration: 128 samples; sequence length inherited from `model.seqlen`.

The sequence-length rule is unsafe for the target models. The code allocates two `[128, seqlen, hidden_size]` buffers. With BF16/FP16 and hidden size 4096, one buffer is roughly 64 GiB at Klear’s 65,536-token maximum and 128 GiB at Granite’s 131,072-token maximum, before the second buffer, weights, hooks, and other activations. The adapter must use an explicit practical calibration length such as 1,024 or 2,048, selected as an experiment parameter rather than inferred from maximum context.

## Architecture and API assumptions

- `model.model.layers`, with a separate duplicated OPT implementation.
- `model.hf_device_map` and keys such as `model.embed_tokens` / `model.layers.{i}`.
- First-block kwargs always contain `attention_mask` and `position_ids`.
- Direct decoder calls accept only `attention_mask` and `position_ids` and return hidden states at index 0.
- Only modules whose exact type is `nn.Linear` are selected.
- Old dependency stack: Python 3.9, PyTorch 1.10.1 + CUDA 11.3, Transformers 4.28.0, datasets 2.11.0, accelerate 0.18.0.

Current Qwen3 and Granite model forwards compute shared rotary `position_embeddings` and `cache_position` before invoking each block. Passing only `position_ids` leaves `position_embeddings=None`; current attention code then attempts to unpack it. Therefore the apparent shared `model.model.layers` path is insufficient.

## Saving and reload

`main.py` optionally calls both `model.save_pretrained` and `tokenizer.save_pretrained`. Because Wanda stores zeros in otherwise standard dense tensors, standard Hugging Face save/reload should work if the original architecture class and config revision are used. Upstream does not include a clean-process equivalence test or sparse runtime format. Saving dense zero tensors does not automatically reduce checkpoint size or inference latency.

## Target-model adaptation

### `Kwai-Klear/Klear-AgentForge-8B` / Qwen3

Difficulty: **Medium**.

- Block path matches, but Qwen3 requires current rotary/cache kwargs for direct block replay.
- It uses GQA (32 query heads / 8 KV heads) and Q/K normalization. Wanda’s per-linear metric is not inherently MHA-specific, but Q/K/V shape coverage is required.
- Maximum-context-derived calibration allocation is unacceptable.
- A Qwen3 adapter should expose blocks, embeddings, native block-forward kwargs, layer devices, and target linear modules without copying Qwen3 forward code.

### `ibm-granite/granite-4.2-8b` / Granite

Difficulty: **Medium**.

- Block path also matches, but current Granite has the same shared rotary/cache requirement.
- Granite includes residual and attention multipliers; replay must invoke the native block forward so these semantics remain intact.
- It also uses GQA (32/8) and has an even longer maximum context.
- Use a Granite adapter rather than treating it as old LLaMA merely because the module names look similar.

## Recommendation

Directly port the compact `WrappedGPT` statistics and Wanda masking logic under MIT attribution. Replace upstream model loading, dataset code, fixed activation allocation, device-map string logic, and decoder replay with project-owned components. Before claiming support, run calibration-capture, target-module enumeration, sparsity, logits, save/reload, and peak-memory tests on small models or tiny randomly initialized configs of the same architecture.

## Implemented project adaptation

The project implementation is independent code based on the algorithm and execution semantics audited at upstream commit `8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6`. The official repository is MIT licensed; no upstream package or complete source file is vendored.

Official invariants retained:

- score is `abs(weight) * sqrt(sum(input_activation ** 2))` per input channel;
- comparison and stable exact-k selection are independent per output row;
- calibration data is required;
- blocks are processed sequentially and the next block receives the preceding pruned block's output;
- masking only sets selected original weights to zero;
- no reconstruction, weight update, gradients, or retraining;
- only basic unstructured Wanda is implemented, not the alpha-search variant or N:M modes.

Project engineering adaptations:

- Qwen3 and Granite traversal remains adapter-owned.
- Native backbone forward hooks capture the actual block positional arguments and kwargs rather than reconstructing RoPE, causal masks, cache positions, or residual behavior.
- Adapters normalize Qwen3's Tensor block output and Granite's tuple block output.
- Calibration length is explicitly 2048 rather than inherited from maximum context.
- The C4 loader uses the modern `allenai/c4`, `en` datasets API with the official English training shard and preserves random-document/random-contiguous-token-span sampling.
- Nested captured tensors are moved to each native block's device for replay.
- Project summaries and manifests distinguish requested masks, pre-existing zeros, newly zeroed weights, and total post-pruning zeros.

These are architecture and dependency compatibility adaptations, not changes to Wanda's importance metric, per-output grouping, or sequential layer-wise algorithm.

The official `WrappedGPT.scaler_row` divides its running squared statistic by the number of calibration samples. This project accumulates the unnormalized squared L2 sum and takes its square root. The omitted `1 / sqrt(num_samples)` factor is identical for every input channel of a Linear, so stable per-row rankings and masks are exactly equivalent for the fixed-shape, batch-one calibration protocol.
