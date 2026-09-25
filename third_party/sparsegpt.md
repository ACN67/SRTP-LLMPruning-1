# SparseGPT upstream audit

## Paper and official implementation

- Elias Frantar and Dan Alistarh, **SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot**, Proceedings of the 40th International Conference on Machine Learning, PMLR 202:10323-10337, 2023.
- [Official PMLR paper page](https://proceedings.mlr.press/v202/frantar23a.html)
- [Official repository: IST-DASLab/sparsegpt](https://github.com/IST-DASLab/sparsegpt)
- License: Apache-2.0.
- Suggested upstream pin: [`147d2159dc4f3e9f73e47b32c04d7b3708f44436`](https://github.com/IST-DASLab/sparsegpt/tree/147d2159dc4f3e9f73e47b32c04d7b3708f44436) (audited repository HEAD on 2026-09-25; commit date 2024-05-26).

The official repository provides separate scripts for OPT, BLOOM, and LLaMA. The README’s main stated support is OPT and BLOOM; the later LLaMA script follows the same interface. There is no Qwen3 or Granite path.

## Core source and algorithm flow

Core files and symbols:

- [`sparsegpt.py:SparseGPT`](https://github.com/IST-DASLab/sparsegpt/blob/147d2159dc4f3e9f73e47b32c04d7b3708f44436/sparsegpt.py): architecture-light per-layer pruning core.
- `SparseGPT.add_batch`: accumulates an empirical input Hessian approximation `H += X X^T` in FP32.
- `SparseGPT.fasterprune`: damping, Cholesky inverse, saliency/mask selection, column-block processing, and sequential error compensation.
- [`llama.py:get_llama/llama_sequential`](https://github.com/IST-DASLab/sparsegpt/blob/147d2159dc4f3e9f73e47b32c04d7b3708f44436/llama.py): LLaMA-specific loading, calibration capture, layer replay, pruning, and optional saving.
- `opt.py` and `bloom.py`: separate architecture-specific orchestration.
- [`datautils.py`](https://github.com/IST-DASLab/sparsegpt/blob/147d2159dc4f3e9f73e47b32c04d7b3708f44436/datautils.py): WikiText2, PTB, and C4 tokenization/sampling.
- `modelutils.py:find_layers`: exact-type traversal for `nn.Conv2d` and `nn.Linear`.

Execution flow:

1. Load an architecture-specific Hugging Face model and calibration dataset, normally 128 sequences of length 2,048.
2. Capture inputs to the first transformer block.
3. Move/replay one block at a time, attach hooks to its linear layers, and build a per-layer FP32 input Gram/Hessian matrix.
4. Add diagonal damping, compute a Cholesky-based inverse factor, select low-saliency weights, and compensate subsequent columns for induced error in blocks of 128 columns.
5. Support unstructured pruning, N:M sparsity, and optional quantization.
6. Propagate the pruned block’s outputs to calibrate the next block.

The pruning granularity is individual weights. It does not remove heads, neurons, hidden dimensions, or transformer blocks.

## Information and resource requirements

- Calibration: required; WikiText2, PTB, or C4 in official scripts.
- Activation information: required.
- Hessian: empirical input Hessian/Gram approximation, one square matrix per active linear layer.
- Gradients/retraining: not required.
- CUDA: effectively required by the reference flow (`DEV = cuda:0`, CUDA synchronization/cache calls).

The Hessian dimension is the linear layer’s input width, independent of output width. For the target models, a 12,288-wide down projection needs about 576 MiB for one FP32 square matrix; a 12,800-wide Granite down projection needs about 625 MiB (binary MiB), before Cholesky workspaces. The reference orchestration builds Hessians for all linear modules in a block concurrently, so peak memory is materially higher than the model layer alone. Exact peak VRAM/RAM must be measured before selecting GPU instances.

## Architecture and API assumptions

- Separate `opt.py`, `bloom.py`, and `llama.py` rather than a model adapter.
- `llama.py` imports `LlamaForCausalLM` directly.
- LLaMA traversal is hardcoded to `model.model.layers`, `model.model.embed_tokens`, and `model.model.norm`.
- LLaMA sequential groups name `self_attn.{q,k,v,o}_proj` and `mlp.{up,gate,down}_proj`.
- Direct block calls pass only `attention_mask`; no current rotary `position_embeddings` or `cache_position` handling.
- Model sequence length is hardcoded to 2,048.
- Dependency baseline is old: PyTorch 1.10.1+cu111, Transformers 4.21.2, and datasets 1.17.0.

The `SparseGPT` class itself accepts ordinary `nn.Linear` and is substantially more portable than the orchestration around it. GQA does not invalidate its matrix math, because Q/K/V projections are pruned separately, but their unequal shapes and memory profiles must be tested.

## Saving and reload

Each architecture script can call `model.save_pretrained`; the LLaMA path does not save the tokenizer. For pruning without the optional custom quantization path, zeroed standard dense tensors should reload through the native Hugging Face class. The project must save tokenizer/config/revision metadata and run a clean-process equivalence test. The dense checkpoint will not automatically be smaller or faster.

## Target-model adaptation

### `Kwai-Klear/Klear-AgentForge-8B` / Qwen3

Difficulty: **High**.

- It cannot load through `LlamaForCausalLM`.
- The block container and common projection names are similar, but direct layer replay lacks Qwen3’s required shared rotary/cache arguments.
- Qwen3 uses GQA and Q/K normalization; native block forward must remain authoritative.
- Hessian and Cholesky memory for 12,288-wide MLP inputs is substantial.
- The core algorithm can be retained; loader, block capture/replay, target selection, device placement, and data pipeline must be replaced.

### `ibm-granite/granite-4.2-8b` / Granite

Difficulty: **High**.

- No Granite-specific official path exists.
- Current Granite block replay also requires shared rotary/cache inputs.
- Native forward contains Granite scaling semantics that must not be duplicated as old LLaMA logic.
- The 12,800-wide intermediate dimension makes the largest Hessian slightly larger than Qwen3’s.
- A Granite adapter plus explicit memory scheduling is required; this is adaptation of orchestration, not a rewrite of SparseGPT’s mathematical core.

## Recommendation

Port the `SparseGPT` per-linear core with Apache-2.0 notices and a record of modifications. Do not import the old architecture scripts wholesale. Reuse the project’s future calibration-capture interface from Wanda, but add a resource planner, per-module processing schedule, numerical-failure reporting for Cholesky, and clean save/reload validation. Keep optional quantization outside the first pruning milestone because it changes artifact compatibility and is not in the current study scope.

