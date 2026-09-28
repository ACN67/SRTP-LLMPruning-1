# Magnitude Pruning upstream audit

## Identity and provenance

“Magnitude pruning” is a family of criteria rather than a uniquely owned modern algorithm. A canonical neural-network pruning reference is:

- Song Han, Jeff Pool, John Tran, and William J. Dally, **Learning both Weights and Connections for Efficient Neural Network**, NeurIPS 2015.
- [NeurIPS paper page](https://proceedings.neurips.cc/paper/2015/hash/ae0eb3eed39d2bcef4622b2499a05fe6-Abstract.html)
- [arXiv:1506.02626](https://arxiv.org/abs/1506.02626)

The historical paper trains, prunes low-magnitude connections, and retrains. That is not identical to the one-shot, no-retraining LLM baseline intended in this project.

Two code sources must therefore be distinguished:

1. **Historical author repository:** [songhan/Deep-Compression-AlexNet](https://github.com/songhan/Deep-Compression-AlexNet), suggested pin `a4ab6859e1bc86dec3607c8cdd53b0a72da6bcda`, BSD-2-Clause. This is a Caffe/AlexNet compressed-model decoding demo, not reusable LLM pruning code.
2. **Audited LLM baseline:** [`prune_magnitude` in the official Wanda repository](https://github.com/locuslab/wanda/blob/8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6/lib/prune.py#L105-L125), suggested pin `8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6`, MIT. This repository is official for Wanda, not for the historical origin of magnitude pruning; it is used because it defines the directly comparable LLM baseline.

## Audited implementation

Core entry points:

- `main.py`: `get_llm`, CLI selection, evaluation, and optional `model.save_pretrained` / `tokenizer.save_pretrained`.
- `lib/prune.py`: `find_layers`, `prune_magnitude`, and `check_sparsity`.
- `lib/prune_opt.py`: separate OPT traversal.

Execution flow:

1. Select each transformer block through `model.model.layers` (or `model.model.decoder.layers` for OPT).
2. Recursively select modules whose exact type is `torch.nn.Linear`.
3. Compute `abs(weight)`.
4. For unstructured pruning, choose a flattened layer-wise threshold and zero values below it. For N:M, choose the `N` smallest values within each group of `M` input columns for each output row.
5. Mutate dense weight tensors in place by setting masked entries to zero.

Granularity is individual weights, either unstructured or N:M semi-structured. It does not remove attention heads, MLP neurons, hidden dimensions, or blocks.

## Data and compute requirements

- Calibration data: none.
- Activation information: none.
- Hessian/second-order information: none.
- Gradients or training: none for the audited LLM baseline.
- Main memory overhead: one weight metric and mask at a time.

The audited code nevertheless contains a device bug/assumption: the unstructured threshold path calls `W_metric.flatten().cuda()`, forcing CUDA device 0 even when the layer is elsewhere. A project implementation must compute the threshold on the layer’s own device and avoid CPU/GPU round trips.

## Saving and reload

The Wanda entry point uses standard Hugging Face `save_pretrained` for the model and tokenizer. Since pruning only replaces dense weight values with zero, a standard reload should preserve those zeros without custom model classes. This must still be tested by comparing:

- tensor-level zero masks before and after reload;
- model config and tokenizer revision;
- logits on a small fixed input;
- measured sparsity over exactly the intended target modules.

This representation does not physically compress dense tensors and does not imply latency improvement without a sparse kernel/export format.

## Target-model adaptation

### `Kwai-Klear/Klear-AgentForge-8B` / Qwen3

Difficulty: **Low**.

- Qwen3 uses `model.model.layers`, so the current path happens to match.
- GQA produces different Q versus K/V projection shapes, but per-matrix magnitude ranking remains valid.
- The module selector should use an adapter-owned target policy rather than prune every discovered `nn.Linear` blindly.
- Remove the forced `.cuda()` call and support BF16 weights cleanly.

### `ibm-granite/granite-4.2-8b` / Granite

Difficulty: **Low**.

- Current Granite also exposes `model.model.layers` and ordinary linear projections.
- Granite-specific residual/attention scaling is executed only during forward and is not altered by weight masking.
- GQA and unequal projection shapes require coverage tests but no algorithm rewrite.

## Recommendation

Implement the small masking rule directly in the project after defining a shared architecture adapter and explicit target-layer policy. Do not vendor the Caffe demo or the full Wanda repository. Record both the historical paper and the Wanda baseline provenance so the experimental method is not misrepresented as the train-prune-retrain procedure from 2015.

## Implemented project adaptation

The project implementation is original code informed by the audited baseline; no Wanda source was copied. Version 1.0 follows Wanda's ordinary **per-Linear flattened magnitude ranking** semantics while making the selected count exact and deterministic:

- targets only `torch.nn.Linear` weights returned from each adapter-enumerated transformer block;
- flattens each Linear matrix and ranks `abs(weight)` independently from other modules;
- uses stable `argsort` and selects exactly `floor(weight.numel() * ratio)` indices, avoiding threshold tie over-pruning;
- applies masks on the weight's current device and preserves shape, bias, and native forward code;
- reports requested masks separately from pre-existing and newly introduced zeros.

Unlike a threshold comparison that may select too many equal-valued weights, stable exact-k selection resolves ties by row-major flattened index. This preserves the intended requested count while retaining the upstream baseline's per-Linear, non-global granularity.
