# Upstream pruning audit

Audit date: 2026-09-25. This directory records source provenance and adaptation risks; it does not vendor upstream code or claim that any method is already compatible with the project models.

## Target models used for the audit

| Project model ID | Hugging Face repository | Audited revision | Architecture facts relevant to pruning |
|---|---|---|---|
| `klear_agentforge_8b` | `Kwai-Klear/Klear-AgentForge-8B` | `fa3d41e92e9ce7a5b4a52a3e7439aa00521f40c9` | `Qwen3ForCausalLM`; 36 blocks; hidden size 4096; intermediate size 12288; 32 attention heads / 8 KV heads (GQA); model config names Transformers 4.51.1 |
| `granite_4_2_8b` | `ibm-granite/granite-4.2-8b` | `f8de16cdcdbc6c779ca517604e050d82cc119e44` | `GraniteForCausalLM`; 40 blocks; hidden size 4096; intermediate size 12800; 32 attention heads / 8 KV heads (GQA); model config names Transformers 4.57.1 |

Both current Hugging Face implementations expose decoder blocks through `model.model.layers`, but both compute shared rotary `position_embeddings` in the model-level forward and pass them, together with `cache_position`, into each block. Old LLaMA orchestration that calls a block with only `attention_mask` and `position_ids` is therefore not directly compatible.

## Comparison

| Method | Paper / publication | Official implementation used for audit | License | Calibration / extra information | Native upstream architectures | Klear / Qwen3 | Granite | Recommended strategy |
|---|---|---|---|---|---|---|---|---|
| Magnitude Pruning | Han et al., NeurIPS 2015 (canonical historical reference; magnitude pruning has no unique single origin) | Historical author demo: `songhan/Deep-Compression-AlexNet`; practical LLM baseline: Wanda repo `prune_magnitude` | BSD-2-Clause for historical demo; MIT for Wanda baseline | None for the one-shot LLM baseline; no activations, Hessian, gradients, or retraining | Historical demo: AlexNet/Caffe; audited LLM baseline: LLaMA/LLaMA-2 and separate OPT path | Low | Low | Implement the small weight-masking rule locally behind the common model adapter; cite the Wanda baseline semantics, not the historical demo as LLM code |
| Wanda | Sun et al., ICLR 2024 | `locuslab/wanda` at `8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6` | MIT | Yes: normally 128 C4 sequences; input activation L2 statistics; no Hessian, gradients, or retraining | LLaMA/LLaMA-2 and OPT | Medium | Medium | Port the compact scoring/masking core; use project-owned calibration and model adapters |
| SparseGPT | Frantar and Alistarh, ICML 2023 | `IST-DASLab/sparsegpt` at `147d2159dc4f3e9f73e47b32c04d7b3708f44436` | Apache-2.0 | Yes: WikiText2/PTB/C4; empirical input Hessian approximation; no gradients or retraining | Separate OPT, BLOOM, and LLaMA scripts | High | High | Port `SparseGPT` layer core under Apache attribution, but replace all architecture/data/device orchestration |
| SLEB | Song et al., ICML 2024 | `jiwonsong-dev/SLEB` at `d07129af60520e751087b8abb04a268a3c7ec861` | MIT; bundled lm-evaluation-harness is separately MIT | Yes: WikiText2 or C4 loss used to score every candidate block; no gradient; many full forward passes | Only OPT and LLaMA name-dispatched wrappers | High | High | Preserve the greedy block-selection idea and wrap it with native-forward-safe block bypass/removal adapters; do not copy the old decoder forwards |

## Cross-cutting findings

- GQA is not itself a blocker for weight-only Magnitude, Wanda, or SparseGPT: they process each `nn.Linear` matrix independently. It is still a required test dimension because Q/K/V shapes differ and old LLaMA forward code predates current GQA/cache APIs.
- The upstream projects do not prune attention head count, MLP width, or hidden size. Magnitude, Wanda, and SparseGPT zero individual weights (unstructured or N:M); SLEB removes complete transformer blocks.
- Wanda and SparseGPT rely on forward hooks and layer-by-layer replay. Their algorithm cores are relatively architecture-neutral, while their orchestration is architecture-specific.
- Standard Hugging Face `save_pretrained` is suitable for zero-valued dense tensors from Magnitude, Wanda, and SparseGPT, subject to an actual save/reload equivalence test. It does not produce a physically sparse checkpoint or automatic inference speedup.
- SLEB upstream does not save a pruned model. Physical block deletion must also update `config.num_hidden_layers`, reindex attention layers where required, save the tokenizer/config, and pass a clean-process reload test.
- The audited dependencies are old: Wanda pins PyTorch 1.10.1 / Transformers 4.28.0; SparseGPT reports PyTorch 1.10.1+cu111 / Transformers 4.21.2 / datasets 1.17.0; SLEB pins PyTorch 2.2.0 / Transformers 4.37.2. None natively covers Qwen3 or Granite 4.2.
- Calibration sequence length must be project-controlled. Wanda assigns `model.seqlen = config.max_position_embeddings`; that would be 65,536 for Klear-AgentForge-8B and 131,072 for Granite-4.2-8B and would make its fixed activation buffers impractical.

## Recommended implementation order

1. Magnitude Pruning: establishes model traversal, target-layer policy, sparsity accounting, and save/reload tests without calibration.
2. Wanda: reuses traversal and adds a shared calibration capture/replay layer.
3. SparseGPT: reuses calibration but adds large per-layer Hessian buffers, numerical damping, and sequential error compensation.
4. SLEB: changes model depth and requires separate block-bypass, config-mutation, cache, generation, and reload validation.

Do not begin an implementation solely from this audit. First decide the calibration corpus/sequence length, target linear-layer policy, artifact format, hardware memory budget, and license-attribution approach.

## Detailed records

- [Magnitude Pruning](magnitude.md)
- [Wanda](wanda.md)
- [SparseGPT](sparsegpt.md)
- [SLEB](sleb.md)

## Primary sources

- [Klear-AgentForge-8B config at audited revision](https://huggingface.co/Kwai-Klear/Klear-AgentForge-8B/blob/fa3d41e92e9ce7a5b4a52a3e7439aa00521f40c9/config.json)
- [Granite-4.2-8B config at audited revision](https://huggingface.co/ibm-granite/granite-4.2-8b/blob/f8de16cdcdbc6c779ca517604e050d82cc119e44/config.json)
- [Transformers Qwen3 implementation, v4.51.1](https://github.com/huggingface/transformers/blob/v4.51.1/src/transformers/models/qwen3/modeling_qwen3.py)
- [Transformers Granite implementation, v4.57.1](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/granite/modeling_granite.py)
