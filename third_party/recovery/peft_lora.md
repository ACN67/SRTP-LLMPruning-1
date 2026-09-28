# PEFT LoRA Recovery

## Pinned implementation

- Upstream: `huggingface/peft`
- Release: `0.18.1`
- Git revision: `e3398fc05c556f64a8e8dc248628bae657107488`
- Project dependency: `peft==0.18.1`
- Official release: https://github.com/huggingface/peft/releases/tag/v0.18.1
- Official LoRA reference: https://huggingface.co/docs/peft/main/package_reference/lora

The project uses PEFT's `LoraConfig`, `get_peft_model`, `PeftModel.from_pretrained`, and `merge_and_unload` rather than reimplementing LoRA layers. PEFT documents `merge_and_unload` as producing a standalone base-model architecture with adapter deltas merged into the model.

## Project role

LoRA is a **Recovery** transformation, not a pruning method. A recovery run consumes a canonical `ModelArtifact` and produces a non-standalone PEFT adapter artifact and, only when explicitly requested, a standalone merged artifact.

`configs/recovery/lora.yaml` is an implementation smoke baseline, not the final experimental protocol. The final recovery corpus, rank/alpha/dropout, training budget, and which pruning conditions receive recovery remain research-design decisions.

## Target modules

The implementation validates configured target-module suffixes against the actual loaded architecture and fails if any configured target has no match. The current implementation baseline covers the standard Qwen3/Granite attention and MLP projections:

`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`.

Tiny Qwen3 and Granite integration tests verify real PEFT adapter attachment, optimizer updates, save/reload, forward, and generation.

## Sparse merge safety

Ordinary LoRA merging applies a dense low-rank delta to the base weight. For weight-sparse artifacts produced by Magnitude, Wanda, or SparseGPT, this can refill zeroed weights and change the achieved sparsity. Therefore:

- `adapter_only` is always allowed;
- `standard_merge` is rejected by default for weight-sparse artifacts;
- an explicit `allow_sparse_merge=true` opt-in is required to perform such a merge;
- the recovery manifest records zero count and sparsity before/after the merge.

For reduced-depth SLEB/TaBP artifacts, a standard merge is allowed but the decoder-block count must remain unchanged across merge and reload.

## Dataset boundary

Recovery data is independent from pruning calibration data and benchmark evaluation data. The implementation supports pinned Hugging Face datasets and local JSON/JSONL/Parquet inputs with `text`, `prompt_completion`, or `messages` schemas. Dataset identity, revision/file hash, split, deterministic sampling seed, formatter version, and sample count are recorded. Formal experiments must additionally perform contamination auditing against the benchmark sets; implementation-smoke data is marked `not_run_for_implementation_smoke`.
