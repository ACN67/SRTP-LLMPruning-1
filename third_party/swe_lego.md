# SWE-Lego-Qwen3-8B upstream audit

## Canonical identity

- Official project: [SWE-Lego/SWE-Lego](https://github.com/SWE-Lego/SWE-Lego)
- Audited project commit: [`94704b69aac886e003660e1e0f69f7de163b284e`](https://github.com/SWE-Lego/SWE-Lego/tree/94704b69aac886e003660e1e0f69f7de163b284e)
- Canonical model repository: [`Lego-X/SWE-Lego-Qwen3-8B`](https://huggingface.co/Lego-X/SWE-Lego-Qwen3-8B). The former `SWE-Lego/...` URL redirects to this repository.
- Immutable model revision: [`00f37992485b78e9579a2674161d512781aee21e`](https://huggingface.co/Lego-X/SWE-Lego-Qwen3-8B/tree/00f37992485b78e9579a2674161d512781aee21e)
- Model license reported by Hugging Face: Apache-2.0.
- No reliable model-specific ModelScope mirror was found during the 2026-09-28 audit. `domestic_modelscope_repo` is therefore `null`; callers must select the official Hugging Face transport explicitly.

The snapshot manifest is derived from the Hugging Face model API with blob metadata enabled at the immutable revision. Ordinary files use their Git blob SHA-1; LFS objects use the published SHA-256 and size. The runtime set contains config, generation config, chat template, tokenizer assets, the safetensors index, and all four referenced weight shards. Documentation images and training arguments are not runtime dependencies.

## Checkpoint structure

The pinned `config.json` declares:

- architecture: `Qwen3ForCausalLM`;
- `model_type`: `qwen3`;
- 36 transformer blocks;
- hidden size 4096;
- intermediate size 12288;
- 32 attention heads and 8 KV heads;
- BF16 weights;
- vocabulary size 151936;
- YaRN factor 4 with original context 40960 and configured maximum 163840;
- standard native Qwen3 layer types (all full attention).

There is no checkpoint evidence of a structural fork from Transformers Qwen3. The project therefore reuses `Qwen3Adapter`; a name-only SWE-Lego adapter would add no behavior and is intentionally not created.

## Generation and Direct benchmark profile

The pinned `generation_config.json` specifies sampling with temperature 0.6, top-p 0.95, and top-k 20. The official SFT recipe uses LLaMA-Factory's `qwen3_nothink` template. The official OpenHands configuration uses temperature 0 and a 16,384-token Agent output budget, but that configuration belongs to the out-of-scope SWE-bench Agent run.

The SRTP Direct profile therefore makes the following explicit choices:

- `enable_thinking: false` for all three Direct benchmarks, matching the no-think SFT template and avoiding Agent reasoning/tool formatting;
- greedy, single-trial HumanEval and MBPP, preserving the repository's fixed Direct policy;
- pinned model sampling values 0.6/0.95/20 for LiveCodeBench;
- a 4096-token LiveCodeBench budget, preserving the Direct code-generation budget rather than importing the Agent run's 16,384-token limit.

Dense and every pruned artifact resolve the same profile by `project_model_id`; pruning does not select or mutate generation parameters.

## Scope boundary

This integration adds model identity, verified snapshot metadata, native Qwen3 loading/pruning/artifact behavior, and HumanEval/MBPP/LiveCodeBench profiles only. It does not vendor or implement OpenHands, SWE-bench, serving, tool schemas, Agent prompts, trajectories, or test-time scaling.
