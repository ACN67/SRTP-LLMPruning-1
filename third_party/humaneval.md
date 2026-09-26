# HumanEval provenance

- Upstream: https://github.com/openai/human-eval
- Pinned commit: `6d43fb980f9fee3c892a914eda09951f772ad10d`
- Evaluator code and dataset license: MIT；完整 notice 见 `third_party/licenses/human_eval_LICENSE`。
- Used semantics: `HumanEval.jsonl.gz`, 164 unique tasks, official prompt plus suffix completion, `check(entry_point)`, timeout, and pass@1.
- Asset SHA256: `b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef`.
- Evaluator: minimally vendored official execution core from the pinned MIT source.
- Prompt/generation reference: EvalPlus commit `26d6d00bb1fd0fa37f39c99d5290da67891d1c5e`; project chat adaptation asks for one complete runnable fenced solution and accepts either full-function or suffix output.
- Project profile: chat-templated greedy generation, one pass@1 trial. The reliability guard is not a security sandbox.
- Pending: comparison on the prefetched real asset and real 8B generations.
