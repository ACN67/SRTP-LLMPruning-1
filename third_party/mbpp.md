# MBPP provenance

- Dataset upstream: https://github.com/google-research/google-research
- Dataset commit: `d36068b845da4c2b24927fee2cea1e6ef98dadda`
- File: `mbpp/mbpp.jsonl`
- File SHA256: `ccf64ceae9c5403bf50a044cb6d505bfd2a2963ee58338ba268fd65beab92a9f`
- Formal test set: task IDs 11 through 510 inclusive, exactly 500 tasks.
- Evaluator reference: BigCode evaluation harness commit `8fc5bae6479c4fbbb28c3f8b644f6a15b3f3b5bd`, Apache-2.0.
- Prompt/generation reference also records EvalPlus commit `26d6d00bb1fd0fa37f39c99d5290da67891d1c5e` for current chat-model practice.
- Used semantics: `mbpp_prompt_v1` docstring prompt containing description and first test; execution includes `test_imports`, `test_setup_code`, and every `test_list` assertion, with timeout and pass@1.
- Project profile: chat-templated greedy generation, one trial.
- Dataset licensing and evaluator licensing are separate; consult the upstream dataset metadata before redistribution.
