# SWT-Bench Verified provenance

- Official code: https://github.com/logic-star-ai/swt-bench
- Selected release: tag `1.3.0`, commit `443e03b385089bf8bd787fb8b79fb9a49b7fb2a0`, MIT license.
- Upstream packaging caveat: the tag is `1.3.0`, but its `pyproject.toml` still declares package version `1.0.1`; both identities are recorded instead of inventing a normalized version.
- Inference dataset: official ZeroShotPlus Verified dataset, canonicalized as `eth-sri/SWT-bench_Verified_bm25_27k_zsp`, `test`, revision `70675e4c5e43093a3e5e60cf6128f9757471ab40`, 433 instances. It is used for selection, `problem_statement`, and Agent generation only; `text` and `hits` are not injected into the canonical task prompt.
- Evaluation source: `SWE-bench/SWE-bench_Verified`, `test`, revision `78f471bf655a3137b2e8a75af1501690ec009ec3`, 500 instances. Setup removes the 67 IDs in the pinned harness's `dataset/filter_cases_verified.txt`, producing `.agent_benchmarks/datasets/swtbench_verified_eval/test.json` with 433 rows and exactly the same ID set as inference.
- Derivation identity: filter SHA256 `726776a13721dc01e4fbeee5cb7f3c3068649e2b32f8be5282e498658aacf58b`; derived output SHA256 `aed655aa72d2d7316817bf1b131b8dd3c2a12db163c1624576108c073f30b559`. The sidecar additionally records source and inference hashes plus SWT harness revision.
- Task: generate a reproducing test patch, not a product fix. The selected protocol is `unit_test`, where applicability and successful fail-before/pass-after behavior are evaluated by SWT's Docker harness. `reproduction_script` remains a distinct optional upstream mode and is not the project default.
- Prediction schema: `instance_id`, `model_name_or_path`, `model_patch`, optional `full_output`.
- Evaluator: `python -m src.main --dataset_name .agent_benchmarks/datasets/swtbench_verified_eval/test.json --predictions_path ... --exec_mode unit_test --patch_types vanilla --run_id RUN`. It intentionally omits `--is_swt true` so the original SWE fields mean `patch` = code fix and `test_patch` = reproducing test. It also omits `--filter_swt` because local JSON returns before the pinned harness's runtime filter and the snapshot is already deterministically filtered.
- Gold smoke uses the same derived original-SWE snapshot: substitute `--predictions_path gold` and pass one `--instance_ids INSTANCE`.

The default protocol is `project_raw_swt_harness_protocol`: a uniform cross-Agent-system raw patch is passed to the pinned SWT harness. The project does not silently apply OpenHands-specific setup/config filtering or newer test-only patch stripping. Those evaluation-time transformations can change historical scores, so results from this protocol are not claimed to be numerically leaderboard-equivalent to the current OpenHands wrapper recipe.

The upstream recommendation is x86_64, about 120 GB free storage, 16 GB RAM, and 8 CPU cores. Local software tests stub evaluator execution; official success/applicability values require the pinned Docker harness.

Sources: https://github.com/logic-star-ai/swt-bench/tree/1.3.0, https://github.com/logic-star-ai/swt-bench#running-evaluation, https://huggingface.co/datasets/eth-sri/SWT-bench_Verified_bm25_27k_zsp, https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified, https://github.com/OpenHands/benchmarks/issues/344, https://github.com/OpenHands/benchmarks/issues/718
