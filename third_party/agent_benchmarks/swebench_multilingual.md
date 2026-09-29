# SWE-bench Multilingual provenance

- Official evaluator: SWE-bench `v5.0.1`, commit `87ab1f6ced28f75ba73ca899dc759b019310944a`, package `5.0.1`, MIT license.
- Dataset: `SWE-bench/SWE-bench_Multilingual`, `test`, revision `846e647b9f33c0b51b739d005d13d85493c9af09`.
- Pinned executable identity: 300 tasks, 41 repositories, and 9 languages (C, C++, Go, Java, JavaScript, TypeScript, PHP, Ruby, Rust). The dataset has no language column, so repository-to-language provenance remains explicit in the adapter.
- Task repository: https://github.com/SWE-bench/swe-bench-multilingual-tasks at `6e08cbcd763c869abd7aae4bcb9e1b3991180777` when audited.
- Executable contract: the official alias form is `swebench eval multilingual -p predictions.jsonl --run-id RUN -j WORKERS`; the project wrapper supplies the exact-revision local JSON snapshot to prevent alias drift.
- Prediction and v5 report schemas are the same as Verified. The dataset currently has no language column, so the adapter derives the per-instance language group from the official Appendix A repository table. JavaScript/TypeScript repositories are reported as the official combined group.

The adapter contains no Python-specific build/test assumptions. Dockerfiles and `eval.sh` in the v5 task data remain authoritative. The prior v4 planned pin is superseded.

Sources: https://www.swebench.com/multilingual.html, https://github.com/SWE-bench/SWE-bench/blob/v5.0.1/docs/reference/cli.md, https://huggingface.co/datasets/SWE-bench/SWE-bench_Multilingual
