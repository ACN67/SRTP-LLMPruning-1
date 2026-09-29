# SWE-bench Verified provenance

- Official code: https://github.com/SWE-bench/SWE-bench
- Selected release: `v5.0.1`; commit `87ab1f6ced28f75ba73ca899dc759b019310944a`; Python package `5.0.1`; MIT license.
- Dataset: `SWE-bench/SWE-bench_Verified`, `test`, revision `78f471bf655a3137b2e8a75af1501690ec009ec3`, 500 instances.
- Task repository: https://github.com/SWE-bench/swe-bench-tasks at `3d07b464b7b311a0cbfb5ed5b2d8a3b96f84a33d` when audited.
- Executable contract: the official alias form is `swebench eval verified -p predictions.jsonl --run-id RUN -j WORKERS`; the project wrapper passes its exact-revision local JSON snapshot instead of the mutable alias. `--task-repo` opts into a local v5 task tree.
- Prediction schema accepted by v5: `instance_id`, `model_name_or_path`, `model_patch`. Project generation never passes `patch`, `test_patch`, `FAIL_TO_PASS`, or `PASS_TO_PASS` to the Agent.
- Result contract: v5 schema-2 report fields including `submitted_ids`, `completed_ids`, `resolved_ids`, `unresolved_ids`, `error_ids`, and `incomplete_ids`.

The former `v4.0.4` planned configuration is historical and superseded. v5 moves the source-of-truth build recipe into task repositories while retaining dataset aliases and the legacy module entry points. Software smoke tests validate command/schema parsing; only the official Docker harness can establish resolved rate.

Sources: https://github.com/SWE-bench/SWE-bench/releases/tag/v5.0.1, https://github.com/SWE-bench/SWE-bench/blob/v5.0.1/docs/reference/cli.md, https://www.swebench.com/SWE-bench/guides/datasets/
