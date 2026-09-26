# LiveCodeBench provenance

- Code upstream: https://github.com/LiveCodeBench/LiveCodeBench
- Code commit: `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24`
- Code license: MIT；完整 notice 见 `third_party/licenses/livecodebench_LICENSE`。
- Dataset: `livecodebench/code_generation_lite`
- Dataset revision: `0fe84c3912ea0c4d4a78037083943e8f0c4dd505`
- Fixed asset: fine-grained `v6` file `test6.jsonl`, SHA256 `bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5`, 175 unique tasks.
- Local runtime path: `/data/datasets/livecodebench/v6/test6.jsonl`; it is read directly and is independent of the `datasets` package.
- Vendored minimal evaluator: `src/evaluation/vendor/livecodebench/testing_util.py`, adapted from the pinned MIT source. Project wrappers add process isolation/status mapping; the full hash is checked before private pickle decoding.
- Prompt/chat adaptation preserves the pinned generic system/user content. Extraction uses the final fenced block and returns empty when no complete fence exists.
- Project generation uses model-by-benchmark profiles and repeated pass@1 trials. Default is one trial; it is not equivalent to leaderboard/model-card repeated-sampling scores.
