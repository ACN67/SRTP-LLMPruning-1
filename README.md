# SRTP LLM 剪枝复现与评测

## 当前状态

本仓库面向 2 个 8B 模型、4 种剪枝方法和 3 个代码能力 benchmark：

| 类别 | 支持对象 |
|---|---|
| 模型 | Klear-AgentForge-8B、Granite-4.2-8B |
| 剪枝 | Magnitude、Wanda、SparseGPT、SLEB |
| 评测 | HumanEval original（164）、MBPP original（500）、LiveCodeBench v6（175） |

本地已经验证配置、下载/校验逻辑、tiny model、剪枝算法回归和 evaluator。真实 8B 权重、Docker image、GPU、正式 pruning 与 full benchmark 仍待服务器验证，仓库不声明任何正式 pass@1 结果。

## 主流程

```text
准备模型与数据
→ build/load Docker image
→ software/GPU/path preflight
→ pruning 并保存 checkpoint
→ benchmark generate
→ 在隔离环境 benchmark evaluate
```

## 支持对象与配置

- 模型身份与结构：`configs/models/`
- canonical runtime 文件清单：`configs/model_snapshots/`
- 剪枝参数：`configs/pruning/`
- benchmark 数据、prompt、extraction：`configs/eval/`
- model × benchmark generation 参数：`configs/evaluation_profiles/`
- 上游 revision、license 与适配说明：`third_party/`

`max_new_tokens`、sampling、chat template 和 trial 数只由 evaluation profile 管理。相同模型的 dense 与各类 pruned artifact 必须使用同一 profile。

## 本地准备

国内默认模型 transport 是 ModelScope，科学身份仍是 canonical Hugging Face repo + exact commit。下载器按仓库内 runtime snapshot manifest 逐文件做 size/hash 校验，使用 `.part` 和 atomic rename；国内失败不会自动切换官方 HF。
`download_models.py` 只负责下载、校验并写入 transport provenance；已有 snapshot 的只读验证统一使用 `verify_model_snapshot.py`，不会改写 provenance sidecar。

```bash
python scripts/setup/download_models.py --all --root /data/models
python scripts/setup/download_models.py --model granite_4_2_8b \
  --download-source official --root /data/models
python scripts/setup/verify_model_snapshot.py --model klear_agentforge_8b \
  --path /data/models/klear_agentforge_8b
```

只验证国内小文件链路、避免下载权重 shard：

```bash
python scripts/setup/download_models.py --all --small-file-smoke --root /tmp/model-smoke
```

benchmark 资产以及 C4/WikiText-2 calibration 文件默认从 `https://hf-mirror.net` 的 exact resolve URL 直接下载。C4 使用固定 gzip JSONL shard；WikiText-2 使用固定 parquet。两者均做 size/SHA256 校验，不依赖 Hub metadata 或 remote dataset script。

```bash
python scripts/setup/prefetch_assets.py --all --root /data/datasets
python scripts/setup/verify_assets.py --root /data/datasets
```

## Docker / Server

持久化目录约定为 `/data/{models,cache,datasets,checkpoints,results}`。canonical image 通过国内 DaoCloud mirror 获取固定 digest 的 PyTorch 2.7.1 / CUDA 12.8 runtime，pip 默认使用清华镜像，面向 Linux amd64 与 NVIDIA GPU。

```bash
scripts/setup/build_image.sh srtp-llm-pruning:server-ready
scripts/setup/smoke_image.sh srtp-llm-pruning:server-ready software
scripts/setup/smoke_image.sh srtp-llm-pruning:server-ready gpu
python scripts/setup/preflight.py --all --json
```

本地 graduation gate 已实际完成 image build、software smoke 与 RTX 5060 Docker GPU smoke。Docker build 不会下载模型或数据。

## 剪枝

```bash
python scripts/run_experiment.py --model klear_agentforge_8b --pruner magnitude \
  --sparsity 0.2 --execute --local-path /data/models/klear_agentforge_8b \
  --output-dir /data/checkpoints/klear_agentforge_8b/magnitude_s020
```

`--local-files-only --datasets-root /data/datasets` 会从已验证的本地 C4 JSON gzip 或 WikiText parquet 建立 dataset，再交给原 calibration provider。C4 seeded contiguous sampling 和 SLEB shuffle/first rows/join/tokenize 语义不变。

## 代码能力评测

generation 与 evaluator 是强制分离的两个 phase，没有组合执行入口。`--limit N` 仅用于 smoke；正式运行使用完整 pinned task set。`--num-trials N` 生成 repeated pass@1，报告每个 trial、mean 和 std，不称为 pass@N。

```bash
python scripts/run_benchmark.py --phase generate --benchmark livecodebench \
  --model klear_agentforge_8b --artifact-kind dense --artifact-label dense \
  --artifact-path /data/models/klear_agentforge_8b --run-id smoke --limit 2 --offline

python scripts/run_benchmark.py --phase evaluate --benchmark livecodebench \
  --model klear_agentforge_8b --artifact-kind dense --artifact-label dense \
  --run-id smoke --limit 2 --offline
```

## 结果与可复现性

结果写入 `/data/results/<benchmark>/<model>/<artifact>/<run-id>/`。manifest 保存 task identity、prompt/extraction protocol、实际 effective generation config、evaluator timeout、trial/seed、模型/剪枝 provenance、source generation config hash、source Git revision/dirty 状态、软件环境、运行时可获得的 image metadata 与结果 hash。完整 Docker image identity 由 `export_image.sh` 生成的 tar SHA256 和 metadata JSON 保存。模型权重、数据、checkpoint、结果和 secrets 不进入 Git。

依赖分为两类：PyTorch/CUDA、Transformers 与固定 source revisions 是方法/兼容性关键项；Accelerate、datasets 5.0.1、PyYAML、huggingface_hub、NumPy 和 tqdm 是经过 clean resolution 的 reproducible runtime lock。LCB 不依赖 `datasets`。

## 安全说明

模型生成的 Python 是不可信代码。reliability guard 不是安全沙箱。建议 GPU container 只负责 generation；CPU evaluator 使用 disposable container，并配置 `--network none`、`--cap-drop ALL`、`no-new-privileges` 及 CPU/memory/PID/wall-time limits。

## 尚待服务器验证

1. 完整下载并验证两个 8B snapshot。
2. 实际 build/export/load Docker image。
3. 运行 CUDA/GPU preflight 与 dense load/forward/generate smoke。
4. prefetch 并验证真实 benchmark/calibration assets。
5. 验证四类 pruned artifact，尤其 SLEB reduced checkpoint。
6. 先运行 `--limit 2`，再根据资源执行 full benchmark。
