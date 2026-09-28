# SRTP LLM 剪枝复现与评测

本仓库用于复现和评测 8B 代码/软件工程模型剪枝。当前目标分为两层：

- **SRTP 结项阶段**：完成 Klear-AgentForge-8B、Granite-4.2-8B 上的剪枝复现，并在 HumanEval、MBPP、LiveCodeBench fine-grained v6 上做固定协议评测。
- **论文非 Agent 阶段**：接入 SWE-Lego-Qwen3-8B、TaBP 与 dense/pruned 成对分析。
- **后续 Agent 阶段**：单独设计 serving、SWE-bench、SWT-Bench 等软件工程 Agent 基准。

本仓库目前不声明任何正式 pass@1、resolved rate 或 leaderboard 结果。真实 8B 权重、GPU 剪枝、完整 benchmark 和论文级 agent 评测仍需要服务器验证。

## 当前支持范围

| 类别 | 已实现 | 规划中 |
| --- | --- | --- |
| 模型 | Klear-AgentForge-8B、Granite-4.2-8B、SWE-Lego-Qwen3-8B | 无 |
| 剪枝 | Magnitude、Wanda、SparseGPT、SLEB、TaBP（内部支持 SSN / DDF） | 无 |
| 分析 | task-level paired bootstrap、retention/delta、Pareto helper | repo-cluster bootstrap、serving TTFT/TPOT |
| SRTP benchmark | HumanEval original 164、MBPP original test 500、LiveCodeBench fine-grained v6 175 | 无 |
| 论文 benchmark | 无 | SWE-bench Verified/Live、SWT-Bench Verified、BFCL V4 Agentic、Terminal-Bench 2.1 子集 |

TaBP 在 registry 中仍是一个 pruning family；后续 empirical study 会将 TaBP-SSN 与 TaBP-DDF 作为两个实验条件，而不是两个顶层算法 family。

当前代码已经覆盖配置注册、模型 snapshot manifest、下载校验、tiny model 剪枝回归、artifact manifest、三项代码生成 benchmark 的 generate/evaluate 分离流程。完整 8B 服务器实验尚待执行。

更多规划见：

- `docs/RESEARCH_PLAN.md`
- `docs/DEPLOYMENT.md`

## 主流程

```text
准备模型与数据
→ 构建或准备运行环境
→ software/GPU/path preflight
→ pruning 并保存 checkpoint
→ benchmark generate
→ 在隔离环境 benchmark evaluate
→ 归档 manifest、结果和环境身份
```

## 目录与配置

- 模型身份与结构：`configs/models/`
- runtime snapshot 清单：`configs/model_snapshots/`
- 剪枝参数：`configs/pruning/`
- benchmark 数据、prompt、extraction：`configs/eval/`
- model × benchmark generation 参数：`configs/evaluation_profiles/`
- 上游 revision、license 与方法说明：`third_party/`

持久化目录约定为：

```text
/data/models
/data/cache
/data/datasets
/data/checkpoints
/data/results
```

模型权重、benchmark 数据、checkpoint、结果和 secrets 不进入 Git。

## 本地与服务器准备

建议使用 Linux/WSL2 或服务器 Linux 环境。普通 Windows 本地不适合作为正式剪枝和 benchmark 运行环境。

安装依赖：

```bash
python3 -m pip install \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  -r requirements.txt
```

基础检查：

```bash
python3 -c "import src.models; print('import-ok')"
python3 scripts/setup/preflight.py --software-only
python3 scripts/setup/preflight.py --paths
python3 scripts/setup/preflight.py --gpu
```

如果使用支持 Docker 的服务器：

```bash
scripts/setup/build_image.sh srtp-llm-pruning:server-ready
scripts/setup/smoke_image.sh srtp-llm-pruning:server-ready software
scripts/setup/smoke_image.sh srtp-llm-pruning:server-ready gpu
```

AutoDL 普通容器实例不支持嵌套 Docker，推荐用平台自定义镜像保存系统盘环境，并把模型、数据和结果放在 `/root/autodl-tmp/srtp`，再通过 `/data/...` 软链接兼容项目路径。详细步骤见 `docs/DEPLOYMENT.md`。

## 模型和数据

国内默认模型 transport 是 ModelScope，科学身份仍然是 canonical Hugging Face repo + exact commit。下载器会按仓库内 snapshot manifest 逐文件校验 size/hash，使用 `.part` 和 atomic rename；国内失败不会自动切换官方 HF。SWE-Lego 没有经过核实的 ModelScope 镜像，必须显式选择官方源。

```bash
python3 scripts/setup/download_models.py --all --root /data/models
python3 scripts/setup/download_models.py --model granite_4_2_8b \
  --download-source official --root /data/models
python3 scripts/setup/download_models.py --model swe_lego_qwen3_8b \
  --download-source official --root /data/models

python3 scripts/setup/verify_model_snapshot.py \
  --model klear_agentforge_8b \
  --path /data/models/klear_agentforge_8b
```

只验证小文件链路，避免下载权重 shard：

```bash
python3 scripts/setup/download_models.py --model klear_agentforge_8b \
  --small-file-smoke --root /tmp/model-smoke
python3 scripts/setup/download_models.py --model swe_lego_qwen3_8b \
  --download-source official --small-file-smoke --root /tmp/model-smoke
```

benchmark 资产以及 C4/WikiText-2 calibration 文件默认从固定 exact resolve URL 下载并校验：

```bash
python3 scripts/setup/prefetch_assets.py --all --root /data/datasets
python3 scripts/setup/verify_assets.py --root /data/datasets
```

## 剪枝

示例：

```bash
python3 scripts/run_experiment.py \
  --model klear_agentforge_8b \
  --pruner magnitude \
  --sparsity 0.3 \
  --execute \
  --local-path /data/models/klear_agentforge_8b \
  --local-files-only \
  --datasets-root /data/datasets \
  --output-dir /data/checkpoints/klear_agentforge_8b/magnitude_s030
```

`--local-files-only --datasets-root /data/datasets` 会从已验证的本地 C4 JSON gzip 或 WikiText parquet 建立 Wanda/SparseGPT/SLEB calibration dataset。TaBP 默认使用固定 revision 的 ARC-Easy，以及 `ssn/latter/entropy/qa/1024/frozen` canonical 配置；可选 DDF 保留 pinned upstream 的 QA/text-generation 分支语义。离线执行需要显式提供 `--tabp-dataset-path`。Magnitude、Wanda、SparseGPT 保存的是普通 dense tensor 中的零值；没有稀疏 kernel 时不应宣称推理加速。SLEB 与 TaBP 会物理删 block，需要 reduced-depth artifact loader 校验。

## 代码生成评测

generation 与 evaluator 是两个独立 phase，没有组合执行入口。`--limit N` 只用于 smoke；正式运行必须使用完整 pinned task set。

```bash
python3 scripts/run_benchmark.py \
  --phase generate \
  --benchmark livecodebench \
  --model klear_agentforge_8b \
  --artifact-kind dense \
  --artifact-label dense \
  --artifact-path /data/models/klear_agentforge_8b \
  --run-id smoke \
  --limit 2 \
  --offline

python3 scripts/run_benchmark.py \
  --phase evaluate \
  --benchmark livecodebench \
  --model klear_agentforge_8b \
  --artifact-kind dense \
  --artifact-label dense \
  --run-id smoke \
  --limit 2 \
  --offline
```

结果写入：

```text
/data/results/<benchmark>/<model>/<artifact-label>/<run-id>/
```

manifest 会保存 task identity、prompt/extraction protocol、effective generation config、trial/seed、模型/剪枝 provenance、Git revision/dirty 状态、软件环境和结果 hash。

## Direct 与 Agent evaluation

两条评测路径保持独立：

```text
Direct: artifact -> Transformers -> HumanEval / MBPP / LiveCodeBench
Agent:  artifact -> vLLM -> model-specific canonical Agent -> repository sandbox -> SWE / SWT
```

`configs/systems/` 与 `src/agent_evaluation/` 当前只提供三模型 canonical system identity、已知/未知参数 provenance 和 config-only validation。Agent serving、runner、repository sandbox 及 SWE/SWT evaluator 尚未实现；Direct benchmark 不 import OpenHands、mini-swe-agent-plus 或 vLLM。

评测会额外写出包含所有任务结果的 `outcomes.jsonl`。可对 identity 完全一致的 dense/pruned run 做 task-level paired bootstrap：

```bash
python3 scripts/analyze_results.py \
  --dense-outcomes /data/results/.../dense/.../outcomes.jsonl \
  --pruned-outcomes /data/results/.../pruned/.../outcomes.jsonl
```

Direct generation runtime 只记录 `model.generate` 的墙钟时间、prompt/generated token 数、generated tokens/s、CUDA peak allocated VRAM（无 CUDA 时为 `null`）和 artifact checkpoint bytes。剪枝执行另外记录 pruner call 墙钟时间、该调用的 peak allocated VRAM，以及不含 `pruning_manifest.json` 的 checkpoint bytes。TTFT、TPOT、serving/concurrent throughput 未测量，manifest 以 `not_measured_without_serving_layer` 明确标记。没有 sparse kernel 时，不从 Magnitude/Wanda/SparseGPT 的零权重推断加速。

## 论文阶段边界

SWE-bench、SWT-Bench、BFCL、Terminal-Bench 不应直接扩进当前 `run_benchmark.py`。它们需要独立 harness、容器隔离、任务 revision lock、成本记录和更复杂的指标。

当前 non-Agent 代码不提供 serving、Agent scaffold、SWE/SWT harness、Agent manifest、repo-cluster bootstrap 或 Docker-capable evaluator。这些能力不能由 Direct benchmark 路径隐式替代。

## 安全说明

模型生成的 Python 是不可信代码。当前 reliability guard 不是安全沙箱。

建议 GPU 环境只负责模型 generation；CPU evaluator 使用 disposable container，并配置 `--network none`、`--cap-drop ALL`、`no-new-privileges` 及 CPU/memory/PID/wall-time limits。AutoDL 普通容器实例不适合无隔离执行模型生成代码。

## 尚待服务器验证

1. 完整下载并验证三个已接入 8B snapshot。
2. 运行 CUDA/GPU preflight 与 dense load/forward/generate smoke。
3. Prefetch 并验证真实 benchmark/calibration assets。
4. 验证五类 pruned artifact，尤其 SLEB/TaBP reduced checkpoint。
5. 先运行 `--limit 2`，再根据资源执行 full benchmark。
6. 在单独阶段设计 serving 与论文级 Agent benchmark。
