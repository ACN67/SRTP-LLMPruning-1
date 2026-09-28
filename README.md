# SRTP LLM 剪枝复现与评测

本仓库用于复现和评测 8B 代码/软件工程模型剪枝。当前目标分为两层：

- **SRTP 结项阶段**：完成 Klear-AgentForge-8B、Granite-4.2-8B 上的剪枝复现，并在 HumanEval、MBPP、LiveCodeBench fine-grained v6 上做固定协议评测。
- **论文阶段**：补充 SWE-Lego-Qwen3-8B、TaBP 候选方法，以及 SWE-bench / SWT-Bench / BFCL / Terminal-Bench 等软件工程 agent 基准。

本仓库目前不声明任何正式 pass@1、resolved rate 或 leaderboard 结果。真实 8B 权重、GPU 剪枝、完整 benchmark 和论文级 agent 评测仍需要服务器验证。

## 当前支持范围

| 类别 | 已实现 | 规划中 |
| --- | --- | --- |
| 模型 | Klear-AgentForge-8B、Granite-4.2-8B | SWE-Lego-Qwen3-8B |
| 剪枝 | Magnitude、Wanda、SparseGPT、SLEB | TaBP-SSN，DDF 待澄清 |
| SRTP benchmark | HumanEval original 164、MBPP original test 500、LiveCodeBench fine-grained v6 175 | 无 |
| 论文 benchmark | 无 | SWE-bench Verified/Live、SWT-Bench Verified、BFCL V4 Agentic、Terminal-Bench 2.1 子集 |

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

国内默认模型 transport 是 ModelScope，科学身份仍然是 canonical Hugging Face repo + exact commit。下载器会按仓库内 snapshot manifest 逐文件校验 size/hash，使用 `.part` 和 atomic rename；国内失败不会自动切换官方 HF。

```bash
python3 scripts/setup/download_models.py --all --root /data/models
python3 scripts/setup/download_models.py --model granite_4_2_8b \
  --download-source official --root /data/models

python3 scripts/setup/verify_model_snapshot.py \
  --model klear_agentforge_8b \
  --path /data/models/klear_agentforge_8b
```

只验证小文件链路，避免下载权重 shard：

```bash
python3 scripts/setup/download_models.py \
  --all --small-file-smoke --root /tmp/model-smoke
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

`--local-files-only --datasets-root /data/datasets` 会从已验证的本地 C4 JSON gzip 或 WikiText parquet 建立 calibration dataset。Magnitude、Wanda、SparseGPT 保存的是普通 dense tensor 中的零值；没有稀疏 kernel 时不应宣称推理加速。SLEB 会物理删 block，需要 reduced-depth artifact loader 校验。

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

## 论文阶段边界

SWE-bench、SWT-Bench、BFCL、Terminal-Bench 不应直接扩进当前 `run_benchmark.py`。它们需要独立 harness、容器隔离、任务 revision lock、成本记录和更复杂的指标。

论文阶段的建议顺序：

1. 补 SWE-Lego-Qwen3-8B 的 model config、snapshot manifest、evaluation profile。
2. 跑三模型 dense baseline。
3. 复现并澄清 TaBP，上游 DDF 排序语义未确认前只实现 TaBP-SSN 候选。
4. 用小规模预注册集合筛选 Pareto 点。
5. 只把少量 Pareto artifact 投入完整 SWE/SWT agent benchmark。

## 安全说明

模型生成的 Python 是不可信代码。当前 reliability guard 不是安全沙箱。

建议 GPU 环境只负责模型 generation；CPU evaluator 使用 disposable container，并配置 `--network none`、`--cap-drop ALL`、`no-new-privileges` 及 CPU/memory/PID/wall-time limits。AutoDL 普通容器实例不适合无隔离执行模型生成代码。

## 尚待服务器验证

1. 完整下载并验证两个已接入 8B snapshot。
2. 运行 CUDA/GPU preflight 与 dense load/forward/generate smoke。
3. Prefetch 并验证真实 benchmark/calibration assets。
4. 验证四类 pruned artifact，尤其 SLEB reduced checkpoint。
5. 先运行 `--limit 2`，再根据资源执行 full benchmark。
6. 补 SWE-Lego-Qwen3-8B、TaBP 和论文级 agent benchmark。
