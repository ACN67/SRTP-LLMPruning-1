# SRTP LLM 剪枝复现与评测

本仓库用于复现和评测 8B 代码/软件工程模型剪枝。当前目标分为两层：

- **SRTP 结项阶段**：完成 Klear-AgentForge-8B、Granite-4.2-8B 上的剪枝复现，并在 HumanEval、MBPP、LiveCodeBench fine-grained v6 上做固定协议评测。
- **论文非 Agent 阶段**：接入 SWE-Lego-Qwen3-8B、TaBP 与 dense/pruned 成对分析。
- **Agent 研究层**：三套 canonical Agent + vLLM runner，以及 SWE-bench Verified、SWE-bench Multilingual、SWT-Bench Verified 的软件接入与官方 evaluator 接口。

本仓库目前不声明任何正式 pass@1、resolved rate 或 leaderboard 结果。真实 8B 权重、GPU 剪枝、完整 benchmark 和论文级 agent 评测仍需要服务器验证。

## 当前支持范围

| 类别 | 已实现 | 规划中 |
| --- | --- | --- |
| 模型 | Klear-AgentForge-8B、Granite-4.2-8B、SWE-Lego-Qwen3-8B | 无 |
| 剪枝 | Magnitude、Wanda、SparseGPT、SLEB、TaBP（内部支持 SSN / DDF） | 无 |
| Recovery | PEFT LoRA：adapter-only、显式 merge、稀疏 merge safety | 最终 recovery corpus / protocol 待实验设计确定 |
| 分析 | task-level paired bootstrap、retention/delta、Pareto helper | repo-cluster bootstrap、serving TTFT/TPOT |
| SRTP benchmark | HumanEval original 164、MBPP original test 500、LiveCodeBench fine-grained v6 175 | 无 |
| Agent runner | vLLM 生命周期、mini-swe-agent-plus、OpenHands、canonical full-checkpoint artifact、patch/trajectory/manifest | 真实 8B GPU 环境 smoke |
| Agent benchmark | SWE-bench Verified（500）、Multilingual（300）、SWT-Bench Verified（433）的 adapter、repo provisioning、prediction、official harness wrapper | 真实 Docker/8B 全量结果；Terminal-Bench、BFCL |

TaBP 在 registry 中仍是一个 pruning family；后续 empirical study 会将 TaBP-SSN 与 TaBP-DDF 作为两个实验条件，而不是两个顶层算法 family。

当前代码已经覆盖配置注册、模型 snapshot manifest、下载校验、tiny model 剪枝回归、canonical artifact lineage、PEFT LoRA recovery、三项代码生成 benchmark 的 generate/evaluate 分离流程，以及完整 Agent + vLLM 运行链。完整 8B 服务器实验仍待执行。

## Agent + vLLM

三套系统都由 `configs/agent_runner/systems/` 中的完整配置启动。重依赖分别安装到隔离环境，不改变现有 `.venv`：

```bash
.venv/bin/python scripts/setup/setup_agent_runner.py --component vllm
.venv/bin/python scripts/setup/setup_agent_runner.py --component mini_swe_agent_plus
.venv/bin/python scripts/setup/setup_agent_runner.py --component openhands
```

可通过 `--index-url https://pypi.tuna.tsinghua.edu.cn/simple` 显式选择国内 PyPI 镜像。启动前检查：

```bash
.venv/bin/python scripts/setup/agent_preflight.py \
  --system klear_agentforge_8b \
  --artifact-path /data/models/klear_agentforge_8b
```

运行本地任务（系统可替换为 `swe_lego_qwen3_8b` 或 `granite_4_2_8b`）：

```bash
.venv/bin/python scripts/run_agent_system.py \
  --system klear_agentforge_8b \
  --artifact-path /data/checkpoints/klear_agentforge_8b/magnitude_s030 \
  --repo-path /path/to/local/repo \
  --task-file task.md \
  --output-dir /data/results/agent_smoke/task001
```

默认自动管理 vLLM；`--endpoint http://host:8000` 复用已有服务。`--port`、`--tensor-parallel-size`、`--gpu-memory-utilization` 和 `--max-num-seqs` 是显式基础设施 override，并会写入 manifest。`--dry-run` 验证 artifact、任务和完整命令但不启动进程；`--keep-server` 仅在成功后保留本轮启动的服务。输出包含 `run_manifest.json`、`serving_manifest.json`、trajectory、Agent/vLLM 日志和 `changes.patch`。

更多规划见：

- `docs/RESEARCH_PLAN.md`
- `docs/DEPLOYMENT.md`

## 主流程

```text
准备模型与数据
→ 构建或准备运行环境
→ software/GPU/path preflight
→ pruning 产生 canonical artifact
→ recovery（可选）产生新的 artifact
→ Direct Evaluation 或 Agent Runner → Agent Benchmarks
→ analysis / 归档 manifest、结果和环境身份
```

## 目录与配置

- 模型身份与结构：`configs/models/`
- runtime snapshot 清单：`configs/model_snapshots/`
- 剪枝参数：`configs/pruning/`
- benchmark 数据、prompt、extraction：`configs/direct_benchmarks/`
- model × benchmark generation 参数：`configs/direct_evaluation/`
- Recovery 参数与数据配置：`configs/recovery/`
- Agent runner 与 system 配置：`configs/agent_runner/`
- Agent benchmark 协议配置：`configs/agent_benchmarks/`
- canonical artifact 实现：`src/artifacts/`
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
python3 scripts/run_pruning.py \
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

## LoRA Recovery

LoRA 是独立于 pruning 的可选 artifact transformation，不属于任何剪枝算法本身。当前实现固定使用 `peft==0.18.1` 作为软件基线，但 `configs/recovery/lora.yaml` 明确标记为 `implementation_baseline_not_final_experiment_protocol`；最终 recovery corpus、rank/alpha、训练预算以及哪些剪枝条件进入恢复实验尚未锁定。

```bash
python3 scripts/run_recovery.py \
  --method lora \
  --model klear_agentforge_8b \
  --artifact-path /data/checkpoints/klear_agentforge_8b/magnitude_s030 \
  --config configs/recovery/lora.yaml \
  --dataset-config configs/recovery/dataset_implementation_smoke.yaml \
  --dataset-path /path/to/local/recovery.jsonl \
  --output-dir /data/checkpoints/klear_agentforge_8b/magnitude_s030_lora \
  --device cuda
```

输出始终包含 PEFT adapter 和 `recovery_manifest.json`。只有显式选择 `standard_merge` 时才生成 standalone merged checkpoint。对 Magnitude/Wanda/SparseGPT 这类 weight-sparse artifact，普通 LoRA merge 默认被拒绝；必须显式允许潜在 sparsity 改变，并记录 merge 前后零值率。SLEB/TaBP reduced-depth artifact 可以显式 merge，但必须保持层数不变。Direct evaluation 可加载 PEFT overlay；vLLM Agent runner 只接受 standalone full checkpoint，因此 adapter-only artifact 会 fail-fast。

## 代码生成评测

generation 与 evaluator 是两个独立 phase，没有组合执行入口。`--limit N` 只用于 smoke；正式运行必须使用完整 pinned task set。

```bash
python3 scripts/run_direct_benchmark.py \
  --phase generate \
  --benchmark livecodebench \
  --model klear_agentforge_8b \
  --artifact-kind dense \
  --artifact-label dense \
  --artifact-path /data/models/klear_agentforge_8b \
  --run-id smoke \
  --limit 2 \
  --offline

python3 scripts/run_direct_benchmark.py \
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
Agent:  artifact -> Agent Runner -> Agent Benchmarks -> official evaluator
```

`agent_runner` 只负责把一个已准备好的 repository task 跑起来：vLLM、mini-swe-agent-plus/OpenHands、trajectory 与 patch。`agent_benchmarks` 负责 benchmark 数据、独立 worktree、prediction、官方 evaluator 与聚合结果；依赖方向固定为 `agent_benchmarks -> agent_runner`。Direct benchmark 不 import OpenHands、mini-swe-agent-plus 或 vLLM。

Agent benchmark 的 generation/evaluation 可独立重跑：

```bash
python3 scripts/setup/setup_agent_benchmarks.py --benchmark swebench_verified

python3 scripts/run_agent_benchmark.py \
  --phase generate --benchmark swebench_verified \
  --system klear_agentforge_8b --artifact-path /data/models/klear_agentforge_8b \
  --output-root /data/agent_benchmarks/results --run-id pilot \
  --repo-cache-root /data/agent_benchmarks/repos \
  --workspace-root /data/agent_benchmarks/worktrees --limit 1

python3 scripts/run_agent_benchmark.py \
  --phase evaluate --benchmark swebench_verified \
  --system klear_agentforge_8b --artifact-path /data/models/klear_agentforge_8b \
  --output-root /data/agent_benchmarks/results --run-id pilot \
  --repo-cache-root /data/agent_benchmarks/repos \
  --workspace-root /data/agent_benchmarks/worktrees --evaluator-workers 1
```

setup 会把 exact-revision dataset 固化为 `.agent_benchmarks/datasets/.../test.json` 并写入带 SHA256 的 sidecar；正式 evaluator 使用该本地 snapshot，避免 alias 漂移。SWE-bench Multilingual 的固定口径是 300 tasks / 41 repositories / 9 languages。SWT-Bench generation 使用 433-row ZSP snapshot；evaluation 则从 exact SWE-bench Verified 500-row source 按 pinned SWT filter 删除 67 题，生成 ID 集合相同的 433-row original-SWE-semantics snapshot。SWE-bench 使用官方 `v5.0.1` CLI；SWT-Bench 使用独立的 `1.3.0` tag evaluator。SWT 默认协议名为 `project_raw_swt_harness_protocol`，保留 raw Agent patch，不等同于当前 OpenHands leaderboard 的额外 patch postprocessing。软件测试、schema 与 exact command 验证不等于 Docker evaluator 已运行，也不产生正式 resolved/success rate。

评测会额外写出包含所有任务结果的 `outcomes.jsonl`。可对 identity 完全一致的 dense/pruned run 做 task-level paired bootstrap：

```bash
python3 scripts/analyze_results.py \
  --dense-outcomes /data/results/.../dense/.../outcomes.jsonl \
  --pruned-outcomes /data/results/.../pruned/.../outcomes.jsonl
```

Direct generation runtime 只记录 `model.generate` 的墙钟时间、prompt/generated token 数、generated tokens/s、CUDA peak allocated VRAM（无 CUDA 时为 `null`）和 artifact checkpoint bytes。剪枝执行另外记录 pruner call 墙钟时间、该调用的 peak allocated VRAM，以及不含 `artifact_manifest.json` 的 checkpoint bytes。TTFT、TPOT、serving/concurrent throughput 未测量，manifest 以 `not_measured_without_serving_layer` 明确标记。没有 sparse kernel 时，不从 Magnitude/Wanda/SparseGPT 的零权重推断加速。

## 论文阶段边界

SWE-bench、SWT-Bench、BFCL、Terminal-Bench 不应直接扩进当前 `run_direct_benchmark.py`。它们需要独立 harness、容器隔离、任务 revision lock、成本记录和更复杂的指标。

当前已实现 SWE/SWT official harness 薄封装与 per-instance worktree provisioning；尚未执行 benchmark-scale Docker evaluation，也尚未实现 repo-cluster bootstrap。

## 安全说明

模型生成的 Python 是不可信代码。当前 reliability guard 不是安全沙箱。

建议 GPU 环境只负责模型 generation；CPU evaluator 使用 disposable container，并配置 `--network none`、`--cap-drop ALL`、`no-new-privileges` 及 CPU/memory/PID/wall-time limits。AutoDL 普通容器实例不适合无隔离执行模型生成代码。

## 尚待服务器验证

1. 完整下载并验证三个已接入 8B snapshot。
2. 运行 CUDA/GPU preflight 与 dense load/forward/generate smoke。
3. Prefetch 并验证真实 benchmark/calibration assets。
4. 验证五类 pruned artifact，尤其 SLEB/TaBP reduced checkpoint。
5. 先运行 `--limit 2`，再根据资源执行 full benchmark。
6. 对 LoRA/merged artifact 做服务器级 smoke；再按部署文档运行三套 Agent benchmark 的官方 gold smoke 和正式 8B 实验。
