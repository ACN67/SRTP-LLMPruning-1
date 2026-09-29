# SRTP 研究路线

本项目分成两个阶段推进：

1. **SRTP 结项阶段**：在固定的代码生成诊断集上复现和评测 8B 模型剪枝。
2. **论文实验阶段**：转向软件工程 agent 任务，重点评估剪枝后模型在真实仓库修复、测试生成、工具调用和终端任务中的能力保留与效率收益。

两阶段共享模型、剪枝 artifact、manifest 和可复现性约束，但不共享同一套 evaluator。SRTP 阶段的结果用于验证流水线和给出阶段性结论；论文阶段结果才作为主要科学贡献。

## 模型选择

| 模型 | 当前状态 | 科学角色 |
| --- | --- | --- |
| Klear-AgentForge-8B | 已有配置、snapshot manifest、evaluation profile | Qwen3 系列的 agent/code 后训练模型，作为主要开发和剪枝验证目标 |
| Granite-4.2-8B | 已有配置、snapshot manifest、evaluation profile | 独立 Granite 架构，用于验证结论是否跨架构成立 |
| SWE-Lego-Qwen3-8B | 已接入模型配置、snapshot、Direct profile 与 Agent system | 与 Klear 同为 Qwen3-8B 路线，适合做“同架构、不同软件工程后训练”的论文对照 |

推荐顺序：

1. 先用 Klear-AgentForge-8B 跑通下载、剪枝、保存、加载和 `--limit 2` benchmark。
2. 再扩到 Granite-4.2-8B，验证 adapter、内存和生成 profile。
3. 最后接入 SWE-Lego-Qwen3-8B，并重新跑三个模型的 dense baseline。

三模型不能简单按原模型卡分数横向排名，因为模型卡使用的 agent scaffold、提示、采样和评测框架通常不同。论文中应主要比较每个模型自身 dense 与 pruned artifact 的成对差异，再讨论跨模型趋势。

## 剪枝方法

| 方法 | 类型 | 校准数据 | 当前状态 | 论文解释边界 |
| --- | --- | --- | --- | --- |
| Magnitude | 非结构化权重置零 | 无 | 已实现 | 只说明质量退化和实际零值率；无稀疏 kernel 时不宣称加速 |
| Wanda | 非结构化权重置零 | C4 | 已实现 | 关注 activation-aware mask；不自动加速 |
| SparseGPT | 非结构化权重置零并更新保留权重 | C4 | 已实现 | 注意 Hessian 内存、damping、实际稀疏率和失败率 |
| SLEB | 结构化删 Transformer block | WikiText-2 | 已实现 | 可报告真实 reduced-depth 延迟和吞吐 |
| TaBP | 结构化删 Transformer block | ARC-Easy / WikiText（按 SSN/DDF 官方分支） | 已实现 | TaBP 是一个 family；SSN 与 DDF 作为两个正式实验条件，DDF 保留 pinned executable behavior |

权重稀疏方法和删 block 方法不应只按同一个 `sparsity` 数字比较。推荐报告：

- 代码能力保留率；
- 实际零值率或实际删层数；
- checkpoint 大小；
- TTFT、TPOT、tokens/s、峰值显存；
- 质量与延迟的 Pareto 曲线。

## Recovery

Recovery 与 pruning 分层。当前仓库已经实现 PEFT LoRA 的训练、adapter save/reload、可选 merge、reduced-depth 校验和 weight-sparse merge safety，但**尚未决定最终论文协议**。后续研究需要单独确定 recovery corpus、LoRA 超参数/训练预算、哪些 pruning condition 进入 recovery，以及非结构化稀疏模型是否采用普通 adapter-only LoRA 或其他 sparsity-preserving recovery。

因此正式结果至少区分 `pruned-only` 与 `pruned + recovery`，不能把恢复后的 checkpoint 直接作为剪枝算法原始结果。Recovery 数据与 pruning calibration、Direct benchmark、Agent benchmark evaluation 数据必须隔离并记录 contamination audit。

## SRTP 阶段 Benchmark

SRTP 阶段只承诺完成以下固定任务集：

| Benchmark | 任务数 | 用途 | 说明 |
| --- | ---: | --- | --- |
| HumanEval original | 164 | Python 函数补全诊断 | 公开时间早，污染风险高，只作为历史诊断 |
| MBPP original test | 500 | 基础 Python 编程能力 | 使用 task 11-510，不等同所有官方协议 |
| LiveCodeBench fine-grained v6 | 175 | 较新的竞赛式代码生成 | 当前项目固定 v6 tranche，不等同累计 release_v6 leaderboard |

这些结果应写成“project pinned protocol”。如果未来要和官方榜单比较，需要重新对齐官方生成数量、温度、pass@k 和 evaluator 版本。

## 论文阶段 Benchmark

论文主基准（软件已接入，正式服务器结果 pending）：

- **SWE-bench Verified**：主 repository-level 修复评估，输出 patch，指标为 resolved rate。
- **SWE-bench Multilingual**：多语言 repository-level 软件工程能力，独立配置与 adapter 文件，但可复用公共 helper。
- **SWT-Bench Verified**：测试生成能力，必须独立于 SWE-bench patch 修复指标报告。generation 使用 ZSP 433-row inference snapshot，evaluation 使用 SWE-bench Verified 500-row source 减去 pinned 67-ID filter 后的 433-row original-semantics snapshot；默认 `project_raw_swt_harness_protocol` 不应用 OpenHands wrapper 的额外 patch postprocessing。

补充基准：

- **Terminal-Bench**：终端/系统级 autonomous task execution 补充；保持官方/native harness 语义。
- **BFCL**：function/tool calling 与 multi-turn agentic 能力补充；保持官方/native harness 语义。

补充 benchmark 的具体版本与子集在实现轮再固定，不提前用当前草案冒充最终协议。

SWE、SWT、BFCL、Terminal-Bench 应保留官方 harness 或官方语义 adapter。不要把它们塞进当前 `run_direct_benchmark.py` 的简单 pass@1 evaluator。

## 推荐实验矩阵

### Pilot

目标是排除 8B、GPU、路径、保存和加载风险。

- 模型：Klear-AgentForge-8B。
- 方法：Magnitude、Wanda、SparseGPT、SLEB。
- 稀疏设置：权重方法 30%，SLEB 约 20% block 删除。
- Benchmark：三个 SRTP benchmark 各 `--limit 2`。
- 输出：完整 canonical artifact manifest、generation/evaluation manifest、显存和耗时记录。

### SRTP 完整矩阵

- 模型：Klear-AgentForge-8B、Granite-4.2-8B。
- 方法：Dense、Magnitude、Wanda、SparseGPT、SLEB。
- 稀疏率：以 30% 作为主设置；资源允许时补 50%。
- Benchmark：HumanEval original、MBPP original test、LiveCodeBench fine-grained v6。

SWE-Lego-Qwen3-8B 如果在结项前完成接入，可以补 dense baseline 和 30% 主设置；否则作为论文阶段扩展，不阻塞 SRTP 结项。

### 论文筛选矩阵

- 模型：三模型。
- 方法：Dense、代表性权重剪枝方法、SLEB、TaBP-SSN、TaBP-DDF；Recovery 作为独立可选实验条件，不与 raw pruning 混为同一结果。
- 先在预注册小集合上筛选 Pareto 点，再跑完整 SWE-bench Verified / SWE-bench Multilingual / SWT-Bench Verified。
- 不建议把所有模型 × 所有剪枝方法 × 所有稀疏率直接乘到完整 agent benchmark，成本过高且难以解释。

## 统计协议

每个结果至少保存：

- dense/pruned 的 task-level 成对结果；
- pass@1 或 resolved rate 的绝对差；
- 相对保留率；
- 10,000 次 task-level paired bootstrap 95% CI；
- 实际稀疏率、实际删层数、checkpoint 大小；
- 生成 tokens、wall time、GPU-hours、CPU-hours；
- 每个 CUDA device 的 peak allocated VRAM、各 device peak 的最大值，以及 RAM；不得把 max-device peak 或各卡独立 peak 的加和表述为同一时刻的多卡总峰值；
- 失败分类：generation error、parse error、timeout、infra error、test failure。

SWE 类 benchmark 建议按 repository 做 cluster bootstrap，避免同一仓库任务过多导致置信区间过窄。

## 代码演进路线

正式实验前的核心骨架固定为：

- `src/models/`：模型身份、结构 adapter 与 dense loading；
- `src/artifacts/`：Pruning/Recovery 与所有下游之间的 canonical artifact boundary；
- `src/pruning/`：Magnitude、Wanda、SparseGPT、SLEB、TaBP；
- `src/recovery/`：独立 post-pruning/post-training recovery，目前实现 PEFT LoRA；
- `src/direct_evaluation/`：HumanEval、MBPP、LiveCodeBench；
- `src/agent_runner/`：vLLM lifecycle 与 model-specific canonical Agent；
- `src/agent_benchmarks/`：benchmark dataset/repository/prediction、official evaluator wrapper 与 benchmark result；
- `src/analysis/`：结果统计与效率分析。

对应顶层入口为 `run_pruning.py`、`run_recovery.py`、`run_direct_benchmark.py`、`run_agent_system.py`、`run_agent_benchmark.py` 与 `analyze_results.py`。架构测试固定依赖方向，并禁止 `agent_runner -> agent_benchmarks`。

SWE-bench Verified、SWE-bench Multilingual（300 tasks / 41 repositories / 9 languages）、SWT-Bench Verified 现已分别拥有独立 adapter、配置、测试和 provenance 文档，并共享最小 repository/prediction/harness helper。状态为 software-ready 与 official-interface-validated；正式 Docker gold smoke、8B generation 和全量分数仍 pending。Terminal-Bench、BFCL 等后续 benchmark 可通过同一 extension contract 接入，不修改更深层的模型、Artifact、Pruning、Recovery 或 Agent runner。
