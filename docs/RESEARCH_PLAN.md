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
| SWE-Lego-Qwen3-8B | 待接入 | 与 Klear 同为 Qwen3-8B 路线，适合做“同架构、不同软件工程后训练”的论文对照 |

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
| TaBP | 候选结构化删 block | 任务相关校准 | 待研究接入 | 先实现定义更清晰的 TaBP-SSN；DDF 排序语义澄清前不进主实验 |

权重稀疏方法和删 block 方法不应只按同一个 `sparsity` 数字比较。推荐报告：

- 代码能力保留率；
- 实际零值率或实际删层数；
- checkpoint 大小；
- TTFT、TPOT、tokens/s、峰值显存；
- 质量与延迟的 Pareto 曲线。

## SRTP 阶段 Benchmark

SRTP 阶段只承诺完成以下固定任务集：

| Benchmark | 任务数 | 用途 | 说明 |
| --- | ---: | --- | --- |
| HumanEval original | 164 | Python 函数补全诊断 | 公开时间早，污染风险高，只作为历史诊断 |
| MBPP original test | 500 | 基础 Python 编程能力 | 使用 task 11-510，不等同所有官方协议 |
| LiveCodeBench fine-grained v6 | 175 | 较新的竞赛式代码生成 | 当前项目固定 v6 tranche，不等同累计 release_v6 leaderboard |

这些结果应写成“project pinned protocol”。如果未来要和官方榜单比较，需要重新对齐官方生成数量、温度、pass@k 和 evaluator 版本。

## 论文阶段 Benchmark

论文主基准：

- **SWE-bench Verified**：主评估，输出 patch，指标为 resolved rate。
- **SWE-bench Live**：低污染补充，必须固定 revision、时间窗、task ID hash。
- **SWT-Bench Verified**：测试生成能力，必须独立于 SWE-bench patch 修复指标报告。

补充基准：

- **BFCL V4 Agentic**：工具调用、web search、memory 类能力补充。只跑 Agentic 时不能报告官方 overall。
- **Terminal-Bench 2.1 子集**：只采用预注册的 resource-bounded subset，不能人工挑“简单题”。如果要官方可比，必须完整 89 题并按官方 trial 要求运行。

SWE、SWT、BFCL、Terminal-Bench 应保留官方 harness 或官方语义 adapter。不要把它们塞进当前 `run_benchmark.py` 的简单 pass@1 evaluator。

## 推荐实验矩阵

### Pilot

目标是排除 8B、GPU、路径、保存和加载风险。

- 模型：Klear-AgentForge-8B。
- 方法：Magnitude、Wanda、SparseGPT、SLEB。
- 稀疏设置：权重方法 30%，SLEB 约 20% block 删除。
- Benchmark：三个 SRTP benchmark 各 `--limit 2`。
- 输出：完整 pruning manifest、generation/evaluation manifest、显存和耗时记录。

### SRTP 完整矩阵

- 模型：Klear-AgentForge-8B、Granite-4.2-8B。
- 方法：Dense、Magnitude、Wanda、SparseGPT、SLEB。
- 稀疏率：以 30% 作为主设置；资源允许时补 50%。
- Benchmark：HumanEval original、MBPP original test、LiveCodeBench fine-grained v6。

SWE-Lego-Qwen3-8B 如果在结项前完成接入，可以补 dense baseline 和 30% 主设置；否则作为论文阶段扩展，不阻塞 SRTP 结项。

### 论文筛选矩阵

- 模型：三模型。
- 方法：Dense、最佳权重剪枝方法、SLEB、TaBP-SSN。
- 先在预注册小集合上筛选 Pareto 点，再跑完整 SWE-bench Verified / SWE-bench Live / SWT-Bench Verified。
- 不建议把所有模型 × 所有剪枝方法 × 所有稀疏率直接乘到完整 agent benchmark，成本过高且难以解释。

## 统计协议

每个结果至少保存：

- dense/pruned 的 task-level 成对结果；
- pass@1 或 resolved rate 的绝对差；
- 相对保留率；
- 10,000 次 task-level paired bootstrap 95% CI；
- 实际稀疏率、实际删层数、checkpoint 大小；
- 生成 tokens、wall time、GPU-hours、CPU-hours；
- 峰值 VRAM/RAM；
- 失败分类：generation error、parse error、timeout、infra error、test failure。

SWE 类 benchmark 建议按 repository 做 cluster bootstrap，避免同一仓库任务过多导致置信区间过窄。

## 代码演进路线

近期只维护当前三项代码生成 benchmark：

- `scripts/run_benchmark.py`
- `src/evaluation/benchmarks.py`
- `src/evaluation/execution.py`

论文阶段另建分层模块：

- `src/evaluation/core/`：通用 manifest、任务身份、成本、指标。
- `src/evaluation/executors/`：OCI/Docker、SWE harness、Harbor/Terminal-Bench executor。
- `src/evaluation/harnesses/`：SWE-bench、SWE-bench Live、SWT-Bench、BFCL、Terminal-Bench 的独立 adapter。
- `src/evaluation/agents/`：patch agent、test agent、tool agent、terminal agent。
- `configs/benchmarks/locks/`：固定官方 revision、task manifest、image digest、许可证说明。

当前仓库不应在基础设施稳定前加入半成品论文 benchmark，以免把 SRTP 结项流程复杂化。
