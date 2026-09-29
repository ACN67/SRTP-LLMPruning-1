# SRTP LLM 剪枝复现与评测

本仓库研究 8B 代码与软件工程模型在非结构化权重剪枝、结构化 block removal、LoRA recovery、代码生成评测和 Agent benchmark 下的行为。当前接入 Klear-AgentForge-8B、Granite-4.2-8B、SWE-Lego-Qwen3-8B，以及 Magnitude、Wanda、SparseGPT、SLEB、TaBP（SSN / DDF 是 TaBP 内部实验条件）。

仓库提供可审查的软件实现、固定配置、上游 provenance、tiny/mock 回归测试和统一 artifact/manifest 边界；它**不发布任何由本仓库产生的正式 8B pass@1、resolved rate 或 leaderboard 结果**。

## 当前验证状态

| 层级 | 当前状态 | 含义 |
| --- | --- | --- |
| Software implemented | 已实现 | 模型 adapter、五类剪枝、LoRA、Direct/Agent workflow 和 manifest 路径存在 |
| Unit / Tiny / Mock validated | 已验证 | CPU-safe unit、真实 Transformers tiny Qwen3/Granite、mock CLI 与 schema 回归测试覆盖 |
| Real 8B validated | **Pending / Not yet validated** | 尚无本仓库记录的三模型真实 8B GPU 完整验证 |
| Full benchmark / formal experiment completed | **Pending / Not yet completed** | 尚无本仓库正式完整 Direct 或 Agent benchmark 结果 |

配置中的公开 resolved rate 位于 `upstream_reference_result.upstream_reported_resolved_rate`，并明确标记 `reproduced_by_this_repository: false`；它们只是上游参考，不是本仓库复现结果。

### Model × pruning compatibility / validation matrix

矩阵表达实现与软件验证状态，不是性能结果。

| 模型 | Magnitude | Wanda | SparseGPT | SLEB | TaBP |
| --- | --- | --- | --- | --- | --- |
| Klear-AgentForge-8B | 已实现；tiny Qwen3 验证；8B Pending | 已实现；tiny Qwen3 验证；8B Pending | 已实现；tiny Qwen3 验证；8B Pending | 已实现；tiny Qwen3 验证；8B Pending | 已实现；tiny Qwen3 SSN/DDF 验证；8B Pending |
| Granite-4.2-8B | 已实现；tiny Granite 验证；8B Pending | 已实现；tiny Granite 验证；8B Pending | 已实现；tiny Granite 验证；8B Pending | 已实现；tiny Granite 验证；8B Pending | 已实现；tiny Granite SSN/DDF 验证；8B Pending |
| SWE-Lego-Qwen3-8B | 已实现；复用 Qwen3 adapter/tiny 验证；8B Pending | 已实现；复用 Qwen3 adapter/tiny 验证；8B Pending | 已实现；复用 Qwen3 adapter/tiny 验证；8B Pending | 已实现；复用 Qwen3 adapter/tiny 验证；8B Pending | 已实现；复用 Qwen3 adapter/tiny SSN/DDF 验证；8B Pending |

## Quick Start：最小端到端闭环

安装和 preflight 可在普通 Linux/WSL CPU 环境完成；完整模型下载、剪枝和 generation 需要有足够磁盘及合适 GPU 的 Linux 服务器。

```bash
git clone https://github.com/ACN67/SRTP-LLMPruning-1.git
cd SRTP-LLMPruning-1
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/setup/preflight.py --software-only
```

下载并验证一个 canonical model snapshot。完整下载成功会写入 verified provenance sidecar；验证命令会逐文件核验并写入可复用的 content identity。

```bash
.venv/bin/python scripts/setup/download_models.py \
  --model klear_agentforge_8b --root /data/models
.venv/bin/python scripts/setup/verify_model_snapshot.py \
  --model klear_agentforge_8b --path /data/models/klear_agentforge_8b

.venv/bin/python scripts/setup/prefetch_assets.py --all --root /data/datasets
.venv/bin/python scripts/setup/verify_assets.py --root /data/datasets
```

在 GPU 服务器运行 Magnitude pruning，并对 HumanEval 生成：

```bash
.venv/bin/python scripts/run_pruning.py \
  --model klear_agentforge_8b --pruner magnitude --sparsity 0.3 \
  --execute --local-path /data/models/klear_agentforge_8b \
  --local-files-only --output-dir /data/checkpoints/klear_agentforge_8b/magnitude_s030

.venv/bin/python scripts/run_direct_benchmark.py \
  --phase generate --benchmark humaneval --model klear_agentforge_8b \
  --artifact-kind pruned --artifact-label magnitude_s030 \
  --artifact-path /data/checkpoints/klear_agentforge_8b/magnitude_s030 \
  --output-root /data/results --run-id smoke --limit 2 --offline
```

> **执行生成代码前的安全边界：** Direct evaluator 会执行模型生成的 Python。当前 timeout/reliability guard **不是安全 sandbox**。正式 evaluate 应在 disposable container/VM 中运行，并禁用网络、drop capabilities、启用 `no-new-privileges`，同时设置 CPU、内存、PID 和 wall-time 限额。

在隔离的 CPU evaluator 环境执行并查看结果：

```bash
.venv/bin/python scripts/run_direct_benchmark.py \
  --phase evaluate --benchmark humaneval --model klear_agentforge_8b \
  --artifact-kind pruned --artifact-label magnitude_s030 \
  --output-root /data/results --run-id smoke --limit 2 --offline

cat /data/results/humaneval/klear_agentforge_8b/magnitude_s030/smoke/evaluation.json
```

`--limit` 只用于 smoke；正式实验必须运行完整 pinned task set。generation 与 evaluate 是两个独立 phase。

## Repository map

```text
configs/       机器实际执行的模型、剪枝、评测、Agent 协议
docs/          详细部署与研究说明
models/        tracked README/说明占位目录，不存模型权重
results/       tracked README/说明占位目录，不存实验结果大文件
scripts/       setup、pruning、recovery、Direct/Agent 入口
src/           实现代码
tests/         unit、tiny architecture、mock/integration 回归
third_party/   upstream repository/paper/revision/license/provenance
.github/       CPU-safe CI；不下载 8B、不要求 GPU/Docker benchmark
```

真正的大文件位置固定为 `/data/models`、`/data/cache`、`/data/datasets`、`/data/checkpoints`、`/data/results`。根目录的 `models/`、`results/` 只是进入 Git 的说明/占位目录；`/data/models`、`/data/results` 才是运行时持久化目录。大文件和 secrets 不进入 Git。

## Main workflow

```text
verified model snapshot + pinned datasets
→ software/GPU/path preflight
→ pruning canonical artifact
→ recovery artifact（可选）
→ Direct generation/evaluation 或 Agent generation/official evaluator
→ paired analysis + manifests/results archive
```

### Source of truth

- `configs/` 是机器实际执行参数与 protocol identity 的 source of truth。
- `third_party/` 是 upstream repository、paper、exact revision、license 与 provenance 的 source of truth。
- `docs/` 保存详细运行和研究说明。
- 本 README 只负责导航、状态和入口；若参数与详细文档重复，应以对应 config/third-party record 为准。

## 模型、snapshot provenance 与 artifact

国内默认 transport 是 ModelScope，科学身份始终是 canonical Hugging Face repository + exact commit。SWE-Lego 没有已核实的 ModelScope 镜像，必须显式使用 `--download-source official`。下载器逐文件校验 size/hash，使用 `.part` 和 atomic rename；国内源失败不会静默切换官方源。

正式入口对 canonical/project-downloaded local dense snapshot 默认要求 `.srtp_model_source.json` 的 verified provenance 与当前 content hash 一致。开发 fixture 可显式传 `--allow-unverified-model`；该 opt-in 会写入 manifest。canonical pruned/recovered artifact 继续以 `artifact_manifest.json`、完整 content hash 和 lineage 为边界。canonical artifact 目录禁止 symlink。

旧 schema v2 sidecar 不会触发重新下载，但正式入口会 fail closed，并提示重新验证。对已有 snapshot 原地运行 `scripts/setup/verify_model_snapshot.py` 会重新核验现有文件、计算 content identity，并把 sidecar 升级为 schema v3；权重 shard 无需重新下载。

只验证下载链路而不下载 shard：

```bash
.venv/bin/python scripts/setup/download_models.py \
  --model klear_agentforge_8b --small-file-smoke --root /tmp/model-smoke
.venv/bin/python scripts/setup/download_models.py \
  --model swe_lego_qwen3_8b --download-source official \
  --small-file-smoke --root /tmp/model-smoke
```

## 剪枝与 Recovery

Wanda/SparseGPT/SLEB 使用 `--datasets-root /data/datasets` 下已验证的 calibration 资产。TaBP canonical 配置固定在 `configs/pruning/tabp.yaml`；离线执行需显式提供 `--tabp-dataset-path`。Magnitude、Wanda、SparseGPT 将零值保存在普通 dense tensor 中，没有 sparse kernel 时不得据此宣称推理加速。SLEB/TaBP 会物理移除 block，loader 会校验 reduced-depth invariant。

LoRA 是独立的 artifact transformation。当前 `configs/recovery/lora.yaml` 是 implementation baseline，不是最终实验协议：

```bash
.venv/bin/python scripts/run_recovery.py \
  --method lora --model klear_agentforge_8b \
  --artifact-path /data/checkpoints/klear_agentforge_8b/magnitude_s030 \
  --config configs/recovery/lora.yaml \
  --dataset-config configs/recovery/dataset_implementation_smoke.yaml \
  --dataset-path /path/to/local/recovery.jsonl \
  --output-dir /data/checkpoints/klear_agentforge_8b/magnitude_s030_lora \
  --device cuda
```

adapter metadata 用 `base_artifact_hash` 作为身份，旧 `base_artifact_path` 只作 location hint。搬运后 Direct/recovery loader 可用 `--base-artifact-path /new/base/path` 重定位，并以 hash + lineage 验证；不匹配会拒绝。vLLM Agent runner 只接受 standalone full checkpoint，因此 adapter-only artifact 会 fail fast。

## Direct evaluation 与指标语义

结果目录为 `/data/results/<benchmark>/<model>/<artifact-label>/<run-id>/`。resume identity 包含 exact artifact content SHA256、benchmark/task set 和 generation/evaluation protocol；同 run-id 换 checkpoint 或协议会 fail closed，必须使用新 run-id 或显式 `--overwrite`。

Direct generation 记录 `model.generate` wall time、tokens/s、`peak_cuda_vram_per_device_bytes` 与 `peak_cuda_vram_max_device_bytes`。后者是各 device peak 的最大值，不是多卡总显存。pruning runtime 使用相同字段语义。TTFT、TPOT、serving/concurrent throughput 未测量。

可对 identity 一致的 dense/pruned outcomes 做 paired bootstrap：

```bash
.venv/bin/python scripts/analyze_results.py \
  --dense-outcomes /data/results/.../dense/.../outcomes.jsonl \
  --pruned-outcomes /data/results/.../pruned/.../outcomes.jsonl
```

## Agent + vLLM 与 Agent benchmarks

重依赖安装在隔离环境，不改变 core `.venv`：

```bash
.venv/bin/python scripts/setup/setup_agent_runner.py --component vllm
.venv/bin/python scripts/setup/setup_agent_runner.py --component mini_swe_agent_plus
.venv/bin/python scripts/setup/setup_agent_runner.py --component openhands
```

single-task preflight 与 dry-run：

```bash
.venv/bin/python scripts/setup/agent_preflight.py \
  --system granite_4_2_8b --artifact-path /data/models/granite_4_2_8b

.venv/bin/python scripts/run_agent_system.py \
  --system granite_4_2_8b --artifact-path /data/models/granite_4_2_8b \
  --repo-path /path/to/local/repo --task-file task.md \
  --output-dir /data/results/agent_smoke/task001 --dry-run
```

Granite single/batch managed vLLM 使用同一个 `resolve_granite_parser()` 路径；其他模型不注入 parser plugin。Agent generation 的成功状态名为 `patch_generated`，只表示 Agent 正常退出并产生非空 patch；它不表示 benchmark resolved。正式 resolved/unresolved 只来自 evaluator。

旧 manifest 中 generation `status: success` 不会被静默解释为 evaluator 成功，也不会在缺少新 `resume_identity` 时继续复用；旧结果仍可审计，继续运行需用 `--overwrite` 明确新建当前 schema 的输出。

Agent benchmark generation/evaluation 示例：

```bash
.venv/bin/python scripts/setup/setup_agent_benchmarks.py --benchmark swebench_verified

.venv/bin/python scripts/run_agent_benchmark.py \
  --phase generate --benchmark swebench_verified \
  --system klear_agentforge_8b --artifact-path /data/models/klear_agentforge_8b \
  --output-root /data/results/agent --run-id pilot \
  --repo-cache-root /data/cache/agent-repos \
  --workspace-root /data/cache/agent-worktrees --limit 1

.venv/bin/python scripts/run_agent_benchmark.py \
  --phase evaluate --benchmark swebench_verified \
  --system klear_agentforge_8b --artifact-path /data/models/klear_agentforge_8b \
  --output-root /data/results/agent --run-id pilot \
  --repo-cache-root /data/cache/agent-repos \
  --workspace-root /data/cache/agent-worktrees --evaluator-workers 1
```

Agent resume identity 同时绑定 artifact content、system config、benchmark config 与选中 task 内容。single Agent 还绑定 problem statement、base/HEAD commit 和 serving config。`--resume` 与 `--overwrite` 互斥。

SWE/SWT setup 会固化 exact-revision local dataset snapshot 和 SHA256 sidecar；正式 evaluator 使用 pinned harness。软件/schema/command 测试不等于 Docker evaluator 已运行。

## 依赖、CI 与部署

`requirements.txt` 和 `configs/agent_runner/runtime_pins.yaml` 固定顶层版本/revision，但当前**没有宣称完整 transitive lock**。core、vLLM、OpenHands、mini-swe-agent-plus 与 benchmark harness 涉及不同 Python/CUDA/Docker 环境；应由项目负责人先确认 canonical platform，再在各隔离环境生成带 hash 的 lock/constraints 并记录其 SHA256。仓库没有用当前开发机的 `pip freeze` 伪装成 canonical lock。

GitHub Actions `cpu-tests` 执行 compile/import smoke 与完整 unit/tiny suite，不下载 8B、不启动 GPU、vLLM 或 Docker benchmark。服务器、Docker/AutoDL 和 `/data` 部署见 [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)，研究边界见 [docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md)。

## 尚待真实服务器验证

1. 完整下载、验证并加载三个 8B snapshot。
2. CUDA/GPU preflight、dense forward/generate smoke。
3. 五类 pruning 的真实 8B artifact，尤其 SLEB/TaBP reduced checkpoint。
4. LoRA/merged artifact 的服务器级 smoke 与最终 recovery protocol 决策。
5. 完整 HumanEval/MBPP/LiveCodeBench，以及三套 Agent benchmark 的官方 evaluator。
6. canonical platform 的 transitive dependency locks。

项目当前没有 root LICENSE，也不据 third-party license 推断项目自身授权；复用或发布前需由项目负责人明确许可证。作者列表、正式项目标题/版本和论文/DOI 尚未形成可核实的 citation metadata，因此未生成 `CITATION.cff`。
