# 服务器与部署建议

本项目的部署目标是让代码、环境、模型、数据和结果各自有清晰边界：

- **代码**：Git 管理，服务器上 `git pull` 或从固定 commit checkout。
- **环境**：Docker image 或 AutoDL 平台自定义镜像管理。
- **模型/数据/checkpoint/结果**：放持久化数据盘，不进入 Git，也不打进镜像。
- **复现记录**：manifest 记录 Git revision、image identity、模型 snapshot、benchmark task hash 和结果 hash。

## 推荐服务器路线

### SRTP 阶段

目标是完成两个已接入 8B 模型、四种剪枝方法和三项代码生成 benchmark。

推荐配置：

- GPU：48GB 显存起步，优先 A40、L40、L40S 或同级。
- CPU 内存：64GB 起步，128GB 更稳。
- 数据盘：至少 500GB。
- 系统：Linux x86_64，Python 3.11，CUDA 与 PyTorch 2.7.1 / CUDA 12.8 对齐。

如果 SparseGPT 或 SLEB pilot 出现显存压力，再临时升级到 A100/H100 80GB。不要一开始就把所有完整 checkpoint 都保存下来，开发阶段优先保存 manifest、mask 或 block removal list，只有进入正式对比的 artifact 保存完整权重。

### 论文阶段

目标是运行 SWE-bench、SWT-Bench、BFCL、Terminal-Bench 等 agent benchmark。

推荐配置：

- GPU：A100/H100 80GB 或同级，用于模型推理和长上下文 generation。
- CPU 内存：128GB 起步。
- 本地 NVMe：约 2TB。
- 必须支持 Docker/OCI 或等价容器运行时。

SWE/SWT/Terminal-Bench 的正式 evaluator 需要大量隔离容器。普通 AutoDL 容器实例不支持嵌套 Docker，因此更适合做剪枝和 generation，不适合作为论文级 evaluator 主机。

## AutoDL 普通容器实例

AutoDL 适合小时租赁做 SRTP 阶段 GPU 计算。推荐结构：

```text
/root/SRTP                         # 代码，系统盘，可保存到 AutoDL 镜像
/root/autodl-tmp/srtp/             # 数据盘，关机保留但不进入系统镜像
├── models
├── datasets
├── cache
├── checkpoints
└── results
/data -> /root/autodl-tmp/srtp     # 通过子目录软链接兼容项目约定
```

初始化目录：

```bash
mkdir -p /root/autodl-tmp/srtp/{models,datasets,cache,checkpoints,results}
mkdir -p /data

for name in models datasets cache checkpoints results; do
  [ -e "/data/$name" ] || ln -s "/root/autodl-tmp/srtp/$name" "/data/$name"
done
```

首次准备代码：

```bash
cd /root
git clone https://github.com/ACN67/SRTP-LLMPruning-1.git SRTP
cd /root/SRTP
git rev-parse HEAD

python3 -m pip install \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  -r requirements.txt
```

基础检查：

```bash
python3 -c "import src.models; print('import-ok')"
python3 scripts/setup/preflight.py --software-only
python3 scripts/setup/preflight.py --gpu
python3 scripts/setup/preflight.py --paths
```

canonical pruned/recovered artifact 使用 schema 2 `artifact_manifest.json`：模型 bytes 的 `content_sha256` 与路径无关的 `artifact_provenance_sha256` 分层验证。旧 schema 1 artifact 会明确拒绝，不能被视为已有 provenance 保护，也不会自动升级；应使用当前 writer 重新生成。

Agent 的 managed vLLM 会把 serving checkpoint 绑定到本地 artifact identity。复用 external endpoint 时，标准 `/v1/models` 只能证明 served model name，不能证明 checkpoint digest；single/batch/preflight 均要求显式传 `--allow-unverified-external-endpoint`，并把规范化 URL 与 unverified 状态写入 manifest。non-gold Agent evaluate 还会验证已完成的 `generate_run_manifest.json`、predictions SHA256 和 generation contract；gold evaluate 会明确记录为 evaluator-only。

确认环境可用后，关机并在 AutoDL 控制台保存平台镜像。该镜像保存系统盘中的代码和 Python 环境，不保存 `/root/autodl-tmp` 数据盘内容。

## 标准 Docker / OCI 服务器

如果服务器支持 Docker，优先使用仓库 Dockerfile 固定环境：

```bash
scripts/setup/build_image.sh srtp-llm-pruning:server-ready
scripts/setup/smoke_image.sh srtp-llm-pruning:server-ready software
scripts/setup/smoke_image.sh srtp-llm-pruning:server-ready gpu
```

运行时挂载 `/data`：

```bash
docker run --gpus all --rm \
  -v /data:/data \
  srtp-llm-pruning:server-ready \
  python scripts/setup/preflight.py --all --json
```

导出镜像身份用于归档：

```bash
scripts/setup/export_image.sh srtp-llm-pruning:server-ready
```

AutoDL 普通容器实例不支持导入和嵌套运行这个 Docker image；它只能作为 Docker-capable 服务器的标准环境。

## 模型和数据

模型、数据和校准资产直接在服务器数据盘下载并校验：

```bash
cd /root/SRTP

python3 scripts/setup/download_models.py --all --root /data/models
python3 scripts/setup/verify_model_snapshot.py \
  --model klear_agentforge_8b \
  --path /data/models/klear_agentforge_8b
python3 scripts/setup/verify_model_snapshot.py \
  --model granite_4_2_8b \
  --path /data/models/granite_4_2_8b

python3 scripts/setup/prefetch_assets.py --all --root /data/datasets
python3 scripts/setup/verify_assets.py --root /data/datasets
```

正式入口要求 local dense snapshot 具有 schema v3 verified sidecar。已有 schema v2 snapshot 不需要重新下载；直接对原目录重跑 `verify_model_snapshot.py`，验证器会核验现有 runtime files、计算 artifact content SHA256，并在全部检查成功后原地升级 `.srtp_model_source.json`。`--allow-unverified-model` 只用于明确的开发 fixture，绕过状态会写入运行 manifest，不应用于正式实验。

两个当前模型权重合计约 34GB。加入 SWE-Lego-Qwen3-8B 后，三份 dense 模型约 46-50GB。每保存一个完整 pruned checkpoint，通常还会接近一份 dense 模型大小。论文阶段建议准备 1-2TB 存储，并定期清理非 Pareto 点 checkpoint。

## 实验运行

剪枝示例：

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

生成与评测必须分离：

```bash
python3 scripts/run_direct_benchmark.py \
  --phase generate \
  --benchmark livecodebench \
  --model klear_agentforge_8b \
  --artifact-kind pruned \
  --artifact-label magnitude_s030 \
  --artifact-path /data/checkpoints/klear_agentforge_8b/magnitude_s030 \
  --run-id pilot_lcb_limit2 \
  --limit 2 \
  --offline

python3 scripts/run_direct_benchmark.py \
  --phase evaluate \
  --benchmark livecodebench \
  --model klear_agentforge_8b \
  --artifact-kind pruned \
  --artifact-label magnitude_s030 \
  --run-id pilot_lcb_limit2 \
  --limit 2 \
  --offline
```

正式运行时不要使用 `--limit`。

三个可 resume 的入口都会把 artifact content、task/protocol 与 system/benchmark identity 写入 resume identity；同 run-id 更换 checkpoint 或配置会 fail closed。Direct/Agent 输出应保留原 manifest，不能只复制结果 JSON 后继续 resume。

## 安全边界

模型生成的 Python 代码是不可信代码。

- GPU 服务器负责模型加载、剪枝和 generation。
- evaluator 应在 disposable container 中运行。
- 正式隔离建议包含 `--network none`、`--cap-drop ALL`、`no-new-privileges`、CPU/memory/PID/wall-time limits。
- 如果使用 AutoDL 普通容器实例，不建议在同一实例直接无隔离执行模型生成代码。可把 generations 和 manifest 同步到本地 WSL Docker 或独立 CPU evaluator 主机。

## 每次开机检查清单

```bash
cd /root/SRTP
git fetch origin
git checkout main
git pull --ff-only
git rev-parse HEAD
git status --short

python3 -c "import src.models; print('import-ok')"
python3 scripts/setup/preflight.py --all --json
```

正式实验开始后，不要在同一批 run 中继续 `git pull`。如果必须更新代码，应新建 run id，并在结果 manifest 中保留新的 Git revision。

## 关机前检查清单

1. 确认结果写入 `/data/results`。
2. 确认 checkpoint 写入 `/data/checkpoints`，且包含 canonical `artifact_manifest.json`。
3. 备份关键目录到本地或对象存储：
   - `/data/results`
   - `/data/checkpoints/*/artifact_manifest.json`
   - generation/evaluation manifest
   - 镜像 tar SHA256 或 AutoDL 镜像名称
4. 记录本次实例类型、GPU 型号、显存、计费时长和异常。
5. 关机停止 GPU 计费。

AutoDL 本地数据盘不应作为唯一副本。长时间关机、欠费、主机下架或本地盘故障都可能导致数据不可恢复。

## Agent benchmark 官方运行环境

Agent runner 与 benchmark evaluator 使用不同环境：现有 `.venv` 保留 pruning/recovery/direct evaluation；SWE-bench v5 使用 `.venv-swebench`，SWT-Bench 使用 `.venv-swtbench`。上游 checkout 位于被 Git 忽略的 `.agent_benchmarks/`。

SWT-Bench 的数据角色不可互换：generation 从 pinned ZeroShotPlus 433-row snapshot 读取 `problem_statement`；evaluation 从 pinned SWE-bench Verified 500-row source 和 pinned SWT 67-ID filter 派生 433-row snapshot。setup 会验证两边 ID 集合相等并记录 source/filter/output SHA256。默认 `project_raw_swt_harness_protocol` 直接评价 raw Agent patch，不执行 OpenHands wrapper 的 setup/config filtering 或 test-only stripping，因此不宣称与其当前 leaderboard recipe 数值严格可比。

```bash
python3 scripts/setup/setup_agent_benchmarks.py

python3 scripts/setup/agent_benchmark_preflight.py \
  --benchmark swebench_verified \
  --task-repo /data/agent_benchmarks/swe-bench-tasks \
  --check-runtime
python3 scripts/setup/agent_benchmark_preflight.py \
  --benchmark swtbench_verified --check-runtime
```

建议持久化目录：

```text
/data/agent_benchmarks/
├── repos/       # read-only-ish bare mirrors/cache
├── worktrees/   # one writable detached worktree per instance
├── datasets/    # exact-revision local snapshots
├── task-repos/  # SWE-bench v5 task trees
└── results/     # predictions, evaluator logs, manifests
```

SWE-bench v5 可使用 registry image，也可通过 `--task-repo` 使用 pinned task repository。正式 evaluator 需要 Docker daemon 和 x86_64；SWT-Bench 上游建议至少 120 GB 可用存储、16 GB RAM、8 CPU cores，并建议 worker 数不超过 CPU 的约 75%（且不超过 24）。运行前预留 Docker image/cache 空间，避免与模型 checkpoint 共用紧张的系统盘。

官方单题 gold smoke（`INSTANCE` 必须换成对应数据集中的真实 id）：

```bash
# SWE-bench Verified v5
.venv-swebench/bin/swebench eval .agent_benchmarks/datasets/swebench_verified/test.json --gold \
  --instance INSTANCE --run-id gold-verified-1 --workers 1 \
  --task-repo /data/agent_benchmarks/task-repos/swe-bench-tasks

# SWE-bench Multilingual v5
.venv-swebench/bin/swebench eval .agent_benchmarks/datasets/swebench_multilingual/test.json --gold \
  --instance INSTANCE --run-id gold-multilingual-1 --workers 1 \
  --task-repo /data/agent_benchmarks/task-repos/swe-bench-multilingual-tasks

# SWT-Bench Verified, unit-test mode
(cd .agent_benchmarks/swt-bench && ../../.venv-swtbench/bin/python -m src.main \
  --dataset_name ../../.agent_benchmarks/datasets/swtbench_verified_eval/test.json \
  --predictions_path gold \
  --instance_ids INSTANCE --max_workers 1 --run_id gold-swt-verified-1 \
  --exec_mode unit_test --patch_types vanilla)
```

项目 wrapper 的 `--phase evaluate --dry-run` 会写出 exact command 而不启动 Docker；generation 和 evaluation 可使用同一个 `run-id` 独立重跑。软件 dry-run 只验证接口，不能替代上述官方 gold smoke。
