# 上游来源、审计与许可证

本目录按职责记录固定上游来源、revision/tag、license、项目适配以及仍待真实硬件验证的边界。正式实验中的模型、剪枝、Recovery、Direct benchmark、Agent runner 与 Agent benchmark 都应能从这里追溯到对应官方来源；benchmark 数据来源与 evaluator code license 分开记录。

## 目录

- `models/`：模型或模型家族特有的上游身份与结构说明。
- `pruning/`：Magnitude、Wanda、SparseGPT、SLEB、TaBP。
- `recovery/`：PEFT LoRA 及后续 recovery 方法。
- `direct_benchmarks/`：HumanEval、MBPP、LiveCodeBench。
- `agent_runner/`：vLLM / Agent system provenance 与 Granite reasoning parser。
- `agent_benchmarks/`：SWE-bench v5 与 SWT-Bench 的 dataset、evaluator、schema 和 protocol pin。
- `licenses/`：vendored evaluator/source license notices。

## 模型与架构结论

| 项目模型 | 架构 | 与剪枝相关的结构 |
|---|---|---|
| `klear_agentforge_8b` | Qwen3ForCausalLM | 36 blocks，hidden 4096，MLP 12288，32 attention / 8 KV heads |
| `granite_4_2_8b` | GraniteForCausalLM | 40 blocks，hidden 4096，MLP 12800，32 attention / 8 KV heads |
| `swe_lego_qwen3_8b` | Qwen3ForCausalLM | 36 blocks，hidden 4096，MLP 12288，32 attention / 8 KV heads；复用 Qwen3 adapter |

## 详细记录

### Models
- [SWE-Lego-Qwen3-8B](models/swe_lego.md)

### Pruning
- [Magnitude](pruning/magnitude.md)
- [Wanda](pruning/wanda.md)
- [SparseGPT](pruning/sparsegpt.md)
- [SLEB](pruning/sleb.md)
- [TaBP](pruning/tabp.md)

### Recovery
- [PEFT LoRA](recovery/peft_lora.md)

### Direct benchmarks
- [HumanEval](direct_benchmarks/humaneval.md)
- [MBPP](direct_benchmarks/mbpp.md)
- [LiveCodeBench](direct_benchmarks/livecodebench.md)

### Agent runner
- [Canonical Agent systems](agent_runner/agent_systems.md)

### Agent benchmarks
- [SWE-bench Verified](agent_benchmarks/swebench_verified.md)
- [SWE-bench Multilingual](agent_benchmarks/swebench_multilingual.md)
- [SWT-Bench Verified](agent_benchmarks/swtbench_verified.md)
