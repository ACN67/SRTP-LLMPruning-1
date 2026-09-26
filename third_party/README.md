# 上游来源、审计与许可证

本目录记录模型、剪枝方法和 benchmark 的固定上游来源、license、项目适配以及仍待真实硬件验证的边界。benchmark 数据来源与 evaluator code license 分开记录；vendored code 的完整 notice 位于 `third_party/licenses/`。

## 模型与架构结论

| 项目模型 | 架构 | 与剪枝相关的结构 |
|---|---|---|
| `klear_agentforge_8b` | Qwen3ForCausalLM | 36 blocks，hidden 4096，MLP 12288，32 attention / 8 KV heads |
| `granite_4_2_8b` | GraniteForCausalLM | 40 blocks，hidden 4096，MLP 12800，32 attention / 8 KV heads |

两个模型都通过 `model.model.layers` 暴露 decoder blocks，并在 model-level forward 计算 rotary position embeddings。旧 LLaMA orchestration 不能直接替代当前 native forward；本项目通过 adapter 与 native capture/replay 处理。

## 剪枝方法结论

- Magnitude：无 calibration，按权重幅值做 deterministic masking。
- Wanda：使用 C4 activation statistics；保留官方 scoring/masking 语义，替换架构和数据 orchestration。
- SparseGPT：使用 C4 Hessian approximation 与 sequential error compensation；保留核心算法并适配当前模型。
- SLEB：使用 WikiText-2 loss 做 dynamic greedy block removal；物理删层后同步 config，并验证 save/reload。

Magnitude/Wanda/SparseGPT 保持原模型深度并产生含零值的 dense tensor checkpoint，不自动带来稀疏推理加速；SLEB 改变 block 数量，必须走 reduced-depth loader。

## 详细记录

- [Magnitude](magnitude.md)
- [Wanda](wanda.md)
- [SparseGPT](sparsegpt.md)
- [SLEB](sleb.md)
- [HumanEval](humaneval.md)
- [MBPP](mbpp.md)
- [LiveCodeBench](livecodebench.md)
