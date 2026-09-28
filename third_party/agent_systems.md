# Agent system identity audit

This document records configuration identity only. The repository does not yet
install or run OpenHands, mini-swe-agent-plus, vLLM, repository sandboxes, or a
SWE/SWT evaluator.

## Shared experimental rule

The paper intervention variables are pruning method and pruning
retention/sparsity. Within one model system, dense and every pruned artifact
must use the same Agent, serving, prompt/tool, context, sampling, iteration, and
benchmark configuration. Different model systems retain their own official
settings; missing upstream details stay `null` rather than being borrowed from
another model.

## SWE-Lego-Qwen3-8B + OpenHands

Audited source: [`SWE-Lego/SWE-Lego` commit
`94704b69aac886e003660e1e0f69f7de163b284e`](https://github.com/SWE-Lego/SWE-Lego/tree/94704b69aac886e003660e1e0f69f7de163b284e).
The canonical checkpoint remains the separately pinned
[`Lego-X/SWE-Lego-Qwen3-8B`](https://huggingface.co/Lego-X/SWE-Lego-Qwen3-8B/tree/00f37992485b78e9579a2674161d512781aee21e).

The official 8B inference script fixes OpenHands 0.53.0, `CodeActAgent`, 500
SWE-bench Verified test instances, one run, `mode=swe`, 100 maximum iterations,
`USE_HINT_TEXT=false`, `swe_default.j2`, plan mode off, and no in-context
example. Its 24 workers are recorded as infrastructure concurrency rather than
a capability variable.

The official vLLM script fixes host `0.0.0.0`, port 8000, tensor parallel 8,
GPU memory utilization 0.9, maximum model length 163840, maximum sequences 24,
and `dummy-key`. The script's old model alias is provenance, not a second model
identity. The vendored OpenHands config fixes temperature 0, 147456 maximum
input tokens, and 16384 maximum output tokens. These fields are therefore
marked `official_score_recipe` or `official_serving_recipe`.

Sources:

- [`infer.sh`](https://github.com/SWE-Lego/SWE-Lego/blob/94704b69aac886e003660e1e0f69f7de163b284e/scripts/swe_lego_qwen3_8b/infer.sh)
- [`serve_vllm.sh`](https://github.com/SWE-Lego/SWE-Lego/blob/94704b69aac886e003660e1e0f69f7de163b284e/scripts/swe_lego_qwen3_8b/serve_vllm.sh)
- [`OpenHands-0.53.0/config.toml`](https://github.com/SWE-Lego/SWE-Lego/blob/94704b69aac886e003660e1e0f69f7de163b284e/OpenHands-0.53.0/config.toml)

## Klear-AgentForge-8B + mini-swe-agent-plus

The [Klear-AgentForge repository at commit
`d37d53ee65dae326b2ec3f61d58a3e0a7e19db21`](https://github.com/Kwai-Klear/Klear-AgentForge/tree/d37d53ee65dae326b2ec3f61d58a3e0a7e19db21)
reports 39.4% on SWE-bench Verified and identifies mini-swe-agent-plus as the
Agent. It also identifies the 66k mini-swe-agent-plus trajectory dataset.

The current official [mini-swe-agent-plus commit
`3dfa5e26831306978ff3cfa2da15b49113ded0e6`](https://github.com/Kwai-Klear/mini-swe-agent-plus/tree/3dfa5e26831306978ff3cfa2da15b49113ded0e6)
documents the Verified/test entry point and hosted or multiple-vLLM-server
operation. Its `swebench_add_edit_tool.yaml` fixes step limit 200, cost limit 3,
`/testbed`, command timeout 60, example model temperature 1.0, `drop_params=true`,
and the string-replacement edit tool.

Those are marked `official_agent_config`, not the exact 39.4 recipe. The
Klear-AgentForge report does not prove that its scored run used the current
config without overrides. Exact score-time overrides, serving topology,
sampling, run/seed policy, and output budget therefore remain
`score_recipe_undisclosed`. Checkpoint-native context 65536 and generation
values 0.6/0.95/20 are recorded separately and are not promoted to score-time
facts.

Sources:

- [Klear-AgentForge README](https://github.com/Kwai-Klear/Klear-AgentForge/tree/d37d53ee65dae326b2ec3f61d58a3e0a7e19db21)
- [`swebench_add_edit_tool.yaml`](https://github.com/Kwai-Klear/mini-swe-agent-plus/blob/3dfa5e26831306978ff3cfa2da15b49113ded0e6/src/minisweagent/config/extra/swebench_add_edit_tool.yaml)
- [Klear-AgentForge-8B checkpoint](https://huggingface.co/Kwai-Klear/Klear-AgentForge-8B/tree/fa3d41e92e9ce7a5b4a52a3e7439aa00521f40c9)

## Granite-4.2-8B + OpenHands

The official [Granite-4.2-8B model card](https://huggingface.co/ibm-granite/granite-4.2-8b)
reports 47.67 on SWE-bench Verified, documents OpenHands compatibility, and
publishes a vLLM recipe: BF16, model length 131072,
`granite_thinking_parser`, its plugin file (vLLM 0.20+), `qwen3_coder`, and
automatic tool choice. It also fixes temperature 1.0, top-p 0.95, sampling on,
thinking enabled by default, an 8192 thinking-mode output budget, and default
history-thinking truncation. The project model identity remains pinned to
checkpoint revision `f8de16cdcdbc6c779ca517604e050d82cc119e44`.

IBM does not publish the exact OpenHands version, Agent class/prompt,
iterations, hint/plan settings, run/seed policy, or complete harness overrides
for the reported 47.67. Those fields remain `null` and the score recipe is
marked `score_recipe_undisclosed`; SWE-Lego's OpenHands values are not reused.
The Granite 4.2 source repository was additionally audited at commit
[`75355e6ef8c17d72a7b3736654ab525352f5e0b8`](https://github.com/ibm-granite/granite-4.2-language-models/tree/75355e6ef8c17d72a7b3736654ab525352f5e0b8).

## SWE-bench Verified benchmark identity

`configs/agent_benchmarks/swebench_verified.yaml` contains benchmark identity
only: `princeton-nlp/SWE-bench_Verified`, test split, 500 tasks, resolved rate,
and official SWE-bench release v4.0.4 at commit
[`c7c22a916c9215e709722bc5ab18df4062dc6248`](https://github.com/SWE-bench/SWE-bench/releases/tag/v4.0.4).
It intentionally contains no Agent, iteration, prompt, generation, or serving
parameters.
