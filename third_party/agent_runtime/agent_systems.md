# Agent and vLLM runtime audit

All active runtime fields are fixed in `configs/agent_runtime/systems/`. “Score recipe
undisclosed” now describes only historical leaderboard provenance; it never
leaves an executable value unset. Dense and pruned artifacts for one system
use the same effective configuration.

## Exact runtime pins

| Component | Version | Exact revision | Basis |
| --- | --- | --- | --- |
| vLLM | 0.20.2 | `bc150f50299199599673614f80d12a196f377655` | Exact bugfix release satisfying Granite's vLLM 0.20+ parser requirement and supporting Qwen3/Granite |
| mini-swe-agent-plus | 1.14.4 | `3dfa5e26831306978ff3cfa2da15b49113ded0e6` | Klear official Agent implementation and config |
| OpenHands | 0.53.0 | `9ee704a25a331d0d2eb9a8e87a4dcff1d948855b` | SWE-Lego's pinned public recipe |

The pins are machine-readable in `configs/agent_runtime/runtime_pins.yaml`.
Each framework and vLLM is installed in its own venv. The adapters invoke
upstream framework classes; they do not reimplement either Agent loop.

## Klear-AgentForge-8B

Official mini-swe-agent-plus values from
[`swebench_add_edit_tool.yaml`](https://github.com/Kwai-Klear/mini-swe-agent-plus/blob/3dfa5e26831306978ff3cfa2da15b49113ded0e6/src/minisweagent/config/extra/swebench_add_edit_tool.yaml)
are step limit 200, cost limit 3, `/testbed` benchmark cwd, timeout 60,
temperature 1, `drop_params=true`, and `edit_via_str_replace`. The local
adapter substitutes the user-provided repository for the benchmark container's
`/testbed` while preserving its effective role.

The checkpoint is pinned at
[`fa3d41e92e9ce7a5b4a52a3e7439aa00521f40c9`](https://huggingface.co/Kwai-Klear/Klear-AgentForge-8B/tree/fa3d41e92e9ce7a5b4a52a3e7439aa00521f40c9).
The 65,536 context is checkpoint-native. TP=1, one sequence, BF16, 0.9 GPU
utilization, 57,344 input and 8,192 output tokens are explicit single-task
project baselines because the published 39.4 score does not expose its full
serving/run recipe.

## SWE-Lego-Qwen3-8B

The project source is pinned at
[`94704b69aac886e003660e1e0f69f7de163b284e`](https://github.com/SWE-Lego/SWE-Lego/tree/94704b69aac886e003660e1e0f69f7de163b284e).
Its [`infer.sh`](https://github.com/SWE-Lego/SWE-Lego/blob/94704b69aac886e003660e1e0f69f7de163b284e/scripts/swe_lego_qwen3_8b/infer.sh),
[`serve_vllm.sh`](https://github.com/SWE-Lego/SWE-Lego/blob/94704b69aac886e003660e1e0f69f7de163b284e/scripts/swe_lego_qwen3_8b/serve_vllm.sh),
and [`config.toml`](https://github.com/SWE-Lego/SWE-Lego/blob/94704b69aac886e003660e1e0f69f7de163b284e/OpenHands-0.53.0/config.toml)
fix OpenHands 0.53.0, CodeActAgent, 100 iterations, one run, SWE mode,
24 benchmark workers, hints/plan/ICL off, `swe_default.j2`, TP=8, GPU
utilization 0.9, model length 163,840, 24 sequences, temperature 0, input
147,456 and output 16,384. Runtime worker count is one for this single-task
entry point; BF16 is the project's explicit dtype baseline because the serving
script does not pass a dtype. Native tool calling is explicitly false because
the official config does not enable it and its vLLM command has no tool parser.

## Granite-4.2-8B

The checkpoint is pinned at
[`f8de16cdcdbc6c779ca517604e050d82cc119e44`](https://huggingface.co/ibm-granite/granite-4.2-8b/tree/f8de16cdcdbc6c779ca517604e050d82cc119e44).
IBM's official model card fixes BF16, 131,072 context,
`granite_thinking_parser`, `qwen3_coder`, automatic tool choice, temperature
1, top-p 0.95, sampling enabled, thinking enabled, 8,192 output tokens and
history-thinking truncation. The official parser is vendored with provenance
at `third_party/agent_runtime/granite_thinking_parser.py`, validated before spawn,
and passed to vLLM by an absolute plugin path for both dense and pruned models.

IBM publishes OpenHands compatibility and a 47.67 score but not the complete
score-time Agent recipe. The executable baseline therefore explicitly uses
OpenHands 0.53.0/CodeActAgent, 100 iterations, one worker, 120-second command
timeout, hints/plan/ICL off, local runtime, and a 122,880 input budget; these
are marked `project_baseline`, not attributed to IBM's score.

## Runtime boundary

The shared loader accepts dense artifacts and Magnitude, Wanda, SparseGPT,
SLEB, TaBP-SSN and TaBP-DDF manifests. Same-depth and reduced-depth invariants
are checked before vLLM starts. The lifecycle layer validates ports, starts or
reuses a server, polls `/v1/models`, verifies the stable alias, persists logs,
detects early death/timeouts, and guarantees terminate/kill cleanup unless the
user explicitly requests `--keep-server` after success.

OpenHands 0.53.0 does not expose a separate stable high-level “run one local
repository task” package API. The adapter therefore uses the revision-pinned
`openhands.core.main.run_controller` entry used by its own application, with
public configuration/action models, and records this dependency here.
