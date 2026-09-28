# TaBP upstream audit and adaptation

## Sources and reproducibility pin

- Paper: *Task-aware Block Pruning with Output Distribution Signals for Large Language Models*, Findings of EACL 2026 ([ACL Anthology](https://aclanthology.org/2026.findings-eacl.320/)).
- Official repository: [`Song-haJo/TaBP`](https://github.com/Song-haJo/TaBP).
- Executable-semantics baseline: commit [`f56a0d2c38f490bec7b223bb98d1bf1740e5e92a`](https://github.com/Song-haJo/TaBP/tree/f56a0d2c38f490bec7b223bb98d1bf1740e5e92a).
- No license file was present at the audited commit. This repository does not vendor upstream source; it independently maps the observed behavior to local adapters.

The audit covered `main.py`, `run.sh`, `pruning/metrics/{ssn,ddf,_hooks}.py`, `pruning/stats.py`, `pruning/pruner.py`, `models/lm_head.py`, and `data/preprocessing.py`. Where comments, README, paper, and executable code differ, the pinned executable path below is preserved rather than normalized.

## Project identity and canonical configuration

TaBP is one top-level project method, registry ID `tabp`. `ssn` and `ddf` are internal `ranking_strategy` values, not additional pruning methods. The canonical experiment configuration is fixed to:

```text
ranking_strategy=ssn, mode=latter, measure=entropy,
calibration=arc_easy, task_type=qa, n_samples=1024,
lm_head_type=frozen
```

The official `run.sh` fixes SSN/latter and loops entropy and confidence; this project fixes entropy so model and retention remain the intended experimental variables. The project ratio API maps sparsity to `ceil(B * ratio)` and records requested and achieved sparsity/retention. Upstream takes explicit block counts; ratio rounding is a project adaptation.

## Pinned executable behavior

Both strategies call one-token `model.generate`, capture native block outputs with hooks, probe the final-token hidden state directly with the LM head (no inserted final normalization), and use the processed final generation score as the target distribution. ARC-Easy prompts use the upstream prefix/question/options/`Answer: ` format and generation is restricted to answer-label token IDs. The ordered first 1,024 train rows are used. Revision `210d026faf9955653af8916fad021475a3f00453` is an additional project reproducibility pin; upstream does not pin that dataset revision.

SSN computes `mean(abs(s_i - s_(i-1)))` once on the dense model. It uses NumPy's default `argsort` behavior. `whole` returns `np.argsort(full_scores)[1:]`; `latter` observes from `B//2`, treats that first observed block as the boundary, and returns only indices after it. Frozen heads are supported. Trained heads are rejected rather than randomly initialized because no explicit compatible upstream-named per-block checkpoints are supplied.

DDF uses the exact `MEASURE_DIRECTION` mapping from upstream `stats.py`. The pinned QA branch increments a penalty when `alpha * delta < 0`, but the text-generation branch increments when `alpha * delta > 0`. It normalizes by pass count, then returns `np.argsort(-block_ddf)` over all blocks. Consequently, executable QA behavior and final descending ordering conflict with the DDF docstring/README description of desirable-frequency ascending pruning; this implementation retains the executable behavior without choosing or correcting one interpretation. `mode` limits hook observation but does not filter the final DDF array/ranking.

WikiText DDF concatenates all train text with double newlines, takes consecutive 32-token windows, performs 32 windows by 32 autoregressive one-token steps by default, and appends each generated token. It is not replaced by SLEB's WikiText calibration path.

All 13 upstream measure identifiers and direction signs are exposed. The pinned `compute_metric` re-indexes an already answer-restricted intermediate distribution for JS and several distance measures; ordinary ARC token IDs can therefore raise an indexing error. That behavior is retained and documented instead of silently repaired. The canonical entropy path is unaffected.

The pinned `main.py` executable default prune counts are `[2, 4, 8]`, while its top-level help text/README still mentions older values such as `1 3 5 6 7`. Project manifests use their explicit ratio-derived count and do not claim either list as ratio semantics.

## Architecture and artifact adaptation

The algorithm never hard-codes `model.model.layers`. `Qwen3Adapter` (shared by Klear-AgentForge and SWE-Lego) and `GraniteAdapter` supply block containers, output normalization, LM heads, block replacement, and metadata finalization. Shared physical removal updates `config.num_hidden_layers`, attention `layer_idx`, Qwen3 `config.layer_types` by retained original index, and Granite native metadata.

The standard `save_pretrained` checkpoint is wrapped by the project canonical `artifact_manifest.json`. TaBP is represented as a reduced-depth pruning lineage operation; the common artifact validator checks model identity, config/native depth, dimensions, adapter compatibility, and the reduced-depth invariant. Strategy/mode/measure, calibration provenance, ranking/scores, removed/retained original indices, parameters, and requested/achieved ratios are retained in the pruning lineage parameters/provenance.
