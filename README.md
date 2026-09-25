# SRTP LLM Pruning

Empirical study scaffold for comparing structured and unstructured LLM pruning methods on code-generation benchmarks. The current repository establishes reproducible configuration, model-adapter, pruning, evaluation, and experiment-record interfaces; it does **not** yet implement the paper algorithms.

## Initial study matrix

Models:

- `Kwai-Klear/Klear-AgentForge-8B` (`klear_agentforge_8b`), using the Qwen3 architecture adapter and pinned to `fa3d41e92e9ce7a5b4a52a3e7439aa00521f40c9`.
- `ibm-granite/granite-4.2-8b` (`granite_4_2_8b`), using the Granite architecture adapter and pinned to `f8de16cdcdbc6c779ca517604e050d82cc119e44`.

Pruning methods: Magnitude Pruning, Wanda, SparseGPT, and SLEB.

Benchmarks: HumanEval, MBPP, and LiveCodeBench.

Model brand and architecture are represented separately. Hugging Face repository IDs live in model YAML files and are not duplicated throughout the codebase.

## Repository layout

```text
configs/          Model, pruning, and evaluation YAML configurations
src/models/       Model specifications, registry, and architecture adapters
src/pruning/      Shared pruner interface and method placeholders
src/evaluation/   Future benchmark runners
src/utils/        Shared utilities
scripts/          Setup/pruning/evaluation helpers and the unified entry point
experiments/      Versionable manifests and experiment definitions
results/          Lightweight CSV/JSON summaries only
models/           Local or mounted weights (ignored except for its README)
third_party/      Upstream paper/repository tracking
tests/            Import, registry, and configuration smoke tests
```

## Local setup

Python 3.11 is the initial supported runtime.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The dependencies provide the future model-loading environment, but the current smoke tests neither download nor load an 8B model.

## Dense model validation

Validate pinned metadata and adapter registration without downloading weights:

```bash
python scripts/validate_model.py --model klear_agentforge_8b --config-only
python scripts/validate_model.py --model granite_4_2_8b --config-only
```

On a suitable GPU server, load the pinned model and tokenizer, validate the native structure, run a minimal forward, and optionally generate a few tokens:

```bash
python scripts/validate_model.py \
  --model klear_agentforge_8b \
  --load \
  --device-map auto \
  --cache-dir /data/cache \
  --generate
```

Use `--local-path /data/models/<directory>` to override the Hugging Face source. Local overrides do not silently claim the configured upstream commit as their resolved revision. Real `--load` attempts always write a success or failure manifest under `experiments/generated/`; `--manifest` selects an explicit path. No adapter replaces the model's native decoder forward.

## Unified experiment entry point

The current CLI validates IDs and configuration, then prints a planned manifest:

```bash
python scripts/run_experiment.py \
  --model klear_agentforge_8b \
  --pruner wanda \
  --sparsity 0.3 \
  --benchmark humaneval
```

Add `--write-manifest` to save the plan under `experiments/generated/`. The manifest records the full Hugging Face repository ID, project model ID, requested revision, and a `resolved_revision` field. The latter remains `null` until a future loader resolves the immutable Hugging Face commit. `--execute` currently fails deliberately because the pruning and evaluation implementations have not been verified.

## Docker and persistent data

The image contains only the CUDA/PyTorch/Python software environment and project code:

```bash
docker build -t srtp-llm-pruning .
docker run --rm --gpus all \
  -v /host/models:/data/models \
  -v /host/cache:/data/cache \
  -v /host/datasets:/data/datasets \
  -v /host/checkpoints:/data/checkpoints \
  -v /host/results:/data/results \
  srtp-llm-pruning \
  python scripts/run_experiment.py --model granite_4_2_8b --pruner magnitude --sparsity 0.3 --benchmark mbpp
```

The Docker build does not include model weights, datasets, checkpoints, caches, or experiment results. Override the base image at build time if the GPU host requires a different CUDA/PyTorch combination:

```bash
docker build --build-arg BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime -t srtp-llm-pruning .
```

## Version-control policy

Commit source code, YAML configurations, manifests, third-party provenance records, and lightweight CSV/JSON summaries. Never commit model weights, Hugging Face or benchmark caches, datasets, checkpoints, pruned models, LoRA/adapter checkpoints, large logs, secrets, or virtual environments. Use `models/` locally or mount persistent server storage at the `/data/*` paths above.

See `third_party/README.md` before importing external pruning implementations. Each implementation must be checked against its paper, official repository, upstream commit, license, originally supported models, and our adaptation changes.
