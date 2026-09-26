# SRTP LLM Pruning

Empirical study scaffold for comparing structured and unstructured LLM pruning methods on code-generation benchmarks. Dense model loading, Magnitude Pruning, and basic unstructured Wanda are implemented. SparseGPT, SLEB, and benchmark execution remain placeholders.

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
src/pruning/      Shared pruning interfaces and implemented pruning methods
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

Tests instantiate tiny random Qwen3 and Granite models locally; they do not download or load an 8B model.

## Magnitude Pruning

The implemented baseline is deterministic **per-module (layer-wise) unstructured magnitude pruning**. For every `torch.nn.Linear` weight inside adapter-enumerated transformer blocks, it flattens and ranks `abs(weight)` over that complete matrix, then zeros exactly `floor(weight.numel() * sparsity_ratio)` entries. Each Linear is handled independently; pruning is not global across modules. Stable sorting resolves ties by row-major flattened index. Tensor shapes and biases are unchanged.

Embeddings, final/all norms, `lm_head`, biases, and every parameter outside transformer blocks are excluded. The summary distinguishes targeted weights, requested mask positions, pre-existing zeros, newly zeroed weights, and post-pruning zeros, both globally and per module.

The test suite validates the algorithm with toy models and with genuine tiny `Qwen3ForCausalLM` and `GraniteForCausalLM` instances, including forward and standard Hugging Face save/reload. This is algorithm- and architecture-level validation only; both configured 8B checkpoints remain pending GPU validation.

## Wanda

The implemented Wanda baseline uses `abs(weight) * input-channel activation L2` and stable exact-k selection independently in every output row. Calibration and pruning are sequential by transformer block: each pruned block is replayed before the next block collects statistics. There is no weight update, reconstruction, retraining, Wanda variant, or N:M support.

The canonical protocol is C4 training data, 128 random tokenized samples, sequence length 2048, and seed 0. Sampling selects a random sufficiently long document and then a random contiguous token span. The explicit 2048-token length is not inferred from a model's maximum context. Tests inject local samples and never download C4.

Native model hooks capture the current Transformers block kwargs, including masks, positions, cache position, and rotary embeddings. Architecture adapters normalize Qwen3's Tensor block output and Granite's tuple output. Tiny real-architecture replay equivalence, sequential propagation, pruning, forward, and save/reload are tested; real 8B and C4/GPU validation remain pending.

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

The CLI prints a planned manifest without loading weights:

```bash
python scripts/run_experiment.py \
  --model klear_agentforge_8b \
  --pruner wanda \
  --sparsity 0.3 \
  --benchmark humaneval
```

Add `--write-manifest` to save the plan under `experiments/generated/`. To execute Magnitude on a suitable GPU host and save a standard Hugging Face model/tokenizer checkpoint plus `pruning_manifest.json`:

```bash
python scripts/run_experiment.py \
  --model klear_agentforge_8b \
  --pruner magnitude \
  --sparsity 0.20 \
  --execute \
  --device-map auto \
  --cache-dir /data/cache \
  --output-dir /data/checkpoints/klear_agentforge_8b/magnitude/s020
```

`--output-dir` is mandatory for execution and must point to external persistent storage. Paths inside the Git repository and paths equal to, above, or below a `--local-path` dense source are rejected. Existing non-empty output directories are rejected by default; `--overwrite-output-dir` explicitly permits Hugging Face to reuse a safe external non-empty directory, but does not delete its existing contents and does not bypass either path restriction. Benchmark flags cannot be combined with execution yet.

Wanda execution uses the canonical calibration defaults, which may be overridden explicitly:

```bash
python scripts/run_experiment.py \
  --model granite_4_2_8b \
  --pruner wanda \
  --sparsity 0.20 \
  --execute \
  --device-map auto \
  --cache-dir /data/cache \
  --calibration-source c4 \
  --calibration-samples 128 \
  --calibration-seqlen 2048 \
  --calibration-seed 0 \
  --output-dir /data/checkpoints/granite_4_2_8b/wanda/s020
```

SparseGPT and SLEB deliberately fail if passed with `--execute`.

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
