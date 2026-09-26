#!/usr/bin/env bash
set -euo pipefail
IMAGE="${1:-srtp-llm-pruning:server-ready}"
MODE="${2:-software}"
if [[ "$MODE" == "gpu" ]]; then
  docker run --rm --gpus all "$IMAGE" python scripts/setup/preflight.py --gpu
else
  docker run --rm "$IMAGE" python scripts/setup/preflight.py --software-only
fi
