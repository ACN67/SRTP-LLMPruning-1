#!/usr/bin/env bash
set -euo pipefail
IMAGE="${1:-srtp-llm-pruning:server-ready}"
OUTPUT="${2:-srtp-llm-pruning-server-ready.tar}"
docker save --output "$OUTPUT" "$IMAGE"
sha256sum "$OUTPUT" > "$OUTPUT.sha256"
docker image inspect "$IMAGE" > "$OUTPUT.metadata.json"
