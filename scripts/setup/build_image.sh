#!/usr/bin/env bash
set -euo pipefail
IMAGE="${1:-srtp-llm-pruning:server-ready}"
GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || printf 'unknown')"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if [[ -n "$(git status --porcelain=v1 --untracked-files=normal)" ]]; then
    SOURCE_DIRTY=true
  else
    SOURCE_DIRTY=false
  fi
else
  SOURCE_DIRTY=unknown
fi
docker build --platform linux/amd64 \
  --build-arg SRTP_PROJECT_GIT_COMMIT="$GIT_COMMIT" \
  --build-arg SRTP_SOURCE_DIRTY="$SOURCE_DIRTY" \
  -t "$IMAGE" .
