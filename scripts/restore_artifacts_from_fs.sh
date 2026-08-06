#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="${SOLOLLM_ARTIFACT_REMOTE:-ben@192.168.0.193}"
REMOTE_ROOT="${SOLOLLM_ARTIFACT_REMOTE_ROOT:-/mnt/raid/26/aimlcs/project-artifacts/soloLLM}"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing required command: $1" >&2
    exit 1
  fi
}

require_command rsync

mkdir -p "$PROJECT_ROOT/docs" "$PROJECT_ROOT/outputs"
rsync -a --delete "$REMOTE:$REMOTE_ROOT/docs/" "$PROJECT_ROOT/docs/"
rsync -a --delete "$REMOTE:$REMOTE_ROOT/outputs/" "$PROJECT_ROOT/outputs/"
echo "Restored SoloLLM docs and outputs artifacts."
