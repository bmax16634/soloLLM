#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="${SOLOLLM_ARTIFACT_REMOTE:-ben@192.168.0.193}"
REMOTE_ROOT="${SOLOLLM_ARTIFACT_REMOTE_ROOT:-/mnt/raid/26/aimlcs/project-artifacts/soloLLM}"
STAMP="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
META="$(mktemp)"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing required command: $1" >&2
    exit 1
  fi
}

require_command rsync
require_command ssh
require_command scp

DOC_FILES=$(find "$PROJECT_ROOT/docs" -type f 2>/dev/null | wc -l)
DOC_BYTES=$(find "$PROJECT_ROOT/docs" -type f -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {print s+0}')
OUT_FILES=$(find "$PROJECT_ROOT/outputs" -type f 2>/dev/null | wc -l)
OUT_BYTES=$(find "$PROJECT_ROOT/outputs" -type f -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {print s+0}')

cat > "$META" <<JSON
{
  "project": "soloLLM",
  "artifactRoot": "$REMOTE_ROOT",
  "capturedAt": "$STAMP",
  "artifacts": {
    "docs": {
      "remotePath": "$REMOTE_ROOT/docs/",
      "sourcePath": "$PROJECT_ROOT/docs/",
      "files": $DOC_FILES,
      "bytes": $DOC_BYTES,
      "purpose": "Internal project docs, research notes, result summaries, and generated docs kept off GitHub."
    },
    "outputs": {
      "remotePath": "$REMOTE_ROOT/outputs/",
      "sourcePath": "$PROJECT_ROOT/outputs/",
      "files": $OUT_FILES,
      "bytes": $OUT_BYTES,
      "purpose": "Small non-checkpoint experiment outputs: eval summaries, configs, logs, reports, and export metadata. Model weights are not stored here."
    }
  }
}
JSON

ssh "$REMOTE" "mkdir -p '$REMOTE_ROOT/docs' '$REMOTE_ROOT/outputs' '$REMOTE_ROOT/metadata'"
if [ -d "$PROJECT_ROOT/docs" ]; then
  rsync -a --delete "$PROJECT_ROOT/docs/" "$REMOTE:$REMOTE_ROOT/docs/"
fi
if [ -d "$PROJECT_ROOT/outputs" ]; then
  rsync -a --delete "$PROJECT_ROOT/outputs/" "$REMOTE:$REMOTE_ROOT/outputs/"
fi
scp "$META" "$REMOTE:$REMOTE_ROOT/metadata/artifacts-latest.json"
mkdir -p "$PROJECT_ROOT/artifacts"
cp "$META" "$PROJECT_ROOT/artifacts/solollm-artifacts.json"
rm -f "$META"
echo "Synced SoloLLM docs and outputs artifacts."
