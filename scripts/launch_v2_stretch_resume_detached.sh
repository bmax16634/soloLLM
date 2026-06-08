#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_DIR="${RUN_DIR:-outputs/sologpt_v2/stretch_5p85b_modern_small_from3b}"
SOURCE_CKPT="${SOURCE_CKPT:-$RUN_DIR/checkpoints/latest.pt}"
BACKEND_URL="${BACKEND_URL:-http://127.0.0.1:8787}"
NOHUP_LOG="${NOHUP_LOG:-/tmp/sologpt_v2_stretch_5p85b_resume.nohup}"
PID_FILE="${PID_FILE:-$RUN_DIR/launcher.pid}"
LOG_PATH="${LOG_PATH:-$RUN_DIR/stretch_5p85b.log}"
MAX_TOKENS="${MAX_TOKENS:-5850000000}"

mkdir -p "$RUN_DIR"

if [[ ! -f "$SOURCE_CKPT" ]]; then
  echo "Resume checkpoint missing: $SOURCE_CKPT" >&2
  exit 2
fi

if pgrep -af "python -m sologpt_v2.pretrain" | grep -F "$RUN_DIR" | grep -v grep >/dev/null; then
  echo "A stretch training run already appears to be running for $RUN_DIR."
  pgrep -af "python -m sologpt_v2.pretrain" | grep -F "$RUN_DIR" | grep -v grep || true
  exit 0
fi

RUN_DIR="$RUN_DIR" \
SOURCE_CKPT="$SOURCE_CKPT" \
BACKEND_URL="$BACKEND_URL" \
MAX_TOKENS="$MAX_TOKENS" \
LOG_PATH="$LOG_PATH" \
nohup bash scripts/run_v2_modern_small_stretch_5p85b.sh >"$NOHUP_LOG" 2>&1 &

pid=$!
echo "$pid" > "$PID_FILE"

echo "Started detached stretch training resume."
echo "pid=$pid"
echo "pid_file=$PID_FILE"
echo "run_dir=$RUN_DIR"
echo "source_ckpt=$SOURCE_CKPT"
echo "log=$LOG_PATH"
echo "nohup_log=$NOHUP_LOG"
echo "max_tokens=$MAX_TOKENS"
echo "started_at_az=$(TZ=America/Phoenix date '+%Y-%m-%d %I:%M:%S %p AZ')"
echo "notification=existing run script posts Ariya reminder when complete or stopped"
