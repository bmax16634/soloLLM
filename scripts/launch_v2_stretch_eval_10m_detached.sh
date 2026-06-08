#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_DIR="${RUN_DIR:-outputs/sologpt_v2/stretch_5p85b_modern_small_from3b}"
NOHUP_LOG="${NOHUP_LOG:-/tmp/sologpt_v2_stretch_eval_10m.nohup}"
PID_FILE="${PID_FILE:-$RUN_DIR/stretch_eval_10m.pid}"
LOG_PATH="${LOG_PATH:-$RUN_DIR/stretch_eval_10m.log}"

mkdir -p "$RUN_DIR"

if pgrep -af "python -m eval.eval|scripts/run_phase4_full_eval.sh" | grep -v grep | grep -q "$RUN_DIR"; then
  echo "A stretch eval appears to already be running for $RUN_DIR."
  pgrep -af "python -m eval.eval|scripts/run_phase4_full_eval.sh" | grep "$RUN_DIR" || true
  exit 0
fi

RUN_DIR="$RUN_DIR" \
V2_CHECKPOINT="$RUN_DIR/checkpoints/latest.pt" \
V2_OUTPUT="$RUN_DIR/eval_v2_5p6b_test_10m.json" \
GPT2_OUTPUT="$RUN_DIR/eval_gpt2_test_10m.json" \
MAX_BATCHES=407 \
LOG_PATH="$LOG_PATH" \
nohup bash scripts/run_phase4_full_eval.sh >"$NOHUP_LOG" 2>&1 &

pid=$!
echo "$pid" > "$PID_FILE"

echo "Started detached stretch eval."
echo "pid=$pid"
echo "pid_file=$PID_FILE"
echo "log=$LOG_PATH"
echo "nohup_log=$NOHUP_LOG"
echo "started_at_az=$(TZ=America/Phoenix date '+%Y-%m-%d %I:%M:%S %p AZ')"
echo "expected_runtime=about 5-8 minutes if CUDA is visible"
