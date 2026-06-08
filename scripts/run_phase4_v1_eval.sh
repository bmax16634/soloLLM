#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

RUN_DIR="${RUN_DIR:-outputs/sologpt_v2/final_3b_modern_small_from300m}"
SHARD_DIR="${SHARD_DIR:-/home/bmx/_projects/soloLLM/data/tokenized_chunks}"
BACKEND_URL="${BACKEND_URL:-http://127.0.0.1:8787}"
V1_CHECKPOINT="${V1_CHECKPOINT:-outputs/sologpt_v1/published/pytorch_model.safetensors}"
V1_CONFIG="${V1_CONFIG:-outputs/sologpt_v1/published/config/soloGPT_v1_config.json}"
SHARDS="${SHARDS:-58:60}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_BATCHES="${MAX_BATCHES:-}"
LOG_PATH="${LOG_PATH:-$RUN_DIR/phase4_v1_eval.log}"
V1_OUTPUT="${V1_OUTPUT:-$RUN_DIR/eval_v1_test_full.json}"

mkdir -p "$RUN_DIR"
exec >> "$LOG_PATH" 2>&1

echo "=== SoloLLM Phase 4 v1 eval ==="
echo "started_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "started_at_az=$(TZ=America/Phoenix date '+%Y-%m-%d %I:%M:%S %p AZ')"
echo "root=$ROOT"
echo "run_dir=$RUN_DIR"
echo "shard_dir=$SHARD_DIR"
echo "shards=$SHARDS"
echo "batch_size=$BATCH_SIZE"
echo "max_batches=${MAX_BATCHES:-full}"
echo "backend_url=$BACKEND_URL"
echo "pid=$$"

post_reminder() {
  local title="$1"
  local body="$2"
  local notes="$3"

  python - "$title" "$body" "$notes" <<'PY' | curl -fsS -X POST "$BACKEND_URL/api/reminders" -H "Content-Type: application/json" --data-binary @- || true
import json
import sys
from datetime import datetime, timezone

title, body, notes = sys.argv[1:4]
print(json.dumps({
    "title": title,
    "body": body,
    "notes": notes,
    "dueAt": datetime.now(timezone.utc).isoformat(),
}))
PY
  echo
}

summary_body() {
  python - "$V1_OUTPUT" <<'PY'
import json
import math
import sys

path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8") as f:
        result = json.load(f)
except Exception as exc:
    print(f"Phase 4 v1 eval finished, but summary read failed: {exc}")
    raise SystemExit(0)

ppl = result.get("perplexity")
loss = result.get("loss")
tokens = result.get("tokens")
parts = ["Phase 4 v1 held-out eval finished"]
if isinstance(tokens, (int, float)):
    parts.append(f"{tokens / 1_000_000:.1f}M tokens")
if isinstance(ppl, (int, float)) and math.isfinite(ppl):
    parts.append(f"v1 PPL {ppl:.2f}")
if isinstance(loss, (int, float)) and math.isfinite(loss):
    parts.append(f"loss {loss:.4f}")
print(", ".join(parts) + ".")
PY
}

max_batches_args=()
if [[ -n "$MAX_BATCHES" ]]; then
  max_batches_args=(--max-batches "$MAX_BATCHES")
fi

if [[ ! -f "$V1_CHECKPOINT" ]]; then
  msg="Missing v1 checkpoint: $V1_CHECKPOINT"
  echo "$msg"
  post_reminder "SoloLLM Phase 4 v1 eval did not start" "$msg" "Run directory: $ROOT/$RUN_DIR"
  exit 2
fi

nice -n 10 ionice -c2 -n7 python -m eval.eval \
  --model v1 \
  --checkpoint "$V1_CHECKPOINT" \
  --config "$V1_CONFIG" \
  --shard-dir "$SHARD_DIR" \
  --shards "$SHARDS" \
  --batch-size "$BATCH_SIZE" \
  "${max_batches_args[@]}" \
  --output-json "$V1_OUTPUT" \
  --device cuda \
  --no-progress
status=$?

echo "finished_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "finished_at_az=$(TZ=America/Phoenix date '+%Y-%m-%d %I:%M:%S %p AZ')"
echo "exit_status=$status"

if [[ "$status" -eq 0 ]]; then
  body="$(summary_body)"
  post_reminder "SoloLLM Phase 4 v1 eval complete" "$body" "Run directory: $ROOT/$RUN_DIR"
else
  body="Phase 4 v1 eval stopped with exit status $status. Check $ROOT/$LOG_PATH."
  post_reminder "SoloLLM Phase 4 v1 eval stopped" "$body" "Run directory: $ROOT/$RUN_DIR"
fi

exit "$status"
