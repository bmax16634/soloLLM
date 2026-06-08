#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

RUN_DIR="${RUN_DIR:-outputs/sologpt_v2/final_3b_modern_small_from300m}"
SHARD_DIR="${SHARD_DIR:-/home/bmx/_projects/soloLLM/data/tokenized_chunks}"
BACKEND_URL="${BACKEND_URL:-http://127.0.0.1:8787}"
V2_CHECKPOINT="${V2_CHECKPOINT:-$RUN_DIR/final_model.pt}"
V2_CONFIG="${V2_CONFIG:-sologpt_v2/config_modern_small.json}"
SHARDS="${SHARDS:-58:60}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_BATCHES="${MAX_BATCHES:-}"
LOG_PATH="${LOG_PATH:-$RUN_DIR/phase4_full_eval.log}"
V2_OUTPUT="${V2_OUTPUT:-$RUN_DIR/eval_v2_test_full.json}"
GPT2_OUTPUT="${GPT2_OUTPUT:-$RUN_DIR/eval_gpt2_test_full.json}"
RUN_GPT2="${RUN_GPT2:-1}"

mkdir -p "$RUN_DIR"
exec >> "$LOG_PATH" 2>&1

echo "=== SoloLLM Phase 4 full eval ==="
echo "started_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "started_at_az=$(TZ=America/Phoenix date '+%Y-%m-%d %I:%M:%S %p AZ')"
echo "root=$ROOT"
echo "run_dir=$RUN_DIR"
echo "shard_dir=$SHARD_DIR"
echo "shards=$SHARDS"
echo "batch_size=$BATCH_SIZE"
echo "max_batches=${MAX_BATCHES:-full}"
echo "run_gpt2=$RUN_GPT2"
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
  python - "$V2_OUTPUT" "$GPT2_OUTPUT" <<'PY'
import json
import math
import sys

v2_path, gpt2_path = sys.argv[1:3]

def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

try:
    v2 = load(v2_path)
    gpt2 = load(gpt2_path)
except Exception as exc:
    print(f"Phase 4 eval finished, but summary read failed: {exc}")
    raise SystemExit(0)

v2_ppl = v2.get("perplexity")
gpt2_ppl = gpt2.get("perplexity")
v2_loss = v2.get("loss")
gpt2_loss = gpt2.get("loss")
tokens = v2.get("tokens")

parts = ["Phase 4 full held-out eval finished"]
if isinstance(tokens, (int, float)):
    parts.append(f"{tokens / 1_000_000:.1f}M tokens")
if isinstance(v2_ppl, (int, float)) and math.isfinite(v2_ppl):
    parts.append(f"v2 PPL {v2_ppl:.2f}")
if isinstance(gpt2_ppl, (int, float)) and math.isfinite(gpt2_ppl):
    parts.append(f"GPT-2 PPL {gpt2_ppl:.2f}")
if all(isinstance(x, (int, float)) for x in [v2_loss, gpt2_loss]):
    parts.append(f"loss gap {v2_loss - gpt2_loss:+.4f}")
print(", ".join(parts) + ".")
PY
}

v2_summary_body() {
  python - "$V2_OUTPUT" <<'PY'
import json
import math
import sys

v2_path = sys.argv[1]

try:
    with open(v2_path, "r", encoding="utf-8") as f:
        v2 = json.load(f)
except Exception as exc:
    print(f"Phase 4 v2 eval finished, but summary read failed: {exc}")
    raise SystemExit(0)

v2_ppl = v2.get("perplexity")
v2_loss = v2.get("loss")
tokens = v2.get("tokens")
elapsed = v2.get("elapsed_sec")

parts = ["Phase 4 v2 full held-out eval finished"]
if isinstance(tokens, (int, float)):
    parts.append(f"{tokens / 1_000_000:.1f}M tokens")
if isinstance(v2_ppl, (int, float)) and math.isfinite(v2_ppl):
    parts.append(f"v2 PPL {v2_ppl:.2f}")
if isinstance(v2_loss, (int, float)) and math.isfinite(v2_loss):
    parts.append(f"loss {v2_loss:.4f}")
if isinstance(elapsed, (int, float)):
    parts.append(f"{elapsed / 60:.1f} min")
print(", ".join(parts) + ".")
PY
}

max_batches_args=()
if [[ -n "$MAX_BATCHES" ]]; then
  max_batches_args=(--max-batches "$MAX_BATCHES")
fi

if [[ ! -f "$V2_CHECKPOINT" ]]; then
  msg="Missing v2 checkpoint: $V2_CHECKPOINT"
  echo "$msg"
  post_reminder "SoloLLM Phase 4 eval did not start" "$msg" "Run directory: $ROOT/$RUN_DIR"
  exit 2
fi

echo "--- v2 eval start ---"
nice -n 10 ionice -c2 -n7 python -m eval.eval \
  --model v2 \
  --checkpoint "$V2_CHECKPOINT" \
  --config "$V2_CONFIG" \
  --shard-dir "$SHARD_DIR" \
  --shards "$SHARDS" \
  --batch-size "$BATCH_SIZE" \
  "${max_batches_args[@]}" \
  --output-json "$V2_OUTPUT" \
  --device cuda \
  --no-progress
v2_status=$?
echo "v2_exit_status=$v2_status"

if [[ "$v2_status" -ne 0 ]]; then
  body="Phase 4 v2 eval stopped with exit status $v2_status. Check $ROOT/$LOG_PATH."
  post_reminder "SoloLLM Phase 4 eval stopped" "$body" "Run directory: $ROOT/$RUN_DIR"
  exit "$v2_status"
fi

if [[ "$RUN_GPT2" == "0" || "$RUN_GPT2" == "false" ]]; then
  echo "finished_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "finished_at_az=$(TZ=America/Phoenix date '+%Y-%m-%d %I:%M:%S %p AZ')"
  body="$(v2_summary_body)"
  post_reminder "SoloLLM Phase 4 v2 eval complete" "$body" "Run directory: $ROOT/$RUN_DIR"
  exit 0
fi

echo "--- GPT-2 eval start ---"
nice -n 10 ionice -c2 -n7 python -m eval.eval \
  --model gpt2 \
  --shard-dir "$SHARD_DIR" \
  --shards "$SHARDS" \
  --batch-size "$BATCH_SIZE" \
  "${max_batches_args[@]}" \
  --output-json "$GPT2_OUTPUT" \
  --device cuda \
  --no-progress
gpt2_status=$?
echo "gpt2_exit_status=$gpt2_status"

echo "finished_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "finished_at_az=$(TZ=America/Phoenix date '+%Y-%m-%d %I:%M:%S %p AZ')"

if [[ "$gpt2_status" -eq 0 ]]; then
  body="$(summary_body)"
  post_reminder "SoloLLM Phase 4 eval complete" "$body" "Run directory: $ROOT/$RUN_DIR"
else
  body="Phase 4 GPT-2 eval stopped with exit status $gpt2_status. Check $ROOT/$LOG_PATH."
  post_reminder "SoloLLM Phase 4 eval stopped" "$body" "Run directory: $ROOT/$RUN_DIR"
fi

exit "$gpt2_status"
