#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

SOURCE_CKPT="${SOURCE_CKPT:-outputs/sologpt_v2/final_3b_modern_small_from300m/checkpoints/latest.pt}"
RUN_DIR="${RUN_DIR:-outputs/sologpt_v2/stretch_5p85b_modern_small_from3b}"
SHARD_DIR="${SHARD_DIR:-/home/bmx/_projects/soloLLM/data/tokenized_chunks}"
BACKEND_URL="${BACKEND_URL:-http://127.0.0.1:8787}"
MAX_TOKENS="${MAX_TOKENS:-5850000000}"
LEARNING_RATE="${LEARNING_RATE:-0.00003}"
EVAL_EVERY_TOKENS="${EVAL_EVERY_TOKENS:-100000000}"
TOKENS_PER_CHECKPOINT="${TOKENS_PER_CHECKPOINT:-100000000}"
LOG_PATH="${LOG_PATH:-$RUN_DIR/stretch_5p85b.log}"

mkdir -p "$RUN_DIR"
exec >> "$LOG_PATH" 2>&1

echo "=== SoloLLM v2 modern-small 5.85B stretch run ==="
echo "started_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "started_at_az=$(TZ=America/Phoenix date '+%Y-%m-%d %I:%M:%S %p AZ')"
echo "root=$ROOT"
echo "source_ckpt=$SOURCE_CKPT"
echo "run_dir=$RUN_DIR"
echo "shard_dir=$SHARD_DIR"
echo "max_tokens=$MAX_TOKENS"
echo "learning_rate=$LEARNING_RATE"
echo "eval_every_tokens=$EVAL_EVERY_TOKENS"
echo "tokens_per_checkpoint=$TOKENS_PER_CHECKPOINT"
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
  python - "$RUN_DIR/metrics_summary.json" <<'PY'
import json
import math
import sys

path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8") as f:
        summary = json.load(f)
except Exception as exc:
    print(f"Finished, but could not read metrics_summary.json: {exc}")
    raise SystemExit(0)

tokens = summary.get("tokens_total")
processed = summary.get("tokens_processed_this_run")
train_loss = summary.get("final_train_loss")
val_ppl = summary.get("best_val_ppl")
vram = summary.get("gpu_peak_mem_gb")
elapsed = summary.get("total_time_sec")
throughput = summary.get("tokens_per_sec_avg")

parts = ["v2-modern-small 5.85B stretch finished"]
if isinstance(tokens, (int, float)):
    parts.append(f"{tokens / 1_000_000_000:.3f}B total tokens")
if isinstance(processed, (int, float)):
    parts.append(f"{processed / 1_000_000_000:.3f}B new tokens")
if isinstance(train_loss, (int, float)):
    parts.append(f"train loss {train_loss:.3f}")
if isinstance(val_ppl, (int, float)) and math.isfinite(val_ppl):
    parts.append(f"best val PPL {val_ppl:.2f}")
if isinstance(throughput, (int, float)):
    parts.append(f"{throughput:,.0f} tok/s")
if isinstance(vram, (int, float)):
    parts.append(f"peak VRAM {vram:.2f}GB")
if isinstance(elapsed, (int, float)):
    parts.append(f"wall time {elapsed / 3600:.1f} hr")
print(", ".join(parts) + ".")
PY
}

if [[ ! -f "$SOURCE_CKPT" ]]; then
  msg="Resume checkpoint is missing: $SOURCE_CKPT"
  echo "$msg"
  post_reminder "SoloLLM v2 stretch did not start" "$msg" "Run directory: $ROOT/$RUN_DIR"
  exit 2
fi

nice -n 10 ionice -c2 -n7 python -m sologpt_v2.pretrain \
  --config sologpt_v2/config_modern_small.json \
  --shard-dir "$SHARD_DIR" \
  --output-dir "$RUN_DIR" \
  --train-shards 0:54 \
  --val-shards 55:57 \
  --resume "$SOURCE_CKPT" \
  --max-tokens "$MAX_TOKENS" \
  --learning-rate "$LEARNING_RATE" \
  --max-eval-tokens 10000000 \
  --eval-every-tokens "$EVAL_EVERY_TOKENS" \
  --tokens-per-checkpoint "$TOKENS_PER_CHECKPOINT" \
  --no-progress

status=$?
echo "finished_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "finished_at_az=$(TZ=America/Phoenix date '+%Y-%m-%d %I:%M:%S %p AZ')"
echo "exit_status=$status"

if [[ "$status" -eq 0 ]]; then
  body="$(summary_body)"
  post_reminder "SoloLLM v2 5.85B stretch complete" "$body" "Run directory: $ROOT/$RUN_DIR"
else
  body="v2-modern-small 5.85B stretch stopped with exit status $status. Check $ROOT/$LOG_PATH."
  post_reminder "SoloLLM v2 5.85B stretch stopped" "$body" "Run directory: $ROOT/$RUN_DIR"
fi

exit "$status"
