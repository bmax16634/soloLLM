#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

SOURCE_CKPT="${SOURCE_CKPT:-outputs/sologpt_v2/pilot_300m_modern_small/checkpoints/ckpt_100M.pt}"
RUN_DIR="${RUN_DIR:-outputs/sologpt_v2/pilot_300m_modern_small_resume100m}"
SHARD_DIR="${SHARD_DIR:-/home/bmx/_projects/soloLLM/data/tokenized_chunks}"
BACKEND_URL="${BACKEND_URL:-http://127.0.0.1:8787}"
LOG_PATH="${LOG_PATH:-$RUN_DIR/resume_300m.log}"

mkdir -p "$RUN_DIR"
exec >> "$LOG_PATH" 2>&1

echo "=== SoloLLM v2 modern-small 300M resume ==="
echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "root=$ROOT"
echo "source_ckpt=$SOURCE_CKPT"
echo "run_dir=$RUN_DIR"
echo "shard_dir=$SHARD_DIR"
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
train_loss = summary.get("final_train_loss")
val_ppl = summary.get("best_val_ppl")
vram = summary.get("gpu_peak_mem_gb")
elapsed = summary.get("total_time_sec")

parts = ["v2-modern-small 300M pilot finished"]
if isinstance(tokens, (int, float)):
    parts.append(f"{tokens / 1_000_000:.1f}M tokens")
if isinstance(train_loss, (int, float)):
    parts.append(f"train loss {train_loss:.3f}")
if isinstance(val_ppl, (int, float)) and math.isfinite(val_ppl):
    parts.append(f"best val PPL {val_ppl:.2f}")
if isinstance(vram, (int, float)):
    parts.append(f"peak VRAM {vram:.2f}GB")
if isinstance(elapsed, (int, float)):
    parts.append(f"resume wall time {elapsed / 60:.1f} min")
print(", ".join(parts) + ".")
PY
}

if [[ ! -f "$SOURCE_CKPT" ]]; then
  msg="Resume checkpoint is missing: $SOURCE_CKPT"
  echo "$msg"
  post_reminder "SoloLLM 300M pilot did not start" "$msg" "Run directory: $ROOT/$RUN_DIR"
  exit 2
fi

nice -n 10 ionice -c2 -n7 python -m sologpt_v2.pretrain \
  --config sologpt_v2/config_modern_small.json \
  --shard-dir "$SHARD_DIR" \
  --output-dir "$RUN_DIR" \
  --train-shards 0:54 \
  --val-shards 55:57 \
  --resume "$SOURCE_CKPT" \
  --max-tokens 300000000 \
  --max-eval-tokens 10000000 \
  --eval-every-tokens 100000000 \
  --tokens-per-checkpoint 100000000 \
  --no-progress

status=$?
echo "finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "exit_status=$status"

if [[ "$status" -eq 0 ]]; then
  body="$(summary_body)"
  post_reminder "SoloLLM 300M pilot complete" "$body" "Run directory: $ROOT/$RUN_DIR"
else
  body="v2-modern-small 300M resume stopped with exit status $status. Check $ROOT/$LOG_PATH."
  post_reminder "SoloLLM 300M pilot stopped" "$body" "Run directory: $ROOT/$RUN_DIR"
fi

exit "$status"
