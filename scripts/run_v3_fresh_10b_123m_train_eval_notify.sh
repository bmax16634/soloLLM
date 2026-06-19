#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/home/bmx/_projects/ariya/userdata/projects/soloLLM}"
RUN_NAME="${RUN_NAME:-v3_fresh_10b_gpt2scale_123m_1024_10bdata}"
TRAIN_OUT_DIR="${TRAIN_OUT_DIR:-${ROOT}/outputs/sologpt_v3/${RUN_NAME}}"
TRAIN_LOG_PATH="${TRAIN_LOG_PATH:-${ROOT}/outputs/sologpt_v3/${RUN_NAME}.log}"
CONFIG_PATH="${CONFIG_PATH:-${ROOT}/sologpt_v3/config_gpt2_scale_1024.json}"
SHARD_DIR="${SHARD_DIR:-/home/bmx/_projects/soloLLM/data/v3_10b_1024/chunks}"
TRAIN_SHARDS="${TRAIN_SHARDS:-0:1168}"
VAL_SHARDS="${VAL_SHARDS:-1169:1180}"

# One clean pass over the train split from data/v3_10b_1024/splits.json.
TARGET_TOKENS="${TARGET_TOKENS:-9800728576}"
EVAL_EVERY_TOKENS="${EVAL_EVERY_TOKENS:-250000000}"
TOKENS_PER_CHECKPOINT="${TOKENS_PER_CHECKPOINT:-500000000}"

EVAL_LABEL="${EVAL_LABEL:-v3_fresh_10b_123m_10bdata}"
EVAL_OUT_DIR="${EVAL_OUT_DIR:-${ROOT}/outputs/eval_suites/${EVAL_LABEL}_gpt2_full_suite}"
EVAL_REPORT_DIR="${EVAL_REPORT_DIR:-${EVAL_OUT_DIR}/reports}"
EVAL_LOG_PATH="${EVAL_LOG_PATH:-${EVAL_OUT_DIR}/v3_eval_suite_full.log}"

METRICS_PATH="${TRAIN_OUT_DIR}/metrics.jsonl"
MILESTONE_SENT_PATH="${TRAIN_OUT_DIR}/.milestones_sent"

notify() {
  local title="$1"
  local body="$2"
  local notes="${3:-}"
  python - "$title" "$body" "$notes" <<'PY'
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone

title, body, notes = sys.argv[1], sys.argv[2], sys.argv[3]
payload = {
    "title": title,
    "body": body,
    "dueAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "notes": notes,
}
request = urllib.request.Request(
    "http://127.0.0.1:8787/api/reminders",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
try:
    with urllib.request.urlopen(request, timeout=10) as response:
        print(response.read().decode("utf-8"))
except (urllib.error.URLError, TimeoutError) as exc:
    print(f"notification_error={exc}")
PY
}

training_summary_body() {
  local status="$1"
  python - "$TRAIN_OUT_DIR" "$TRAIN_LOG_PATH" "$status" <<'PY'
import json
import math
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
log_path = Path(sys.argv[2])
status = int(sys.argv[3])
summary_path = out_dir / "metrics_summary.json"
metrics_path = out_dir / "metrics.jsonl"

parts = []
if summary_path.exists():
    summary = json.loads(summary_path.read_text())
    tokens = int(summary.get("tokens_total", 0) or 0)
    processed = int(summary.get("tokens_processed_this_run", 0) or 0)
    elapsed = float(summary.get("total_time_sec", 0.0) or 0.0)
    avg_tps = float(summary.get("tokens_per_sec_avg", 0.0) or 0.0)
    final_train = summary.get("final_train_loss")
    best_val = summary.get("best_val_loss")
    best_ppl = summary.get("best_val_ppl")
    peak = summary.get("gpu_peak_mem_gb")
    parts.append(f"{tokens / 1_000_000_000:.2f}B total tok")
    parts.append(f"{processed / 1_000_000_000:.2f}B new tok")
    if elapsed:
        parts.append(f"{elapsed / 3600:.2f}h")
    if avg_tps:
        parts.append(f"{avg_tps / 1000:.1f}k tok/s")
    if final_train is not None:
        parts.append(f"train loss {float(final_train):.4f}")
    if best_val is not None:
        val_text = f"best val {float(best_val):.4f}"
        if best_ppl is not None and math.isfinite(float(best_ppl)):
            val_text += f" / PPL {float(best_ppl):.2f}"
        parts.append(val_text)
    if peak:
        parts.append(f"peak {float(peak):.1f}GB")

if not parts and metrics_path.exists():
    parts.append("metrics_summary missing; metrics.jsonl exists")
if not parts:
    parts.append(f"exit status {status}")

parts.append(f"log: {log_path}")
print("; ".join(parts))
PY
}

monitor_milestones() {
  mkdir -p "$TRAIN_OUT_DIR"
  touch "$MILESTONE_SENT_PATH"

  while true; do
    python - "$METRICS_PATH" "$MILESTONE_SENT_PATH" "$TRAIN_LOG_PATH" "$TARGET_TOKENS" <<'PY' |
import json
import math
import sys
from pathlib import Path

metrics_path = Path(sys.argv[1])
sent_path = Path(sys.argv[2])
log_path = Path(sys.argv[3])
target_tokens = int(sys.argv[4])
thresholds = [2_000_000_000, 4_000_000_000, 6_000_000_000, 8_000_000_000]

if not metrics_path.exists():
    raise SystemExit

sent = set()
if sent_path.exists():
    sent = {line.strip() for line in sent_path.read_text().splitlines() if line.strip()}

latest = None
last_val = None
with metrics_path.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "tokens_total" in row:
            latest = row
        if row.get("type") == "validation":
            last_val = row

if not latest:
    raise SystemExit

tokens = int(latest.get("tokens_total", 0) or 0)
tps = float(latest.get("tokens_per_sec", 0.0) or 0.0)
train_loss = latest.get("train_loss")

for threshold in thresholds:
    key = str(threshold)
    if tokens < threshold or key in sent:
        continue

    parts = [f"{threshold / 1_000_000_000:.0f}B tokens reached"]
    parts.append(f"progress {100 * tokens / max(target_tokens, 1):.1f}%")
    if tps:
        remaining_sec = max(target_tokens - tokens, 0) / tps
        parts.append(f"{tps / 1000:.1f}k tok/s")
        parts.append(f"ETA train {remaining_sec / 3600:.1f}h")
    if train_loss is not None:
        parts.append(f"train loss {float(train_loss):.4f}")
    if last_val:
        val = float(last_val.get("val_loss", float("nan")))
        ppl = last_val.get("val_ppl")
        if math.isfinite(val):
            text = f"last val {val:.4f}"
            if ppl is not None and math.isfinite(float(ppl)):
                text += f" / PPL {float(ppl):.2f}"
            parts.append(text)
    parts.append(f"log: {log_path}")
    print(f"{key}\tSoloLLM v3 123M training milestone\t{'; '.join(parts)}")
PY
    while IFS=$'\t' read -r key title body; do
      if [ -n "${key:-}" ] && [ -n "${title:-}" ] && [ -n "${body:-}" ]; then
        notify "$title" "$body" "Run directory: ${TRAIN_OUT_DIR}"
        echo "$key" >>"$MILESTONE_SENT_PATH"
      fi
    done
    sleep 120
  done
}

mkdir -p "$(dirname "$TRAIN_LOG_PATH")" "$TRAIN_OUT_DIR"
cd "$ROOT" || exit 2

exec >>"$TRAIN_LOG_PATH" 2>&1

echo "=== SoloLLM v3 fresh 10B-dataset 123M train plus full eval ==="
echo "started_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
TZ=America/Phoenix date "+started_at_az=%Y-%m-%d %I:%M:%S %p AZ"
echo "root=$ROOT"
echo "train_out_dir=$TRAIN_OUT_DIR"
echo "train_log_path=$TRAIN_LOG_PATH"
echo "config_path=$CONFIG_PATH"
echo "shard_dir=$SHARD_DIR"
echo "train_shards=$TRAIN_SHARDS"
echo "val_shards=$VAL_SHARDS"
echo "target_tokens=$TARGET_TOKENS"
echo "eval_label=$EVAL_LABEL"
echo "eval_out_dir=$EVAL_OUT_DIR"
echo "eval_log_path=$EVAL_LOG_PATH"
echo "pid=$$"

if [ -f "$METRICS_PATH" ] || [ -f "${TRAIN_OUT_DIR}/final_model.pt" ]; then
  echo "refusing_to_mix_with_existing_run=$TRAIN_OUT_DIR"
  notify "SoloLLM v3 123M run not started" "Output directory already contains metrics or final_model.pt; log: ${TRAIN_LOG_PATH}" "Run directory: ${TRAIN_OUT_DIR}"
  exit 2
fi

monitor_milestones &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT

python -m sologpt_v2.pretrain \
  --config "$CONFIG_PATH" \
  --output-dir "$TRAIN_OUT_DIR" \
  --shard-dir "$SHARD_DIR" \
  --train-shards "$TRAIN_SHARDS" \
  --val-shards "$VAL_SHARDS" \
  --max-tokens "$TARGET_TOKENS" \
  --max-eval-tokens 4194304 \
  --eval-every-tokens "$EVAL_EVERY_TOKENS" \
  --tokens-per-checkpoint "$TOKENS_PER_CHECKPOINT" \
  --log-every-opt-steps 100 \
  --device cuda \
  --no-progress

train_status=$?
kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
trap - EXIT

echo "training_finished_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
TZ=America/Phoenix date "+training_finished_at_az=%Y-%m-%d %I:%M:%S %p AZ"
echo "training_exit_status=$train_status"

train_body="$(training_summary_body "$train_status")"
if [ "$train_status" -eq 0 ]; then
  notify "SoloLLM v3 123M 10B-dataset training complete" "$train_body" "Run directory: ${TRAIN_OUT_DIR}"
else
  notify "SoloLLM v3 123M 10B-dataset training failed" "$train_body" "Run directory: ${TRAIN_OUT_DIR}"
  exit "$train_status"
fi

echo "=== starting full eval suite ==="
LABEL="$EVAL_LABEL" \
OUT_DIR="$EVAL_OUT_DIR" \
REPORT_DIR="$EVAL_REPORT_DIR" \
LOG_PATH="$EVAL_LOG_PATH" \
CANDIDATE_CHECKPOINT="${TRAIN_OUT_DIR}/final_model.pt" \
CANDIDATE_CONFIG="$CONFIG_PATH" \
  bash "${ROOT}/scripts/run_v3_1b_full_eval_notify.sh"

eval_status=$?

echo "eval_finished_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
TZ=America/Phoenix date "+eval_finished_at_az=%Y-%m-%d %I:%M:%S %p AZ"
echo "eval_exit_status=$eval_status"

exit "$eval_status"
