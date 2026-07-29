#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/home/bmx/_projects/soloLLM}"
export PYTHONPATH="${ROOT}/src:${ROOT}:${PYTHONPATH:-}"
RUN_NAME="${RUN_NAME:-v4_smollm_reasoning_10b_plus_150m_1024}"
DATA_DIR="${DATA_DIR:-/home/bmx/_projects/soloLLM/data/v4_smollm_reasoning_10b_1024}"

TRAIN_OUT_DIR="${TRAIN_OUT_DIR:-${ROOT}/outputs/sologpt_v4/${RUN_NAME}}"
TRAIN_LOG_PATH="${TRAIN_LOG_PATH:-${ROOT}/outputs/sologpt_v4/${RUN_NAME}_resume_4b_to_10b.log}"
CONFIG_PATH="${CONFIG_PATH:-${ROOT}/configs/solollm-0.3-152m-base.json}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-${TRAIN_OUT_DIR}/checkpoints/latest.pt}"
SHARD_DIR="${SHARD_DIR:-${DATA_DIR}/chunks}"
TRAIN_SHARDS="${TRAIN_SHARDS:-477:1168}"
VAL_SHARDS="${VAL_SHARDS:-1169:1180}"
TRAIN_TARGET_TOKENS="${TRAIN_TARGET_TOKENS:-9800188928}"
EVAL_EVERY_TOKENS="${EVAL_EVERY_TOKENS:-250000000}"
TOKENS_PER_CHECKPOINT="${TOKENS_PER_CHECKPOINT:-2000000000}"
MAX_EVAL_TOKENS="${MAX_EVAL_TOKENS:-4194304}"
DEVICE="${DEVICE:-cuda}"

EVAL_LABEL="${EVAL_LABEL:-v4_smollm_reasoning_10b_150m}"
EVAL_OUT_DIR="${EVAL_OUT_DIR:-${ROOT}/outputs/eval_suites/${EVAL_LABEL}_gpt2_full_suite}"
EVAL_REPORT_DIR="${EVAL_REPORT_DIR:-${EVAL_OUT_DIR}/reports}"
EVAL_LOG_PATH="${EVAL_LOG_PATH:-${EVAL_OUT_DIR}/v3_eval_suite_full.log}"

METRICS_PATH="${TRAIN_OUT_DIR}/metrics.jsonl"
MILESTONE_SENT_PATH="${TRAIN_OUT_DIR}/.resume_4b_milestones_sent"

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

checkpoint_body() {
  python - "$RESUME_CHECKPOINT" <<'PY'
import sys
import torch
from pathlib import Path

path = Path(sys.argv[1])
ckpt = torch.load(path, map_location="cpu", weights_only=False)
meta = ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}
tokens = int(meta.get("tokens_total", 0) or 0)
steps = int(meta.get("optimizer_steps", 0) or 0)
status = meta.get("status", "unknown")
micro = meta.get("micro_step", "unknown")
print(f"resume checkpoint {tokens / 1_000_000_000:.2f}B tok, steps {steps}, status {status}, micro_step {micro}")
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
    run_status = summary.get("status")
    if run_status:
        parts.append(f"status {run_status}")
    parts.append(f"{tokens / 1_000_000_000:.2f}B total tok")
    parts.append(f"{processed / 1_000_000_000:.2f}B new tok")
    if elapsed:
        parts.append(f"{elapsed / 3600:.2f}h")
    if avg_tps:
        parts.append(f"{avg_tps / 1000:.1f}k tok/s")
    if final_train is not None:
        parts.append(f"train loss {float(final_train):.4f}")
    if best_val is not None:
        text = f"best val {float(best_val):.4f}"
        if best_ppl is not None and math.isfinite(float(best_ppl)):
            text += f" / PPL {float(best_ppl):.2f}"
        parts.append(text)
    if peak:
        parts.append(f"peak {float(peak):.1f}GB")

if not parts:
    parts.append(f"exit status {status}")
parts.append(f"log: {log_path}")
print("; ".join(parts))
PY
}

training_run_status() {
  python - "$TRAIN_OUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1]) / "metrics_summary.json"
if not summary_path.exists():
    print("missing")
else:
    print(json.loads(summary_path.read_text()).get("status", "unknown"))
PY
}

monitor_milestones() {
  mkdir -p "$TRAIN_OUT_DIR"
  touch "$MILESTONE_SENT_PATH"

  while true; do
    python - "$METRICS_PATH" "$MILESTONE_SENT_PATH" "$TRAIN_LOG_PATH" "$TRAIN_TARGET_TOKENS" <<'PY' |
import json
import math
import sys
from pathlib import Path

metrics_path = Path(sys.argv[1])
sent_path = Path(sys.argv[2])
log_path = Path(sys.argv[3])
target_tokens = int(sys.argv[4])
thresholds = [6_000_000_000, 8_000_000_000]

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
    print(f"{key}\tSoloLLM v4 150M resume milestone\t{'; '.join(parts)}")
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

mkdir -p "$(dirname "$TRAIN_LOG_PATH")" "$TRAIN_OUT_DIR" "$EVAL_OUT_DIR" "$EVAL_REPORT_DIR"
cd "$ROOT" || exit 2
exec >>"$TRAIN_LOG_PATH" 2>&1

echo "=== SoloLLM v4 SmolLM-reasoning 10B 150M resume from 4B plus full eval ==="
echo "started_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
TZ=America/Phoenix date "+started_at_az=%Y-%m-%d %I:%M:%S %p AZ"
echo "root=$ROOT"
echo "data_dir=$DATA_DIR"
echo "shard_dir=$SHARD_DIR"
echo "train_out_dir=$TRAIN_OUT_DIR"
echo "train_log_path=$TRAIN_LOG_PATH"
echo "config_path=$CONFIG_PATH"
echo "resume_checkpoint=$RESUME_CHECKPOINT"
echo "train_shards=$TRAIN_SHARDS"
echo "val_shards=$VAL_SHARDS"
echo "train_target_tokens=$TRAIN_TARGET_TOKENS"
echo "eval_label=$EVAL_LABEL"
echo "eval_out_dir=$EVAL_OUT_DIR"
echo "eval_log_path=$EVAL_LOG_PATH"
echo "pid=$$"

if [ ! -f "$RESUME_CHECKPOINT" ]; then
  notify "SoloLLM v4 resume did not start" "resume checkpoint missing: ${RESUME_CHECKPOINT}" "Run directory: ${TRAIN_OUT_DIR}"
  exit 2
fi
if [ ! -d "$SHARD_DIR" ]; then
  notify "SoloLLM v4 resume did not start" "shard dir missing: ${SHARD_DIR}" "Run directory: ${TRAIN_OUT_DIR}"
  exit 2
fi

ckpt_summary="$(checkpoint_body)"
notify \
  "SoloLLM v4 150M resume started" \
  "${ckpt_summary}; train shards ${TRAIN_SHARDS}; val shards ${VAL_SHARDS}; target ${TRAIN_TARGET_TOKENS}; log: ${TRAIN_LOG_PATH}" \
  "Run directory: ${TRAIN_OUT_DIR}"

monitor_milestones &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT

python -m solollm.train \
  --config "$CONFIG_PATH" \
  --output-dir "$TRAIN_OUT_DIR" \
  --shard-dir "$SHARD_DIR" \
  --train-shards "$TRAIN_SHARDS" \
  --val-shards "$VAL_SHARDS" \
  --resume "$RESUME_CHECKPOINT" \
  --max-tokens "$TRAIN_TARGET_TOKENS" \
  --max-eval-tokens "$MAX_EVAL_TOKENS" \
  --eval-every-tokens "$EVAL_EVERY_TOKENS" \
  --tokens-per-checkpoint "$TOKENS_PER_CHECKPOINT" \
  --log-every-opt-steps 100 \
  --device "$DEVICE" \
  --no-progress
train_status=$?

kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
trap - EXIT

echo "training_finished_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
TZ=America/Phoenix date "+training_finished_at_az=%Y-%m-%d %I:%M:%S %p AZ"
echo "training_exit_status=$train_status"

train_body="$(training_summary_body "$train_status")"
run_status="$(training_run_status)"
if [ "$train_status" -eq 0 ] && [ "$run_status" = "complete" ]; then
  notify "SoloLLM v4 150M 10B resume training complete" "$train_body" "Run directory: ${TRAIN_OUT_DIR}"
else
  notify "SoloLLM v4 150M 10B resume training stopped" "$train_body" "Run directory: ${TRAIN_OUT_DIR}"
  if [ "$train_status" -eq 0 ]; then
    exit 130
  fi
  exit "$train_status"
fi

echo "=== starting v4 full eval suite ==="
notify \
  "SoloLLM v4 full eval started" \
  "candidate ${TRAIN_OUT_DIR}/final_model.pt; output ${EVAL_OUT_DIR}; log: ${EVAL_LOG_PATH}" \
  "Run directory: ${EVAL_OUT_DIR}"

ROOT="$ROOT" \
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

if [ "$eval_status" -eq 0 ]; then
  notify "SoloLLM v4 150M 10B resume train plus eval complete" "training and full eval completed; eval log: ${EVAL_LOG_PATH}; train log: ${TRAIN_LOG_PATH}" "Run directory: ${TRAIN_OUT_DIR}"
else
  notify "SoloLLM v4 full eval failed" "training completed but eval failed with exit ${eval_status}; eval log: ${EVAL_LOG_PATH}; train log: ${TRAIN_LOG_PATH}" "Run directory: ${EVAL_OUT_DIR}"
fi

exit "$eval_status"
