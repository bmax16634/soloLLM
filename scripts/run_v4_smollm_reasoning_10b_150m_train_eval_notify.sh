#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/home/bmx/_projects/ariya/userdata/projects/soloLLM}"
RUN_NAME="${RUN_NAME:-v4_smollm_reasoning_10b_plus_150m_1024}"
MANIFEST="${MANIFEST:-${ROOT}/sologpt_v4/data_mix_smollm_reasoning_10b.yaml}"
DATA_DIR="${DATA_DIR:-/home/bmx/_projects/soloLLM/data/v4_smollm_reasoning_10b_1024}"
DATASET_TARGET_TOKENS="${DATASET_TARGET_TOKENS:-10000000000}"
BUILD_IF_MISSING="${BUILD_IF_MISSING:-1}"
OVERWRITE_DATA="${OVERWRITE_DATA:-0}"

TRAIN_OUT_DIR="${TRAIN_OUT_DIR:-${ROOT}/outputs/sologpt_v4/${RUN_NAME}}"
TRAIN_LOG_PATH="${TRAIN_LOG_PATH:-${ROOT}/outputs/sologpt_v4/${RUN_NAME}.log}"
CONFIG_PATH="${CONFIG_PATH:-${ROOT}/sologpt_v3/config_plus_150m_1024.json}"
EVAL_EVERY_TOKENS="${EVAL_EVERY_TOKENS:-250000000}"
TOKENS_PER_CHECKPOINT="${TOKENS_PER_CHECKPOINT:-2000000000}"
MAX_EVAL_TOKENS="${MAX_EVAL_TOKENS:-4194304}"
DEVICE="${DEVICE:-cuda}"

EVAL_LABEL="${EVAL_LABEL:-v4_smollm_reasoning_10b_150m}"
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

dataset_body() {
  python - "$DATA_DIR" "$MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

data_dir = Path(sys.argv[1])
manifest = Path(sys.argv[2])
stats_path = data_dir / "build_stats.json"
parts = [f"manifest {manifest.name}"]
if stats_path.exists():
    stats = json.loads(stats_path.read_text())
    progress = stats.get("progress", {})
    accepted = int(progress.get("accepted_tokens", 0) or 0)
    target = int(progress.get("target_tokens", 0) or 0)
    elapsed = float(progress.get("elapsed_sec", 0.0) or 0.0)
    tps = float(progress.get("tokens_per_sec", 0.0) or 0.0)
    parts.append(f"{accepted / 1_000_000_000:.2f}B/{target / 1_000_000_000:.2f}B tokens")
    if elapsed:
        parts.append(f"{elapsed / 3600:.2f}h")
    if tps:
        parts.append(f"{tps / 1000:.0f}k tok/s")
    ranges = stats.get("shard_ranges", {})
    train = ranges.get("train", {})
    val = ranges.get("val", {})
    if train:
        parts.append(f"train {train.get('range')} ({int(train.get('tokens', 0)) / 1_000_000_000:.2f}B tok)")
    if val:
        parts.append(f"val {val.get('range')}")
else:
    parts.append("build_stats missing")
parts.append(f"data: {data_dir}")
print("; ".join(parts))
PY
}

read_split_info() {
  python - "$DATA_DIR" <<'PY'
import json
import sys
from pathlib import Path

data_dir = Path(sys.argv[1])
splits = json.loads((data_dir / "splits.json").read_text())
ranges = splits["shard_ranges"]
print(ranges["train"]["range"])
print(ranges["val"]["range"])
print(splits["chunks_dir"])
print(int(ranges["train"]["tokens"]))
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
        text = f"best val {float(best_val):.4f}"
        if best_ppl is not None and math.isfinite(float(best_ppl)):
            text += f" / PPL {float(best_ppl):.2f}"
        parts.append(text)
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
    python - "$METRICS_PATH" "$MILESTONE_SENT_PATH" "$TRAIN_LOG_PATH" "$TRAIN_TARGET_TOKENS" <<'PY' |
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
    print(f"{key}\tSoloLLM v4 150M training milestone\t{'; '.join(parts)}")
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

echo "=== SoloLLM v4 SmolLM-reasoning 10B 150M train plus full eval ==="
echo "started_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
TZ=America/Phoenix date "+started_at_az=%Y-%m-%d %I:%M:%S %p AZ"
echo "root=$ROOT"
echo "manifest=$MANIFEST"
echo "data_dir=$DATA_DIR"
echo "dataset_target_tokens=$DATASET_TARGET_TOKENS"
echo "train_out_dir=$TRAIN_OUT_DIR"
echo "train_log_path=$TRAIN_LOG_PATH"
echo "config_path=$CONFIG_PATH"
echo "eval_label=$EVAL_LABEL"
echo "eval_out_dir=$EVAL_OUT_DIR"
echo "eval_log_path=$EVAL_LOG_PATH"
echo "pid=$$"

notify \
  "SoloLLM v4 150M 10B run started" \
  "building dataset if needed, then fresh 150M train, 2B milestones, and full eval; log: ${TRAIN_LOG_PATH}" \
  "Run directory: ${TRAIN_OUT_DIR}"

if [ ! -f "$DATA_DIR/splits.json" ]; then
  if [ "$BUILD_IF_MISSING" != "1" ]; then
    echo "dataset_missing=$DATA_DIR"
    notify "SoloLLM v4 dataset missing" "Dataset missing and BUILD_IF_MISSING != 1; log: ${TRAIN_LOG_PATH}" "Dataset: ${DATA_DIR}"
    exit 2
  fi
  build_args=()
  if [ "$OVERWRITE_DATA" = "1" ]; then
    build_args+=(--overwrite)
  fi
  echo "=== building v4 10B dataset ==="
  python scripts/build_v3_pilot_dataset.py \
    --manifest "$MANIFEST" \
    --output-dir "$DATA_DIR" \
    --target-tokens "$DATASET_TARGET_TOKENS" \
    --stats-every-docs 10000 \
    "${build_args[@]}"
  build_status=$?
  build_summary="$(dataset_body)"
  if [ "$build_status" -eq 0 ]; then
    notify "SoloLLM v4 10B dataset built" "$build_summary" "Dataset: ${DATA_DIR}"
  else
    notify "SoloLLM v4 10B dataset build failed" "$build_summary; log: ${TRAIN_LOG_PATH}" "Dataset: ${DATA_DIR}"
    exit "$build_status"
  fi
else
  echo "dataset_exists=$DATA_DIR"
  notify "SoloLLM v4 10B dataset found" "$(dataset_body)" "Dataset: ${DATA_DIR}"
fi

if [ -f "$METRICS_PATH" ] || [ -f "${TRAIN_OUT_DIR}/final_model.pt" ]; then
  echo "refusing_to_mix_with_existing_run=$TRAIN_OUT_DIR"
  notify "SoloLLM v4 150M run not started" "Output directory already contains metrics or final_model.pt; log: ${TRAIN_LOG_PATH}" "Run directory: ${TRAIN_OUT_DIR}"
  exit 2
fi

mapfile -t split_info < <(read_split_info)
TRAIN_SHARDS="${TRAIN_SHARDS:-${split_info[0]}}"
VAL_SHARDS="${VAL_SHARDS:-${split_info[1]}}"
SHARD_DIR="${SHARD_DIR:-${split_info[2]}}"
TRAIN_TARGET_TOKENS="${TRAIN_TARGET_TOKENS:-${split_info[3]}}"

echo "train_shards=$TRAIN_SHARDS"
echo "val_shards=$VAL_SHARDS"
echo "shard_dir=$SHARD_DIR"
echo "train_target_tokens=$TRAIN_TARGET_TOKENS"

notify \
  "SoloLLM v4 150M training started" \
  "train shards ${TRAIN_SHARDS}; val shards ${VAL_SHARDS}; target ${TRAIN_TARGET_TOKENS} tokens; checkpoints every ${TOKENS_PER_CHECKPOINT}; log: ${TRAIN_LOG_PATH}" \
  "Run directory: ${TRAIN_OUT_DIR}"

monitor_milestones &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT

python -m sologpt_v2.pretrain \
  --config "$CONFIG_PATH" \
  --output-dir "$TRAIN_OUT_DIR" \
  --shard-dir "$SHARD_DIR" \
  --train-shards "$TRAIN_SHARDS" \
  --val-shards "$VAL_SHARDS" \
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
if [ "$train_status" -eq 0 ]; then
  notify "SoloLLM v4 150M 10B training complete" "$train_body" "Run directory: ${TRAIN_OUT_DIR}"
else
  notify "SoloLLM v4 150M 10B training failed" "$train_body" "Run directory: ${TRAIN_OUT_DIR}"
  exit "$train_status"
fi

echo "=== starting v4 full eval suite ==="
notify \
  "SoloLLM v4 full eval started" \
  "candidate ${TRAIN_OUT_DIR}/final_model.pt; output ${EVAL_OUT_DIR}; log: ${EVAL_LOG_PATH}" \
  "Run directory: ${EVAL_OUT_DIR}"

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
  notify "SoloLLM v4 150M 10B train plus eval complete" "training and full eval completed; eval log: ${EVAL_LOG_PATH}; train log: ${TRAIN_LOG_PATH}" "Run directory: ${TRAIN_OUT_DIR}"
else
  notify "SoloLLM v4 full eval failed" "training completed but eval failed with exit ${eval_status}; eval log: ${EVAL_LOG_PATH}; train log: ${TRAIN_LOG_PATH}" "Run directory: ${EVAL_OUT_DIR}"
fi

exit "$eval_status"
