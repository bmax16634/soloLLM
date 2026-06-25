#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/home/bmx/_projects/ariya/userdata/projects/soloLLM}"
MIX_NAME="${MIX_NAME:-quality_web}"
MANIFEST="${MANIFEST:-${ROOT}/sologpt_v4/data_mix_${MIX_NAME}_300m.yaml}"
DATA_DIR="${DATA_DIR:-/home/bmx/_projects/soloLLM/data/v4_${MIX_NAME}_300m_1024}"
RUN_OUT_DIR="${RUN_OUT_DIR:-${ROOT}/outputs/sologpt_v4/data_proxy_${MIX_NAME}_57m_300m}"
RUN_DIR="${RUN_DIR:-${ROOT}/outputs/sologpt_v4/data_proxy_${MIX_NAME}}"
LOG_PATH="${LOG_PATH:-${RUN_DIR}/data_proxy_${MIX_NAME}.log}"
REPORT_PATH="${REPORT_PATH:-${ROOT}/docs/results/v4_data_proxy_${MIX_NAME}_300m.md}"
CONFIG_PATH="${CONFIG_PATH:-${ROOT}/sologpt_v3/config_proxy_v3_style_60m_1024.json}"
TARGET_TOKENS="${TARGET_TOKENS:-300000000}"
MAX_EVAL_TOKENS="${MAX_EVAL_TOKENS:-2097152}"
EVAL_EVERY_TOKENS="${EVAL_EVERY_TOKENS:-50000000}"
TOKENS_PER_CHECKPOINT="${TOKENS_PER_CHECKPOINT:-100000000}"
DEVICE="${DEVICE:-cuda}"
BUILD_IF_MISSING="${BUILD_IF_MISSING:-1}"
OVERWRITE_DATA="${OVERWRITE_DATA:-0}"
NOTIFY="${NOTIFY:-1}"

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
  python - "$DATA_DIR" "$MIX_NAME" <<'PY'
import json
import sys
from pathlib import Path

data_dir = Path(sys.argv[1])
mix_name = sys.argv[2]
stats_path = data_dir / "build_stats.json"
parts = [mix_name]
if stats_path.exists():
    stats = json.loads(stats_path.read_text())
    progress = stats.get("progress", {})
    accepted = int(progress.get("accepted_tokens", 0) or 0)
    target = int(progress.get("target_tokens", 0) or 0)
    elapsed = float(progress.get("elapsed_sec", 0.0) or 0.0)
    tps = float(progress.get("tokens_per_sec", 0.0) or 0.0)
    parts.append(f"{accepted / 1_000_000:.1f}M/{target / 1_000_000:.1f}M tokens")
    if elapsed:
        parts.append(f"{elapsed / 60:.1f}m")
    if tps:
        parts.append(f"{tps / 1000:.0f}k tok/s")
    ranges = stats.get("shard_ranges", {})
    train = ranges.get("train", {})
    val = ranges.get("val", {})
    if train:
        parts.append(f"train {train.get('range')}")
    if val:
        parts.append(f"val {val.get('range')}")
else:
    parts.append("build_stats missing")
parts.append(f"data: {data_dir}")
print("; ".join(parts))
PY
}

train_body() {
  local status="$1"
  python - "$RUN_OUT_DIR" "$MIX_NAME" "$status" <<'PY'
import json
import math
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
mix_name = sys.argv[2]
status = int(sys.argv[3])
summary_path = run_dir / "metrics_summary.json"
parts = [mix_name, f"exit {status}"]
if summary_path.exists():
    summary = json.loads(summary_path.read_text())
    tokens = int(summary.get("tokens_total", 0) or 0)
    elapsed = float(summary.get("total_time_sec", 0.0) or 0.0)
    tps = float(summary.get("tokens_per_sec_avg", 0.0) or 0.0)
    train = summary.get("final_train_loss")
    val = summary.get("best_val_loss")
    ppl = summary.get("best_val_ppl")
    peak = summary.get("gpu_peak_mem_gb")
    parts.append(f"{tokens / 1_000_000:.1f}M tok")
    if elapsed:
        parts.append(f"{elapsed / 3600:.2f}h")
    if tps:
        parts.append(f"{tps / 1000:.1f}k tok/s")
    if train is not None:
        parts.append(f"train {float(train):.4f}")
    if val is not None:
        text = f"best val {float(val):.4f}"
        if ppl is not None and math.isfinite(float(ppl)):
            text += f" / PPL {float(ppl):.2f}"
        parts.append(text)
    if peak:
        parts.append(f"peak {float(peak):.1f}GB")
parts.append(f"out: {run_dir}")
print("; ".join(parts))
PY
}

read_ranges() {
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
PY
}

write_report() {
  python - "$RUN_OUT_DIR" "$DATA_DIR" "$REPORT_PATH" "$MIX_NAME" <<'PY'
import json
import math
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
data_dir = Path(sys.argv[2])
report_path = Path(sys.argv[3])
mix_name = sys.argv[4]
summary = json.loads((run_dir / "metrics_summary.json").read_text())
stats = json.loads((data_dir / "build_stats.json").read_text())

def fmt(value, precision=4):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return f"{value:.{precision}f}"
    return str(value)

lines = [
    f"# V4 Data Proxy: {mix_name}",
    "",
    "57M v3-style proxy trained on a candidate v4 data mix.",
    "",
    f"- Dataset: `{data_dir}`",
    f"- Run: `{run_dir}`",
    f"- Accepted tokens: `{int(stats['progress']['accepted_tokens']):,}`",
    "",
    "| Metric | Value |",
    "|---|---:|",
    f"| Params | {fmt(summary.get('parameter_count'), 0)} |",
    f"| Tokens | {fmt(summary.get('tokens_total'), 0)} |",
    f"| Best val loss | {fmt(summary.get('best_val_loss'))} |",
    f"| Best val PPL | {fmt(summary.get('best_val_ppl'), 2)} |",
    f"| Final train loss | {fmt(summary.get('final_train_loss'))} |",
    f"| Tok/s | {fmt(summary.get('tokens_per_sec_avg'), 1)} |",
    f"| Hours | {fmt(float(summary.get('total_time_sec', 0.0) or 0.0) / 3600, 2)} |",
    f"| Peak GB | {fmt(summary.get('gpu_peak_mem_gb'), 1)} |",
    "",
    "Compare against the v3 pilot proxy baseline: best val loss `4.1530`, PPL `63.63`.",
    "",
]
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text("\n".join(lines), encoding="utf-8")
print(report_path)
PY
}

mkdir -p "$RUN_DIR" "$(dirname "$REPORT_PATH")"
cd "$ROOT" || exit 2
exec >>"$LOG_PATH" 2>&1

echo "=== SoloLLM v4 data proxy mix ==="
echo "started_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
TZ=America/Phoenix date "+started_at_az=%Y-%m-%d %I:%M:%S %p AZ"
echo "mix_name=$MIX_NAME"
echo "manifest=$MANIFEST"
echo "data_dir=$DATA_DIR"
echo "run_out_dir=$RUN_OUT_DIR"
echo "target_tokens=$TARGET_TOKENS"
echo "log_path=$LOG_PATH"
echo "report_path=$REPORT_PATH"
echo "pid=$$"

if [ ! -f "$DATA_DIR/splits.json" ]; then
  if [ "$BUILD_IF_MISSING" != "1" ]; then
    echo "dataset_missing=$DATA_DIR"
    exit 2
  fi
  build_args=()
  if [ "$OVERWRITE_DATA" = "1" ]; then
    build_args+=(--overwrite)
  fi
  python scripts/build_v3_pilot_dataset.py \
    --manifest "$MANIFEST" \
    --output-dir "$DATA_DIR" \
    --target-tokens "$TARGET_TOKENS" \
    --stats-every-docs 10000 \
    "${build_args[@]}"
  build_status=$?
  build_summary="$(dataset_body)"
  if [ "$NOTIFY" = "1" ]; then
    notify "SoloLLM v4 data proxy dataset built: ${MIX_NAME}" "$build_summary" "Dataset: ${DATA_DIR}"
  fi
  if [ "$build_status" -ne 0 ]; then
    exit "$build_status"
  fi
else
  echo "dataset_exists=$DATA_DIR"
fi

if [ -f "$RUN_OUT_DIR/metrics_summary.json" ] || [ -f "$RUN_OUT_DIR/final_model.pt" ]; then
  echo "refusing_to_mix_existing_run=$RUN_OUT_DIR"
  exit 2
fi

mapfile -t ranges < <(read_ranges)
TRAIN_SHARDS="${ranges[0]}"
VAL_SHARDS="${ranges[1]}"
SHARD_DIR="${ranges[2]}"

echo "train_shards=$TRAIN_SHARDS"
echo "val_shards=$VAL_SHARDS"
echo "shard_dir=$SHARD_DIR"

python -m sologpt_v2.pretrain \
  --config "$CONFIG_PATH" \
  --output-dir "$RUN_OUT_DIR" \
  --shard-dir "$SHARD_DIR" \
  --train-shards "$TRAIN_SHARDS" \
  --val-shards "$VAL_SHARDS" \
  --max-tokens "$TARGET_TOKENS" \
  --max-eval-tokens "$MAX_EVAL_TOKENS" \
  --eval-every-tokens "$EVAL_EVERY_TOKENS" \
  --tokens-per-checkpoint "$TOKENS_PER_CHECKPOINT" \
  --device "$DEVICE" \
  --no-progress
train_status=$?

if [ "$train_status" -eq 0 ]; then
  write_report
fi

body="$(train_body "$train_status")"
echo "finished_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
TZ=America/Phoenix date "+finished_at_az=%Y-%m-%d %I:%M:%S %p AZ"
echo "train_status=$train_status"

if [ "$NOTIFY" = "1" ]; then
  if [ "$train_status" -eq 0 ]; then
    notify "SoloLLM v4 data proxy complete: ${MIX_NAME}" "$body; report: ${REPORT_PATH}; log: ${LOG_PATH}" "Run directory: ${RUN_OUT_DIR}"
  else
    notify "SoloLLM v4 data proxy failed: ${MIX_NAME}" "$body; log: ${LOG_PATH}" "Run directory: ${RUN_OUT_DIR}"
  fi
fi

exit "$train_status"
