#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MANIFEST="${MANIFEST:-${ROOT}/data_mixes/solollm-1.0-broad-blend-10b.yaml}"
OUT_DIR="${OUT_DIR:-${ROOT}/data/v4_2_dropout0_v4_1_blend_10b_1024}"
LOG_PATH="${LOG_PATH:-${ROOT}/outputs/sologpt_v4/build_v4_2_dropout0_v4_1_blend_10b_1024.log}"
TARGET_TOKENS="${TARGET_TOKENS:-10000000000}"
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

summary_body() {
  local status="$1"
  python - "$OUT_DIR" "$LOG_PATH" "$status" <<'PY'
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
log_path = Path(sys.argv[2])
status = int(sys.argv[3])
parts = [f"exit {status}"]
stats_path = out_dir / "build_stats.json"
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
    test = ranges.get("test", {})
    if train:
        parts.append(f"train {train.get('range')} ({int(train.get('tokens', 0)) / 1_000_000_000:.2f}B tok)")
    if val and test:
        parts.append(f"val {val.get('range')} test {test.get('range')}")
parts.append(f"data: {out_dir}")
parts.append(f"log: {log_path}")
print("; ".join(parts))
PY
}

mkdir -p "$(dirname "$LOG_PATH")"
cd "$ROOT" || exit 2
exec >>"$LOG_PATH" 2>&1

echo "=== SoloLLM v4.2 dropout0 v4.1 blend 10B dataset build ==="
echo "started_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
TZ=America/Phoenix date "+started_at_az=%Y-%m-%d %I:%M:%S %p AZ"
echo "root=$ROOT"
echo "manifest=$MANIFEST"
echo "out_dir=$OUT_DIR"
echo "log_path=$LOG_PATH"
echo "target_tokens=$TARGET_TOKENS"
echo "overwrite_data=$OVERWRITE_DATA"
echo "pid=$$"

if [ -e "$OUT_DIR" ] && [ "$(find "$OUT_DIR" -mindepth 1 -maxdepth 1 2>/dev/null | head -n 1)" ]; then
  if [ "$OVERWRITE_DATA" != "1" ]; then
    echo "refusing_to_overwrite_existing_dataset=$OUT_DIR"
    body="$(summary_body 2)"
    if [ "$NOTIFY" = "1" ]; then
      notify "SoloLLM v4.2 10B dataset build refused" "$body" "Dataset exists: ${OUT_DIR}"
    fi
    exit 2
  fi
  overwrite_args=(--overwrite)
else
  overwrite_args=()
fi

if [ "$NOTIFY" = "1" ]; then
  notify "SoloLLM v4.2 10B dataset build started" \
    "target ${TARGET_TOKENS}; manifest ${MANIFEST}; output ${OUT_DIR}; log ${LOG_PATH}" \
    "Dataset: ${OUT_DIR}"
fi

python scripts/build_v3_pilot_dataset.py \
  --manifest "$MANIFEST" \
  --output-dir "$OUT_DIR" \
  --target-tokens "$TARGET_TOKENS" \
  --stats-every-docs 10000 \
  "${overwrite_args[@]}"
status=$?

echo "finished_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
TZ=America/Phoenix date "+finished_at_az=%Y-%m-%d %I:%M:%S %p AZ"
echo "exit_status=$status"

body="$(summary_body "$status")"
if [ "$NOTIFY" = "1" ]; then
  if [ "$status" -eq 0 ]; then
    notify "SoloLLM v4.2 10B dataset build complete" "$body" "Dataset: ${OUT_DIR}"
  else
    notify "SoloLLM v4.2 10B dataset build failed" "$body" "Dataset: ${OUT_DIR}"
  fi
fi

exit "$status"
