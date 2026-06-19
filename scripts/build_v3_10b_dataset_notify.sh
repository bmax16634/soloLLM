#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/home/bmx/_projects/ariya/userdata/projects/soloLLM}"
OUT_DIR="${OUT_DIR:-/home/bmx/_projects/soloLLM/data/v3_10b_1024}"
LOG_PATH="${LOG_PATH:-${ROOT}/outputs/sologpt_v3/build_v3_10b_1024.log}"
TARGET_TOKENS="${TARGET_TOKENS:-10000000000}"
MANIFEST="${MANIFEST:-${ROOT}/sologpt_v3/data_sources.yaml}"

mkdir -p "$(dirname "$LOG_PATH")"
cd "$ROOT" || exit 2

exec >>"$LOG_PATH" 2>&1

echo "=== SoloLLM v3 10B dataset build ==="
echo "started_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
TZ=America/Phoenix date "+started_at_az=%Y-%m-%d %I:%M:%S %p AZ"
echo "root=$ROOT"
echo "out_dir=$OUT_DIR"
echo "log_path=$LOG_PATH"
echo "manifest=$MANIFEST"
echo "target_tokens=$TARGET_TOKENS"
echo "pid=$$"

python scripts/build_v3_pilot_dataset.py \
  --manifest "$MANIFEST" \
  --output-dir "$OUT_DIR" \
  --target-tokens "$TARGET_TOKENS" \
  --stats-every-docs 10000

status=$?

echo "finished_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
TZ=America/Phoenix date "+finished_at_az=%Y-%m-%d %I:%M:%S %p AZ"
echo "exit_status=$status"

python - "$OUT_DIR" "$LOG_PATH" "$status" <<'PY'
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

out_dir = Path(sys.argv[1])
log_path = Path(sys.argv[2])
status = int(sys.argv[3])

stats_path = out_dir / "build_stats.json"
splits_path = out_dir / "splits.json"
parts = []

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
    split_ranges = stats.get("shard_ranges", {})
    train = split_ranges.get("train", {})
    val = split_ranges.get("val", {})
    test = split_ranges.get("test", {})
    if train:
        parts.append(f"train {train.get('range')} ({train.get('tokens', 0) / 1_000_000_000:.2f}B)")
    if val and test:
        parts.append(f"val {val.get('range')} test {test.get('range')}")

if splits_path.exists() and not any(part.startswith("train ") for part in parts):
    splits = json.loads(splits_path.read_text())
    ranges = splits.get("shard_ranges", {})
    train = ranges.get("train", {})
    if train:
        parts.append(f"train {train.get('range')} ({train.get('tokens', 0) / 1_000_000_000:.2f}B)")

if not parts:
    parts.append(f"exit status {status}")

title = "SoloLLM v3 10B dataset build complete" if status == 0 else "SoloLLM v3 10B dataset build failed"
body = "; ".join(parts) + f"; log: {log_path}"
payload = {
    "title": title,
    "body": body,
    "dueAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "notes": f"Dataset: {out_dir}",
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

exit "$status"
