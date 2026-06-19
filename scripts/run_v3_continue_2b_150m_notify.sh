#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/home/bmx/_projects/ariya/userdata/projects/soloLLM}"
RUN_NAME="${RUN_NAME:-v3_full_2b_plus_150m_1024}"
OUT_DIR="${OUT_DIR:-${ROOT}/outputs/sologpt_v3/${RUN_NAME}}"
LOG_PATH="${LOG_PATH:-${ROOT}/outputs/sologpt_v3/${RUN_NAME}.log}"
CONFIG_PATH="${CONFIG_PATH:-${ROOT}/sologpt_v3/config_plus_150m_1024.json}"
RESUME_PATH="${RESUME_PATH:-${ROOT}/outputs/sologpt_v3/v3_full_1b_plus_150m_1024/checkpoints/latest.pt}"

mkdir -p "$(dirname "$LOG_PATH")"
cd "$ROOT" || exit 2

exec >>"$LOG_PATH" 2>&1

echo "=== SoloLLM v3 continue to 2B 150M run ==="
echo "started_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
TZ=America/Phoenix date "+started_at_az=%Y-%m-%d %I:%M:%S %p AZ"
echo "root=$ROOT"
echo "out_dir=$OUT_DIR"
echo "log_path=$LOG_PATH"
echo "config_path=$CONFIG_PATH"
echo "resume_path=$RESUME_PATH"
echo "target_tokens=2000000000"
echo "continuation_learning_rate=0.0001"
echo "pid=$$"

python -m sologpt_v2.pretrain \
  --config "$CONFIG_PATH" \
  --output-dir "$OUT_DIR" \
  --resume "$RESUME_PATH" \
  --max-tokens 2000000000 \
  --learning-rate 0.0001 \
  --max-eval-tokens 2097152 \
  --eval-every-tokens 100000000 \
  --tokens-per-checkpoint 100000000 \
  --log-every-opt-steps 50 \
  --device cuda \
  --no-progress

status=$?

echo "finished_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
TZ=America/Phoenix date "+finished_at_az=%Y-%m-%d %I:%M:%S %p AZ"
echo "exit_status=$status"

python - "$OUT_DIR" "$LOG_PATH" "$status" <<'PY'
import json
import math
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

out_dir = Path(sys.argv[1])
log_path = Path(sys.argv[2])
status = int(sys.argv[3])

summary_path = out_dir / "metrics_summary.json"
metrics_path = out_dir / "metrics.jsonl"

details = []
title = "SoloLLM v3 2B continuation complete" if status == 0 else "SoloLLM v3 2B continuation failed"

if summary_path.exists():
    summary = json.loads(summary_path.read_text())
    tokens = summary.get("tokens_total", 0)
    processed = summary.get("tokens_processed_this_run", 0)
    elapsed = float(summary.get("total_time_sec", 0.0) or 0.0)
    avg_tps = float(summary.get("tokens_per_sec_avg", 0.0) or 0.0)
    final_train = summary.get("final_train_loss")
    best_val = summary.get("best_val_loss")
    best_ppl = summary.get("best_val_ppl")
    details.append(f"{tokens / 1_000_000:.1f}M total tok")
    details.append(f"{processed / 1_000_000:.1f}M new tok")
    if elapsed:
        details.append(f"{elapsed / 3600:.2f}h")
    if avg_tps:
        details.append(f"{avg_tps / 1000:.1f}k tok/s")
    if final_train is not None:
        details.append(f"train loss {float(final_train):.4f}")
    if best_val is not None:
        val_text = f"best val {float(best_val):.4f}"
        if best_ppl is not None and math.isfinite(float(best_ppl)):
            val_text += f" / PPL {float(best_ppl):.2f}"
        details.append(val_text)

if metrics_path.exists():
    peak = 0.0
    last_val = None
    for line in metrics_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        peak = max(peak, float(row.get("gpu_peak_mem_gb", 0.0) or 0.0))
        if row.get("type") == "validation":
            last_val = row
    if peak:
        details.append(f"peak {peak:.1f}GB")
    if last_val:
        details.append(f"last val {float(last_val['val_loss']):.4f} / PPL {float(last_val['val_ppl']):.2f}")

if not details:
    details.append(f"exit status {status}")

body = "; ".join(details) + f"; log: {log_path}"
payload = {
    "title": title,
    "body": body,
    "dueAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "notes": f"Run directory: {out_dir}",
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
