#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/home/bmx/_projects/ariya/userdata/projects/soloLLM}"
TARGET_TOKENS="${TARGET_TOKENS:-300000000}"
MAX_EVAL_TOKENS="${MAX_EVAL_TOKENS:-2097152}"
EVAL_EVERY_TOKENS="${EVAL_EVERY_TOKENS:-50000000}"
TOKENS_PER_CHECKPOINT="${TOKENS_PER_CHECKPOINT:-100000000}"
DEVICE="${DEVICE:-cuda}"

V3_CONFIG="${V3_CONFIG:-${ROOT}/sologpt_v3/config_proxy_v3_style_60m_1024.json}"
SMOLLM2_CONFIG="${SMOLLM2_CONFIG:-${ROOT}/sologpt_v3/config_proxy_smollm2_style_60m_1024.json}"
V3_OUT_DIR="${V3_OUT_DIR:-${ROOT}/outputs/sologpt_v4/phase1_proxy_v3_style_60m_300m}"
SMOLLM2_OUT_DIR="${SMOLLM2_OUT_DIR:-${ROOT}/outputs/sologpt_v4/phase1_proxy_smollm2_style_60m_300m}"
RUN_DIR="${RUN_DIR:-${ROOT}/outputs/sologpt_v4/phase1_proxy_architecture_300m}"
REPORT_PATH="${REPORT_PATH:-${ROOT}/docs/results/v4_phase1_proxy_architecture_300m.md}"
LOG_PATH="${LOG_PATH:-${RUN_DIR}/phase1_proxy_pair.log}"
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
  local label="$1"
  local out_dir="$2"
  local status="$3"
  python - "$label" "$out_dir" "$status" <<'PY'
import json
import math
import sys
from pathlib import Path

label = sys.argv[1]
out_dir = Path(sys.argv[2])
status = int(sys.argv[3])
summary_path = out_dir / "metrics_summary.json"
parts = [label, f"exit {status}"]
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
parts.append(f"out: {out_dir}")
print("; ".join(parts))
PY
}

write_report() {
  python - "$V3_OUT_DIR" "$SMOLLM2_OUT_DIR" "$REPORT_PATH" "$TARGET_TOKENS" <<'PY'
import json
import math
import sys
from pathlib import Path

v3_dir = Path(sys.argv[1])
smollm2_dir = Path(sys.argv[2])
report_path = Path(sys.argv[3])
target_tokens = int(sys.argv[4])

def load_summary(label, path):
    summary_path = path / "metrics_summary.json"
    if not summary_path.exists():
        return {"label": label, "path": str(path), "missing": True}
    summary = json.loads(summary_path.read_text())
    return {
        "label": label,
        "path": str(path),
        "status": summary.get("status"),
        "params": summary.get("parameter_count"),
        "tokens": summary.get("tokens_total"),
        "train_loss": summary.get("final_train_loss"),
        "best_val_loss": summary.get("best_val_loss"),
        "best_val_ppl": summary.get("best_val_ppl"),
        "hours": float(summary.get("total_time_sec", 0.0) or 0.0) / 3600,
        "tok_per_sec": summary.get("tokens_per_sec_avg"),
        "peak_gb": summary.get("gpu_peak_mem_gb"),
    }

rows = [
    load_summary("v3-style proxy", v3_dir),
    load_summary("SmolLM2-style proxy", smollm2_dir),
]

def fmt(value, precision=4):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return f"{value:.{precision}f}"
    return str(value)

lines = [
    "# V4 Phase 1 Architecture Proxy Results",
    "",
    "Matched-parameter proxy comparison for the SoloLLM-Modern architecture hypothesis.",
    "",
    f"- Target tokens per proxy: `{target_tokens:,}`",
    "- Dataset: `v3_pilot_1b_1024`",
    "- Purpose: compare wider/shallower v3-style shape against deeper/narrower SmolLM2-style shape before a full model train.",
    "",
    "| Proxy | Params | Tokens | Best val loss | Best val PPL | Final train loss | Tok/s | Hours | Peak GB |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]

for row in rows:
    lines.append(
        "| "
        + " | ".join(
            [
                row["label"],
                fmt(row.get("params"), 0),
                fmt(row.get("tokens"), 0),
                fmt(row.get("best_val_loss")),
                fmt(row.get("best_val_ppl"), 2),
                fmt(row.get("train_loss")),
                fmt(row.get("tok_per_sec"), 1),
                fmt(row.get("hours"), 2),
                fmt(row.get("peak_gb"), 1),
            ]
        )
        + " |"
    )

lines.extend(
    [
        "",
        "Lower validation loss/PPL is better. This proxy is a first architecture signal, not the final SoloLLM-Modern result.",
        "",
    ]
)

report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text("\n".join(lines), encoding="utf-8")
print(report_path)
PY
}

run_proxy() {
  local label="$1"
  local config="$2"
  local out_dir="$3"

  if [ -f "${out_dir}/metrics_summary.json" ] || [ -f "${out_dir}/final_model.pt" ]; then
    echo "refusing_to_mix_existing_run=${out_dir}"
    return 2
  fi

  echo "=== ${label} proxy train ==="
  echo "started_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  TZ=America/Phoenix date "+started_at_az=%Y-%m-%d %I:%M:%S %p AZ"
  echo "config=${config}"
  echo "out_dir=${out_dir}"

  python -m sologpt_v2.pretrain \
    --config "$config" \
    --output-dir "$out_dir" \
    --max-tokens "$TARGET_TOKENS" \
    --max-eval-tokens "$MAX_EVAL_TOKENS" \
    --eval-every-tokens "$EVAL_EVERY_TOKENS" \
    --tokens-per-checkpoint "$TOKENS_PER_CHECKPOINT" \
    --device "$DEVICE" \
    --no-progress
}

mkdir -p "$RUN_DIR" "$(dirname "$REPORT_PATH")"
cd "$ROOT" || exit 2
exec >>"$LOG_PATH" 2>&1

echo "=== SoloLLM v4 phase 1 proxy pair ==="
echo "started_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
TZ=America/Phoenix date "+started_at_az=%Y-%m-%d %I:%M:%S %p AZ"
echo "root=$ROOT"
echo "target_tokens=$TARGET_TOKENS"
echo "max_eval_tokens=$MAX_EVAL_TOKENS"
echo "eval_every_tokens=$EVAL_EVERY_TOKENS"
echo "tokens_per_checkpoint=$TOKENS_PER_CHECKPOINT"
echo "device=$DEVICE"
echo "log_path=$LOG_PATH"
echo "report_path=$REPORT_PATH"
echo "pid=$$"

run_proxy "v3-style" "$V3_CONFIG" "$V3_OUT_DIR"
v3_status=$?
v3_body="$(summary_body "v3-style proxy" "$V3_OUT_DIR" "$v3_status")"
if [ "$NOTIFY" = "1" ]; then
  notify "SoloLLM v4 phase 1 v3-style proxy done" "$v3_body" "Run directory: ${V3_OUT_DIR}"
fi

if [ "$v3_status" -eq 0 ]; then
  run_proxy "SmolLM2-style" "$SMOLLM2_CONFIG" "$SMOLLM2_OUT_DIR"
  smollm2_status=$?
else
  smollm2_status=2
fi

smollm2_body="$(summary_body "SmolLM2-style proxy" "$SMOLLM2_OUT_DIR" "$smollm2_status")"
if [ "$NOTIFY" = "1" ]; then
  notify "SoloLLM v4 phase 1 SmolLM2-style proxy done" "$smollm2_body" "Run directory: ${SMOLLM2_OUT_DIR}"
fi

status=0
if [ "$v3_status" -ne 0 ] || [ "$smollm2_status" -ne 0 ]; then
  status=1
fi

if [ "$status" -eq 0 ]; then
  write_report
fi

echo "finished_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
TZ=America/Phoenix date "+finished_at_az=%Y-%m-%d %I:%M:%S %p AZ"
echo "v3_status=$v3_status"
echo "smollm2_status=$smollm2_status"
echo "exit_status=$status"

final_body="${v3_body}; ${smollm2_body}; report: ${REPORT_PATH}; log: ${LOG_PATH}"
if [ "$NOTIFY" = "1" ]; then
  if [ "$status" -eq 0 ]; then
    notify "SoloLLM v4 phase 1 proxy pair complete" "$final_body" "Run directory: ${RUN_DIR}"
  else
    notify "SoloLLM v4 phase 1 proxy pair failed" "$final_body" "Run directory: ${RUN_DIR}"
  fi
fi

exit "$status"
