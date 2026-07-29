#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DEVICE="${DEVICE:-cuda}"
NOTIFY="${NOTIFY:-1}"
BOARD_LOG="${BOARD_LOG:-${ROOT}/outputs/sologpt_v5/v5_1_v5_2_proxy_board.log}"

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

preflight_cuda() {
  if [ "$DEVICE" != "cuda" ]; then
    return 0
  fi
  nvidia-smi >/dev/null
  python - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    sys.exit("torch.cuda.is_available() is false")
PY
}

run_proxy() {
  local key="$1"
  local display="$2"
  local description="$3"
  local mix_name="$4"
  local manifest="$5"
  local config="$6"
  local data_dir="$7"
  local run_out_dir="$8"
  local report_stem="$9"
  local eval_label="${10}"

  local log_dir="${ROOT}/outputs/sologpt_v5/${key}_logs"
  local eval_name="${key}"

  echo "=== ${display} ==="
  echo "started_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  TZ=America/Phoenix date "+started_at_az=%Y-%m-%d %I:%M:%S %p AZ"
  echo "manifest=${manifest}"
  echo "config=${config}"
  echo "data_dir=${data_dir}"
  echo "run_out_dir=${run_out_dir}"

  EXPERIMENT_DISPLAY_NAME="$display" \
  EXPERIMENT_DESCRIPTION="$description" \
  PROXY_LABEL="$display" \
  MIX_NAME="$mix_name" \
  MANIFEST="$manifest" \
  CONFIG_PATH="$config" \
  DATA_DIR="$data_dir" \
  RUN_OUT_DIR="$run_out_dir" \
  RUN_DIR="$log_dir" \
  TRAIN_LOG_PATH="${log_dir}/train.log" \
  MAIN_LOG_PATH="${log_dir}/train_eval.log" \
  TRAIN_REPORT_PATH="${ROOT}/docs/results/${report_stem}_train.md" \
  SUMMARY_REPORT_PATH="${ROOT}/docs/results/${report_stem}_summary.md" \
  EVAL_RUN_NAME="$eval_name" \
  EVAL_OUT_DIR="${ROOT}/outputs/eval_suites/${eval_name}" \
  EVAL_REPORT_DIR="${ROOT}/docs/results/${eval_name}" \
  EVAL_LOG_PATH="${ROOT}/outputs/eval_suites/${eval_name}/modern_eval.log" \
  EVAL_LABEL="$eval_label" \
  DEVICE="$DEVICE" \
  NOTIFY="$NOTIFY" \
  bash "${ROOT}/scripts/run_v5_proxy_smol_quality_1b_train_eval_notify.sh"

  echo "finished_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  TZ=America/Phoenix date "+finished_at_az=%Y-%m-%d %I:%M:%S %p AZ"
}

mkdir -p "$(dirname "$BOARD_LOG")"
cd "$ROOT"
exec >>"$BOARD_LOG" 2>&1

echo "=== SoloLLM V5.1/V5.2 proxy board ==="
echo "started_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
TZ=America/Phoenix date "+started_at_az=%Y-%m-%d %I:%M:%S %p AZ"
echo "root=$ROOT"
echo "device=$DEVICE"
echo "board_log=$BOARD_LOG"
echo "pid=$$"

if ! preflight_cuda; then
  body="CUDA preflight failed. Fix NVIDIA driver/CUDA state before launching the V5 proxy board."
  echo "$body"
  if [ "$NOTIFY" = "1" ]; then
    notify "SoloLLM V5 proxy board not started" "$body" "Log: ${BOARD_LOG}"
  fi
  exit 1
fi

if [ "$NOTIFY" = "1" ]; then
  notify \
    "SoloLLM V5 proxy board started" \
    "Queued V5.1 high-quality web/edu then V5.2 math/edu, each 60M/1B with modern eval." \
    "Log: ${BOARD_LOG}"
fi

run_proxy \
  "v5_1_high_quality_web_60m_1b" \
  "V5.1 High-Quality Web/Edu 60M/1B Proxy" \
  "This proxy tests whether a cleaner high-quality web and education mix improves broad base-model quality without the heavier math/synthetic slice." \
  "v5_high_quality_web" \
  "${ROOT}/data_mixes/solollm-1.1-highqualityweb-1b.yaml" \
  "${ROOT}/configs/solollm-1.1-57m-highqualityweb-baseproxy.json" \
  "${ROOT}/data/v5_high_quality_web_1b_1024" \
  "${ROOT}/outputs/sologpt_v5/proxy_high_quality_web_60m_1b" \
  "v5_1_high_quality_web_60m_1b" \
  "solollm-v5-1-high-quality-web-60m-1b"

run_proxy \
  "v5_2_math_edu_60m_1b" \
  "V5.2 Math/Edu 60M/1B Proxy" \
  "This proxy tests whether a heavier math and synthetic education slice improves reasoning-style metrics without giving back broad language quality." \
  "v5_math_edu" \
  "${ROOT}/data_mixes/solollm-1.1-mathedu-1b.yaml" \
  "${ROOT}/configs/solollm-1.1-57m-mathedu-baseproxy.json" \
  "${ROOT}/data/v5_math_edu_1b_1024" \
  "${ROOT}/outputs/sologpt_v5/proxy_math_edu_60m_1b" \
  "v5_2_math_edu_60m_1b" \
  "solollm-v5-2-math-edu-60m-1b"

if [ "$NOTIFY" = "1" ]; then
  notify \
    "SoloLLM V5 proxy board complete" \
    "V5.1 and V5.2 proxy train+eval queue finished." \
    "Log: ${BOARD_LOG}"
fi

echo "finished_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
TZ=America/Phoenix date "+finished_at_az=%Y-%m-%d %I:%M:%S %p AZ"
