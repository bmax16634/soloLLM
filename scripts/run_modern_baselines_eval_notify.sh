#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/home/bmx/_projects/ariya/userdata/projects/soloLLM}"
RUN_NAME="${RUN_NAME:-modern_baselines_v3_150m}"
OUT_DIR="${OUT_DIR:-${ROOT}/outputs/eval_suites/${RUN_NAME}}"
REPORT_DIR="${REPORT_DIR:-${ROOT}/docs/results/${RUN_NAME}}"
LOG_PATH="${LOG_PATH:-${OUT_DIR}/modern_baselines_eval.log}"

V2_LABEL="${V2_LABEL:-solollm-v3-150m}"
V2_CHECKPOINT="${V2_CHECKPOINT:-${ROOT}/outputs/sologpt_v3/v3_fresh_10b_plus_150m_1024_10bdata/final_model.pt}"
V2_CONFIG="${V2_CONFIG:-${ROOT}/sologpt_v3/config_plus_150m_1024.json}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-1024}"
MAX_WIKITEXT_TOKENS="${MAX_WIKITEXT_TOKENS:-0}"
MAX_LAMBADA_EXAMPLES="${MAX_LAMBADA_EXAMPLES:-0}"
MAX_MC_EXAMPLES="${MAX_MC_EXAMPLES:-0}"
DEVICE="${DEVICE:-cuda}"
NOTIFY="${NOTIFY:-1}"

# MobileLLM-125M is a useful comparison but is gated on Hugging Face, so it is
# intentionally excluded from the default public-baseline run.
MODELS="${MODELS:-v2 gpt2 distilgpt2 pythia-70m=hf:EleutherAI/pythia-70m-deduped pythia-160m=hf:EleutherAI/pythia-160m-deduped smollm-135m=hf:HuggingFaceTB/SmolLM-135M smollm2-135m=hf:HuggingFaceTB/SmolLM2-135M smollm2-360m=hf:HuggingFaceTB/SmolLM2-360M}"

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
  python - "$status" "$OUT_DIR" "$REPORT_DIR" "$LOG_PATH" <<'PY'
import json
import sys
from pathlib import Path

status = int(sys.argv[1])
out_dir = Path(sys.argv[2])
report_dir = Path(sys.argv[3])
log_path = Path(sys.argv[4])

parts = [f"exit {status}"]
for name in ["external_modern_baselines.json", "multiple_choice_modern_baselines.json"]:
    path = out_dir / name
    if not path.exists():
        parts.append(f"{name} missing")
        continue
    data = json.loads(path.read_text())
    parts.append(f"{name}: {len(data.get('models', {}))} models, {float(data.get('elapsed_sec', 0.0)) / 60:.1f} min")

parts.append(f"reports: {report_dir}")
parts.append(f"log: {log_path}")
print("; ".join(parts))
PY
}

mkdir -p "$OUT_DIR" "$REPORT_DIR"
cd "$ROOT" || exit 2
exec >>"$LOG_PATH" 2>&1

read -r -a MODEL_ARGS <<< "$MODELS"
TRUST_REMOTE_CODE_ARGS=()
if [ "${TRUST_REMOTE_CODE:-0}" = "1" ]; then
  TRUST_REMOTE_CODE_ARGS=(--trust-remote-code)
fi

echo "=== SoloLLM modern baseline eval ==="
echo "started_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
TZ=America/Phoenix date "+started_at_az=%Y-%m-%d %I:%M:%S %p AZ"
echo "root=$ROOT"
echo "out_dir=$OUT_DIR"
echo "report_dir=$REPORT_DIR"
echo "log_path=$LOG_PATH"
echo "v2_label=$V2_LABEL"
echo "v2_checkpoint=$V2_CHECKPOINT"
echo "v2_config=$V2_CONFIG"
echo "context_length=$CONTEXT_LENGTH"
echo "models=$MODELS"
echo "pid=$$"

python -m eval.external_benchmarks \
  --models "${MODEL_ARGS[@]}" \
  --v2-label "$V2_LABEL" \
  --v2-checkpoint "$V2_CHECKPOINT" \
  --v2-config "$V2_CONFIG" \
  --context-length "$CONTEXT_LENGTH" \
  --max-wikitext-tokens "$MAX_WIKITEXT_TOKENS" \
  --max-lambada-examples "$MAX_LAMBADA_EXAMPLES" \
  --device "$DEVICE" \
  --output-json "$OUT_DIR/external_modern_baselines.json" \
  --output-md "$REPORT_DIR/external_modern_baselines.md" \
  --no-progress \
  "${TRUST_REMOTE_CODE_ARGS[@]}"
external_status=$?

python -m eval.multiple_choice_benchmarks \
  --models "${MODEL_ARGS[@]}" \
  --v2-label "$V2_LABEL" \
  --v2-checkpoint "$V2_CHECKPOINT" \
  --v2-config "$V2_CONFIG" \
  --context-length "$CONTEXT_LENGTH" \
  --max-examples "$MAX_MC_EXAMPLES" \
  --device "$DEVICE" \
  --output-json "$OUT_DIR/multiple_choice_modern_baselines.json" \
  --output-md "$REPORT_DIR/multiple_choice_modern_baselines.md" \
  --no-progress \
  "${TRUST_REMOTE_CODE_ARGS[@]}"
mc_status=$?

status=0
if [ "$external_status" -ne 0 ] || [ "$mc_status" -ne 0 ]; then
  status=1
fi

echo "finished_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
TZ=America/Phoenix date "+finished_at_az=%Y-%m-%d %I:%M:%S %p AZ"
echo "external_status=$external_status"
echo "multiple_choice_status=$mc_status"
echo "exit_status=$status"

body="$(summary_body "$status")"
if [ "$status" -eq 0 ]; then
  if [ "$NOTIFY" = "1" ]; then
    notify "SoloLLM modern baseline eval complete" "$body" "Run directory: ${OUT_DIR}"
  fi
else
  if [ "$NOTIFY" = "1" ]; then
    notify "SoloLLM modern baseline eval failed" "$body" "Run directory: ${OUT_DIR}"
  fi
fi

exit "$status"
