#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LABEL="${LABEL:-v4_2_dropout0_v4_1_blend_proxy_57m}"
RUN_NAME="${RUN_NAME:-${LABEL}_lockbox}"
OUT_DIR="${OUT_DIR:-${ROOT}/outputs/eval_suites/${RUN_NAME}}"
REPORT_DIR="${REPORT_DIR:-${ROOT}/docs/results/${RUN_NAME}}"
LOG_PATH="${LOG_PATH:-${OUT_DIR}/lockbox_eval.log}"
CHECKPOINT="${CHECKPOINT:-${ROOT}/outputs/sologpt_v4/arch_proxy_v4_2_dropout0_v4_1_blend_57m_300m/final_model.pt}"
CONFIG="${CONFIG:-${ROOT}/configs/solollm-0.4.2-57m-broad-blend-baseproxy.json}"
SHARD_DIR="${SHARD_DIR:-${ROOT}/data/tokenized_chunks}"
SHARDS="${SHARDS:-58:60}"
HELDOUT_MAX_BATCHES="${HELDOUT_MAX_BATCHES:-512}"
DEVICE="${DEVICE:-cuda}"
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

write_summary() {
  python - "$LABEL" "$OUT_DIR" "$REPORT_DIR/summary.md" "$HELDOUT_MAX_BATCHES" <<'PY'
import json
import math
import sys
from pathlib import Path

label = sys.argv[1]
out_dir = Path(sys.argv[2])
summary_path = Path(sys.argv[3])
heldout_max_batches = int(sys.argv[4])

candidate = json.loads((out_dir / f"{label}_heldout.json").read_text())
gpt2 = json.loads((out_dir / "gpt2_heldout.json").read_text())
gen = json.loads((out_dir / "generation_metrics_candidate_gpt2.json").read_text())


def fmt(value, precision=4):
    if value is None:
        return "n/a"
    value = float(value)
    if not math.isfinite(value):
        return str(value)
    return f"{value:.{precision}f}"


lines = [
    "# V4.2 Dropout-0 V4.1 Blend Lockbox Eval",
    "",
    "Lockbox-style verification for the 57M proxy scale candidate.",
    "",
    f"- Held-out shards: `{candidate['shards'][0]}` to `{candidate['shards'][-1]}`",
    f"- Max batches per shard: `{heldout_max_batches}`",
    "",
    "## Held-Out PPL",
    "",
    "| Model | Tokens | Loss | PPL |",
    "|---|---:|---:|---:|",
    f"| {label} | {int(candidate['tokens']):,} | {fmt(candidate['loss'])} | {fmt(candidate['perplexity'], 2)} |",
    f"| gpt2 | {int(gpt2['tokens']):,} | {fmt(gpt2['loss'])} | {fmt(gpt2['perplexity'], 2)} |",
    "",
    "## Generation Metrics",
    "",
    "| Model | Samples | Avg tokens | Distinct-1 | Distinct-2 | Repeated bigrams | Bad loops |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for model, row in gen["summary"].items():
    lines.append(
        f"| {model} | {int(row['samples'])} | {fmt(row['avg_tokens'], 1)} | "
        f"{fmt(row['corpus_distinct_1'])} | {fmt(row['corpus_distinct_2'])} | "
        f"{fmt(row['mean_repeated_bigram_fraction'])} | {int(row['bad_loop_count'])} |"
    )

lines.extend(
    [
        "",
        "## Decision",
        "",
        "Use this as a final proxy sanity check before building the full 10B-token dataset.",
        "",
    ]
)

summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text("\n".join(lines), encoding="utf-8")
print(summary_path)
PY
}

mkdir -p "$OUT_DIR" "$REPORT_DIR"
cd "$ROOT" || exit 2
exec >>"$LOG_PATH" 2>&1

echo "=== SoloLLM v4.2 lockbox eval ==="
echo "started_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
TZ=America/Phoenix date "+started_at_az=%Y-%m-%d %I:%M:%S %p AZ"
echo "label=$LABEL"
echo "checkpoint=$CHECKPOINT"
echo "config=$CONFIG"
echo "out_dir=$OUT_DIR"
echo "report_dir=$REPORT_DIR"
echo "shards=$SHARDS"
echo "heldout_max_batches=$HELDOUT_MAX_BATCHES"
echo "pid=$$"

if [ "$NOTIFY" = "1" ]; then
  notify "SoloLLM v4.2 lockbox eval started" \
    "label ${LABEL}; shards ${SHARDS}; max batches ${HELDOUT_MAX_BATCHES}; log: ${LOG_PATH}" \
    "Run directory: ${OUT_DIR}"
fi

python -m eval.v3_eval_suite \
  --candidate-label "$LABEL" \
  --candidate-loader solollm \
  --candidate-checkpoint "$CHECKPOINT" \
  --candidate-config "$CONFIG" \
  --output-dir "$OUT_DIR" \
  --report-dir "$REPORT_DIR" \
  --shard-dir "$SHARD_DIR" \
  --shards "$SHARDS" \
  --batch-size 16 \
  --heldout-max-batches "$HELDOUT_MAX_BATCHES" \
  --device "$DEVICE" \
  --context-length 1024 \
  --components heldout generation generation_metrics \
  --execute \
  --skip-existing \
  --no-progress
status=$?

if [ "$status" -eq 0 ]; then
  write_summary
fi

echo "finished_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
TZ=America/Phoenix date "+finished_at_az=%Y-%m-%d %I:%M:%S %p AZ"
echo "exit_status=$status"

if [ "$NOTIFY" = "1" ]; then
  if [ "$status" -eq 0 ]; then
    notify "SoloLLM v4.2 lockbox eval complete" \
      "exit 0; summary: ${REPORT_DIR}/summary.md; log: ${LOG_PATH}" \
      "Run directory: ${OUT_DIR}"
  else
    notify "SoloLLM v4.2 lockbox eval failed" \
      "exit ${status}; log: ${LOG_PATH}" \
      "Run directory: ${OUT_DIR}"
  fi
fi

exit "$status"
