#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/home/bmx/_projects/ariya/userdata/projects/soloLLM}"
LABEL="${LABEL:-v3_1b_150m}"
OUT_DIR="${OUT_DIR:-${ROOT}/outputs/eval_suites/v3_1b_150m_gpt2_full_suite}"
REPORT_DIR="${REPORT_DIR:-${OUT_DIR}/reports}"
LOG_PATH="${LOG_PATH:-${OUT_DIR}/v3_eval_suite_full.log}"
CANDIDATE_CHECKPOINT="${CANDIDATE_CHECKPOINT:-${ROOT}/outputs/sologpt_v3/v3_full_1b_plus_150m_1024/final_model.pt}"
CANDIDATE_CONFIG="${CANDIDATE_CONFIG:-${ROOT}/sologpt_v3/config_plus_150m_1024.json}"

mkdir -p "$OUT_DIR" "$REPORT_DIR"
cd "$ROOT" || exit 2

exec >>"$LOG_PATH" 2>&1

echo "=== SoloLLM v3 full eval suite ==="
echo "started_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
TZ=America/Phoenix date "+started_at_az=%Y-%m-%d %I:%M:%S %p AZ"
echo "root=$ROOT"
echo "label=$LABEL"
echo "out_dir=$OUT_DIR"
echo "report_dir=$REPORT_DIR"
echo "log_path=$LOG_PATH"
echo "candidate_checkpoint=$CANDIDATE_CHECKPOINT"
echo "candidate_config=$CANDIDATE_CONFIG"
echo "pid=$$"

python -m eval.v3_eval_suite \
  --candidate-label "$LABEL" \
  --candidate-loader v2 \
  --candidate-checkpoint "$CANDIDATE_CHECKPOINT" \
  --candidate-config "$CANDIDATE_CONFIG" \
  --baseline-label gpt2 \
  --output-dir "$OUT_DIR" \
  --report-dir "$REPORT_DIR" \
  --device cuda \
  --context-length 1024 \
  --execute \
  --no-progress

status=$?

echo "finished_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
TZ=America/Phoenix date "+finished_at_az=%Y-%m-%d %I:%M:%S %p AZ"
echo "exit_status=$status"

python - "$OUT_DIR" "$REPORT_DIR" "$LOG_PATH" "$LABEL" "$status" <<'PY'
import json
import math
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

out_dir = Path(sys.argv[1])
report_dir = Path(sys.argv[2])
log_path = Path(sys.argv[3])
label = sys.argv[4]
status = int(sys.argv[5])


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def fmt(value, digits=2):
    if value is None:
        return "n/a"
    value = float(value)
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


parts = []

candidate_heldout = load_json(out_dir / f"{label}_heldout.json")
gpt2_heldout = load_json(out_dir / "gpt2_heldout.json")
if candidate_heldout and gpt2_heldout:
    parts.append(
        f"heldout PPL {fmt(candidate_heldout.get('perplexity'))} vs GPT-2 {fmt(gpt2_heldout.get('perplexity'))}"
    )

external = load_json(out_dir / "external_full_candidate_gpt2.json")
if external:
    by_key = {(r.get("model"), r.get("benchmark")): r for r in external.get("results", [])}
    for benchmark in ("wikitext2", "lambada"):
        cand = by_key.get((label, benchmark))
        gpt2 = by_key.get(("gpt2", benchmark))
        if cand and gpt2:
            parts.append(
                f"{benchmark} PPL {fmt(cand.get('perplexity'))} vs GPT-2 {fmt(gpt2.get('perplexity'))}"
            )
    lambada = by_key.get((label, "lambada"))
    gpt2_lambada = by_key.get(("gpt2", "lambada"))
    if lambada and gpt2_lambada:
        parts.append(
            "LAMBADA word "
            f"{fmt(100 * lambada.get('last_word_greedy_exact_accuracy', 0), 2)}% "
            f"vs GPT-2 {fmt(100 * gpt2_lambada.get('last_word_greedy_exact_accuracy', 0), 2)}%"
        )

mc = load_json(out_dir / "multiple_choice_full_candidate_gpt2.json")
if mc:
    acc = {}
    for model in (label, "gpt2"):
        rows = [r for r in mc.get("results", []) if r.get("model") == model]
        if rows:
            acc[model] = sum(float(r.get("accuracy_norm", 0.0)) for r in rows) / len(rows)
    if label in acc and "gpt2" in acc:
        parts.append(f"MC avg norm {fmt(100 * acc[label], 2)}% vs GPT-2 {fmt(100 * acc['gpt2'], 2)}%")

gen = load_json(out_dir / "generation_metrics_candidate_gpt2.json")
if gen:
    summary = gen.get("summary", {})
    cand = summary.get(label)
    gpt2 = summary.get("gpt2")
    if cand and gpt2:
        parts.append(
            "repeat bigram "
            f"{fmt(100 * cand.get('mean_repeated_bigram_fraction', 0), 2)}% "
            f"vs GPT-2 {fmt(100 * gpt2.get('mean_repeated_bigram_fraction', 0), 2)}%"
        )

if not parts:
    parts.append(f"exit status {status}")

title_prefix = "SoloLLM v4" if label.startswith("v4") else "SoloLLM v3"
title = f"{title_prefix} full eval complete" if status == 0 else f"{title_prefix} full eval failed"
body = "; ".join(parts) + f"; log: {log_path}"
payload = {
    "title": title,
    "body": body,
    "dueAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "notes": f"Output: {out_dir}\nReports: {report_dir}",
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
