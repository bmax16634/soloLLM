#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MIX_NAME="${MIX_NAME:-v5_smol_quality}"
CONFIG_PATH="${CONFIG_PATH:-${ROOT}/configs/solollm-1.1-57m-baseproxy.json}"
MANIFEST="${MANIFEST:-${ROOT}/data_mixes/solollm-1.1-smol-quality-1b.yaml}"
DATA_DIR="${DATA_DIR:-${ROOT}/data/v5_smol_quality_1b_1024}"
RUN_OUT_DIR="${RUN_OUT_DIR:-${ROOT}/outputs/sologpt_v5/proxy_smol_quality_60m_1b}"
RUN_DIR="${RUN_DIR:-${ROOT}/outputs/sologpt_v5/proxy_smol_quality_60m_1b_logs}"
TRAIN_LOG_PATH="${TRAIN_LOG_PATH:-${RUN_DIR}/train.log}"
MAIN_LOG_PATH="${MAIN_LOG_PATH:-${RUN_DIR}/train_eval.log}"
TRAIN_REPORT_PATH="${TRAIN_REPORT_PATH:-${ROOT}/docs/results/v5_proxy_smol_quality_60m_1b_train.md}"
SUMMARY_REPORT_PATH="${SUMMARY_REPORT_PATH:-${ROOT}/docs/results/v5_proxy_smol_quality_60m_1b_summary.md}"

EVAL_RUN_NAME="${EVAL_RUN_NAME:-v5_proxy_smol_quality_60m_1b}"
EVAL_OUT_DIR="${EVAL_OUT_DIR:-${ROOT}/outputs/eval_suites/${EVAL_RUN_NAME}}"
EVAL_REPORT_DIR="${EVAL_REPORT_DIR:-${ROOT}/docs/results/${EVAL_RUN_NAME}}"
EVAL_LOG_PATH="${EVAL_LOG_PATH:-${EVAL_OUT_DIR}/modern_eval.log}"
EVAL_LABEL="${EVAL_LABEL:-solollm-v5-proxy-smol-quality-60m-1b}"

EXPERIMENT_DISPLAY_NAME="${EXPERIMENT_DISPLAY_NAME:-V5 Smol-Quality 60M/1B Proxy}"
EXPERIMENT_DESCRIPTION="${EXPERIMENT_DESCRIPTION:-This proxy is aimed at moving toward modern SmolLM2-style small-model quality instead of repeatedly tuning against GPT-2.}"
PROXY_LABEL="${PROXY_LABEL:-v5 smol-quality 60M/1B proxy}"
CONTROL_LABEL="${CONTROL_LABEL:-v4.2 57M/300M proxy}"
CONTROL_TRAIN_DIR="${CONTROL_TRAIN_DIR:-${ROOT}/outputs/sologpt_v4/arch_proxy_v4_2_dropout0_v4_1_blend_57m_300m}"
CONTROL_EVAL_DIR="${CONTROL_EVAL_DIR:-${ROOT}/outputs/eval_suites/v4_2_dropout0_v4_1_blend_proxy_57m}"

TARGET_TOKENS="${TARGET_TOKENS:-1000000000}"
MAX_EVAL_TOKENS="${MAX_EVAL_TOKENS:-2097152}"
EVAL_EVERY_TOKENS="${EVAL_EVERY_TOKENS:-100000000}"
TOKENS_PER_CHECKPOINT="${TOKENS_PER_CHECKPOINT:-250000000}"
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

write_summary() {
  python - "$ROOT" "$RUN_OUT_DIR" "$DATA_DIR" "$EVAL_OUT_DIR" "$SUMMARY_REPORT_PATH" "$EXPERIMENT_DISPLAY_NAME" "$EXPERIMENT_DESCRIPTION" "$PROXY_LABEL" "$CONTROL_LABEL" "$CONTROL_TRAIN_DIR" "$CONTROL_EVAL_DIR" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
run_dir = Path(sys.argv[2])
data_dir = Path(sys.argv[3])
eval_dir = Path(sys.argv[4])
summary_path = Path(sys.argv[5])
experiment_display_name = sys.argv[6]
experiment_description = sys.argv[7]
proxy_label = sys.argv[8]
control_label = sys.argv[9]
control_train_dir = Path(sys.argv[10])
control_eval_dir = Path(sys.argv[11])


def fmt(value, precision=4):
    if value is None:
        return "n/a"
    value = float(value)
    if not math.isfinite(value):
        return str(value)
    return f"{value:.{precision}f}"


def load_proxy_row(label, train_dir, eval_suite_dir):
    train_summary = json.loads((train_dir / "metrics_summary.json").read_text())
    external = {
        row["benchmark"]: row
        for row in json.loads((eval_suite_dir / "external_modern_baselines.json").read_text())["results"]
        if row["model"].startswith("solollm-")
    }
    mc_rows = [
        row
        for row in json.loads((eval_suite_dir / "multiple_choice_modern_baselines.json").read_text())["results"]
        if row["model"].startswith("solollm-")
    ]
    return {
        "label": label,
        "params": train_summary.get("parameter_count"),
        "tokens": train_summary.get("tokens_total"),
        "best_val_ppl": train_summary.get("best_val_ppl"),
        "train_loss": train_summary.get("final_train_loss"),
        "wikitext_bpb": external.get("wikitext2", {}).get("bits_per_byte"),
        "lambada_bpb": external.get("lambada", {}).get("bits_per_byte"),
        "lambada_tok": external.get("lambada", {}).get("last_token_accuracy"),
        "lambada_word": external.get("lambada", {}).get("last_word_greedy_exact_accuracy"),
        "mc_avg_norm": sum(row["accuracy_norm"] for row in mc_rows) / len(mc_rows) if mc_rows else None,
    }


rows = []
comparisons = [
    (control_label, control_train_dir, control_eval_dir),
    (proxy_label, run_dir, eval_dir),
]
for label, train_dir, eval_suite_dir in comparisons:
    if (train_dir / "metrics_summary.json").exists() and (eval_suite_dir / "external_modern_baselines.json").exists():
        rows.append(load_proxy_row(label, train_dir, eval_suite_dir))

stats_path = data_dir / "build_stats.json"
stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}

lines = [
    f"# {experiment_display_name} Summary",
    "",
    experiment_description,
    "",
    f"- Dataset: `{data_dir}`",
    f"- Train run: `{run_dir}`",
    f"- Eval suite: `{eval_dir}`",
]
if stats:
    lines.append(f"- Accepted tokens: `{int(stats['progress']['accepted_tokens']):,}`")
lines.extend(
    [
        "",
        "## Gate Board",
        "",
        "| Proxy | Params | Train tokens | Val PPL | Train loss | WikiText BPB | LAMBADA BPB | LAMBADA tok | LAMBADA word | MC avg norm |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
)
for row in rows:
    lines.append(
        f"| {row['label']} | {fmt(row['params'], 0)} | {fmt(row['tokens'], 0)} | "
        f"{fmt(row['best_val_ppl'], 2)} | {fmt(row['train_loss'])} | "
        f"{fmt(row['wikitext_bpb'])} | {fmt(row['lambada_bpb'])} | "
        f"{fmt(row['lambada_tok'])} | {fmt(row['lambada_word'])} | {fmt(row['mc_avg_norm'])} |"
    )

lines.extend(
    [
        "",
        "## Scale Decision Rule",
        "",
        "Scale only if the v5 proxy improves broad BPB and LAMBADA continuation",
        "without giving back the multiple-choice gains that v4.2 found.",
        "",
        "A mixed result means run another cheap proxy. A clear regression means do",
        "not launch a 123M/150M v5 full run from this mix.",
        "",
    ]
)
summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text("\n".join(lines), encoding="utf-8")
print(summary_path)
PY
}

status_body() {
  local status="$1"
  python - "$status" "$RUN_OUT_DIR" "$EVAL_OUT_DIR" "$SUMMARY_REPORT_PATH" <<'PY'
import json
import sys
from pathlib import Path

status = int(sys.argv[1])
run_dir = Path(sys.argv[2])
eval_dir = Path(sys.argv[3])
summary_path = Path(sys.argv[4])
parts = [f"exit {status}"]
train_summary = run_dir / "metrics_summary.json"
if train_summary.exists():
    data = json.loads(train_summary.read_text())
    parts.append(f"{int(data.get('tokens_total', 0)) / 1_000_000:.1f}M tokens")
    parts.append(f"{float(data.get('tokens_per_sec_avg', 0.0)) / 1000:.1f}k tok/s")
    parts.append(f"best val PPL {float(data.get('best_val_ppl', 0.0)):.2f}")
if (eval_dir / "external_modern_baselines.json").exists():
    parts.append("external eval complete")
if (eval_dir / "multiple_choice_modern_baselines.json").exists():
    parts.append("MC eval complete")
if summary_path.exists():
    parts.append(f"summary: {summary_path}")
print("; ".join(parts))
PY
}

mkdir -p "$RUN_DIR" "$(dirname "$SUMMARY_REPORT_PATH")" "$EVAL_OUT_DIR" "$EVAL_REPORT_DIR"
cd "$ROOT" || exit 2
exec >>"$MAIN_LOG_PATH" 2>&1

echo "=== SoloLLM v5 smol-quality proxy train + eval ==="
echo "started_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
TZ=America/Phoenix date "+started_at_az=%Y-%m-%d %I:%M:%S %p AZ"
echo "root=$ROOT"
echo "manifest=$MANIFEST"
echo "config=$CONFIG_PATH"
echo "data_dir=$DATA_DIR"
echo "run_out_dir=$RUN_OUT_DIR"
echo "eval_out_dir=$EVAL_OUT_DIR"
echo "main_log=$MAIN_LOG_PATH"
echo "train_log=$TRAIN_LOG_PATH"
echo "eval_log=$EVAL_LOG_PATH"
echo "target_tokens=$TARGET_TOKENS"
echo "pid=$$"

if [ "$NOTIFY" = "1" ]; then
  notify \
    "SoloLLM ${EXPERIMENT_DISPLAY_NAME} started" \
    "${MIX_NAME}; dataset build if missing, then train and modern eval." \
    "Run: ${RUN_OUT_DIR}; log: ${MAIN_LOG_PATH}"
fi

ROOT="$ROOT" \
MIX_NAME="$MIX_NAME" \
MANIFEST="$MANIFEST" \
DATA_DIR="$DATA_DIR" \
RUN_OUT_DIR="$RUN_OUT_DIR" \
RUN_DIR="${RUN_DIR}/train_inner" \
LOG_PATH="$TRAIN_LOG_PATH" \
REPORT_PATH="$TRAIN_REPORT_PATH" \
CONFIG_PATH="$CONFIG_PATH" \
TARGET_TOKENS="$TARGET_TOKENS" \
MAX_EVAL_TOKENS="$MAX_EVAL_TOKENS" \
EVAL_EVERY_TOKENS="$EVAL_EVERY_TOKENS" \
TOKENS_PER_CHECKPOINT="$TOKENS_PER_CHECKPOINT" \
DEVICE="$DEVICE" \
BUILD_IF_MISSING="$BUILD_IF_MISSING" \
OVERWRITE_DATA="$OVERWRITE_DATA" \
NOTIFY=0 \
bash "${ROOT}/scripts/run_v4_data_proxy_mix_notify.sh"
train_status=$?

if [ "$train_status" -ne 0 ]; then
  body="$(status_body "$train_status")"
  if [ "$NOTIFY" = "1" ]; then
    notify "SoloLLM ${EXPERIMENT_DISPLAY_NAME} train failed" "$body" "Log: ${TRAIN_LOG_PATH}"
  fi
  exit "$train_status"
fi

ROOT="$ROOT" \
RUN_NAME="$EVAL_RUN_NAME" \
OUT_DIR="$EVAL_OUT_DIR" \
REPORT_DIR="$EVAL_REPORT_DIR" \
LOG_PATH="$EVAL_LOG_PATH" \
V2_LABEL="$EVAL_LABEL" \
V2_CHECKPOINT="${RUN_OUT_DIR}/final_model.pt" \
V2_CONFIG="$CONFIG_PATH" \
MODELS="solollm gpt2 pythia-70m=hf:EleutherAI/pythia-70m-deduped pythia-160m=hf:EleutherAI/pythia-160m-deduped smollm2-135m=hf:HuggingFaceTB/SmolLM2-135M smollm2-360m=hf:HuggingFaceTB/SmolLM2-360M" \
CONTEXT_LENGTH=1024 \
MAX_WIKITEXT_TOKENS=0 \
MAX_LAMBADA_EXAMPLES=0 \
MAX_MC_EXAMPLES=0 \
DEVICE="$DEVICE" \
NOTIFY=0 \
bash "${ROOT}/scripts/run_modern_baselines_eval_notify.sh"
eval_status=$?

if [ "$eval_status" -eq 0 ]; then
  write_summary
fi

status="$eval_status"
echo "finished_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
TZ=America/Phoenix date "+finished_at_az=%Y-%m-%d %I:%M:%S %p AZ"
echo "train_status=$train_status"
echo "eval_status=$eval_status"

body="$(status_body "$status")"
if [ "$NOTIFY" = "1" ]; then
  if [ "$status" -eq 0 ]; then
    notify "SoloLLM ${EXPERIMENT_DISPLAY_NAME} complete" "$body" "Summary: ${SUMMARY_REPORT_PATH}"
  else
    notify "SoloLLM ${EXPERIMENT_DISPLAY_NAME} eval failed" "$body" "Eval log: ${EVAL_LOG_PATH}"
  fi
fi

exit "$status"
