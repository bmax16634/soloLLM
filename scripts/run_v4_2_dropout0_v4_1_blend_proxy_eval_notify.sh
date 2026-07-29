#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG_PATH="${CONFIG_PATH:-${ROOT}/configs/solollm-0.4.2-57m-broad-blend-baseproxy.json}"
DATA_DIR="${DATA_DIR:-${ROOT}/data/v4_1_blend_300m_1024}"
MANIFEST="${MANIFEST:-${ROOT}/data_mixes/solollm-0.4.1-broad-blend-300m.yaml}"

MIX_NAME="${MIX_NAME:-v4_2_dropout0_v4_1_blend}"
RUN_OUT_DIR="${RUN_OUT_DIR:-${ROOT}/outputs/sologpt_v4/arch_proxy_v4_2_dropout0_v4_1_blend_57m_300m}"
RUN_DIR="${RUN_DIR:-${ROOT}/outputs/sologpt_v4/arch_proxy_v4_2_dropout0_v4_1_blend}"
REPORT_PATH="${REPORT_PATH:-${ROOT}/docs/results/v4_arch_proxy_v4_2_dropout0_v4_1_blend_57m_300m.md}"
LOG_PATH="${LOG_PATH:-${RUN_DIR}/arch_proxy_v4_2_dropout0_v4_1_blend.log}"

EVAL_RUN_NAME="${EVAL_RUN_NAME:-v4_2_dropout0_v4_1_blend_proxy_57m}"
EVAL_OUT_DIR="${EVAL_OUT_DIR:-${ROOT}/outputs/eval_suites/${EVAL_RUN_NAME}}"
EVAL_REPORT_DIR="${EVAL_REPORT_DIR:-${ROOT}/docs/results/${EVAL_RUN_NAME}}"
EVAL_LOG_PATH="${EVAL_LOG_PATH:-${EVAL_OUT_DIR}/modern_eval.log}"
EVAL_LABEL="${EVAL_LABEL:-solollm-v4-2-dropout0-v4-1-blend-proxy-57m}"

export ROOT
export MIX_NAME
export MANIFEST
export DATA_DIR
export RUN_OUT_DIR
export RUN_DIR
export REPORT_PATH
export LOG_PATH
export CONFIG_PATH
export BUILD_IF_MISSING=0

write_eval_summary() {
  python - "$ROOT" "$RUN_OUT_DIR" "$EVAL_OUT_DIR" "$EVAL_REPORT_DIR/summary.md" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
current_train_dir = Path(sys.argv[2])
current_eval_dir = Path(sys.argv[3])
summary_path = Path(sys.argv[4])

comparisons = [
    (
        "v4.1 blend dropout0.1",
        root / "outputs/sologpt_v4/data_proxy_v4_1_blend_57m_300m",
        root / "outputs/eval_suites/v4_proxy_v4_1_blend_57m",
    ),
    (
        "v4.1b blend dropout0.1",
        root / "outputs/sologpt_v4/data_proxy_v4_1b_blend_57m_300m",
        root / "outputs/eval_suites/v4_proxy_v4_1b_blend_57m",
    ),
    (
        "v4.1b blend dropout0.0",
        root / "outputs/sologpt_v4/arch_proxy_v4_2_dropout0_57m_300m",
        root / "outputs/eval_suites/v4_2_dropout0_proxy_57m",
    ),
    (
        "v4.1 blend dropout0.0",
        current_train_dir,
        current_eval_dir,
    ),
]


def fmt(value, precision=4):
    if value is None:
        return "n/a"
    value = float(value)
    if not math.isfinite(value):
        return str(value)
    return f"{value:.{precision}f}"


def load_row(train_dir, eval_dir):
    summary = json.loads((train_dir / "metrics_summary.json").read_text())
    ext = {
        row["benchmark"]: row
        for row in json.loads((eval_dir / "external_modern_baselines.json").read_text())["results"]
    }
    mc_rows = json.loads((eval_dir / "multiple_choice_modern_baselines.json").read_text())["results"]
    mc = {row["benchmark"]: row for row in mc_rows}
    mc_avg = sum(row["accuracy_norm"] for row in mc_rows) / len(mc_rows)
    return {
        "val_ppl": summary["best_val_ppl"],
        "train_loss": summary["final_train_loss"],
        "wikitext_ppl": ext["wikitext2"]["perplexity"],
        "lambada_ppl": ext["lambada"]["perplexity"],
        "lambada_tok_acc": ext["lambada"]["last_token_accuracy"],
        "mc_avg_norm": mc_avg,
        "hellaswag": mc["hellaswag"]["accuracy_norm"],
        "piqa": mc["piqa"]["accuracy_norm"],
        "arc_easy": mc["arc_easy"]["accuracy_norm"],
        "arc_challenge": mc["arc_challenge"]["accuracy_norm"],
        "winogrande": mc["winogrande"]["accuracy_norm"],
    }


rows = []
for name, train_dir, eval_dir in comparisons:
    if (train_dir / "metrics_summary.json").exists() and (
        eval_dir / "external_modern_baselines.json"
    ).exists():
        rows.append((name, load_row(train_dir, eval_dir)))

lines = [
    "# V4.2 Dropout-0 On V4.1 Blend Proxy Summary",
    "",
    "Controlled 57M/300M-token proxy using the broader v4.1 blend data. The intended change versus v4.1 is dropout `0.1 -> 0.0`.",
    "",
    "| Proxy | Val PPL ↓ | Train loss ↓ | WikiText PPL ↓ | LAMBADA PPL ↓ | LAMBADA tok acc ↑ | MC avg norm ↑ |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for name, data in rows:
    lines.append(
        f"| {name} | {fmt(data['val_ppl'], 2)} | {fmt(data['train_loss'])} | "
        f"{fmt(data['wikitext_ppl'], 2)} | {fmt(data['lambada_ppl'], 2)} | "
        f"{fmt(data['lambada_tok_acc'])} | {fmt(data['mc_avg_norm'])} |"
    )

lines.extend(
    [
        "",
        "## Multiple Choice",
        "",
        "| Proxy | HellaSwag | PIQA | ARC-Easy | ARC-Challenge | WinoGrande |",
        "|---|---:|---:|---:|---:|---:|",
    ]
)
for name, data in rows:
    lines.append(
        f"| {name} | {fmt(data['hellaswag'])} | {fmt(data['piqa'])} | "
        f"{fmt(data['arc_easy'])} | {fmt(data['arc_challenge'])} | {fmt(data['winogrande'])} |"
    )

lines.extend(
    [
        "",
        "## Decision",
        "",
        "Scale only if dropout `0.0` keeps the v4.1 broad external advantage while materially improving MC and/or validation.",
        "",
    ]
)

summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text("\n".join(lines), encoding="utf-8")
print(summary_path)
PY
}

"${ROOT}/scripts/run_v4_data_proxy_mix_notify.sh"
proxy_status=$?
if [ "$proxy_status" -ne 0 ]; then
  exit "$proxy_status"
fi

ROOT="$ROOT" \
RUN_NAME="$EVAL_RUN_NAME" \
OUT_DIR="$EVAL_OUT_DIR" \
REPORT_DIR="$EVAL_REPORT_DIR" \
LOG_PATH="$EVAL_LOG_PATH" \
V2_LABEL="$EVAL_LABEL" \
V2_CHECKPOINT="${RUN_OUT_DIR}/final_model.pt" \
V2_CONFIG="$CONFIG_PATH" \
MODELS=solollm \
"${ROOT}/scripts/run_modern_baselines_eval_notify.sh"
eval_status=$?

if [ "$eval_status" -eq 0 ]; then
  write_eval_summary
fi

exit "$eval_status"
