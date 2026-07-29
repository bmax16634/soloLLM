#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MIX_NAME="${MIX_NAME:-v4_1b_blend}"
MANIFEST="${MANIFEST:-${ROOT}/data_mixes/solollm-0.4.1b-balanced-blend-300m.yaml}"
DATA_DIR="${DATA_DIR:-${ROOT}/data/v4_1b_blend_300m_1024}"
RUN_OUT_DIR="${RUN_OUT_DIR:-${ROOT}/outputs/sologpt_v4/data_proxy_v4_1b_blend_57m_300m}"
RUN_DIR="${RUN_DIR:-${ROOT}/outputs/sologpt_v4/data_proxy_v4_1b_blend}"
REPORT_PATH="${REPORT_PATH:-${ROOT}/docs/results/v4_data_proxy_v4_1b_blend_300m.md}"
LOG_PATH="${LOG_PATH:-${RUN_DIR}/data_proxy_v4_1b_blend.log}"
CONFIG_PATH="${CONFIG_PATH:-${ROOT}/configs/solollm-0.4-57m-v3-style-baseproxy.json}"

EVAL_RUN_NAME="${EVAL_RUN_NAME:-v4_proxy_v4_1b_blend_57m}"
EVAL_OUT_DIR="${EVAL_OUT_DIR:-${ROOT}/outputs/eval_suites/${EVAL_RUN_NAME}}"
EVAL_REPORT_DIR="${EVAL_REPORT_DIR:-${ROOT}/docs/results/${EVAL_RUN_NAME}}"
EVAL_LOG_PATH="${EVAL_LOG_PATH:-${EVAL_OUT_DIR}/modern_eval.log}"
EVAL_LABEL="${EVAL_LABEL:-solollm-v4-1b-blend-proxy-57m}"

export ROOT
export MIX_NAME
export MANIFEST
export DATA_DIR
export RUN_OUT_DIR
export RUN_DIR
export REPORT_PATH
export LOG_PATH
export CONFIG_PATH

write_eval_summary() {
  python - "$ROOT" "$EVAL_OUT_DIR" "$EVAL_REPORT_DIR/summary.md" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
current_dir = Path(sys.argv[2])
summary_path = Path(sys.argv[3])

comparisons = [
    ("smollm_reasoning", root / "outputs/eval_suites/v4_proxy_smollm_reasoning_57m"),
    ("v4.1_blend", root / "outputs/eval_suites/v4_proxy_v4_1_blend_57m"),
    ("v4.1b_blend", current_dir),
]


def fmt(value, precision=4):
    if value is None:
        return "n/a"
    value = float(value)
    if not math.isfinite(value):
        return str(value)
    return f"{value:.{precision}f}"


def load_eval(run_dir):
    ext_path = run_dir / "external_modern_baselines.json"
    mc_path = run_dir / "multiple_choice_modern_baselines.json"
    if not ext_path.exists() or not mc_path.exists():
        return None
    ext = {row["benchmark"]: row for row in json.loads(ext_path.read_text())["results"]}
    mc_rows = json.loads(mc_path.read_text())["results"]
    mc = {row["benchmark"]: row for row in mc_rows}
    mc_avg = sum(row["accuracy_norm"] for row in mc_rows) / len(mc_rows)
    return {
        "wikitext_ppl": ext["wikitext2"]["perplexity"],
        "wikitext_bpb": ext["wikitext2"]["bits_per_byte"],
        "lambada_ppl": ext["lambada"]["perplexity"],
        "lambada_bpb": ext["lambada"]["bits_per_byte"],
        "lambada_tok_acc": ext["lambada"]["last_token_accuracy"],
        "mc_avg_norm": mc_avg,
        "hellaswag": mc["hellaswag"]["accuracy_norm"],
        "piqa": mc["piqa"]["accuracy_norm"],
        "arc_easy": mc["arc_easy"]["accuracy_norm"],
        "arc_challenge": mc["arc_challenge"]["accuracy_norm"],
        "winogrande": mc["winogrande"]["accuracy_norm"],
    }


rows = [(name, load_eval(path)) for name, path in comparisons]
rows = [(name, data) for name, data in rows if data is not None]

lines = [
    "# V4.1b Blend Proxy Modern Eval Summary",
    "",
    "The 57M `v4.1b_blend` proxy rebalances v4.1 by adding more SmolLM/Cosmopedia/FineMath signal while keeping most of the broad-web recovery.",
    "",
    "| Proxy | WikiText BPB ↓ | WikiText PPL ↓ | LAMBADA BPB ↓ | LAMBADA PPL ↓ | LAMBADA tok acc ↑ | MC avg norm ↑ |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for name, data in rows:
    lines.append(
        f"| {name} | {fmt(data['wikitext_bpb'])} | {fmt(data['wikitext_ppl'], 2)} | "
        f"{fmt(data['lambada_bpb'])} | {fmt(data['lambada_ppl'], 2)} | "
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
        "Use this proxy to decide whether the extra reasoning slice restores MC without losing the v4.1 broad-LM recovery. Scale only if v4.1b is clearly better than v4.1 on MC while staying close on WikiText/LAMBADA.",
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
