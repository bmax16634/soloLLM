#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MIX_NAME="${MIX_NAME:-v4_1_blend}"
MANIFEST="${MANIFEST:-${ROOT}/sologpt_v4/data_mix_v4_1_blend_300m.yaml}"
DATA_DIR="${DATA_DIR:-${ROOT}/data/v4_1_blend_300m_1024}"
RUN_OUT_DIR="${RUN_OUT_DIR:-${ROOT}/outputs/sologpt_v4/data_proxy_v4_1_blend_57m_300m}"
RUN_DIR="${RUN_DIR:-${ROOT}/outputs/sologpt_v4/data_proxy_v4_1_blend}"
REPORT_PATH="${REPORT_PATH:-${ROOT}/docs/results/v4_data_proxy_v4_1_blend_300m.md}"
LOG_PATH="${LOG_PATH:-${RUN_DIR}/data_proxy_v4_1_blend.log}"

export ROOT
export MIX_NAME
export MANIFEST
export DATA_DIR
export RUN_OUT_DIR
export RUN_DIR
export REPORT_PATH
export LOG_PATH

exec "${ROOT}/scripts/run_v4_data_proxy_mix_notify.sh"
