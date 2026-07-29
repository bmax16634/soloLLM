#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/bmx/_projects/soloLLM}"

export ROOT
export RUN_NAME="${RUN_NAME:-v4_2_dropout0_v4_1_blend_10b_150m_1024}"
export MANIFEST="${MANIFEST:-${ROOT}/data_mixes/solollm-1.0-broad-blend-10b.yaml}"
export DATA_DIR="${DATA_DIR:-${ROOT}/data/v4_2_dropout0_v4_1_blend_10b_1024}"
export BUILD_IF_MISSING="${BUILD_IF_MISSING:-0}"
export OVERWRITE_DATA="${OVERWRITE_DATA:-0}"

export TRAIN_OUT_DIR="${TRAIN_OUT_DIR:-${ROOT}/outputs/sologpt_v4/${RUN_NAME}}"
export TRAIN_LOG_PATH="${TRAIN_LOG_PATH:-${ROOT}/outputs/sologpt_v4/${RUN_NAME}.log}"
export CONFIG_PATH="${CONFIG_PATH:-${ROOT}/configs/solollm-1.0-152m-base.json}"
export SHARD_DIR="${SHARD_DIR:-${ROOT}/data/v4_2_dropout0_v4_1_blend_10b_1024/chunks}"
export TRAIN_SHARDS="${TRAIN_SHARDS:-0:1168}"
export VAL_SHARDS="${VAL_SHARDS:-1169:1180}"
export TRAIN_TARGET_TOKENS="${TRAIN_TARGET_TOKENS:-9800571904}"

export EVAL_LABEL="${EVAL_LABEL:-v4_2_dropout0_v4_1_blend_10b_150m}"
export EVAL_OUT_DIR="${EVAL_OUT_DIR:-${ROOT}/outputs/eval_suites/${EVAL_LABEL}_gpt2_full_suite}"
export EVAL_REPORT_DIR="${EVAL_REPORT_DIR:-${EVAL_OUT_DIR}/reports}"
export EVAL_LOG_PATH="${EVAL_LOG_PATH:-${EVAL_OUT_DIR}/v4_2_eval_suite_full.log}"

export EVAL_EVERY_TOKENS="${EVAL_EVERY_TOKENS:-250000000}"
export TOKENS_PER_CHECKPOINT="${TOKENS_PER_CHECKPOINT:-2000000000}"
export MAX_EVAL_TOKENS="${MAX_EVAL_TOKENS:-4194304}"
export DEVICE="${DEVICE:-cuda}"

exec "${ROOT}/scripts/run_v4_smollm_reasoning_10b_150m_train_eval_notify.sh"
