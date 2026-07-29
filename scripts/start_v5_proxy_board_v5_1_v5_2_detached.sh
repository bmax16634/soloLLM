#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SESSION="${SESSION:-sologpt_v5_proxy_board_v5_1_v5_2}"
LOG_PATH="${LOG_PATH:-${ROOT}/outputs/sologpt_v5/v5_1_v5_2_proxy_board.log}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session_exists=$SESSION"
  exit 0
fi

mkdir -p "$(dirname "$LOG_PATH")"
tmux new-session -d -s "$SESSION" "cd '$ROOT' && NOTIFY=1 BOARD_LOG='$LOG_PATH' bash scripts/run_v5_proxy_board_v5_1_v5_2_notify.sh"
echo "session_started=$SESSION"
echo "log=$LOG_PATH"
