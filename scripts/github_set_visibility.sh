#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_FILE="${ENV_FILE:-${ROOT}/.env.local}"
OWNER="${GITHUB_OWNER:-bmax16634}"
REPO="${GITHUB_REPO:-soloLLM}"
VISIBILITY="${1:-private}"

if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

if [ -z "${GITHUB_TOKEN:-}" ]; then
  echo "GITHUB_TOKEN is missing. Put it in ${ENV_FILE} or export it in the shell." >&2
  exit 2
fi

case "$VISIBILITY" in
  private)
    private_value=true
    ;;
  public)
    private_value=false
    ;;
  *)
    echo "usage: $0 [private|public]" >&2
    exit 2
    ;;
esac

response="$(
  curl -fsS \
    -X PATCH \
    -H "Authorization: Bearer ${GITHUB_TOKEN}" \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "https://api.github.com/repos/${OWNER}/${REPO}" \
    -d "{\"private\":${private_value}}"
)"

python - "$response" <<'PY'
import json
import sys

repo = json.loads(sys.argv[1])
print(f"{repo['full_name']} private={repo['private']} visibility={repo.get('visibility', 'n/a')}")
PY
