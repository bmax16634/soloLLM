#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLONE_ROOT="${SOLOLLM_HF_CLONE_ROOT:-/home/bmx/_hf_repos/solollm/models}"
EXPORT_ROOT="${SOLOLLM_HF_EXPORT_ROOT:-$PROJECT_ROOT/outputs/huggingface}"
OWNER="${SOLOLLM_HF_OWNER:-bmax16634}"
SOURCE="${SOLOLLM_MODEL_SOURCE:-hf}"

repos=(
  "solollm-1.0-123m-base"
  "solollm-1.0-152m-base"
)

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing required command: $1" >&2
    exit 1
  fi
}

require_command git
require_command rsync

if ! git lfs version >/dev/null 2>&1; then
  echo "missing required git-lfs support: install git-lfs, then run git lfs install" >&2
  exit 1
fi

mkdir -p "$CLONE_ROOT" "$EXPORT_ROOT"

repo_url() {
  local repo="$1"
  case "$SOURCE" in
    hf)
      printf 'https://huggingface.co/%s/%s\n' "$OWNER" "$repo"
      ;;
    fs)
      printf 'ben@192.168.0.193:/mnt/raid/26/aimlcs/hf_repos/solollm/models/%s.git\n' "$repo"
      ;;
    *)
      echo "unsupported SOLOLLM_MODEL_SOURCE=$SOURCE; use hf or fs" >&2
      exit 1
      ;;
  esac
}

for repo in "${repos[@]}"; do
  clone_dir="$CLONE_ROOT/$repo"
  url="$(repo_url "$repo")"
  echo "== $repo =="

  if [ ! -d "$clone_dir/.git" ]; then
    git clone "$url" "$clone_dir"
  else
    if git -C "$clone_dir" remote get-url origin >/dev/null 2>&1; then
      git -C "$clone_dir" remote set-url origin "https://huggingface.co/$OWNER/$repo"
    else
      git -C "$clone_dir" remote add origin "https://huggingface.co/$OWNER/$repo"
    fi

    if git -C "$clone_dir" remote get-url fs >/dev/null 2>&1; then
      git -C "$clone_dir" remote set-url fs "ben@192.168.0.193:/mnt/raid/26/aimlcs/hf_repos/solollm/models/$repo.git"
    else
      git -C "$clone_dir" remote add fs "ben@192.168.0.193:/mnt/raid/26/aimlcs/hf_repos/solollm/models/$repo.git"
    fi

    remote_name="$SOURCE"
    [ "$SOURCE" = "hf" ] && remote_name="origin"
    git -C "$clone_dir" fetch "$remote_name" --prune
    git -C "$clone_dir" checkout main
    git -C "$clone_dir" pull --ff-only "$remote_name" main
  fi

  git -C "$clone_dir" lfs pull
  rsync -a --delete --exclude '.git/' "$clone_dir/" "$EXPORT_ROOT/$repo/"
  echo "restored $EXPORT_ROOT/$repo"
done
