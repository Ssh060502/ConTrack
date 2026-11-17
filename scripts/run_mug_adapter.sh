#!/usr/bin/env bash
# Activate env_isaaclab before running. Remaining flags go to oracle_adapter.py.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
GPU="0"
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)
            if [[ $# -lt 2 || ! "$2" =~ ^(0|[1-9][0-9]*)$ ]]; then
                echo '--gpu requires a nonnegative integer' >&2
                exit 2
            fi
            GPU="$2"; shift 2 ;;
        *) ARGS+=("$1"); shift ;;
    esac
done
mkdir -p logs/oracle_adapter
LOG="$(mktemp "$REPO_ROOT/logs/oracle_adapter/launcher_$(date +%Y%m%d_%H%M%S)_XXXXXX.log")"
printf 'GPU: %s\nLauncher log: %s\n' "$GPU" "$LOG"
CUDA_VISIBLE_DEVICES="$GPU" python -u scripts/rsl_rl/oracle_adapter.py \
    --device cuda:0 --headless "${ARGS[@]}" 2>&1 | tee "$LOG"
