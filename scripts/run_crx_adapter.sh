#!/usr/bin/env bash
# CRX base defaults remain separate from the existing RBY1A launcher.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$REPO_ROOT/scripts/run_mug_adapter.sh" --robot crx --tool hammer "$@"
