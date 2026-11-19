#!/usr/bin/env bash
# Same train/evaluate flags as run_mug_adapter.sh, with waterbottle defaults.
# Defaults: model_1999.pt, conditions 1:2,3:2, correction scale 0.1, task-only.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$REPO_ROOT/scripts/run_mug_adapter.sh" --tool waterbottle "$@"
