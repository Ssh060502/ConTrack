#!/usr/bin/env bash
# Compatibility entry point; without arguments this runs original defaults.
set -euo pipefail
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/run_hammer.sh" "$@"
