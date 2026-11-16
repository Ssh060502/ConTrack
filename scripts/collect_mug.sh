#!/usr/bin/env bash
# Run on the Linux server after: conda activate env_isaaclab
set -euo pipefail

usage() {
    echo 'Usage: bash scripts/collect_mug.sh [--episodes COUNT] [--gpu INDEX] [--mass-scale NUMBER] [--friction NUMBER]'
    echo 'Defaults: GPU 0; mass scale 1.0; friction retains original material inheritance.'
}
MASS_SCALE="1.0"
FRICTION=""
GPU="0"
EPISODES="10"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mass-scale|--friction|--gpu|--episodes)
            if [[ $# -lt 2 || "$2" == --* || -z "$2" ]]; then
                printf 'Missing value for %s\n' "$1" >&2
                exit 2
            fi
            case "$1" in
                --mass-scale) MASS_SCALE="$2" ;;
                --friction) FRICTION="$2" ;;
                --gpu) GPU="$2" ;;
                --episodes) EPISODES="$2" ;;
            esac
            shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ ! "$GPU" =~ ^(0|[1-9][0-9]*)$ ]]; then
    printf 'Invalid --gpu: expected a nonnegative integer, got %s\n' "$GPU" >&2
    exit 2
fi

if [[ ! "$EPISODES" =~ ^[1-9][0-9]*$ ]]; then
    printf 'Invalid --episodes: expected a positive integer\n' >&2
    exit 2
fi

# Validate and normalize before creating outputs or starting Isaac Sim.
NORMALIZED="$(python - "$MASS_SCALE" "$FRICTION" <<'PY'
import math
import sys
try:
    mass = float(sys.argv[1])
    if not math.isfinite(mass) or mass <= 0 or not math.isfinite(200.0 * mass):
        raise ValueError('--mass-scale must be finite and > 0, with finite scaled density')
    friction = None if sys.argv[2] == '' else float(sys.argv[2])
    if friction is not None and (not math.isfinite(friction) or friction < 0):
        raise ValueError('--friction must be finite and >= 0')
except ValueError as exc:
    sys.exit(f'Invalid physics parameter: {exc}')
print(f'{mass} {"original" if friction is None else friction}')
PY
)"
read -r MASS_SCALE FRICTION_TAG <<< "$NORMALIZED"

# Resolve the repo from this script, regardless of the terminal's current directory.
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SESSION="$REPO_ROOT/logs/rsl_rl/sharpa_mug_sweep/grab-s4_mug_pass_1-rby1a_sharpa-2026_10_14-03_57_03-sharpa_mug_sweep_trial"
CHECKPOINT="$SESSION/model_1500.pt"
DATA="/mnt/ssd1/shenghe/datasets/GRAB_contrack/grab-s4_mug_pass_1-rby1a_sharpa.h5"

for required_file in "$CHECKPOINT" "$DATA" "$SESSION/params/agent.yaml" "$SESSION/params/env.yaml"; do
    if [[ ! -f "$required_file" ]]; then
        printf 'Missing required file: %s\n' "$required_file" >&2
        exit 1
    fi
done

# A fresh output directory for each run, including runs started in the same second.
mkdir -p outputs/physics_validation
OUTPUT_DIR="$(mktemp -d "$REPO_ROOT/outputs/physics_validation/mug_collect_mass${MASS_SCALE}_friction${FRICTION_TAG}_$(date +%Y%m%d_%H%M%S)_XXXXXX")"

# Keep the original mass and material inheritance even if overrides were exported
# in the calling terminal. This script changes only its own process environment.
export CONTRACK_SHARPA_OBJECT_MASS_SCALE="$MASS_SCALE"
unset CONTRACK_SHARPA_OBJECT_FRICTION
if [[ "$FRICTION_TAG" != original ]]; then
    export CONTRACK_SHARPA_OBJECT_FRICTION="$FRICTION_TAG"
fi
export CUDA_VISIBLE_DEVICES="$GPU"

printf 'Checkpoint: %s\nReference: %s\nGPU (CUDA_VISIBLE_DEVICES): %s\nMass scale: %s\nObject friction: %s\nOutputs: %s\n' \
    "$CHECKPOINT" "$DATA" "$GPU" "$MASS_SCALE" "$FRICTION_TAG" "$OUTPUT_DIR" | tee "$OUTPUT_DIR/run.log"

python -u scripts/rsl_rl/play.py \
    --task Isaac-Rby1a-Sharpa-Mimic-Manager-v0 \
    --session "$SESSION" \
    --checkpoint "$CHECKPOINT" \
    --data "$DATA" \
    --num_envs 1 \
    --device cuda:0 \
    --seed 42 \
    --headless \
    --episodes "$EPISODES" \
    --collection_max_steps 100000 \
    --collect_trajectories "$OUTPUT_DIR/episodes.h5" \
    2>&1 | tee -a "$OUTPUT_DIR/run.log"

printf '\nFinished. Outputs: %s\n' "$OUTPUT_DIR"
