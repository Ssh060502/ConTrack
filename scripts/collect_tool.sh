#!/usr/bin/env bash
# Run on the Linux server after: conda activate env_isaaclab
set -euo pipefail

usage() {
    echo 'Usage: bash scripts/collect_tool.sh --tool flashlight|waterbottle|cylinder [--video] [--episodes COUNT] [--gpu INDEX] [--mass-scale NUMBER] [--friction NUMBER]'
    echo 'Optional: --session DIR --checkpoint FILE --data FILE --pose-break-mode strict|log-only --trace-reset --diagnostic-no-self-collisions'
    echo 'Defaults: GPU 0; mass scale 1.0; friction retains original material inheritance.'
}
MASS_SCALE="1.0"
FRICTION=""
GPU="0"
EPISODES="3"
TOOL=""
SESSION=""
CHECKPOINT=""
DATA=""
VIDEO_ARGS=()
TRACE_ARGS=()
COLLISION_ARGS=()
COLLISION_TAG=""
POSE_BREAK_MODE="log-only"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --diagnostic-no-self-collisions) COLLISION_ARGS=(--diagnostic_no_self_collisions); COLLISION_TAG="_noself_DIAGNOSTIC"; shift ;;
        --trace-reset) TRACE_ARGS=(--trace_reset); shift ;;
        --video) VIDEO_ARGS=(--video); shift ;;
        --tool|--session|--checkpoint|--data|--mass-scale|--friction|--gpu|--episodes|--pose-break-mode)
            if [[ $# -lt 2 || "$2" == --* || -z "$2" ]]; then
                printf 'Missing value for %s\n' "$1" >&2
                exit 2
            fi
            case "$1" in
                --tool) TOOL="$2" ;;
                --session) SESSION="$2" ;;
                --checkpoint) CHECKPOINT="$2" ;;
                --data) DATA="$2" ;;
                --mass-scale) MASS_SCALE="$2" ;;
                --friction) FRICTION="$2" ;;
                --gpu) GPU="$2" ;;
                --episodes) EPISODES="$2" ;;
                --pose-break-mode) POSE_BREAK_MODE="$2" ;;
            esac
            shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ "$TOOL" != flashlight && "$TOOL" != waterbottle && "$TOOL" != cylinder ]]; then
    echo 'Required: --tool flashlight|waterbottle|cylinder' >&2
    exit 2
fi
if [[ "$POSE_BREAK_MODE" != strict && "$POSE_BREAK_MODE" != log-only ]]; then
    printf 'Invalid --pose-break-mode: use strict or log-only\n' >&2
    exit 2
fi

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

if [[ -z "$SESSION" ]]; then
    case "$TOOL" in
        cylinder) SESSION="$REPO_ROOT/logs/rsl_rl/sharpa_cylindermedium_lift_baseline/grab-s1_cylindermedium_lift-rby1a_sharpa-2026_11_16-17_40_55" ;;
        flashlight) SESSION="$REPO_ROOT/logs/rsl_rl/sharpa_flashlight_lift_baseline/grab-s1_flashlight_lift-rby1a_sharpa-2026_11_16-15_01_57" ;;
        waterbottle) SESSION="$REPO_ROOT/logs/rsl_rl/sharpa_waterbottle_baseline/grab-s1_waterbottle_offhand_1-rby1a_sharpa-24_504-2026_11_16-12_27_58" ;;
    esac
fi
CHECKPOINT="${CHECKPOINT:-$SESSION/model_1999.pt}"
if [[ -z "$DATA" ]]; then
    DATA="$(python - "$SESSION/params/env.yaml" <<'PYDATA'
import sys
import yaml
with open(sys.argv[1]) as f:
    cfg = yaml.load(f, Loader=yaml.BaseLoader)
path = cfg.get("reference_path")
if not isinstance(path, str) or not path:
    sys.exit("Missing reference_path in env.yaml; specify --data FILE")
print(path)
PYDATA
)"
fi

for required_file in "$CHECKPOINT" "$DATA" "$SESSION/params/agent.yaml" "$SESSION/params/env.yaml"; do
    if [[ ! -f "$required_file" ]]; then
        printf 'Missing required file: %s\n' "$required_file" >&2
        exit 1
    fi
done

# A fresh output directory for each run, including runs started in the same second.
mkdir -p outputs/physics_validation
OUTPUT_DIR="$(mktemp -d "$REPO_ROOT/outputs/physics_validation/${TOOL}_collect_${POSE_BREAK_MODE}_mass${MASS_SCALE}_friction${FRICTION_TAG}${COLLISION_TAG}_$(date +%Y%m%d_%H%M%S)_XXXXXX")"

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

python -X faulthandler -u scripts/rsl_rl/play.py \
    --task Isaac-Rby1a-Sharpa-Mimic-Manager-v0 \
    --session "$SESSION" \
    --checkpoint "$CHECKPOINT" \
    --data "$DATA" \
    --num_envs 1 \
    --device cuda:0 \
    --seed 42 \
    --headless \
    "${VIDEO_ARGS[@]}" \
    "${TRACE_ARGS[@]}" \
    "${COLLISION_ARGS[@]}" \
    --episodes "$EPISODES" \
    --pose_break_mode "$POSE_BREAK_MODE" \
    --collection_max_steps 100000 \
    --collect_trajectories "$OUTPUT_DIR/episodes.h5" \
    2>&1 | tee -a "$OUTPUT_DIR/run.log"

printf '\nFinished. Outputs: %s\n' "$OUTPUT_DIR"
