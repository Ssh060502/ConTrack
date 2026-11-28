#!/usr/bin/env bash
# Single-env CRX base-policy evaluation, one reference-start attempt per video.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
MASS=1.0
FRICTION=original
GPU=0
EPISODES=10
EXTRA=()
VIDEO=1
DIAGNOSTIC_SUFFIX=
SESSION="$REPO_ROOT/logs/rsl_rl/sharpa_crx_hammer_baseline/grab-s1_hammer_use_1-crx_sharpa-slow3-cal1122-2026_11_24-14_17_44_cal1122_env8000"
MODEL=model_1999.pt
while [[ $# -gt 0 ]]; do
    case "$1" in
        --trace-reset) EXTRA+=(--trace_reset); shift;;
        --no-video) VIDEO=0; shift;;
        --diagnostic-no-self-collisions)
            EXTRA+=(--diagnostic_no_self_collisions)
            DIAGNOSTIC_SUFFIX=_diagnostic_no_self_collisions
            shift;;
        --mass-scale|--friction|--gpu|--episodes|--session|--model)
            if [[ $# -lt 2 ]]; then echo "Missing value for $1" >&2; exit 2; fi
            case "$1" in
                --mass-scale) MASS="$2";;
                --friction) FRICTION="$2";;
                --gpu) GPU="$2";;
                --episodes) EPISODES="$2";;
                --session) SESSION="$2";;
                --model) MODEL="$2";;
            esac
            shift 2;;
        --help|-h)
            echo 'Usage: bash scripts/collect_crx_hammer.sh --mass-scale 6 --friction original --gpu 0 --episodes 10'
            echo 'Optional: --trace-reset (records reset and first physics substeps in episodes.h5)'
            echo 'Optional: --no-video (skip rendering and video encoding)'
            echo 'Diagnostic only: --diagnostic-no-self-collisions (keeps hand-object collisions)'
            exit 0;;
        *) echo "Unknown option: $1" >&2; exit 2;;
    esac
done
[[ "$GPU" =~ ^[0-9]+$ && "$EPISODES" =~ ^[1-9][0-9]*$ ]] || {
    echo 'GPU must be nonnegative and episodes positive integers' >&2; exit 2;
}
[[ -f "$SESSION/$MODEL" ]] || { echo "Missing checkpoint: $SESSION/$MODEL" >&2; exit 2; }
mkdir -p outputs/physics_validation
OUTPUT="$(mktemp -d "$REPO_ROOT/outputs/physics_validation/crx_hammer_mass${MASS}_friction${FRICTION}${DIAGNOSTIC_SUFFIX}_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
echo "Outputs: $OUTPUT"
PHYSICS=(env -u CONTRACK_SHARPA_OBJECT_FRICTION
    "CUDA_VISIBLE_DEVICES=$GPU" "CONTRACK_SHARPA_OBJECT_MASS_SCALE=$MASS")
if [[ "$FRICTION" != original ]]; then
    PHYSICS+=("CONTRACK_SHARPA_OBJECT_FRICTION=$FRICTION")
fi
if [[ "$VIDEO" == 1 ]]; then
    EXTRA+=(--video --video_folder "$OUTPUT/videos")
fi
"${PHYSICS[@]}" python -u scripts/rsl_rl/play.py \
    --task Isaac-Crx-Sharpa-Mimic-Manager-Play-v0 \
    --session "$SESSION" --checkpoint "$SESSION/$MODEL" \
    --num_envs 1 --seed 42 --deterministic_start \
    --collect_trajectories "$OUTPUT/episodes.h5" --episodes "$EPISODES" \
    --pose_break_mode log-only \
    --headless --device cuda:0 "${EXTRA[@]}" 2>&1 | tee "$OUTPUT/run.log"
python - "$OUTPUT/episodes.h5" "$EPISODES" <<'PY'
import sys
import h5py
with h5py.File(sys.argv[1], 'r') as f:
    if not f.attrs.get('collection_complete', False) or int(f.attrs.get('completed_episodes', 0)) != int(sys.argv[2]):
        raise RuntimeError('Collection did not complete; inspect run.log')
print('Collection verified complete:', sys.argv[1])
PY
echo "Finished. Outputs: $OUTPUT"
