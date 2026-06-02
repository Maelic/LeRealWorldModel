#!/usr/bin/env bash
# Deploy the trained JEPA + GC-IDM policy on the real SO-100 arm.
#
# This script uses the CUSTOM deploy loop (lewm_robot.deploy_jepa_so100) which
# supports dual-camera, interactive goal capture, per-step latency logging, and
# dry-run mode. For lerobot-rollout integration see deploy_lerobot_rollout.sh.
#
# Usage:
#   ./scripts/deploy_jepa.sh <run_dir> [--capture-goal | --goal-image <path>] [extra args...]
#
# Examples:
#   # Capture goal from live cameras
#   ./scripts/deploy_jepa.sh ~/.stable_worldmodel/models/abc123 --capture-goal
#
#   # Load goal from file
#   ./scripts/deploy_jepa.sh ~/.stable_worldmodel/models/abc123 --goal-image ./goal.jpg
#
#   # Dry run (no hardware)
#   ./scripts/deploy_jepa.sh ~/.stable_worldmodel/models/abc123 --goal-image ./goal.jpg \
#       --dry-run-replay-from maelicneau/stack_cubes \
#       --dry-run-replay-root ./datasets/stack_cubes
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <run_dir> [--capture-goal | --goal-image <path>] [args...]"
    exit 1
fi

RUN_DIR="${1%/}"
shift

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ── Configurable hardware defaults ───────────────────────────────────────────
PORT="${PORT:-/dev/ttyACM0}"
CAM_UP_INDEX="${CAM_UP_INDEX:-0}"
CAM_SIDE_INDEX="${CAM_SIDE_INDEX:-2}"
FPS="${FPS:-30}"
MAX_STEPS="${MAX_STEPS:-300}"
MAX_RELATIVE_TARGET="${MAX_RELATIVE_TARGET:-}"   # leave empty to use robot default
# ─────────────────────────────────────────────────────────────────────────────

# Auto-detect latest checkpoint
WM_CKPT=$(ls -t "$RUN_DIR"/lewm_so100_epoch_*_object.ckpt 2>/dev/null | head -1)
if [[ -z "$WM_CKPT" ]]; then
    echo "ERROR: No lewm_so100_epoch_*_object.ckpt in $RUN_DIR"
    exit 1
fi

GC_IDM_PATH="$RUN_DIR/gc_idm.pt"
if [[ ! -f "$GC_IDM_PATH" ]]; then
    echo "ERROR: gc_idm.pt not found — run train_stage2.sh first."
    exit 1
fi

NORM_ARG=""
NORM_PATH="$RUN_DIR/lewm_so100_normalizers.pt"
[[ -f "$NORM_PATH" ]] && NORM_ARG="--normalizers-path $NORM_PATH"

SAFETY_ARG=""
[[ -n "$MAX_RELATIVE_TARGET" ]] && SAFETY_ARG="--max-relative-target $MAX_RELATIVE_TARGET"

echo "============================================================"
echo "  GC-IDM deployment on SO-100"
echo "  World model: $WM_CKPT"
echo "  GC-IDM:      $GC_IDM_PATH"
echo "  Port:        $PORT"
echo "  Cameras:     up=$CAM_UP_INDEX  side=$CAM_SIDE_INDEX"
echo "  FPS:         $FPS | Max steps: $MAX_STEPS"
echo "============================================================"
echo ""

python -m lewm_robot.deploy_jepa_so100 \
    --world-model-path "$WM_CKPT" \
    --gc-idm-path "$GC_IDM_PATH" \
    $NORM_ARG \
    --port "$PORT" \
    --camera-up-index "$CAM_UP_INDEX" \
    --camera-side-index "$CAM_SIDE_INDEX" \
    --fps "$FPS" \
    --max-steps "$MAX_STEPS" \
    $SAFETY_ARG \
    "$@"
