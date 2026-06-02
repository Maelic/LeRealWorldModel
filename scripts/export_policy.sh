#!/usr/bin/env bash
# Export trained JEPA + GC-IDM as a LeRobot-compatible checkpoint directory.
#
# Usage:
#   ./scripts/export_policy.sh <run_dir> [goal_image]
#
#   <run_dir>     Directory containing lewm_so100_epoch_N_object.ckpt,
#                 gc_idm.pt, and lewm_so100_normalizers.pt
#   [goal_image]  Optional path to a goal PNG/JPG (can also set later at deploy time)
#
# Examples:
#   ./scripts/export_policy.sh ~/.stable_worldmodel/models/abc123
#   ./scripts/export_policy.sh ~/.stable_worldmodel/models/abc123 ./goal.jpg
#
# Output: checkpoints/jepa_so100/
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <run_dir> [goal_image]"
    exit 1
fi

RUN_DIR="${1%/}"    # strip trailing slash
GOAL_IMAGE="${2:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Auto-detect the latest epoch checkpoint
WM_CKPT=$(ls -t "$RUN_DIR"/lewm_so100_epoch_*_object.ckpt 2>/dev/null | head -1)
if [[ -z "$WM_CKPT" ]]; then
    echo "ERROR: No lewm_so100_epoch_*_object.ckpt found in $RUN_DIR"
    exit 1
fi

GC_IDM_PATH="$RUN_DIR/gc_idm.pt"
if [[ ! -f "$GC_IDM_PATH" ]]; then
    echo "ERROR: gc_idm.pt not found in $RUN_DIR — run train_stage2.sh first."
    exit 1
fi

NORM_PATH="$RUN_DIR/lewm_so100_normalizers.pt"
NORM_ARG=""
if [[ -f "$NORM_PATH" ]]; then
    NORM_ARG="--normalizers-path $NORM_PATH"
fi

GOAL_ARG=""
if [[ -n "$GOAL_IMAGE" ]]; then
    GOAL_ARG="--goal-image-path $GOAL_IMAGE"
fi

OUTPUT_DIR="checkpoints/jepa_so100"

echo "============================================================"
echo "  Export JEPAPolicy checkpoint"
echo "  World model:  $WM_CKPT"
echo "  GC-IDM:       $GC_IDM_PATH"
echo "  Normalizers:  ${NORM_PATH:-none}"
echo "  Output dir:   $OUTPUT_DIR"
echo "============================================================"

python export_policy.py \
    --world-model-path "$WM_CKPT" \
    --gc-idm-path "$GC_IDM_PATH" \
    $NORM_ARG \
    $GOAL_ARG \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "Checkpoint ready at: $OUTPUT_DIR"
echo "Deploy with: ./scripts/deploy_jepa.sh"
