#!/usr/bin/env bash
# Train a JEPA image decoder and generate visualizations.
#
# Usage:
#   ./scripts/train_decoder.sh [run_dir] [extra args...]
#
# run_dir defaults to ./checkpoints/so100_topcam (auto-detected if omitted).
#
# Examples:
#   ./scripts/train_decoder.sh
#   ./scripts/train_decoder.sh checkpoints/so100_dualcam
#   ./scripts/train_decoder.sh checkpoints/so100_topcam --steps 30000
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ $# -gt 0 && "${1}" != --* ]]; then
    RUN_DIR="${1%/}"
    shift
else
    RUN_DIR=$(ls -td "$REPO_ROOT"/checkpoints/*/ 2>/dev/null | head -1)
    if [[ -z "$RUN_DIR" ]]; then
        echo "ERROR: No run_dir given and no subdirectory found under checkpoints/"
        exit 1
    fi
    echo "Auto-selected run_dir: $RUN_DIR"
fi

WM_CKPT=$(ls -t "$RUN_DIR"/lewm_*_epoch_*_object.ckpt 2>/dev/null | head -1)
if [[ -z "$WM_CKPT" ]]; then
    echo "ERROR: No world model checkpoint found in $RUN_DIR"
    exit 1
fi

GC_IDM_PATH="$RUN_DIR/gc_idm.pt"
GC_IDM_ARG=""
if [[ -f "$GC_IDM_PATH" ]]; then
    GC_IDM_ARG="--gc-idm-path $GC_IDM_PATH"
    echo "GC-IDM found:    $GC_IDM_PATH (will generate GC-IDM rollout viz)"
fi

echo "============================================================"
echo "  JEPA decoder training"
echo "  World model: $WM_CKPT"
echo "  Run dir:     $RUN_DIR"
echo "============================================================"

python train_jepa_decoder.py \
    --world-model-path "$WM_CKPT" \
    --run-dir "$RUN_DIR" \
    $GC_IDM_ARG \
    "$@"

echo ""
echo "Decoder saved to: $RUN_DIR/decoder.pt"
echo "Visualizations:   $RUN_DIR/decoder_recon.png"
echo "                  $RUN_DIR/decoder_rollout.png"
if [[ -n "$GC_IDM_ARG" ]]; then
    echo "                  $RUN_DIR/decoder_gcidm_rollout.png"
fi
