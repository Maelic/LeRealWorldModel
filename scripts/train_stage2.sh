#!/usr/bin/env bash
# Stage 2: Train the GC-IDM amortized planner on frozen JEPA encoder embeddings.
#
# Prereq: Stage 1 must have completed and produced a *_object.ckpt checkpoint.
#
# Usage:
#   ./scripts/train_stage2.sh <world_model_ckpt> [config_name] [extra hydra overrides...]
#
# config_name:
#   gc_idm          — dual camera top+side (default, auto-selected for dualcam checkpoints)
#   gc_idm_topcam   — top camera only (auto-selected for *topcam* checkpoint paths)
#
# Examples:
#   ./scripts/train_stage2.sh ~/.stable_worldmodel/so100_topcam/lewm_so100_topcam_epoch_50_object.ckpt
#   ./scripts/train_stage2.sh ~/.stable_worldmodel/so100_dualcam/lewm_so100_dualcam_epoch_50_object.ckpt
#   ./scripts/train_stage2.sh /path/to/ckpt steps=100000 batch_size=512
#
# Output: gc_idm.pt saved alongside the world model checkpoint.
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <world_model_ckpt_path> [config_name] [hydra overrides...]"
    echo ""
    echo "Example:"
    echo "  $0 ~/.stable_worldmodel/so100_topcam/lewm_so100_topcam_epoch_50_object.ckpt"
    exit 1
fi

WM_CKPT="$1"
shift

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f "$WM_CKPT" ]]; then
    echo "ERROR: World model checkpoint not found: $WM_CKPT"
    exit 1
fi

# Pick config: explicit second arg (if it doesn't look like a hydra override),
# otherwise auto-detect from checkpoint path.
if [[ $# -gt 0 && "${1}" != *"="* ]]; then
    CONFIG_NAME="$1"
    shift
elif [[ "$WM_CKPT" == *topcam* ]]; then
    CONFIG_NAME="gc_idm_topcam"
else
    CONFIG_NAME="gc_idm"
fi

echo "============================================================"
echo "  Stage 2: GC-IDM amortized planner training"
echo "  World model: $WM_CKPT"
echo "  Config:      config/train/${CONFIG_NAME}.yaml"
echo "============================================================"

python train_gc_idm.py --config-name "$CONFIG_NAME" "world_model_path=$WM_CKPT" "$@"

GC_IDM_PATH="$(dirname "$WM_CKPT")/gc_idm.pt"
echo ""
echo "Stage 2 complete. GC-IDM saved to: $GC_IDM_PATH"
