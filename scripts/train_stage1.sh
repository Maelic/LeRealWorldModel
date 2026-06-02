#!/usr/bin/env bash
# Stage 1: Train JEPA world model on the SO-100 dataset.
#
# Usage:
#   ./scripts/train_stage1.sh [config_name] [extra hydra overrides...]
#
# config_name choices:
#   lewm_so100_topcam   — top camera only,   50 epochs  (ablation)
#   lewm_so100_dualcam  — top + side fused,  50 epochs  (full model)
#   lewm_so100          — dual cam,         100 epochs  (default)
#
# Examples:
#   ./scripts/train_stage1.sh                           # dual cam, 100 epochs
#   ./scripts/train_stage1.sh lewm_so100_topcam         # top only, 50 epochs
#   ./scripts/train_stage1.sh lewm_so100_dualcam        # dual cam, 50 epochs
#   ./scripts/train_stage1.sh lewm_so100 trainer.max_epochs=200
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# First positional arg is the config name if it doesn't contain '='
# (i.e. it's not a Hydra override); defaults to lewm_so100.
if [[ $# -gt 0 && "${1}" != *"="* ]]; then
    CONFIG_NAME="$1"
    shift
else
    CONFIG_NAME="lewm_so100"
fi

echo "============================================================"
echo "  Stage 1: JEPA world model training"
echo "  Config:   config/train/${CONFIG_NAME}.yaml"
echo "============================================================"

python train_lewm.py --config-name "$CONFIG_NAME" "$@"

echo ""
echo "Stage 1 complete."
