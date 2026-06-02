#!/usr/bin/env bash
# Stage 1: Train JEPA world model on the SO-100 dual-camera dataset.
#
# Usage:
#   ./scripts/train_stage1.sh [extra hydra overrides...]
#
# Examples:
#   ./scripts/train_stage1.sh                            # defaults (stack_cubes, 100 epochs)
#   ./scripts/train_stage1.sh trainer.max_epochs=50      # quicker run
#   ./scripts/train_stage1.sh compile=False              # disable torch.compile for debugging
#   ./scripts/train_stage1.sh wandb.enabled=True         # enable W&B logging
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "============================================================"
echo "  Stage 1: JEPA world model training"
echo "  Config:   config/train/lewm_so100.yaml"
echo "  Output:   \$STABLEWM_HOME/models/<run_id>/"
echo "============================================================"

python train_lewm.py --config-name lewm_so100 "$@"

echo ""
echo "Stage 1 complete."
echo "Find the latest run in: \$(python -c \"import stable_worldmodel as swm; print(swm.data.utils.get_cache_dir())\")/models/"
