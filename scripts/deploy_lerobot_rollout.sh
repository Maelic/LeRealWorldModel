#!/usr/bin/env bash
# Deploy via lerobot-rollout (uses exported checkpoint directory).
#
# This path requires a pre-exported checkpoint (run export_policy.sh first)
# and a pre-captured goal image (passed via --policy.goal_image_path).
# For interactive goal capture use deploy_jepa.sh instead.
#
# Usage:
#   ./scripts/deploy_lerobot_rollout.sh <checkpoint_dir> <goal_image> [extra args...]
#
# Examples:
#   ./scripts/deploy_lerobot_rollout.sh checkpoints/jepa_so100 goal.jpg
#   ./scripts/deploy_lerobot_rollout.sh checkpoints/jepa_so100 goal.jpg --duration=120
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <checkpoint_dir> <goal_image> [lerobot-rollout args...]"
    exit 1
fi

CHECKPOINT_DIR="${1%/}"
GOAL_IMAGE="$2"
shift 2

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -d "$CHECKPOINT_DIR" ]]; then
    echo "ERROR: Checkpoint directory not found: $CHECKPOINT_DIR"
    echo "Run ./scripts/export_policy.sh first."
    exit 1
fi

# ── Hardware defaults ─────────────────────────────────────────────────────────
PORT="${PORT:-/dev/ttyACM0}"
CAM_UP_INDEX="${CAM_UP_INDEX:-0}"
CAM_SIDE_INDEX="${CAM_SIDE_INDEX:-2}"
FPS="${FPS:-30}"
# ─────────────────────────────────────────────────────────────────────────────

echo "============================================================"
echo "  lerobot-rollout with JEPAPolicy"
echo "  Checkpoint:  $CHECKPOINT_DIR"
echo "  Goal image:  $GOAL_IMAGE"
echo "  Port:        $PORT"
echo "============================================================"

python -m lewm_robot.rollout_jepa \
    --policy.type=jepa \
    --policy.path="$CHECKPOINT_DIR" \
    --policy.goal_image_path="$GOAL_IMAGE" \
    --robot.type=so100_follower \
    --robot.port="$PORT" \
    --robot.cameras="{
        \"up\":   {\"type\": \"opencv\", \"index_or_path\": $CAM_UP_INDEX,   \"width\": 640, \"height\": 480, \"fps\": $FPS},
        \"side\": {\"type\": \"opencv\", \"index_or_path\": $CAM_SIDE_INDEX, \"width\": 640, \"height\": 480, \"fps\": $FPS}
    }" \
    --fps="$FPS" \
    --inference.type=sync \
    "$@"
