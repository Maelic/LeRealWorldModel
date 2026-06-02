#!/usr/bin/env bash
# Collect teleoperated demonstrations on the dual-camera SO-100.
#
# Prerequisites:
#   - Follower arm on FOLLOWER_PORT
#   - Leader arm on LEADER_PORT
#   - Two USB cameras: up (index 0) and side (index 2)
#
# Usage:
#   ./scripts/collect_data.sh [num_episodes] [repo_id] [task_description]
#
# Examples:
#   ./scripts/collect_data.sh 20 maelicneau/stack_cubes "Stack the cubes"
#   ./scripts/collect_data.sh 5  maelicneau/stack_cubes_test "Stack the cubes"
set -euo pipefail

# ── Configurable defaults ────────────────────────────────────────────────────
NUM_EPISODES="${1:-20}"
REPO_ID="${2:-maelicneau/stack_cubes}"
TASK="${3:-Stack three cubes.}"

FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/ttyACM0}"
LEADER_PORT="${LEADER_PORT:-/dev/ttyACM1}"
FOLLOWER_ID="${FOLLOWER_ID:-follower_1}"
LEADER_ID="${LEADER_ID:-leader_1}"

FPS=30
CAM_W=640
CAM_H=480
CAM_UP_INDEX=0
CAM_SIDE_INDEX=2
# ─────────────────────────────────────────────────────────────────────────────

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "============================================================"
echo "  Data collection: $REPO_ID"
echo "  Task:            $TASK"
echo "  Episodes:        $NUM_EPISODES"
echo "  Follower port:   $FOLLOWER_PORT"
echo "  Leader port:     $LEADER_PORT"
echo "============================================================"

lerobot-record \
    --robot.type=so100_follower \
    --robot.port="$FOLLOWER_PORT" \
    --robot.id="$FOLLOWER_ID" \
    --robot.cameras="{
        \"up\":   {\"type\": \"opencv\", \"index_or_path\": $CAM_UP_INDEX,   \"width\": $CAM_W, \"height\": $CAM_H, \"fps\": $FPS},
        \"side\": {\"type\": \"opencv\", \"index_or_path\": $CAM_SIDE_INDEX, \"width\": $CAM_W, \"height\": $CAM_H, \"fps\": $FPS}
    }" \
    --teleop.type=so100_leader \
    --teleop.port="$LEADER_PORT" \
    --teleop.id="$LEADER_ID" \
    --dataset.repo_id="$REPO_ID" \
    --dataset.num_episodes="$NUM_EPISODES" \
    --dataset.single_task="$TASK" \
    --dataset.root="./datasets/$(basename $REPO_ID)" \
    --dataset.streaming_encoding=true \
    --dataset.encoder_threads=2 \
    --display_data=true \
    "$@"

echo "Done. Dataset saved to ./datasets/$(basename $REPO_ID)"
echo "Remember to update info.json if total_episodes/total_frames are 0."
