#!/bin/bash
# ============================================================
# train_decoder_folding.sh
#
# Train a lightweight image decoder on top of a FROZEN JEPA encoder
# (Stage-1 folding world model), to visually verify the representations.
# Runs on a single GPU; the encoder is frozen so this is cheap — the only
# slow part is the one-off embedding pre-pass (AV1 video decode bound).
#
# Usage (from repo root on HPC):
#   sbatch scripts/slurm/train_decoder_folding.sh
#
# Override defaults, e.g. train on the in-progress rolling checkpoint and
# more episodes:
#   sbatch --export=ALL,NUM_EPISODES=120,STEPS=30000 \
#          scripts/slurm/train_decoder_folding.sh
#
# Env vars (all optional):
#   HPC_REPO     — abs path to this repo            (default below)
#   CKPT         — JEPA *_object.ckpt to decode     (default: rolling latest)
#   RUN_DIR      — where decoder.pt + viz land       (default checkpoints/folding_topcam)
#   NUM_EPISODES — episodes for the embedding pre-pass (default 60; 0 → all 1200)
#   STEPS        — decoder training steps            (default 20000)
#   BATCH_SIZE   — decoder batch size                (default 256)
#   EMB_WORKERS  — dataloader workers for pre-pass   (default 16)
#   GRES         — GPU gres string                   (default A100fat:1; A100:1 also fine)
# ============================================================
#SBATCH --job-name=dec_folding

#SBATCH -A naiss2026-4-349 -p alvis
#SBATCH -N 1 --gpus-per-node=A100fat:1     # 40 GB A100 is plenty; A100fat just for availability
#SBATCH -c 16                              # embedding pre-pass is video-decode bound
#SBATCH -t 04:00:00
#SBATCH -e /mimer/NOBACKUP/groups/naiss2026-4-349/LeRealWorldModel/jobs/%J.err
#SBATCH -o /mimer/NOBACKUP/groups/naiss2026-4-349/LeRealWorldModel/jobs/%J.out

set -euo pipefail

module purge > /dev/null 2>&1
ml Python/3.12.3-GCCcore-13.3.0 CUDA/12.6.0 FFmpeg/7.0.2-GCCcore-13.3.0

# ── Paths / defaults ──────────────────────────────────────────────────────────
HPC_REPO="${HPC_REPO:-/mimer/NOBACKUP/groups/naiss2026-4-349/LeRealWorldModel}"
export HF_HOME="/mimer/NOBACKUP/groups/naiss2026-4-349/.cache/huggingface"

RUN_DIR="${RUN_DIR:-${HPC_REPO}/checkpoints/folding_topcam}"
CKPT="${CKPT:-${RUN_DIR}/lewm_folding_topcam_latest_object.ckpt}"
NUM_EPISODES="${NUM_EPISODES:-60}"
STEPS="${STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EMB_WORKERS="${EMB_WORKERS:-16}"

# ── Offline behaviour for an air-gapped compute node ─────────────────────────
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# ── Freeze a point-in-time copy of the checkpoint ────────────────────────────
# The Stage-1 job overwrites *_latest_object.ckpt live; copy it so we don't read
# a half-written file and so it can't change mid-pre-pass.
FROZEN_CKPT="${RUN_DIR}/decoder_src_${SLURM_JOB_ID}.ckpt"
cp -f "$CKPT" "$FROZEN_CKPT"

echo "============================================================"
echo "  Decoder training — folding JEPA (base cam)"
echo "  HPC_REPO     = $HPC_REPO"
echo "  SRC CKPT     = $CKPT"
echo "  FROZEN COPY  = $FROZEN_CKPT"
echo "  RUN_DIR      = $RUN_DIR"
echo "  NUM_EPISODES = $NUM_EPISODES   STEPS = $STEPS   BATCH = $BATCH_SIZE"
echo "  EMB_WORKERS  = $EMB_WORKERS"
echo "  NODE         = $(hostname)   JOB = ${SLURM_JOB_ID:-n/a}"
echo "  GPU          = $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo N/A)"
echo "============================================================"

# ── Environment ───────────────────────────────────────────────────────────────
source "${HPC_REPO}/.venv/bin/activate"
cd "${HPC_REPO}"

START_TS=$(date +%s)

srun python train_jepa_decoder.py \
  --world-model-path "$FROZEN_CKPT" \
  --run-dir "$RUN_DIR" \
  --repo-id lerobot/high_quality_folding \
  --data-root "" \
  --num-episodes "$NUM_EPISODES" \
  --image-key observation.images.base \
  --encoder-scale tiny \
  --embed-dim 192 \
  --history-size 3 \
  --img-size 224 \
  --patch-size 14 \
  --frameskip 5 \
  --steps "$STEPS" \
  --batch-size "$BATCH_SIZE" \
  --emb-workers "$EMB_WORKERS" \
  --emb-batch-size 128 \
  --device cuda

END_TS=$(date +%s); WALL=$((END_TS - START_TS))
echo ""
echo "--- Decoder done. Wall time: ${WALL}s ($(( WALL/3600 ))h $(( (WALL%3600)/60 ))m) ---"
rm -f "$FROZEN_CKPT"

echo ""
echo "--- Outputs in: ${RUN_DIR}/ ---"
ls -lh "${RUN_DIR}"/decoder*.{pt,png} 2>/dev/null || true
