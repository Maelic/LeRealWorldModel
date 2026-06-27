#!/bin/bash
# ============================================================
# train_jepa_folding.sh
#
# Stage 1: train the JEPA latent world model on the bimanual cloth-folding
# dataset (lerobot/high_quality_folding), base camera only, on a single
# 80 GB GPU (A100fat / H100).
#
# Usage (from repo root on HPC):
#   sbatch scripts/slurm/train_jepa_folding.sh
#
# Override defaults:
#   sbatch --export=ALL,CONFIG=lewm_folding_topcam,BATCH_SIZE=384,EPOCHS=100 \
#          scripts/slurm/train_jepa_folding.sh
#
# ── IMPORTANT: Alvis compute nodes have NO internet ─────────────────────────
# Warm the HF cache + (optionally) the wandb run ON A LOGIN NODE first:
#
#   export HF_HOME=/mimer/NOBACKUP/groups/naiss2026-4-349/.cache/huggingface
#   huggingface-cli download lerobot/high_quality_folding --repo-type dataset
#
# Then this job runs with HF_HUB_OFFLINE=1 (cache-only) and WANDB_MODE=offline.
# After the job, sync metrics from a login node:
#   wandb sync wandb/offline-run-*    (or:  wandb sync --sync-all)
#
# NOTE: the #SBATCH -e/-o log dir below must exist before submitting:
#   mkdir -p /mimer/NOBACKUP/groups/naiss2026-4-349/LeRealWorldModel/jobs
# (SBATCH paths can't expand shell vars; edit them if your repo lives elsewhere.)
#
# Env vars (all optional):
#   HPC_REPO   — abs path to this repo on HPC   (default below)
#   HF_HOME    — persistent HF cache dir         (default below; keep off $HOME)
#   CONFIG     — Hydra config name               (default lewm_folding_topcam)
#   BATCH_SIZE — loader.batch_size               (default 256)
#   EPOCHS     — trainer.max_epochs              (default 100)
#   LR         — optimizer.lr                    (default unset → use config)
#   EPISODES   — Hydra list to subset episodes   (e.g. '[0,1,2,3]'; default full)
#   WANDB_MODE — online | offline | disabled     (default offline)
#   EXTRA      — extra Hydra overrides, appended verbatim
# ============================================================
#SBATCH --job-name=jepa_folding

#SBATCH -A naiss2026-4-349 -p alvis
#SBATCH -N 1 --gpus-per-node=A100fat:1     # 80 GB A100. 40 GB: A100:1 | H100 cluster: adjust gres
#SBATCH -c 16                              # data loading is video-decode bound — give it cores
#SBATCH -t 2-00:00:00
#SBATCH -e /mimer/NOBACKUP/groups/naiss2026-4-349/LeRealWorldModel/jobs/%J.err
#SBATCH -o /mimer/NOBACKUP/groups/naiss2026-4-349/LeRealWorldModel/jobs/%J.out

set -euo pipefail

# ── Paths / defaults ──────────────────────────────────────────────────────────
HPC_REPO="${HPC_REPO:-/mimer/NOBACKUP/groups/naiss2026-4-349/LeRealWorldModel}"
export HF_HOME="${HF_HOME:-/mimer/NOBACKUP/groups/naiss2026-4-349/.cache/huggingface}"

CONFIG="${CONFIG:-lewm_folding_topcam}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EPOCHS="${EPOCHS:-100}"
EPISODES="${EPISODES:-}"          # empty → use whatever the config sets (full dataset)
LR="${LR:-}"                      # empty → use config lr
EXTRA="${EXTRA:-}"

# ── Offline behaviour for an air-gapped compute node ─────────────────────────
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"   # read dataset from the pre-warmed cache only
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"     # sync later from a login node
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

echo "============================================================"
echo "  Stage 1: JEPA world model — cloth folding (base cam)"
echo "  HPC_REPO   = $HPC_REPO"
echo "  HF_HOME    = $HF_HOME"
echo "  CONFIG     = $CONFIG"
echo "  BATCH_SIZE = $BATCH_SIZE   EPOCHS = $EPOCHS"
echo "  EPISODES   = ${EPISODES:-<full dataset>}   LR = ${LR:-<config>}"
echo "  HF_HUB_OFFLINE=$HF_HUB_OFFLINE   WANDB_MODE=$WANDB_MODE"
echo "  NODE       = $(hostname)"
echo "  SLURM_JOB_ID = ${SLURM_JOB_ID:-n/a}   CPUS = ${SLURM_CPUS_PER_TASK:-?}"
echo "  GPU        = $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================================"

# ── Environment ───────────────────────────────────────────────────────────────
source "${HPC_REPO}/.venv/bin/activate"
cd "${HPC_REPO}"

# ── Assemble Hydra overrides ──────────────────────────────────────────────────
OVERRIDES=(
  "--config-name" "$CONFIG"
  "loader.batch_size=${BATCH_SIZE}"
  "num_workers=${SLURM_CPUS_PER_TASK:-16}"
  "trainer.max_epochs=${EPOCHS}"
  "wandb.enabled=true"
)
[[ -n "$EPISODES" ]] && OVERRIDES+=("data.dataset.episodes=${EPISODES}")
[[ -n "$LR" ]]       && OVERRIDES+=("optimizer.lr=${LR}")
# shellcheck disable=SC2206
[[ -n "$EXTRA" ]]    && OVERRIDES+=(${EXTRA})

echo ""
echo "--- GPU state before training ---"
nvidia-smi 2>/dev/null || true

echo ""
echo "--- Starting training ---"
echo "python train_lewm.py ${OVERRIDES[*]}"
START_TS=$(date +%s)

srun python train_lewm.py "${OVERRIDES[@]}"

END_TS=$(date +%s)
WALL=$((END_TS - START_TS))
echo ""
echo "--- Training done. Wall time: ${WALL}s ($(( WALL/3600 ))h $(( (WALL%3600)/60 ))m) ---"

# ── Outputs ───────────────────────────────────────────────────────────────────
RUN_DIR="${HPC_REPO}/checkpoints/folding_topcam"
echo ""
echo "--- Outputs in: ${RUN_DIR}/ ---"
ls -lh "${RUN_DIR}" 2>/dev/null || true
echo ""
echo "Next steps:"
echo "  • sync wandb:        wandb sync wandb/offline-run-*   (from a login node)"
echo "  • identifiability:   python analysis/run_identifiability_so100.py \\"
echo "                         --ckpt ${RUN_DIR}/lewm_folding_topcam_epoch_${EPOCHS}_object.ckpt \\"
echo "                         --normalizers ${RUN_DIR}/lewm_folding_topcam_normalizers.pt"
echo ""
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
