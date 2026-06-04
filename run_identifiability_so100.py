"""
Identifiability & representation-quality evaluation for the SO-100 JEPA model
trained on lerobot/svla_so100_pickplace.

Unlike the PushT version there is no simulation ground-truth state beyond
the robot's proprioception (joint angles), so we use proprio (6-D) as the
ground-truth factor for all linear/nonlinear probe metrics.

Metrics computed
----------------
  Equivariance    — 1-step MSE and cosine similarity between predicted and
                    true next embedding.
  k-step rollout  — Autoregressive equivariance degradation over k=1..K_MAX.
  Affine probe    — Linear regression z → proprio: per-joint R², Pearson r.
  Nonlinear probe — MLP regression z → proprio.
  Probe generalization — train on seen episodes, test on held-out ones.
  Action diversity     — Effective rank and entropy of the action distribution.
  Temporal contrastivity — Does changing the action change the embedding?
  Action invertibility   — Can we decode Δz → action?

Usage
-----
    python run_identifiability_so100.py
    python run_identifiability_so100.py --ckpt ~/.stable_worldmodel/lewm_so100_epoch_19_object.ckpt
    python run_identifiability_so100.py --corruption 0.5
    python run_identifiability_so100.py --n-episodes 300 --output my_results.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import v2 as T

sys.path.insert(0, str(Path(__file__).parent))

from jepa import JEPA
from identifiability import (
    affine_identifiability,
    nonlinear_identifiability,
    action_diversity,
    temporal_contrastivity,
    equivariance_error,
    action_invertibility,
    probe_generalization,
    dci_metrics,
)
from action_diversity import corrupt_actions

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FRAMESKIP        = 5
ACTION_DIM       = 6                       # SO-100 has 6 joints
EFFECTIVE_ACT    = FRAMESKIP * ACTION_DIM  # = 30
EMBED_DIM        = 192
HISTORY_SIZE     = 3
IMG_SIZE         = 224
IMAGENET_MEAN    = [0.485, 0.456, 0.406]
IMAGENET_STD     = [0.229, 0.224, 0.225]
REPO_ID          = "maelicneau/stack_cubes"
IMAGE_KEY        = "observation.images.up"
PROPRIO_KEY      = "observation.state"
ACTION_KEY       = "action"
JOINT_NAMES      = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
K_MAX            = 7                       # max k-step rollout horizon


# ---------------------------------------------------------------------------
# Image transform (matches training)
# ---------------------------------------------------------------------------

def make_transform() -> T.Compose:
    return T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        T.Resize(size=IMG_SIZE, antialias=True),
    ])


# ---------------------------------------------------------------------------
# Dataset sampling via LeRobotDataset
# ---------------------------------------------------------------------------

def sample_trajectories(
    repo_id: str,
    n_episodes: int,
    seq_len: int,
    seed: int = 42,
    corruption: float = 0.0,
    action_mean: np.ndarray | None = None,
    action_std: np.ndarray | None = None,
    root: str | None = None,
    image_key: str = IMAGE_KEY,
):
    """Sample `n_episodes` short trajectories from a LeRobot dataset.

    Returns
    -------
    pixels   : torch.Tensor (N, seq_len, 3, H, W)  ImageNet-normalized float32
    actions  : np.ndarray   (N, seq_len, EFFECTIVE_ACT)
    proprios : np.ndarray   (N, seq_len, 6)
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    rng = np.random.default_rng(seed)
    transform = make_transform()

    lrds = LeRobotDataset(repo_id=repo_id, root=root)
    meta = lrds.meta
    fps  = meta.fps

    episodes_meta = meta.episodes
    ep_from = np.asarray(episodes_meta["dataset_from_index"], dtype=np.int64)
    ep_to   = np.asarray(episodes_meta["dataset_to_index"],   dtype=np.int64)
    ep_lens = (ep_to - ep_from).tolist()
    n_total = len(ep_lens)

    min_len_frames = seq_len * FRAMESKIP + FRAMESKIP
    valid_eps = [i for i, l in enumerate(ep_lens) if l >= min_len_frames]
    chosen = rng.choice(
        valid_eps, size=min(n_episodes, len(valid_eps)), replace=False
    )
    chosen = np.sort(chosen)

    hf = lrds.hf_dataset
    reader = lrds.reader

    all_pixels  = []
    all_actions = []
    all_proprios = []

    for ep_idx in chosen:
        g_start = int(ep_from[ep_idx])
        ep_len  = int(ep_lens[ep_idx])

        max_start_local = ep_len - seq_len * FRAMESKIP
        if max_start_local <= 0:
            continue
        start_local = int(rng.integers(0, max_start_local))

        # Global frame indices for each latent timestep
        frame_rows = [
            g_start + start_local + t * FRAMESKIP
            for t in range(seq_len)
        ]
        # Timestamps for the video reader (local to episode)
        local_indices = [start_local + t * FRAMESKIP for t in range(seq_len)]
        timestamps    = [idx / fps for idx in local_indices]

        # ---- pixels via video reader ----------------------------------------
        frames_dict = reader._query_videos({image_key: timestamps}, int(ep_idx))
        frames = frames_dict[image_key]   # (seq_len, C, H, W) uint8
        if frames.ndim == 3:
            frames = frames.unsqueeze(0)
        pixel_frames = []
        for t in range(seq_len):
            pixel_frames.append(transform(frames[t]))
        pixels_seq = torch.stack(pixel_frames)          # (seq_len, 3, H, W)

        # ---- actions (concatenate FRAMESKIP raw steps per latent step) -------
        act_blocks = []
        for row in frame_rows:
            raw_rows = hf.select(range(row, min(row + FRAMESKIP, int(ep_to[ep_idx]))))
            acts = [torch.as_tensor(a) for a in raw_rows[ACTION_KEY]]
            acts = torch.stack(acts)                    # (<= FRAMESKIP, 6)
            if len(acts) < FRAMESKIP:
                pad = torch.zeros(FRAMESKIP - len(acts), ACTION_DIM)
                acts = torch.cat([acts, pad], dim=0)
            act_blocks.append(acts.flatten().numpy())   # (30,)
        actions_seq = np.array(act_blocks, dtype=np.float32)   # (seq_len, 30)

        # ---- proprio --------------------------------------------------------
        proprio_rows = hf.select(frame_rows)
        proprio_seq = np.stack([np.asarray(p) for p in proprio_rows[PROPRIO_KEY]])

        all_pixels.append(pixels_seq)
        all_actions.append(actions_seq)
        all_proprios.append(proprio_seq)

    pixels   = torch.stack(all_pixels)                  # (N, T, 3, H, W)
    actions  = np.array(all_actions,  dtype=np.float32) # (N, T, 30)
    proprios = np.array(all_proprios, dtype=np.float32)  # (N, T, 6)

    # Normalize actions (use saved normalizer if provided, else z-score)
    if action_mean is not None and action_std is not None:
        act_dim = action_mean.shape[0]  # = 6
        a_reshaped = actions.reshape(-1, FRAMESKIP, act_dim)
        a_reshaped = (a_reshaped - action_mean) / (action_std + 1e-8)
        actions = a_reshaped.reshape(-1, seq_len, EFFECTIVE_ACT)
    else:
        act_flat = actions.reshape(-1, EFFECTIVE_ACT)
        act_mean = act_flat.mean(0, keepdims=True)
        act_std  = act_flat.std(0,  keepdims=True) + 1e-8
        actions  = (actions - act_mean) / act_std

    if corruption > 0.0:
        print(f"  Applying action corruption = {corruption:.2f}...")
        actions = corrupt_actions(actions, corruption)

    return pixels, actions, proprios


# ---------------------------------------------------------------------------
# Encode with model
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_trajectories(
    model: JEPA,
    pixels: torch.Tensor,
    actions: np.ndarray,
    batch_size: int = 32,
    device: str = "cuda",
):
    """Returns z (N, T, D) and z_pred (N, HISTORY_SIZE, D)."""
    model.eval()
    N = pixels.shape[0]
    all_z, all_z_pred = [], []

    for i in range(0, N, batch_size):
        pix_b = pixels[i: i + batch_size].to(device)
        act_b = torch.from_numpy(actions[i: i + batch_size]).float().to(device)

        info = model.encode({"pixels": pix_b, "action": act_b})
        emb     = info["emb"]      # (B, T, D)
        act_emb = info["act_emb"]  # (B, T, D_a)

        pred_emb = model.predict(
            emb[:, :HISTORY_SIZE], act_emb[:, :HISTORY_SIZE]
        )                          # (B, HISTORY_SIZE, D)

        all_z.append(emb.cpu())
        all_z_pred.append(pred_emb.cpu())

        if (i // batch_size) % 10 == 0:
            print(f"  Encoded {min(i + batch_size, N)}/{N}...", end="\r")

    print()
    return (
        torch.cat(all_z,      dim=0).numpy(),
        torch.cat(all_z_pred, dim=0).numpy(),
    )


# ---------------------------------------------------------------------------
# K-step autoregressive rollout
# ---------------------------------------------------------------------------

@torch.no_grad()
def kstep_rollout(
    model: JEPA,
    pixels: torch.Tensor,
    actions: np.ndarray,
    k_max: int = K_MAX,
    batch_size: int = 32,
    device: str = "cuda",
) -> dict:
    """Autoregressive k-step equivariance degradation curve."""
    model.eval()
    N = pixels.shape[0]
    all_errors = {k: [] for k in range(1, k_max + 1)}
    all_cos    = {k: [] for k in range(1, k_max + 1)}

    for i in range(0, N, batch_size):
        pix_b = pixels[i: i + batch_size].to(device)
        act_b = torch.from_numpy(actions[i: i + batch_size]).float().to(device)
        B = pix_b.shape[0]

        info = model.encode({"pixels": pix_b, "action": act_b})
        z_all       = info["emb"]
        act_emb_all = info["act_emb"]

        z_ctx   = z_all[:, :HISTORY_SIZE].clone()
        act_ctx = act_emb_all[:, :HISTORY_SIZE].clone()

        for k in range(1, k_max + 1):
            pred_out    = model.predict(z_ctx, act_ctx)
            z_next_pred = pred_out[:, -1:]                      # (B, 1, D)
            z_true_k    = z_all[:, HISTORY_SIZE + k - 1]       # (B, D)

            err = ((z_next_pred[:, 0] - z_true_k) ** 2).mean(dim=-1)
            cos = F.cosine_similarity(z_next_pred[:, 0], z_true_k, dim=-1)
            all_errors[k].append(err.cpu().numpy())
            all_cos[k].append(cos.cpu().numpy())

            if k < k_max:
                next_act = act_emb_all[:, HISTORY_SIZE + k - 1: HISTORY_SIZE + k]
                z_ctx    = torch.cat([z_ctx[:, 1:],   z_next_pred], dim=1)
                act_ctx  = torch.cat([act_ctx[:, 1:], next_act],    dim=1)

        if (i // batch_size) % 5 == 0:
            print(f"  k-step rollout: {min(i + batch_size, N)}/{N}...", end="\r")

    print()
    return {
        "k_values":      list(range(1, k_max + 1)),
        "mse_per_k":     [float(np.concatenate(all_errors[k]).mean()) for k in range(1, k_max + 1)],
        "cos_sim_per_k": [float(np.concatenate(all_cos[k]).mean())    for k in range(1, k_max + 1)],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _repo_root = Path(__file__).resolve().parent
    _ckpt_dir  = _repo_root / "checkpoints" / "so100_topcam"
    default_ckpt = str(next(iter(sorted(_ckpt_dir.glob("lewm_*_epoch_*_object.ckpt"))), _ckpt_dir / "model.ckpt"))
    default_norm = str(next(iter(sorted(_ckpt_dir.glob("*_normalizers.pt"))), _ckpt_dir / "normalizers.pt"))
    default_data = str(_repo_root / "datasets" / "stack_cubes")

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",        default=default_ckpt)
    parser.add_argument("--normalizers", default=default_norm)
    parser.add_argument("--repo-id",     default=REPO_ID)
    parser.add_argument("--image-key",   default=IMAGE_KEY)
    parser.add_argument("--data-root",   default=default_data)
    parser.add_argument("--n-episodes",  type=int,   default=400)
    parser.add_argument("--batch-size",  type=int,   default=32)
    parser.add_argument("--corruption",  type=float, default=0.0)
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output",      default="so100_identifiability_results.txt")
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("  LeWM Identifiability Analysis — SO-100 pickplace")
    print(f"{'='*60}")
    print(f"  Checkpoint:  {args.ckpt}")
    print(f"  Dataset:     {args.repo_id}  ({args.image_key})")
    print(f"  Device:      {args.device}")
    print(f"  Episodes:    {args.n_episodes}")
    print(f"  Corruption:  {args.corruption}")

    # 1. Load model
    print("\nLoading model...")
    model = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    model.eval().to(args.device)
    model.requires_grad_(False)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params / 1e6:.1f}M")

    # 2. Load action normalizer stats
    action_mean = action_std = None
    if Path(args.normalizers).exists():
        stats = torch.load(args.normalizers, map_location="cpu")
        if "action" in stats:
            action_mean = stats["action"]["mean"].numpy()  # (6,)
            action_std  = stats["action"]["std"].numpy()
            print(f"  Loaded action normalizer: mean {action_mean.shape}")

    # 3. Sample trajectories (seq_len = HISTORY_SIZE + 1 for basic metrics)
    seq_len = HISTORY_SIZE + 1
    print(f"\nSampling {args.n_episodes} trajectories (seq_len={seq_len})...")
    pixels, actions, proprios = sample_trajectories(
        args.repo_id,
        n_episodes=args.n_episodes,
        seq_len=seq_len,
        seed=args.seed,
        corruption=args.corruption,
        action_mean=action_mean,
        action_std=action_std,
        root=args.data_root,
        image_key=args.image_key,
    )
    print(f"  pixels:   {tuple(pixels.shape)}")
    print(f"  actions:  {actions.shape}")
    print(f"  proprios: {proprios.shape}")

    # 4. Encode
    print(f"\nEncoding ({args.device})...")
    z, z_pred = encode_trajectories(
        model, pixels, actions,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(f"  z: {z.shape},  z_pred: {z_pred.shape}")

    # 5. Prepare flat arrays for probe metrics
    z_flat   = z.reshape(-1, z.shape[-1])           # (N*T, D)
    p_flat   = proprios.reshape(-1, proprios.shape[-1])  # (N*T, 6)
    zp_flat  = z_pred.reshape(-1, z_pred.shape[-1])

    # Normalize proprio
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    p_flat_norm = scaler.fit_transform(p_flat)

    # 6. Run all metrics
    print("\n[1/8] Affine probe (linear regression z → proprio)...")
    aff = affine_identifiability(z_flat, p_flat_norm, train_frac=0.8, seed=args.seed)

    print("[2/8] Nonlinear probe (MLP z → proprio)...")
    nl = nonlinear_identifiability(z_flat, p_flat_norm, max_iter=2000, train_frac=0.8, seed=args.seed)

    print("[3/8] Probe generalization (episode split)...")
    # Use per-episode proprios normalized by the same scaler
    proprios_norm = scaler.transform(proprios.reshape(-1, 6)).reshape(proprios.shape)
    pg = probe_generalization(z, proprios_norm, train_frac=0.8)

    print("[4/8] Action diversity...")
    ad = action_diversity(actions)

    print("[5/8] Temporal contrastivity...")
    tc = temporal_contrastivity(z, actions)

    print("[6/8] Equivariance (1-step)...")
    # z: (N, H+1, D); z_pred: (N, H, D) — predictions for t=1..H given t=0..H-1
    z_t     = z[:, :HISTORY_SIZE].reshape(-1, z.shape[-1])      # (N*H, D)
    z_tp1   = z[:, 1:HISTORY_SIZE+1].reshape(-1, z.shape[-1])   # (N*H, D)
    z_tp1_p = z_pred.reshape(-1, z_pred.shape[-1])               # (N*H, D)
    eq = equivariance_error(z_t, z_tp1, z_tp1_p)

    print("[7/8] Action invertibility (decode Δz → action)...")
    ai = action_invertibility(z, actions)

    print("[8/8] DCI metrics (disentanglement / completeness / informativeness)...")
    dci = dci_metrics(z_flat, p_flat_norm)

    # 7. K-step rollout (needs longer sequences)
    print("\nSampling longer trajectories for k-step rollout...")
    kseq_len = HISTORY_SIZE + K_MAX
    pixels_long, actions_long, _ = sample_trajectories(
        args.repo_id,
        n_episodes=min(args.n_episodes, 300),
        seq_len=kseq_len,
        seed=args.seed + 1,
        action_mean=action_mean,
        action_std=action_std,
        root=args.data_root,
        image_key=args.image_key,
    )
    print("K-step autoregressive rollout...")
    kstep = kstep_rollout(
        model, pixels_long, actions_long,
        k_max=K_MAX,
        batch_size=args.batch_size,
        device=args.device,
    )

    # 8. Assemble and print
    results = {
        "affine":               aff,
        "nonlinear":            nl,
        "probe_generalization": pg,
        "action_diversity":     ad,
        "temporal_contrastivity": tc,
        "equivariance":         eq,
        "action_invertibility": ai,
        "dci":                  dci,
        "kstep_equivariance":   kstep,
        "summary": {
            "affine_r2":              aff["r2_mean"],
            "nonlinear_r2":           nl["r2_mean"],
            "identifiability_gap":    nl["r2_mean"] - aff["r2_mean"],
            "probe_test_r2":          pg["test_r2_mean"],
            "generalization_gap":     pg["generalization_gap"],
            "action_diversity_rank":  ad["effective_rank"],
            "action_entropy_norm":    ad["normalized_entropy"],
            "action_dependence":      tc["action_dependence_score"],
            "equivariance_mse":       eq["equivariance_mse"],
            "action_invertibility_r2": ai["r2_mean"],
            "dci_disentanglement":    dci["disentanglement_mean"],
            "dci_completeness":       dci["completeness_mean"],
            "dci_informativeness":    dci["informativeness_mean"],
            "dci_score":              dci["dci_score"],
        },
    }

    # Concise summary of the metrics we actually compute for SO-100.
    print(f"\n{'='*60}")
    print("  SUMMARY — SO-100 LeWM identifiability")
    print(f"{'='*60}")
    print(f"  Affine probe R²      : {aff['r2_mean']:.4f}")
    print(f"  Nonlinear probe R²   : {nl['r2_mean']:.4f}")
    print(f"  Identifiability gap  : {nl['r2_mean'] - aff['r2_mean']:.4f}  (small → linearly structured)")
    print(f"  Probe test R²        : {pg['test_r2_mean']:.4f}  (gen. gap: {pg['generalization_gap']:.4f})")
    print(f"  Action eff. rank     : {ad['effective_rank']:.2f} / {EFFECTIVE_ACT}")
    print(f"  Action entropy (norm): {ad['normalized_entropy']:.4f}")
    print(f"  Action dependence    : {tc['action_dependence_score']:.4f}")
    print(f"  Equivariance MSE     : {eq['equivariance_mse']:.6f}")
    print(f"  Action invert. R²    : {ai['r2_mean']:.4f}")
    print(f"  DCI score (D,C H-mean): {dci['dci_score']:.4f}")
    print()
    print(f"  Per-joint affine R² | nonlinear R² | probe-gen test R²:")
    for j, name in enumerate(JOINT_NAMES):
        print(
            f"    {name:8s}: {aff['r2_per_factor'][j]:.4f} | "
            f"{nl['r2_per_factor'][j]:.4f} | "
            f"{pg['test_r2_per_factor'][j]:.4f}"
        )
    print()
    print(f"  K-step equivariance degradation:")
    for k, (mse_k, cos_k) in enumerate(
        zip(kstep["mse_per_k"], kstep["cos_sim_per_k"]), start=1
    ):
        print(f"    k={k}: MSE={mse_k:.5f}  cos_sim={cos_k:.4f}")

    # 9. Save
    out_path = Path(args.output)
    with out_path.open("w") as f:
        f.write("LeWM SO-100 Identifiability Analysis\n")
        f.write(f"checkpoint: {args.ckpt}\n")
        f.write(f"n_episodes: {args.n_episodes}\n")
        f.write(f"action_corruption: {args.corruption}\n\n")

        f.write("=== SUMMARY ===\n")
        for k, v in results["summary"].items():
            f.write(f"  {k}: {v:.6f}\n")

        f.write("\n=== PER-JOINT AFFINE PROBE ===\n")
        for j, name in enumerate(JOINT_NAMES):
            f.write(f"  {name}: R2={aff['r2_per_factor'][j]:.6f}  r={aff['pearson_r_per_factor'][j]:.6f}  MSE={aff['mse_per_factor'][j]:.6f}\n")

        f.write("\n=== PER-JOINT NONLINEAR PROBE ===\n")
        for j, name in enumerate(JOINT_NAMES):
            f.write(f"  {name}: R2={nl['r2_per_factor'][j]:.6f}  r={nl['pearson_r_per_factor'][j]:.6f}  MSE={nl['mse_per_factor'][j]:.6f}\n")

        f.write("\n=== PROBE GENERALIZATION (per-joint test R²) ===\n")
        for j, name in enumerate(JOINT_NAMES):
            f.write(f"  {name}: train={pg['train_r2_per_factor'][j]:.6f}  test={pg['test_r2_per_factor'][j]:.6f}\n")

        f.write("\n=== ACTION DIVERSITY ===\n")
        for k, v in ad.items():
            f.write(f"  {k}: {v}\n")

        f.write("\n=== EQUIVARIANCE (1-step) ===\n")
        for k, v in eq.items():
            f.write(f"  {k}: {v:.8f}\n")

        f.write("\n=== ACTION INVERTIBILITY ===\n")
        f.write(f"  r2_mean: {ai['r2_mean']:.6f}\n")
        for idx, r2 in enumerate(ai.get("r2_mean_raw_dims", [])):
            f.write(f"  step_{idx}_r2: {r2:.6f}\n")

        f.write("\n=== DCI METRICS ===\n")
        f.write(f"  disentanglement: {dci['disentanglement_mean']:.6f}\n")
        f.write(f"  completeness:    {dci['completeness_mean']:.6f}\n")
        f.write(f"  informativeness: {dci['informativeness_mean']:.6f}\n")
        f.write(f"  dci_score:       {dci['dci_score']:.6f}\n")

        f.write("\n=== K-STEP EQUIVARIANCE ===\n")
        for k, (mse_k, cos_k) in enumerate(
            zip(kstep["mse_per_k"], kstep["cos_sim_per_k"]), start=1
        ):
            f.write(f"  k={k}: mse={mse_k:.8f}  cos_sim={cos_k:.6f}\n")

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
