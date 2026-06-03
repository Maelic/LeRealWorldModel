"""Stage 2 training: GC-IDM amortized planner on frozen JEPA encoder embeddings.

Usage:
    python train_gc_idm.py world_model_path=<path/to/lewm_epoch_N_object.ckpt>

The script:
  1. Loads the frozen JEPA world model from Stage 1.
  2. Pre-computes encoder embeddings for every frame in the dataset (cached in
     RAM — typically <4 GB for 20 SO-100 episodes at ViT-Tiny resolution).
  3. Trains GC-IDM via supervised MSE regression on (z_t, z_{t+h}, h) → a_t
     triples sampled uniformly from [1, max_horizon].
  4. Saves gc_idm.pt alongside the world-model checkpoint.

~20 min on a single GPU for the stack_cubes dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# ── Make repo root importable ──────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import stable_pretraining as spt  # noqa: E402

from jepa import JEPA  # noqa: E402
from lewm.data.lerobot_adapter import LeRobotWMDataset  # noqa: E402
from lewm_robot.policies.jepa.modeling_gc_idm import GCIDM  # noqa: E402
from lewm_robot.policies.jepa.modeling_jepa import build_world_model  # noqa: E402
from lewm_robot.policies.jepa.configuration_jepa import JEPAConfig  # noqa: E402
from utils import get_img_preprocessor  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# Embedding pre-computation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def precompute_embeddings(
    world_model: JEPA,
    dataset: LeRobotWMDataset,
    device: torch.device,
    image_keys: list[str],
    img_preprocessors: dict,
    batch_size: int = 64,
    num_workers: int = 4,
) -> list[torch.Tensor]:
    """Encode every frame in every episode with the frozen encoder.

    Returns a list of (N_frames_ep, D) tensors, one per episode.
    Uses a DataLoader over the underlying LeRobotDataset for efficient
    sequential video decoding (avoids per-frame seek overhead).
    """
    from torch.utils.data import DataLoader, Subset

    world_model.eval()
    lerobot_ds = dataset._lerobot
    n_episodes = len(dataset.lengths)
    emb_cache: list[torch.Tensor] = []

    print(f"Pre-computing embeddings for {n_episodes} episodes...")
    for ep_idx in tqdm(range(n_episodes)):
        ep_len = int(dataset.lengths[ep_idx])
        ep_offset = int(dataset.offsets[ep_idx])

        # One DataLoader per episode → sequential decode, no random seeks
        subset = Subset(lerobot_ds, range(ep_offset, ep_offset + ep_len))
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
        )

        ep_embs: list[torch.Tensor] = []
        for batch in loader:
            info: dict[str, torch.Tensor] = {}
            for slot_idx, key in enumerate(image_keys):
                frames = batch[key]             # (B, C, H, W), float [0,1]
                if frames.is_floating_point():
                    frames = (frames * 255.0).clamp(0, 255).to(torch.uint8)
                pre = img_preprocessors[key]({"pixels": frames})["pixels"].to(device)
                slot = "pixels" if slot_idx == 0 else "pixels2"
                info[slot] = pre.unsqueeze(1)   # (B, 1, C, H, W)

            encoded = world_model.encode(info)
            z = encoded["emb"][:, 0].cpu()      # (B, D)
            ep_embs.append(z)

        emb_cache.append(torch.cat(ep_embs, dim=0))    # (ep_len, D)

    print("Done. Total frames encoded:", sum(e.shape[0] for e in emb_cache))
    return emb_cache


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="./config/train", config_name="gc_idm")
def run(cfg) -> None:
    print(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Build and load world model ───────────────────────────────────────────
    jepa_cfg = JEPAConfig(
        encoder_scale=cfg.wm.encoder_scale,
        patch_size=cfg.wm.patch_size,
        img_size=cfg.wm.img_size,
        embed_dim=cfg.wm.embed_dim,
        history_size=cfg.wm.history_size,
        action_dim=cfg.wm.action_dim,
        frameskip=cfg.wm.frameskip,
        image_keys=list(cfg.data.dataset.image_keys),
        predictor_depth=cfg.wm.predictor.depth,
        predictor_heads=cfg.wm.predictor.heads,
        predictor_mlp_dim=cfg.wm.predictor.mlp_dim,
        predictor_dim_head=cfg.wm.predictor.dim_head,
        predictor_dropout=cfg.wm.predictor.dropout,
        predictor_emb_dropout=cfg.wm.predictor.emb_dropout,
    )

    world_model = build_world_model(jepa_cfg, device)
    wm_path = Path(cfg.world_model_path).expanduser()
    print(f"Loading world model from {wm_path}")
    loaded = torch.load(wm_path, map_location=device, weights_only=False)
    if isinstance(loaded, JEPA):
        world_model.load_state_dict(loaded.state_dict())
    elif isinstance(loaded, dict):
        world_model.load_state_dict(loaded)
    else:
        raise ValueError(f"Unexpected world model type: {type(loaded)}")
    world_model.eval()
    for p in world_model.parameters():
        p.requires_grad_(False)

    # ── Dataset ──────────────────────────────────────────────────────────────
    image_keys: list[str] = list(cfg.data.dataset.image_keys)
    img_preprocessors = {
        key: get_img_preprocessor(source="pixels", target="pixels", img_size=jepa_cfg.img_size)
        for key in image_keys
    }

    has_second_cam = len(image_keys) > 1
    keys_to_load = ["pixels", "action"] + (["pixels2"] if has_second_cam else [])

    dataset = LeRobotWMDataset(
        repo_id=cfg.data.dataset.repo_id,
        root=cfg.data.dataset.root,
        image_key=cfg.data.dataset.image_key,
        image_key2=cfg.data.dataset.get("image_key2") if has_second_cam else None,
        frameskip=cfg.data.dataset.frameskip,
        # Enough steps to sample any horizon up to max_horizon
        num_steps=cfg.max_horizon + 1,
        keys_to_load=keys_to_load,
        return_uint8=True,
    )

    # ── Pre-compute embeddings ────────────────────────────────────────────────
    emb_cache_cpu = precompute_embeddings(
        world_model, dataset, device, image_keys, img_preprocessors
    )

    # Pre-load all actions too (already in RAM via parquet)
    n_episodes = len(dataset.lengths)
    action_cache_cpu: list[torch.Tensor] = []
    print("Loading action cache...")
    for ep_idx in range(n_episodes):
        ep_len = int(dataset.lengths[ep_idx])
        ep_offset = int(dataset.offsets[ep_idx])
        rows = dataset._lerobot.hf_dataset.select(
            range(ep_offset, ep_offset + ep_len)
        )
        acts = torch.stack([torch.as_tensor(a) for a in rows["action"]])  # (ep_len, A)
        action_cache_cpu.append(acts)

    # Load normalizer stats to train in normalised action space
    action_mean = torch.zeros(jepa_cfg.action_dim)
    action_std = torch.ones(jepa_cfg.action_dim)
    norm_path = wm_path.parent / f"{wm_path.stem.split('_epoch')[0]}_normalizers.pt"
    if norm_path.exists():
        stats = torch.load(norm_path, map_location="cpu")
        if "action" in stats:
            action_mean = stats["action"]["mean"]
            action_std = stats["action"]["std"]
            print(f"Loaded action normalizers from {norm_path}")

    action_mean = action_mean.to(device)
    action_std = action_std.to(device)

    # Move caches to GPU once — avoids 256 individual .to(device) calls per step
    print("Moving caches to GPU...")
    emb_cache = [e.to(device) for e in emb_cache_cpu]
    action_cache = [
        ((a.to(device) - action_mean) / action_std) for a in action_cache_cpu
    ]
    del emb_cache_cpu, action_cache_cpu
    print(f"Caches on GPU. Starting {cfg.steps} training steps...", flush=True)

    # ── GC-IDM ───────────────────────────────────────────────────────────────
    gc_idm = GCIDM(
        latent_dim=jepa_cfg.embed_dim,
        action_dim=jepa_cfg.action_dim,    # predict one native-frame action
        hidden_dim=cfg.gc_idm.hidden_dim,
        horizon_dim=cfg.gc_idm.horizon_dim,
        max_horizon=cfg.max_horizon,
    ).to(device)

    optimizer = AdamW(gc_idm.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.steps)

    # ── Training loop ────────────────────────────────────────────────────────
    gc_idm.train()
    ep_lengths = [int(l) for l in dataset.lengths]
    rng = torch.Generator().manual_seed(cfg.seed)

    log_interval = max(1, cfg.steps // 200)
    running_loss = 0.0

    for step in range(cfg.steps):
        # Sample a random episode proportional to length
        ep_idx = torch.multinomial(
            torch.tensor(ep_lengths, dtype=torch.float32),
            num_samples=cfg.batch_size,
            replacement=True,
            generator=rng,
        ).tolist()

        z_ts, z_goals, horizons, a_targets = [], [], [], []
        for ep in ep_idx:
            ep_len = ep_lengths[ep]
            # Sample start t and horizon h such that t+h < ep_len
            max_h = min(cfg.max_horizon, ep_len - 1)
            t = torch.randint(0, ep_len - 1, (1,), generator=rng).item()
            h = torch.randint(1, min(max_h, ep_len - t), (1,), generator=rng).item()

            z_ts.append(emb_cache[ep][t])       # already on GPU
            z_goals.append(emb_cache[ep][t + h])
            horizons.append(h)
            a_targets.append(action_cache[ep][t])  # already normalised + on GPU

        z_t = torch.stack(z_ts)                                 # (B, D) — on GPU
        z_goal = torch.stack(z_goals)                           # (B, D) — on GPU
        h_tensor = torch.tensor(horizons, device=device, dtype=torch.long)  # (B,)
        a_target = torch.stack(a_targets)                       # (B, A) — on GPU

        a_pred = gc_idm(z_t, z_goal, h_tensor)
        loss = F.mse_loss(a_pred, a_target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gc_idm.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        if (step + 1) % log_interval == 0:
            avg = running_loss / log_interval
            print(f"[{step+1:>6}/{cfg.steps}] loss={avg:.5f}  lr={scheduler.get_last_lr()[0]:.2e}", flush=True)
            running_loss = 0.0

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = wm_path.parent / "gc_idm.pt"
    torch.save(gc_idm.state_dict(), out_path)
    print(f"Saved GC-IDM → {out_path}", flush=True)


if __name__ == "__main__":
    run()
