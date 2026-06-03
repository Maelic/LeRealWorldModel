"""Train a lightweight image decoder on top of a frozen JEPA encoder.

The decoder maps 192-dim CLS embeddings back to images so we can visually
verify that the JEPA representations are meaningful, and also inspect what
the predictor "imagines" for future frames.

Usage:
    python train_jepa_decoder.py \\
        --world-model-path checkpoints/so100_topcam/lewm_so100_topcam_epoch_50_object.ckpt \\
        --run-dir checkpoints/so100_topcam

Outputs (all written to <run_dir>/):
    decoder.pt             — decoder weights
    decoder_recon.png      — reconstruction grid: GT vs decoded(z_enc)
    decoder_rollout.png    — rollout grid: GT | decoded(z_enc) | decoded(z_pred)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import stable_pretraining as spt  # noqa: F401

from jepa import JEPA  # noqa: E402
from lewm.data.lerobot_adapter import LeRobotWMDataset  # noqa: E402
from lewm_robot.decoder import JEPADecoder  # noqa: E402
from lewm_robot.policies.jepa.configuration_jepa import JEPAConfig  # noqa: E402
from lewm_robot.policies.jepa.modeling_gc_idm import GCIDM  # noqa: E402
from lewm_robot.policies.jepa.modeling_jepa import build_world_model  # noqa: E402
from utils import get_img_preprocessor  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# Paired dataset: (z, image) for supervised decoder training
# ──────────────────────────────────────────────────────────────────────────────

class ZImageDataset(Dataset):
    """Pairs pre-computed CLS embeddings with their source images.

    Images are loaded on-the-fly from ``lerobot_ds`` and resized to
    ``img_size × img_size`` (same view that the ViT encoder saw).
    """

    def __init__(
        self,
        emb_flat: torch.Tensor,    # (N, D) on CPU
        lerobot_ds,
        image_key: str,
        img_size: int = 224,
    ) -> None:
        assert len(emb_flat) == len(lerobot_ds)
        self.emb_flat = emb_flat
        self.lerobot_ds = lerobot_ds
        self.image_key = image_key
        self.resize = transforms.Compose([
            transforms.Resize((img_size, img_size), antialias=True),
        ])

    def __len__(self) -> int:
        return len(self.emb_flat)

    def __getitem__(self, idx: int):
        z = self.emb_flat[idx]                      # (D,)
        img = self.lerobot_ds[idx][self.image_key]  # (C, H, W) float [0,1]
        img = self.resize(img)                       # (C, 224, 224)
        return z, img


# ──────────────────────────────────────────────────────────────────────────────
# Embedding pre-computation (reused from train_gc_idm pattern)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def precompute_embeddings_flat(
    world_model: JEPA,
    lerobot_ds,
    device: torch.device,
    image_key: str,
    img_preprocessor,
    batch_size: int = 128,
    num_workers: int = 4,
) -> torch.Tensor:
    """Encode every frame → flat (N_total, D) tensor on CPU."""
    world_model.eval()
    N = len(lerobot_ds)
    loader = DataLoader(
        lerobot_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )
    embs: list[torch.Tensor] = []
    print(f"Pre-computing embeddings for {N} frames…")
    for batch in tqdm(loader):
        frames = batch[image_key]                              # (B, C, H, W) float
        if frames.is_floating_point():
            frames = (frames * 255).clamp(0, 255).to(torch.uint8)
        pre = img_preprocessor({"pixels": frames})["pixels"]  # (B, C, 224, 224)
        pre = pre.unsqueeze(1).to(device)                     # (B, 1, C, H, W)
        z = world_model.encode({"pixels": pre})["emb"][:, 0] # (B, D)
        embs.append(z.cpu())
    return torch.cat(embs, dim=0)  # (N, D)


# ──────────────────────────────────────────────────────────────────────────────
# Rollout visualization
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def rollout_episode(
    world_model: JEPA,
    decoder: JEPADecoder,
    lerobot_ds,
    image_key: str,
    img_preprocessor,
    device: torch.device,
    ep_start: int,
    ep_len: int,
    history_size: int = 3,
    n_rollout: int = 8,
) -> dict[str, torch.Tensor]:
    """Encode history, roll out predictor, decode all embeddings.

    Returns dict with keys:
        gt_imgs       (T, 3, 224, 224)  — ground truth frames
        enc_imgs      (T, 3, 224, 224)  — decoded from encoder z
        pred_imgs     (n_rollout, 3, 224, 224)  — decoded from predictor z
    where T = history_size + n_rollout.
    """
    world_model.eval()
    decoder.eval()
    T = history_size + n_rollout

    # Load T frames + actions starting from ep_start
    resize = transforms.Resize((224, 224), antialias=True)
    raw_frames, gt_imgs, actions = [], [], []
    for i in range(T):
        idx = ep_start + i
        sample = lerobot_ds[idx]
        img = sample[image_key]           # (C, H, W) float
        gt_imgs.append(resize(img))
        frames_u8 = (img * 255).clamp(0, 255).to(torch.uint8)
        pre = img_preprocessor({"pixels": frames_u8.unsqueeze(0)})["pixels"]  # (1,C,H,W)
        raw_frames.append(pre)
        actions.append(torch.as_tensor(sample["action"], dtype=torch.float32))

    gt_imgs = torch.stack(gt_imgs)        # (T, 3, 224, 224)
    frames_pre = torch.stack(raw_frames)  # (T, 1, C, 224, 224) → need (1, T, C, H, W)
    frames_pre = frames_pre[:, 0].unsqueeze(0).to(device)  # (1, T, C, H, W)
    actions_t = torch.stack(actions).to(device)            # (T, A)

    # Encode all T frames individually for ground truth enc_imgs
    enc_embs: list[torch.Tensor] = []
    for t in range(T):
        z = world_model.encode({"pixels": frames_pre[:, t:t+1]})["emb"][:, 0]  # (1, D)
        enc_embs.append(z)
    enc_embs_t = torch.cat(enc_embs, dim=0)  # (T, D)

    # Encode history context (with delta actions) for rollout
    hist_frames = frames_pre[:, :history_size]                        # (1, H, C, H, W)
    hist_acts = actions_t[:history_size].unsqueeze(0)                  # (1, H, A)
    # delta actions: act[t] - act[t-1], with act[-1]=0 at episode start
    prev = torch.cat([torch.zeros(1, 1, hist_acts.shape[-1], device=device),
                      hist_acts[:, :-1]], dim=1)
    delta_hist = hist_acts - prev                                       # (1, H, A)
    info = world_model.encode({"pixels": hist_frames, "action": delta_hist})
    emb_win = info["emb"].clone()                                      # (1, H, D)
    act_emb_win = world_model.action_encoder(delta_hist)               # (1, H, A_emb)
    prev_act = actions_t[history_size - 1]                             # (A,)

    # Autoregressive rollout
    pred_embs: list[torch.Tensor] = []
    for t in range(n_rollout):
        emb_trunc = emb_win[:, -history_size:]
        act_trunc = act_emb_win[:, -history_size:]
        z_pred = world_model.predict(emb_trunc, act_trunc)[:, -1:]     # (1, 1, D)
        pred_embs.append(z_pred[:, 0])                                  # (1, D)
        emb_win = torch.cat([emb_win, z_pred], dim=1)

        next_act = actions_t[history_size + t]
        delta_next = (next_act - prev_act).unsqueeze(0).unsqueeze(0)   # (1, 1, A)
        act_emb_next = world_model.action_encoder(delta_next)
        act_emb_win = torch.cat([act_emb_win, act_emb_next], dim=1)
        prev_act = next_act

    pred_embs_t = torch.cat(pred_embs, dim=0)                          # (n_rollout, D)

    # Decode
    enc_imgs = decoder(enc_embs_t.to(device)).cpu()                    # (T, 3, 224, 224)
    pred_imgs = decoder(pred_embs_t.to(device)).cpu()                  # (n_rollout, 3, 224, 224)

    return {"gt_imgs": gt_imgs, "enc_imgs": enc_imgs, "pred_imgs": pred_imgs}


@torch.no_grad()
def rollout_episode_gc_idm(
    world_model: JEPA,
    gc_idm: GCIDM,
    decoder: JEPADecoder,
    lerobot_ds,
    image_key: str,
    img_preprocessor,
    device: torch.device,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    ep_start: int,
    ep_len: int,
    history_size: int = 3,
    n_rollout: int = 8,
    max_horizon: int = 50,
) -> dict[str, torch.Tensor]:
    """Roll out JEPA + GC-IDM planner using predicted actions instead of GT.

    Goal = last frame of the episode.  GC-IDM predicts the action at each step,
    which is encoded and fed to the world-model predictor for forward simulation.

    Returns dict with keys:
        gt_imgs   (T, 3, 224, 224)          — ground truth frames
        pred_imgs (n_rollout, 3, 224, 224)  — decoded from GC-IDM rollout
        goal_img  (1, 3, 224, 224)          — decoded goal embedding
    """
    world_model.eval()
    gc_idm.eval()
    decoder.eval()
    T = history_size + n_rollout

    resize = transforms.Resize((224, 224), antialias=True)
    raw_frames, gt_imgs, gt_actions = [], [], []
    for i in range(T):
        idx = ep_start + i
        sample = lerobot_ds[idx]
        img = sample[image_key]
        gt_imgs.append(resize(img))
        frames_u8 = (img * 255).clamp(0, 255).to(torch.uint8)
        pre = img_preprocessor({"pixels": frames_u8.unsqueeze(0)})["pixels"]
        raw_frames.append(pre)
        gt_actions.append(torch.as_tensor(sample["action"], dtype=torch.float32))

    gt_imgs = torch.stack(gt_imgs)                                       # (T, 3, H, W)
    frames_pre = torch.stack(raw_frames)[:, 0].unsqueeze(0).to(device)  # (1, T, C, H, W)
    actions_t = torch.stack(gt_actions).to(device)                       # (T, A)

    # Encode goal = last frame of episode
    goal_sample = lerobot_ds[ep_start + ep_len - 1]
    goal_u8 = (goal_sample[image_key] * 255).clamp(0, 255).to(torch.uint8)
    goal_pre = (
        img_preprocessor({"pixels": goal_u8.unsqueeze(0)})["pixels"]
        .unsqueeze(0).to(device)
    )                                                                    # (1, 1, C, H, W)
    z_goal = world_model.encode({"pixels": goal_pre})["emb"][:, 0]      # (1, D)
    goal_img = decoder(z_goal).cpu()                                     # (1, 3, H, W)

    # Encode history context with GT delta actions (same as GT rollout)
    hist_frames = frames_pre[:, :history_size]
    hist_acts = actions_t[:history_size].unsqueeze(0)                    # (1, H, A)
    prev = torch.cat(
        [torch.zeros(1, 1, hist_acts.shape[-1], device=device), hist_acts[:, :-1]],
        dim=1,
    )
    delta_hist = hist_acts - prev
    info = world_model.encode({"pixels": hist_frames, "action": delta_hist})
    emb_win = info["emb"].clone()                                        # (1, H, D)
    act_emb_win = world_model.action_encoder(delta_hist)                 # (1, H, A_emb)

    action_mean = action_mean.to(device)
    action_std = action_std.to(device)

    # Autoregressive GC-IDM rollout
    pred_embs: list[torch.Tensor] = []
    for t in range(n_rollout):
        # Predict next embedding from current context window
        emb_trunc = emb_win[:, -history_size:]
        act_trunc = act_emb_win[:, -history_size:]
        z_next = world_model.predict(emb_trunc, act_trunc)[:, -1:]      # (1, 1, D)
        pred_embs.append(z_next[:, 0])                                   # (1, D)
        emb_win = torch.cat([emb_win, z_next], dim=1)

        # GC-IDM: predict what action to take FROM z_next toward z_goal
        h = torch.tensor([max(1, max_horizon - t - 1)], device=device)
        a_norm = gc_idm(z_next[:, 0], z_goal, h)                        # (1, A) normalised
        a_delta = a_norm * action_std + action_mean                      # de-normalise
        act_emb_next = world_model.action_encoder(a_delta.unsqueeze(1)) # (1, 1, A_emb)
        act_emb_win = torch.cat([act_emb_win, act_emb_next], dim=1)

    pred_embs_t = torch.cat(pred_embs, dim=0)                           # (n_rollout, D)
    pred_imgs = decoder(pred_embs_t).cpu()                               # (n_rollout, 3, H, W)

    return {"gt_imgs": gt_imgs, "pred_imgs": pred_imgs, "goal_img": goal_img}


def save_image_grid(
    rows: dict[str, torch.Tensor],
    out_path: Path,
    n_cols: int | None = None,
) -> None:
    """Save a labelled image grid: one row per key in `rows`."""
    try:
        from torchvision.utils import make_grid
        import PIL.Image as PIL_Image
        import PIL.ImageDraw as PIL_Draw
        import numpy as np

        tensors = list(rows.values())
        labels = list(rows.keys())
        n = min(t.shape[0] for t in tensors)
        if n_cols is None:
            n_cols = n

        row_imgs = []
        for label, t in zip(labels, tensors):
            grid = make_grid(t[:n_cols].clamp(0, 1), nrow=n_cols, padding=2)
            # grid: (3, H, W+pad)
            row_imgs.append((label, grid))

        # Stitch rows vertically with label bar
        C, H, W = row_imgs[0][1].shape
        label_h = 20
        full_h = (H + label_h) * len(row_imgs)
        canvas = PIL_Image.new("RGB", (W, full_h), (30, 30, 30))
        draw = PIL_Draw.Draw(canvas)

        y = 0
        for label, grid in row_imgs:
            draw.text((4, y + 2), label, fill=(220, 220, 220))
            img_pil = PIL_Image.fromarray(
                (grid.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            )
            canvas.paste(img_pil, (0, y + label_h))
            y += H + label_h

        canvas.save(out_path)
        print(f"Saved grid → {out_path}")
    except ImportError as e:
        print(f"Could not save grid (missing dep: {e})")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--world-model-path", required=True)
    p.add_argument("--run-dir", required=True,
                   help="Directory to save decoder.pt and visualizations")
    p.add_argument("--repo-id", default="maelicneau/stack_cubes")
    p.add_argument("--data-root",
                   default="/home/maelicneau/Documents/tmp/leWorldRobot/datasets/stack_cubes")
    p.add_argument("--image-key", default="observation.images.up")
    p.add_argument("--encoder-scale", default="tiny")
    p.add_argument("--embed-dim", type=int, default=192)
    p.add_argument("--history-size", type=int, default=3)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--patch-size", type=int, default=14)
    p.add_argument("--decoder-dim", type=int, default=256)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n-rollout", type=int, default=8,
                   help="Number of autoregressive steps in rollout visualization")
    p.add_argument("--gc-idm-path", default=None,
                   help="Path to gc_idm.pt — if given, also generate GC-IDM rollout viz")
    p.add_argument("--max-horizon", type=int, default=50,
                   help="GC-IDM planning horizon (steps to goal)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── Load frozen world model ──────────────────────────────────────────────
    jepa_cfg = JEPAConfig(
        encoder_scale=args.encoder_scale,
        embed_dim=args.embed_dim,
        history_size=args.history_size,
        img_size=args.img_size,
        patch_size=args.patch_size,
        image_keys=[args.image_key],
    )
    world_model = build_world_model(jepa_cfg, device)
    wm_path = Path(args.world_model_path).expanduser()
    loaded = torch.load(wm_path, map_location=device, weights_only=False)
    if isinstance(loaded, JEPA):
        world_model.load_state_dict(loaded.state_dict())
    elif isinstance(loaded, dict):
        world_model.load_state_dict(loaded)
    world_model.eval()
    for p in world_model.parameters():
        p.requires_grad_(False)
    print(f"Loaded world model from {wm_path}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    lerobot_ds = LeRobotDataset(
        repo_id=args.repo_id,
        root=Path(args.data_root).expanduser(),
        tolerance_s=0.04,
    )
    img_preprocessor = get_img_preprocessor(
        source="pixels", target="pixels", img_size=args.img_size
    )

    # ── Pre-compute embeddings ────────────────────────────────────────────────
    emb_flat = precompute_embeddings_flat(
        world_model, lerobot_ds, device, args.image_key, img_preprocessor
    )
    print(f"Embeddings shape: {emb_flat.shape}")  # (N, 192)

    # ── Decoder ──────────────────────────────────────────────────────────────
    decoder = JEPADecoder(
        embed_dim=args.embed_dim,
        img_size=args.img_size,
        patch_size=args.patch_size,
        decoder_dim=args.decoder_dim,
    ).to(device)
    n_params = sum(p.numel() for p in decoder.parameters())
    print(f"Decoder: {n_params/1e6:.1f}M params")

    # ── Training loop ─────────────────────────────────────────────────────────
    paired_ds = ZImageDataset(emb_flat, lerobot_ds, args.image_key, args.img_size)
    loader = DataLoader(
        paired_ds, batch_size=args.batch_size, shuffle=True, num_workers=4,
        pin_memory=True, drop_last=True,
    )

    optimizer = AdamW(decoder.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.steps)

    decoder.train()
    log_interval = max(1, args.steps // 100)
    running_loss = 0.0
    step = 0
    loader_iter = iter(loader)

    print(f"Training decoder for {args.steps} steps…", flush=True)
    while step < args.steps:
        try:
            z_batch, img_batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            z_batch, img_batch = next(loader_iter)

        z_batch = z_batch.to(device)
        img_batch = img_batch.to(device)

        recon = decoder(z_batch)
        loss = F.mse_loss(recon, img_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        step += 1
        if step % log_interval == 0:
            avg = running_loss / log_interval
            print(f"[{step:>6}/{args.steps}] mse={avg:.5f}  lr={scheduler.get_last_lr()[0]:.2e}",
                  flush=True)
            running_loss = 0.0

    # ── Save decoder ──────────────────────────────────────────────────────────
    out_path = run_dir / "decoder.pt"
    torch.save(decoder.state_dict(), out_path)
    print(f"Saved decoder → {out_path}", flush=True)

    # ── Reconstruction visualization ─────────────────────────────────────────
    decoder.eval()
    print("Generating reconstruction visualization…")

    # Pick 8 random frames spread across the dataset
    n_vis = 8
    indices = torch.linspace(0, len(lerobot_ds) - 1, n_vis).long().tolist()
    resize = transforms.Resize((args.img_size, args.img_size), antialias=True)

    gt_vis, recon_vis = [], []
    with torch.no_grad():
        for idx in indices:
            z = emb_flat[idx].unsqueeze(0).to(device)
            recon_vis.append(decoder(z)[0].cpu())
            img = resize(lerobot_ds[idx][args.image_key])
            gt_vis.append(img)

    save_image_grid(
        {
            "ground truth": torch.stack(gt_vis),
            "decoded (z_enc)": torch.stack(recon_vis),
        },
        run_dir / "decoder_recon.png",
        n_cols=n_vis,
    )

    # ── Rollout visualization ─────────────────────────────────────────────────
    print("Generating rollout visualization…")

    # Find episode 0 boundaries
    hf = lerobot_ds.hf_dataset
    ep0_frames = [i for i, e in enumerate(hf["episode_index"]) if e == 0]
    ep_start = ep0_frames[0]
    ep_len = len(ep0_frames)

    n_rollout = min(args.n_rollout, ep_len - args.history_size - 1)

    with torch.no_grad():
        rollout = rollout_episode(
            world_model, decoder, lerobot_ds, args.image_key,
            img_preprocessor, device,
            ep_start=ep_start,
            ep_len=ep_len,
            history_size=args.history_size,
            n_rollout=n_rollout,
        )

    T = args.history_size + n_rollout

    # Align rows: pad enc/pred rows to T columns
    # Context frames: z_enc decoded; future: z_pred decoded
    pred_row = torch.cat([
        rollout["enc_imgs"][:args.history_size],   # context decoded from encoder
        rollout["pred_imgs"],                       # future decoded from predictor
    ], dim=0)

    save_image_grid(
        {
            "ground truth": rollout["gt_imgs"],
            "decoded (z_enc)": rollout["enc_imgs"],
            f"decoded (z_pred, {n_rollout} steps)": pred_row,
        },
        run_dir / "decoder_rollout.png",
        n_cols=T,
    )
    # ── GC-IDM rollout visualization (optional) ──────────────────────────────
    if args.gc_idm_path:
        print("Generating GC-IDM rollout visualization…", flush=True)
        gc_idm_path = Path(args.gc_idm_path).expanduser()

        # Infer architecture dims from saved weights
        state = torch.load(gc_idm_path, map_location="cpu", weights_only=True)
        action_dim = state["fc3.weight"].shape[0]
        hidden_dim = state["fc1.weight"].shape[0]
        latent_dim = state["fc1.weight"].shape[1] // 2
        horizon_dim = state["horizon_embed.enc"].shape[1]
        max_horizon_saved = state["horizon_embed.enc"].shape[0] - 1

        gc_idm = GCIDM(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            horizon_dim=horizon_dim,
            max_horizon=max_horizon_saved,
        ).to(device)
        gc_idm.load_state_dict(state)
        gc_idm.eval()
        for p in gc_idm.parameters():
            p.requires_grad_(False)
        print(f"Loaded GC-IDM from {gc_idm_path}  "
              f"(action_dim={action_dim}, hidden={hidden_dim}, horizon={max_horizon_saved})",
              flush=True)

        # Load action normalizers (same auto-detect logic as train_gc_idm.py)
        wm_path = Path(args.world_model_path).expanduser()
        norm_path = wm_path.parent / f"{wm_path.stem.split('_epoch')[0]}_normalizers.pt"
        action_mean = torch.zeros(action_dim)
        action_std = torch.ones(action_dim)
        if norm_path.exists():
            stats = torch.load(norm_path, map_location="cpu", weights_only=False)
            action_mean = stats["action"]["mean"]
            action_std = stats["action"]["std"]
            print(f"Loaded action normalizers from {norm_path}", flush=True)
        else:
            print(f"No normalizers found at {norm_path} — using identity normalization",
                  flush=True)

        max_horizon_vis = min(args.max_horizon, max_horizon_saved)

        with torch.no_grad():
            gc_idm_rollout = rollout_episode_gc_idm(
                world_model, gc_idm, decoder, lerobot_ds, args.image_key,
                img_preprocessor, device,
                action_mean=action_mean,
                action_std=action_std,
                ep_start=ep_start,
                ep_len=ep_len,
                history_size=args.history_size,
                n_rollout=n_rollout,
                max_horizon=max_horizon_vis,
            )

        gc_idm_pred_row = torch.cat([
            gc_idm_rollout["gt_imgs"][:args.history_size],   # context from encoder
            gc_idm_rollout["pred_imgs"],                       # GC-IDM predicted frames
        ], dim=0)

        # Repeat goal frame to fill the T-column grid
        goal_row = gc_idm_rollout["goal_img"].expand(T, -1, -1, -1)

        save_image_grid(
            {
                "ground truth": gc_idm_rollout["gt_imgs"],
                f"decoded (z_gc_idm, {n_rollout} steps)": gc_idm_pred_row,
                "decoded (z_goal)": goal_row,
            },
            run_dir / "decoder_gcidm_rollout.png",
            n_cols=T,
        )
        print(f"Saved GC-IDM rollout → {run_dir}/decoder_gcidm_rollout.png", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
