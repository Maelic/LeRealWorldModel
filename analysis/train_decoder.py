"""
Train a lightweight transformer decoder to reconstruct pixel observations
from the frozen CLS-token latent embedding (192-dim) of a LeWM encoder.

Inspired by Figure 8 of the LeWM paper (arXiv:2603.19312):
  "As training progresses, the latent representation increasingly captures the
   information required to reconstruct the visual scene, even though no
   reconstruction loss is used during training."

Architecture (from Appendix D):
  - The [CLS] embedding (192-dim) is projected and used as key/value in
    cross-attention.
  - 196 learnable query tokens (one per 16x16 patch of the 224x224 image)
    interact with the CLS representation through cross-attention layers
    with residual MLP blocks.
  - Patch embeddings are linearly projected to 16x16x3 pixel patches and
    rearranged to produce a 224x224 RGB image.

Usage:
  # Train decoder for tworoom TCR (epoch 10)
  python train_decoder.py \
      --checkpoint /path/to/lewm_tworoom_tcr_epoch_10_object.ckpt \
      --dataset tworoom \
      --epochs 50 \
      --output-dir ./decoder_weights/tworoom_tcr

  # Train decoder for pusht baseline (epoch 10)
  python train_decoder.py \
      --checkpoint /path/to/lewm_baseline_10ep_epoch_10_object.ckpt \
      --dataset pusht \
      --epochs 50 \
      --output-dir ./decoder_weights/pusht_baseline

  # Visualize imagined rollouts
  python train_decoder.py \
      --checkpoint /path/to/lewm_tworoom_tcr_epoch_10_object.ckpt \
      --dataset tworoom \
      --decoder-weights ./decoder_weights/tworoom_tcr/decoder.pt \
      --visualize \
      --num-rollouts 5 \
      --output-dir ./decoded_rollouts/tworoom_tcr

  # Compare forward vs backward bidirectional checkpoints on the same rollout
  python train_decoder.py \
      --compare-rollouts \
      --checkpoint experiments/tworoom_bidir2/tworoom_bidir2_epoch_5_fwd.ckpt \
      --checkpoint-bwd experiments/tworoom_bidir2/tworoom_bidir2_epoch_5_bwd_object.ckpt \
      --decoder-weights ./decoder_weights/tworoom_bidir2_fwd/decoder.pt \
      --decoder-weights-bwd ./decoder_weights/tworoom_bidir2_bwd/decoder.pt \
      --dataset tworoom \
      --num-rollouts 5 \
      --output-dir ./decoded_rollouts/tworoom_bidir2_compare
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from PIL import Image
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import stable_worldmodel as swm
from stable_pretraining import data as dt


# ──────────────────────────────────────────────────────────────
#  Decoder Architecture (follows Appendix D of arXiv:2603.19312)
# ──────────────────────────────────────────────────────────────


class CrossAttention(nn.Module):
    """Multi-head cross-attention: queries attend to key/value from CLS."""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, queries, context):
        """
        queries: (B, N_q, D)
        context: (B, N_kv, D)   -- for us N_kv=1 (the CLS token)
        """
        q = self.to_q(self.norm_q(queries))
        kv = self.to_kv(self.norm_kv(context))
        k, v = kv.chunk(2, dim=-1)

        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.heads)

        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class DecoderBlock(nn.Module):
    """Cross-attention + residual MLP block."""

    def __init__(self, dim, heads=8, dim_head=64, mlp_dim=1024, dropout=0.0):
        super().__init__()
        self.cross_attn = CrossAttention(dim, heads, dim_head, dropout)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, queries, context):
        queries = queries + self.cross_attn(queries, context)
        queries = queries + self.mlp(self.norm(queries))
        return queries


class PatchDecoder(nn.Module):
    """
    Transformer decoder that reconstructs a 224x224 RGB image from a single
    CLS-token embedding.

    Architecture:
      1. Project CLS embedding to decoder hidden dim.
      2. 196 learnable query tokens cross-attend to the projected CLS.
      3. Several DecoderBlocks refine the patch representations.
      4. Each patch is linearly projected to 16x16x3 = 768 pixel values.
      5. Patches are rearranged into a 224x224x3 image.
    """

    def __init__(
        self,
        embed_dim=192,
        hidden_dim=384,
        num_patches=196,  # (224/16)^2
        patch_size=16,
        depth=4,
        heads=8,
        dim_head=48,
        mlp_dim=1024,
        dropout=0.0,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.grid_size = int(math.sqrt(num_patches))

        # Project CLS embedding into decoder space
        self.cls_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Learnable query tokens
        self.query_tokens = nn.Parameter(torch.randn(1, num_patches, hidden_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, hidden_dim) * 0.02)

        # Cross-attention decoder blocks
        self.blocks = nn.ModuleList(
            [DecoderBlock(hidden_dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)]
        )

        self.norm = nn.LayerNorm(hidden_dim)

        # Project each patch to pixel values
        self.patch_proj = nn.Linear(hidden_dim, patch_size * patch_size * 3)

    def forward(self, cls_emb):
        """
        cls_emb: (B, D) -- the 192-dim CLS embedding
        Returns: (B, 3, 224, 224) -- reconstructed image (in normalized space)
        """
        B = cls_emb.size(0)

        # Project CLS to hidden dim and add sequence dim: (B, 1, H)
        context = self.cls_proj(cls_emb).unsqueeze(1)

        # Learnable queries + positional embeddings
        queries = self.query_tokens.expand(B, -1, -1) + self.pos_embed

        # Cross-attention decoder blocks
        for block in self.blocks:
            queries = block(queries, context)

        queries = self.norm(queries)

        # Project to pixel patches: (B, P, patch_size^2 * 3)
        patches = self.patch_proj(queries)

        # Rearrange patches into image: (B, 3, H, W)
        G = self.grid_size
        P = self.patch_size
        img = rearrange(
            patches,
            "b (gh gw) (ph pw c) -> b c (gh ph) (gw pw)",
            gh=G, gw=G, ph=P, pw=P, c=3,
        )
        return img


# ──────────────────────────────────────────────────────────────
#  Data helpers
# ──────────────────────────────────────────────────────────────

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


def get_dataset(dataset_name, img_size=224, num_steps=1, frameskip=1):
    """Load HDF5 dataset with image preprocessing."""
    dataset = swm.data.HDF5Dataset(
        name=dataset_name,
        num_steps=num_steps,
        frameskip=frameskip,
        keys_to_load=["pixels", "action"],
        keys_to_cache=["action"],
    )

    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source="pixels", target="pixels")
    resize = dt.transforms.Resize(img_size, source="pixels", target="pixels")
    transform = dt.transforms.Compose(to_image, resize)
    dataset.transform = transform

    return dataset


def get_lerobot_dataset(
    repo_id: str,
    normalizers_path: str | None = None,
    img_size: int = 224,
    num_steps: int = 1,
    frameskip: int = 5,
    image_key: str = "observation.images.top",
):
    """Load a LeRobot dataset via the lewm adapter, with matching image preprocessing.

    The adapter returns uint8 (T,C,H,W) pixel tensors. We apply a torchvision
    transform that handles the temporal batch dimension in-place so the decoder
    training loop receives the same ImageNet-normalized float tensors as training.
    """
    from lewm_robot.data.lerobot_adapter import LeRobotWMDataset
    from torchvision.transforms import v2 as Tv2

    _mean = torch.tensor([0.485, 0.456, 0.406])
    _std  = torch.tensor([0.229, 0.224, 0.225])

    # Load optional action normalizer stats
    act_mean = act_std = act_dim = None
    if normalizers_path and Path(normalizers_path).exists():
        stats = torch.load(normalizers_path, map_location="cpu")
        if "action" in stats:
            act_mean = stats["action"]["mean"]   # (6,)
            act_std  = stats["action"]["std"]
            act_dim  = act_mean.shape[0]

    def _transform(batch: dict) -> dict:
        # pixels: (T, C, H, W) uint8 → float32 ImageNet-normalized, resized
        pix = batch["pixels"].float() / 255.0          # (T, C, H, W) in [0,1]
        pix = Tv2.functional.resize(pix, [img_size, img_size], antialias=True)
        pix = (pix - _mean[:, None, None]) / _std[:, None, None]
        batch["pixels"] = pix

        # action: (T_act, effective_act_dim) → normalize
        if act_mean is not None and "action" in batch:
            act = batch["action"].float()              # (T, frameskip*act_dim)
            orig_shape = act.shape
            act = act.reshape(-1, frameskip, act_dim)
            act = (act - act_mean) / (act_std + 1e-8)
            batch["action"] = act.reshape(orig_shape)
        return batch

    dataset = LeRobotWMDataset(
        repo_id=repo_id,
        image_key=image_key,
        frameskip=frameskip,
        num_steps=num_steps,
        return_uint8=True,
        keys_to_load=["pixels", "action"],
        transform=_transform,
    )
    return dataset


def denormalize(img_tensor):
    """Convert normalized image tensor back to [0, 255] uint8."""
    mean = IMAGENET_MEAN.to(img_tensor.device).view(3, 1, 1)
    std = IMAGENET_STD.to(img_tensor.device).view(3, 1, 1)
    img = img_tensor * std + mean
    img = img.clamp(0, 1) * 255
    return img.byte()


# ──────────────────────────────────────────────────────────────
#  Training
# ──────────────────────────────────────────────────────────────


@torch.no_grad()
def encode_batch(model, pixels):
    """Encode pixels through frozen encoder+projector, return CLS embeddings."""
    flat = rearrange(pixels, "b t c h w -> (b t) c h w")
    output = model.encoder(flat, interpolate_pos_encoding=True)
    cls_emb = output.last_hidden_state[:, 0]
    cls_emb = model.projector(cls_emb)
    return cls_emb  # (B*T, D)


def load_world_model(checkpoint_path, device):
    """Load a JEPA-like object checkpoint, with fallback to sibling *_object.ckpt."""
    checkpoint_path = Path(checkpoint_path)
    print(f"Loading world model from {checkpoint_path}")
    obj = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if hasattr(obj, "encoder") and hasattr(obj, "predict"):
        return obj

    # Some training runs save Lightning/state_dict checkpoints at *.ckpt and the
    # model object at *_object.ckpt. If needed, fall back automatically.
    if checkpoint_path.suffix == ".ckpt" and not checkpoint_path.stem.endswith("_object"):
        alt = checkpoint_path.with_name(f"{checkpoint_path.stem}_object.ckpt")
        if alt.exists():
            print(f"Checkpoint is not a model object, falling back to {alt}")
            alt_obj = torch.load(alt, map_location=device, weights_only=False)
            if hasattr(alt_obj, "encoder") and hasattr(alt_obj, "predict"):
                return alt_obj

    raise ValueError(
        "Could not load a model object from checkpoint. "
        "Please provide a *_object.ckpt file or a sibling *_object.ckpt "
        "next to the provided checkpoint."
    )


def get_embed_dim(model):
    """Best-effort embed-dim extraction from JEPA-like model."""
    projector = getattr(model, "projector", None)
    if projector is None:
        raise ValueError("Model has no projector; cannot infer embedding dimension.")

    # MLP projector used in this repo
    if hasattr(projector, "net") and len(projector.net) > 0:
        last = projector.net[-1]
        if hasattr(last, "out_features"):
            return last.out_features

    # Identity projector fallback
    if isinstance(projector, nn.Identity) and hasattr(model.encoder, "config"):
        if hasattr(model.encoder.config, "hidden_size"):
            return model.encoder.config.hidden_size

    # Generic Linear fallback
    if hasattr(projector, "out_features"):
        return projector.out_features

    raise ValueError("Unable to infer embed_dim from model.projector.")


@torch.no_grad()
def rollout_latents(model, pixels, actions, history_size, reverse_time=False):
    """
    Roll out latent predictions autoregressively.

    If reverse_time=True, rollout is done on time-reversed trajectories and then
    flipped back to original timeline so outputs are always aligned with real time.
    """
    HS = history_size
    if reverse_time:
        pixels = pixels.flip(1)
        actions = actions.flip(1)

    T = pixels.size(1)

    # Encode context frames
    ctx_pixels = pixels[:, :HS]
    flat_ctx = rearrange(ctx_pixels, "b t c h w -> (b t) c h w")
    out = model.encoder(flat_ctx, interpolate_pos_encoding=True)
    cls_ctx = out.last_hidden_state[:, 0]
    emb_ctx = model.projector(cls_ctx)
    emb = rearrange(emb_ctx, "(b t) d -> b t d", b=pixels.size(0))

    # Autoregressive rollout
    act = actions[:, :HS]
    n_steps = T - HS

    for t in range(n_steps):
        act_emb = model.action_encoder(act)
        emb_trunc = emb[:, -HS:]
        act_trunc = act_emb[:, -HS:]
        pred = model.predict(emb_trunc, act_trunc)[:, -1:]
        emb = torch.cat([emb, pred], dim=1)
        next_act = actions[:, HS + t : HS + t + 1]
        act = torch.cat([act, next_act], dim=1)

    # Match previous rollout behavior: append one extra final prediction.
    act_emb = model.action_encoder(act)
    emb_trunc = emb[:, -HS:]
    act_trunc = act_emb[:, -HS:]
    pred = model.predict(emb_trunc, act_trunc)[:, -1:]
    emb = torch.cat([emb, pred], dim=1)

    aligned = emb[:, :T]
    if reverse_time:
        aligned = aligned.flip(1)
    return aligned


@torch.no_grad()
def get_decoder_latents(model, pixels, actions, args):
    """Get latents used as decoder inputs for either frame or rollout training."""
    if args.train_on_rollouts:
        latents = rollout_latents(
            model,
            pixels,
            actions,
            history_size=args.history_size,
            reverse_time=args.train_reverse_time,
        )
        return rearrange(latents, "b t d -> (b t) d")

    return encode_batch(model, pixels)


def train_decoder(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load frozen world model
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for training mode.")
    model = load_world_model(args.checkpoint, device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    embed_dim = get_embed_dim(model)
    print(f"Encoder embed_dim = {embed_dim}")

    # Dataset
    if args.train_on_rollouts:
        if args.decoder_seq_len < (args.history_size + 1):
            raise ValueError(
                "--decoder-seq-len must be at least history_size + 1 when "
                "--train-on-rollouts is enabled."
            )
        print(
            "Training decoder on rollout latents "
            f"(seq_len={args.decoder_seq_len}, history={args.history_size}, "
            f"reverse_time={args.train_reverse_time})."
        )
    train_num_steps = args.decoder_seq_len if args.train_on_rollouts else 1
    train_frameskip = args.frameskip if args.train_on_rollouts else 1

    if args.lerobot_repo:
        print(f"Loading LeRobot dataset: {args.lerobot_repo}")
        dataset = get_lerobot_dataset(
            repo_id=args.lerobot_repo,
            normalizers_path=args.lerobot_normalizers,
            img_size=224,
            num_steps=train_num_steps,
            frameskip=train_frameskip,
        )
    else:
        print(f"Loading dataset: {args.dataset}")
        dataset = get_dataset(
            args.dataset,
            img_size=224,
            num_steps=train_num_steps,
            frameskip=train_frameskip,
        )

    dataset_len = len(dataset)
    if dataset_len == 0:
        raise ValueError(
            "No training sequences available for the requested dataset settings. "
            "Try reducing --decoder-seq-len and/or --frameskip."
        )

    rng = torch.Generator().manual_seed(42)
    train_len = max(1, int(0.9 * dataset_len))
    val_len = dataset_len - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len], generator=rng)

    if val_len == 0:
        print("Validation split is empty; using train split for periodic reconstruction samples.")

    drop_last = len(train_set) >= args.batch_size

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=drop_last,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # Decoder
    decoder = PatchDecoder(
        embed_dim=embed_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        heads=args.heads,
        dim_head=args.dim_head,
        mlp_dim=args.mlp_dim,
    ).to(device)

    n_params = sum(p.numel() for p in decoder.parameters())
    print(f"Decoder parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    sample_loader = val_loader if len(val_set) > 0 else train_loader

    for epoch in range(args.epochs):
        # ── Train ──
        decoder.train()
        train_loss = 0.0
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [train]"):
            pixels = batch["pixels"].to(device)  # (B, T, C, H, W)
            actions = batch["action"].to(device)
            target = rearrange(pixels, "b t c h w -> (b t) c h w")

            cls_emb = get_decoder_latents(model, pixels, actions, args)
            recon = decoder(cls_emb)

            loss = F.mse_loss(recon, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = train_loss / max(n_batches, 1)

        # ── Validate ──
        decoder.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                pixels = batch["pixels"].to(device)
                actions = batch["action"].to(device)
                target = rearrange(pixels, "b t c h w -> (b t) c h w")
                cls_emb = get_decoder_latents(model, pixels, actions, args)
                recon = decoder(cls_emb)
                val_loss += F.mse_loss(recon, target).item()
                n_val += 1

        avg_val_loss = val_loss / max(n_val, 1)
        print(f"  train_loss={avg_train_loss:.5f}  val_loss={avg_val_loss:.5f}  lr={scheduler.get_last_lr()[0]:.2e}")

        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(decoder.state_dict(), output_dir / "decoder_best.pt")
            print(f"  -> saved best decoder (val_loss={best_val_loss:.5f})")

        # Save periodic
        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
            torch.save(decoder.state_dict(), output_dir / f"decoder_epoch_{epoch+1}.pt")

        # Save sample reconstructions every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_reconstruction_samples(model, decoder, sample_loader, device, output_dir, epoch + 1, args)

    # Final save
    torch.save(decoder.state_dict(), output_dir / "decoder.pt")
    print(f"Training complete. Decoder saved to {output_dir / 'decoder.pt'}")


@torch.no_grad()
def save_reconstruction_samples(model, decoder, loader, device, output_dir, epoch, args, n_samples=8):
    """Save a grid of original vs reconstructed images."""
    decoder.eval()
    batch = next(iter(loader))
    pixels = batch["pixels"][:n_samples].to(device)
    actions = batch["action"][:n_samples].to(device)
    target = rearrange(pixels, "b t c h w -> (b t) c h w")

    cls_emb = get_decoder_latents(model, pixels, actions, args)
    recon = decoder(cls_emb)

    originals = denormalize(target[:n_samples])
    reconstructed = denormalize(recon[:n_samples])

    # Build comparison grid: original on top, reconstruction on bottom
    rows = []
    for i in range(min(n_samples, originals.size(0))):
        orig = originals[i].permute(1, 2, 0).cpu().numpy()
        rec = reconstructed[i].permute(1, 2, 0).cpu().numpy()
        rows.append(np.concatenate([orig, rec], axis=1))  # side by side

    grid = np.concatenate(rows, axis=0)
    img = Image.fromarray(grid)
    img.save(output_dir / f"recon_epoch_{epoch}.png")
    print(f"  -> saved reconstruction samples to recon_epoch_{epoch}.png")


# ──────────────────────────────────────────────────────────────
#  Visualization: Decode imagined rollouts
# ──────────────────────────────────────────────────────────────


@torch.no_grad()
def visualize_rollouts(args):
    """
    Load a world model + trained decoder, roll out imagined latent
    trajectories from the dataset, decode them to pixels, and produce
    side-by-side GIFs comparing ground truth with imagined reconstructions.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load world model
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for --visualize mode.")
    model = load_world_model(args.checkpoint, device)
    model.eval()

    embed_dim = get_embed_dim(model)

    # Load decoder
    print(f"Loading decoder from {args.decoder_weights}")
    decoder = PatchDecoder(
        embed_dim=embed_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        heads=args.heads,
        dim_head=args.dim_head,
        mlp_dim=args.mlp_dim,
    ).to(device)
    decoder.load_state_dict(torch.load(args.decoder_weights, map_location=device, weights_only=True))
    decoder.eval()

    # Load dataset with longer sequences for rollouts
    rollout_len = args.rollout_steps + args.history_size
    if args.lerobot_repo:
        dataset = get_lerobot_dataset(
            repo_id=args.lerobot_repo,
            normalizers_path=args.lerobot_normalizers,
            img_size=224,
            num_steps=rollout_len,
            frameskip=args.frameskip,
        )
    else:
        dataset = swm.data.HDF5Dataset(
            name=args.dataset,
            num_steps=rollout_len,
            frameskip=args.frameskip,
            keys_to_load=["pixels", "action"],
            keys_to_cache=["action"],
        )
        imagenet_stats = dt.dataset_stats.ImageNet
        to_image = dt.transforms.ToImage(**imagenet_stats, source="pixels", target="pixels")
        resize = dt.transforms.Resize(224, source="pixels", target="pixels")
        dataset.transform = dt.transforms.Compose(to_image, resize)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    for idx, batch in enumerate(loader):
        if idx >= args.num_rollouts:
            break

        pixels = batch["pixels"].to(device)   # (1, T, C, H, W)
        actions = batch["action"].to(device)   # (1, T, A)
        T = pixels.size(1)
        HS = args.history_size

        all_emb = rollout_latents(
            model,
            pixels,
            actions,
            history_size=HS,
            reverse_time=args.reverse_time,
        )[0]

        # --- Decode all embeddings to pixels ---
        decoded = decoder(all_emb)  # (T, 3, 224, 224)
        decoded_imgs = denormalize(decoded)  # (T, 3, 224, 224) uint8

        # --- Also decode the real frames for comparison ---
        flat_all = rearrange(pixels, "b t c h w -> (b t) c h w")
        real_imgs = denormalize(flat_all)  # (T, 3, 224, 224) uint8

        # --- Build side-by-side GIF frames ---
        frames = []
        for t in range(T):
            real = real_imgs[t].permute(1, 2, 0).cpu().numpy()
            imagined = decoded_imgs[t].permute(1, 2, 0).cpu().numpy()

            # Mark context frames vs predicted frames in displayed (forward) time.
            # For reverse-time rollouts, backward context corresponds to the sequence tail.
            if args.reverse_time:
                is_context = t >= (T - HS)
            else:
                is_context = t < HS

            if is_context:
                label_color = (0, 200, 0)  # context
            else:
                label_color = (0, 100, 255)  # predicted

            # Add thin colored border to imagined frame
            bordered = imagined.copy()
            bordered[:3, :] = label_color
            bordered[-3:, :] = label_color
            bordered[:, :3] = label_color
            bordered[:, -3:] = label_color

            combined = np.concatenate([real, bordered], axis=1)
            frames.append(Image.fromarray(combined))

        # Save as GIF
        out_path = output_dir / f"rollout_{idx}.gif"
        frames[0].save(
            str(out_path),
            save_all=True,
            append_images=frames[1:],
            duration=200,
            loop=0,
        )
        print(f"Saved rollout {idx} -> {out_path}  ({T} frames, {HS} context)")

    print(f"\nDone! {args.num_rollouts} rollouts saved to {output_dir}")


@torch.no_grad()
def visualize_rollouts_compare(args):
    """
    Compare forward and backward bidirectional models on the same sampled
    trajectories. Output GIF columns: [real | forward-model | backward-model].
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.checkpoint is None or args.checkpoint_bwd is None:
        raise ValueError("--checkpoint and --checkpoint-bwd are required with --compare-rollouts")
    if args.decoder_weights is None or args.decoder_weights_bwd is None:
        raise ValueError("--decoder-weights and --decoder-weights-bwd are required with --compare-rollouts")

    model_fwd = load_world_model(args.checkpoint, device)
    model_bwd = load_world_model(args.checkpoint_bwd, device)
    model_fwd.eval()
    model_bwd.eval()

    embed_dim_fwd = get_embed_dim(model_fwd)
    embed_dim_bwd = get_embed_dim(model_bwd)

    print(f"Loading forward decoder from {args.decoder_weights}")
    decoder_fwd = PatchDecoder(
        embed_dim=embed_dim_fwd,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        heads=args.heads,
        dim_head=args.dim_head,
        mlp_dim=args.mlp_dim,
    ).to(device)
    decoder_fwd.load_state_dict(torch.load(args.decoder_weights, map_location=device, weights_only=True))
    decoder_fwd.eval()

    print(f"Loading backward decoder from {args.decoder_weights_bwd}")
    decoder_bwd = PatchDecoder(
        embed_dim=embed_dim_bwd,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        heads=args.heads,
        dim_head=args.dim_head,
        mlp_dim=args.mlp_dim,
    ).to(device)
    decoder_bwd.load_state_dict(torch.load(args.decoder_weights_bwd, map_location=device, weights_only=True))
    decoder_bwd.eval()

    rollout_len = args.rollout_steps + args.history_size
    dataset = swm.data.HDF5Dataset(
        name=args.dataset,
        num_steps=rollout_len,
        frameskip=args.frameskip,
        keys_to_load=["pixels", "action"],
        keys_to_cache=["action"],
    )
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source="pixels", target="pixels")
    resize = dt.transforms.Resize(224, source="pixels", target="pixels")
    dataset.transform = dt.transforms.Compose(to_image, resize)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    for idx, batch in enumerate(loader):
        if idx >= args.num_rollouts:
            break

        pixels = batch["pixels"].to(device)
        actions = batch["action"].to(device)
        T = pixels.size(1)
        HS = args.history_size

        emb_fwd = rollout_latents(model_fwd, pixels, actions, history_size=HS, reverse_time=False)[0]
        emb_bwd = rollout_latents(model_bwd, pixels, actions, history_size=HS, reverse_time=True)[0]

        decoded_fwd = denormalize(decoder_fwd(emb_fwd))
        decoded_bwd = denormalize(decoder_bwd(emb_bwd))

        flat_all = rearrange(pixels, "b t c h w -> (b t) c h w")
        real_imgs = denormalize(flat_all)

        frames = []
        for t in range(T):
            real = real_imgs[t].permute(1, 2, 0).cpu().numpy()
            fwd = decoded_fwd[t].permute(1, 2, 0).cpu().numpy()
            bwd = decoded_bwd[t].permute(1, 2, 0).cpu().numpy()

            # Green means context frame, blue means predicted for forward model.
            fwd_color = (0, 200, 0) if t < HS else (0, 100, 255)
            # For backward model context is at sequence tail after time alignment.
            bwd_color = (0, 200, 0) if t >= (T - HS) else (255, 140, 0)

            fwd_bordered = fwd.copy()
            fwd_bordered[:3, :] = fwd_color
            fwd_bordered[-3:, :] = fwd_color
            fwd_bordered[:, :3] = fwd_color
            fwd_bordered[:, -3:] = fwd_color

            bwd_bordered = bwd.copy()
            bwd_bordered[:3, :] = bwd_color
            bwd_bordered[-3:, :] = bwd_color
            bwd_bordered[:, :3] = bwd_color
            bwd_bordered[:, -3:] = bwd_color

            combined = np.concatenate([real, fwd_bordered, bwd_bordered], axis=1)
            frames.append(Image.fromarray(combined))

        out_path = output_dir / f"rollout_compare_{idx}.gif"
        frames[0].save(
            str(out_path),
            save_all=True,
            append_images=frames[1:],
            duration=200,
            loop=0,
        )
        print(f"Saved compare rollout {idx} -> {out_path}  ({T} frames, HS={HS})")

    print(f"\nDone! {args.num_rollouts} compare rollouts saved to {output_dir}")


# ──────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Train/visualize LeWM pixel decoder")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to LeWM checkpoint (*.ckpt or *_object.ckpt)")
    parser.add_argument("--checkpoint-bwd", type=str, default=None, help="Backward model checkpoint for compare mode")
    parser.add_argument("--dataset", type=str, default="tworoom", help="HDF5 dataset name (tworoom, pusht_expert_train). Ignored when --lerobot-repo is set.")
    parser.add_argument("--lerobot-repo", type=str, default=None, help="LeRobot HuggingFace repo id (e.g. lerobot/svla_so100_pickplace). Takes priority over --dataset.")
    parser.add_argument("--lerobot-normalizers", type=str, default=None, help="Path to lewm_*_normalizers.pt produced by train.py.")
    parser.add_argument("--output-dir", type=str, default="./decoder_outputs", help="Output directory")

    # Decoder architecture
    parser.add_argument("--hidden-dim", type=int, default=384, help="Decoder hidden dim")
    parser.add_argument("--depth", type=int, default=4, help="Number of decoder blocks")
    parser.add_argument("--heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--dim-head", type=int, default=48, help="Dimension per head")
    parser.add_argument("--mlp-dim", type=int, default=1024, help="MLP hidden dim in decoder blocks")

    # Training
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--train-on-rollouts", action="store_true", help="Train decoder on autoregressive rollout latents instead of direct encoded frame latents")
    parser.add_argument("--train-reverse-time", action="store_true", help="Use reversed-time rollouts when --train-on-rollouts is enabled (for backward dynamics)")
    parser.add_argument("--decoder-seq-len", type=int, default=1, help="Sequence length for decoder training dataset")

    # Visualization mode
    parser.add_argument("--visualize", action="store_true", help="Visualize decoded rollouts instead of training")
    parser.add_argument("--compare-rollouts", action="store_true", help="Compare forward and backward model rollouts in one GIF")
    parser.add_argument("--decoder-weights", type=str, default=None, help="Path to trained decoder .pt file")
    parser.add_argument("--decoder-weights-bwd", type=str, default=None, help="Path to backward decoder .pt file for compare mode")
    parser.add_argument("--num-rollouts", type=int, default=5, help="Number of rollouts to visualize")
    parser.add_argument("--rollout-steps", type=int, default=20, help="Number of prediction steps")
    parser.add_argument("--history-size", type=int, default=3, help="Context history size")
    parser.add_argument("--frameskip", type=int, default=5, help="Frameskip for rollout dataset")
    parser.add_argument("--reverse-time", action="store_true", help="Use reversed-time latent rollout during --visualize (for backward models)")

    args = parser.parse_args()

    if args.compare_rollouts:
        visualize_rollouts_compare(args)
    elif args.visualize:
        if args.decoder_weights is None:
            parser.error("--decoder-weights is required when using --visualize")
        visualize_rollouts(args)
    else:
        train_decoder(args)


if __name__ == "__main__":
    main()
