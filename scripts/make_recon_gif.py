"""Render a long side-by-side GT | decoded reconstruction GIF for the README.

Runs the frozen JEPA encoder + trained decoder over a dense, contiguous frame
sequence from a single episode and lays Ground-Truth next to the decoded image
(left | right) for every frame, producing a smooth animation.

Usage:
    python scripts/make_recon_gif.py \
        --world-model-path checkpoints/so100_topcam/lewm_so100_topcam_epoch_50_object.ckpt \
        --decoder-path     checkpoints/so100_topcam/decoder.pt \
        --out             checkpoints/so100_topcam/decoder_recon_sidebyside.gif \
        --episode 0 --num-frames 120 --fps 15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torchvision import transforms
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import stable_pretraining as spt  # noqa: E402

from jepa import JEPA  # noqa: E402
from module import ARPredictor, Embedder, MLP  # noqa: E402
from lewm_robot.decoder import JEPADecoder  # noqa: E402
from lewm_robot.policies.jepa.configuration_jepa import JEPAConfig  # noqa: E402
from utils import get_img_preprocessor  # noqa: E402


def build_world_model(cfg: JEPAConfig, device: torch.device) -> JEPA:
    """Inline copy of lewm_robot.policies.jepa.modeling_jepa.build_world_model.

    Replicated here so this script never imports modeling_jepa, which pulls in
    lerobot's full policy registry (GR00T config crashes dataclass init on
    Python 3.13). We only need the frozen encoder for reconstruction.
    """
    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale, patch_size=cfg.patch_size, image_size=cfg.img_size,
        pretrained=False, use_mask_token=False,
    )
    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.embed_dim
    predictor = ARPredictor(
        num_frames=cfg.history_size, input_dim=embed_dim, hidden_dim=hidden_dim,
        output_dim=hidden_dim, depth=cfg.predictor_depth, heads=cfg.predictor_heads,
        mlp_dim=cfg.predictor_mlp_dim, dim_head=cfg.predictor_dim_head,
        dropout=cfg.predictor_dropout, emb_dropout=cfg.predictor_emb_dropout,
    )
    action_encoder = Embedder(input_dim=cfg.effective_action_dim, emb_dim=embed_dim)
    projector = MLP(input_dim=hidden_dim, output_dim=embed_dim, hidden_dim=2048,
                    norm_fn=nn.BatchNorm1d)
    pred_proj = MLP(input_dim=hidden_dim, output_dim=embed_dim, hidden_dim=2048,
                    norm_fn=nn.BatchNorm1d)
    n_cams = len(cfg.image_keys)
    cam_fuser = (MLP(input_dim=n_cams * embed_dim, hidden_dim=2048, output_dim=embed_dim)
                 if n_cams > 1 else None)
    return JEPA(
        encoder=encoder, predictor=predictor, action_encoder=action_encoder,
        projector=projector, pred_proj=pred_proj, cam_fuser=cam_fuser,
    ).to(device)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--world-model-path", required=True)
    p.add_argument("--decoder-path", required=True)
    p.add_argument("--out", required=True, help="Output .gif path")
    p.add_argument("--repo-id", default="maelicneau/stack_cubes")
    p.add_argument("--data-root",
                   default="leWorldRobot/datasets/stack_cubes")
    p.add_argument("--image-key", default="observation.images.up")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--num-frames", type=int, default=120,
                   help="Frames sampled evenly across the episode")
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--encoder-scale", default="tiny")
    p.add_argument("--embed-dim", type=int, default=192)
    p.add_argument("--history-size", type=int, default=3)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--out-h", type=int, default=240)
    p.add_argument("--out-w", type=int, default=320)
    p.add_argument("--patch-size", type=int, default=14)
    p.add_argument("--decoder-dim", type=int, default=256)
    p.add_argument("--gap", type=int, default=6, help="Pixel gap between panels")
    p.add_argument("--colors", type=int, default=128,
                   help="GIF palette size (lower = smaller file)")
    p.add_argument("--mp4", action="store_true",
                   help="Also write a compact .mp4 next to the .gif")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def to_uint8_hwc(img: torch.Tensor) -> np.ndarray:
    """(C, H, W) float[0,1] → (H, W, C) uint8."""
    return (img.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    # ── World model (frozen encoder) ─────────────────────────────────────────
    jepa_cfg = JEPAConfig(
        encoder_scale=args.encoder_scale,
        embed_dim=args.embed_dim,
        history_size=args.history_size,
        img_size=args.img_size,
        patch_size=args.patch_size,
        image_keys=[args.image_key],
    )
    world_model = build_world_model(jepa_cfg, device)
    loaded = torch.load(Path(args.world_model_path).expanduser(),
                        map_location=device, weights_only=False)
    if isinstance(loaded, JEPA):
        world_model.load_state_dict(loaded.state_dict())
    elif isinstance(loaded, dict):
        world_model.load_state_dict(loaded)
    world_model.eval()
    for p in world_model.parameters():
        p.requires_grad_(False)

    # ── Decoder ──────────────────────────────────────────────────────────────
    decoder = JEPADecoder(
        embed_dim=args.embed_dim, img_size=args.img_size, patch_size=args.patch_size,
        decoder_dim=args.decoder_dim, out_h=args.out_h, out_w=args.out_w,
    ).to(device)
    decoder.load_state_dict(torch.load(Path(args.decoder_path).expanduser(),
                                       map_location=device, weights_only=True))
    decoder.eval()

    # ── Episode slice (read directly from mp4; bypass torchcodec) ────────────
    # LeRobot v3 concatenates every episode into one continuous mp4, so the
    # global frame index equals the mp4 frame number. Episode boundaries come
    # from the episodes-meta parquet (dataset_from_index / dataset_to_index).
    import glob

    import av
    import pandas as pd

    data_root = Path(args.data_root).expanduser()
    meta_files = sorted(glob.glob(str(data_root / "meta/episodes/**/*.parquet"), recursive=True))
    ep_df = pd.concat([pd.read_parquet(f) for f in meta_files], ignore_index=True)
    row = ep_df[ep_df["episode_index"] == args.episode]
    if row.empty:
        raise SystemExit(f"Episode {args.episode} not found in episodes meta.")
    g0 = int(row["dataset_from_index"].iloc[0])
    g1 = int(row["dataset_to_index"].iloc[0])

    # Locate the camera video file for this episode.
    vchunk = int(row[f"videos/{args.image_key}/chunk_index"].iloc[0])
    vfile = int(row[f"videos/{args.image_key}/file_index"].iloc[0])
    video_path = data_root / "videos" / args.image_key / f"chunk-{vchunk:03d}" / f"file-{vfile:03d}.mp4"
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    n = min(args.num_frames, g1 - g0)
    sel = sorted({g0 + int(i) for i in torch.linspace(0, g1 - g0 - 1, n).long().tolist()})
    sel_set = set(sel)
    print(f"Episode {args.episode}: frames [{g0}, {g1}) ({g1 - g0}) → sampling {len(sel)}")
    print(f"Reading from {video_path}")

    # Sequential decode of the global frame range with PyAV (libdav1d handles
    # AV1 in software; OpenCV's FFmpeg only attempts unavailable HW decode).
    container = av.open(str(video_path))
    vstream = container.streams.video[0]
    frames_by_idx: dict[int, torch.Tensor] = {}
    gi = 0
    for frame in container.decode(vstream):
        if gi >= g1 or len(frames_by_idx) == len(sel):
            break
        if gi in sel_set:
            rgb = frame.to_ndarray(format="rgb24")  # (H, W, 3) uint8
            frames_by_idx[gi] = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        gi += 1
    container.close()

    img_pre = get_img_preprocessor(source="pixels", target="pixels", img_size=args.img_size)
    resize = transforms.Resize((args.out_h, args.out_w), antialias=True)

    # ── Encode → decode each frame ───────────────────────────────────────────
    panels: list[Image.Image] = []
    label_h = 22
    with torch.no_grad():
        for idx in tqdm(sel, desc="rendering"):
            raw = frames_by_idx[idx]                           # (C, H, W) float[0,1]
            frame_u8 = (raw * 255).clamp(0, 255).to(torch.uint8).unsqueeze(0)  # (1,C,H,W)
            pre = img_pre({"pixels": frame_u8})["pixels"].unsqueeze(1).to(device)  # (1,1,C,224,224)
            z = world_model.encode({"pixels": pre})["emb"][:, 0]                   # (1, D)
            recon = decoder(z)[0].cpu()                                            # (C, out_h, out_w)
            gt = resize(raw)                                                       # (C, out_h, out_w)

            gt_np, rec_np = to_uint8_hwc(gt), to_uint8_hwc(recon)
            h, w = gt_np.shape[:2]
            canvas = Image.new("RGB", (w * 2 + args.gap, h + label_h), (20, 20, 20))
            canvas.paste(Image.fromarray(gt_np), (0, label_h))
            canvas.paste(Image.fromarray(rec_np), (w + args.gap, label_h))
            draw = ImageDraw.Draw(canvas)
            draw.text((6, 5), "Ground truth", fill=(235, 235, 235))
            draw.text((w + args.gap + 6, 5), "Decoded (JEPA 192-d CLS -> image)",
                      fill=(120, 220, 140))
            panels.append(canvas)

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # GIF: quantize every frame to a shared adaptive palette to keep the file
    # small (full-colour frames balloon the size; optimize alone isn't enough).
    pal = panels[0].quantize(colors=args.colors, method=Image.MEDIANCUT)
    quant = [p.quantize(colors=args.colors, palette=pal, dither=Image.NONE)
             for p in panels]
    quant[0].save(
        out_path, save_all=True, append_images=quant[1:],
        loop=0, duration=int(1000 / args.fps), optimize=True,
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f"Saved {out_path}  ({len(panels)} frames @ {args.fps:g} fps, "
          f"{args.colors} colors, {size_mb:.1f} MB)")

    # Bonus: a much smaller, higher-quality MP4 (handy if the README can embed video).
    if args.mp4:
        mp4_path = out_path.with_suffix(".mp4")
        W, H = panels[0].size
        # H.264 needs even dimensions.
        W2, H2 = W - (W % 2), H - (H % 2)
        out = av.open(str(mp4_path), mode="w")
        vs = out.add_stream("libx264", rate=int(round(args.fps)))
        vs.width, vs.height, vs.pix_fmt = W2, H2, "yuv420p"
        for p in panels:
            arr = np.asarray(p.convert("RGB"))[:H2, :W2]
            out.mux(vs.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")))
        out.mux(vs.encode(None))
        out.close()
        print(f"Saved {mp4_path}  ({mp4_path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
