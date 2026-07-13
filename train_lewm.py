import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack


class _AddIsFirst:
    """Wraps a dataset to tag each window with is_first=True when it starts at episode position 0."""
    def __init__(self, dataset):
        self._ds = dataset

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        _, start = self._ds.clip_indices[idx]
        item = self._ds[idx]
        item['is_first'] = torch.tensor(start == 0, dtype=torch.bool)
        return item

    def __getattr__(self, name):
        return getattr(self._ds, name)


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    # Strip anchor pixel frame — it was only loaded to compute action deltas
    for k in list(batch.keys()):
        if k.startswith("pixels"):
            batch[k] = batch[k][:, 1:]

    # Compute action deltas: delta[t] = action[t] - action[t-1]
    # act[:, 0] is the anchor (actual preceding step, or zero for episode starts)
    act = batch["action"]  # (B, T+1, A)
    anchor = act[:, :1].clone()
    if "is_first" in batch:
        mask = batch["is_first"].to(act.dtype).view(-1, 1, 1)
        anchor = anchor * (1 - mask)
    prev = torch.cat([anchor, act[:, 1:-1]], dim=1)  # (B, T, A)
    batch["action"] = act[:, 1:] - prev  # (B, T, A)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    tgt_emb = emb[:, n_preds:] # label
    pred_emb = self.model.predict(ctx_emb, ctx_act) # pred

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]  

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset = hydra.utils.instantiate(
        cfg.data.dataset,
        transform=None,
        num_steps=cfg.wm.history_size + cfg.wm.num_preds + 1,
    )
    pixel_keys = [k for k in cfg.data.dataset.keys_to_load if k.startswith("pixels")]
    transforms = [get_img_preprocessor(source=k, target=k, img_size=cfg.img_size) for k in pixel_keys]

    normalizer_stats: dict[str, dict[str, torch.Tensor]] = {}
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue

            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)
            # Capture mean/std so deploy can apply the same per-column scaling.
            normalizer_stats[col] = {
                "mean": normalizer.mean.detach().clone(),
                "std": normalizer.std.detach().clone(),
            }

            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform
    dataset = _AddIsFirst(dataset)

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################

    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )

    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,
    )

    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)

    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    n_cams = len([k for k in cfg.data.dataset.keys_to_load if k.startswith("pixels")])
    cam_fuser = (
        MLP(input_dim=n_cams * embed_dim, hidden_dim=2048, output_dim=embed_dim)
        if n_cams > 1 else None
    )

    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
        cam_fuser=cam_fuser,
    )
    if cfg.get("compile", True):
        world_model = torch.compile(world_model, dynamic=True)

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)

    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    _repo_root = Path(__file__).resolve().parent
    run_dir = Path(swm.data.utils.get_cache_dir(override_root=_repo_root / "checkpoints"), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    if normalizer_stats:
        torch.save(
            normalizer_stats,
            run_dir / f"{cfg.output_model_name}_normalizers.pt",
        )

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir, filename=cfg.output_model_name, epoch_interval=1,
    )

    # Full trainer-state checkpoint (optimizer/epoch/LR schedule) for resume —
    # the pickled objects above only hold model weights, and nothing ever wrote
    # the legacy {name}_weights.ckpt the resume logic looked for.
    lightning_ckpt_callback = ModelCheckpoint(
        dirpath=run_dir, save_last=True, save_top_k=0, every_n_epochs=1
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback, lightning_ckpt_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    # Resume preference: Lightning last.ckpt (full state) → legacy weights file
    last_ckpt = run_dir / "last.ckpt"
    legacy_ckpt = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    resume_ckpt = last_ckpt if last_ckpt.exists() else legacy_ckpt
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=resume_ckpt if resume_ckpt.exists() else None,
    )

    manager()

    # Export world model in safetensors format for GC-IDM training and
    # JEPAPolicy loading (in addition to the pickled object saved by the callback).
    try:
        from safetensors.torch import save_file
        raw_model = world_model.model
        if hasattr(raw_model, "_orig_mod"):     # unwrap torch.compile
            raw_model = raw_model._orig_mod
        state = {k: v.cpu().contiguous() for k, v in raw_model.state_dict().items()}
        safetensors_path = run_dir / f"{cfg.output_model_name}.safetensors"
        save_file(state, str(safetensors_path))
        print(f"Saved safetensors checkpoint → {safetensors_path}")
    except Exception as exc:
        print(f"Warning: safetensors export failed ({exc}); pickle checkpoint still valid.")

    return


if __name__ == "__main__":
    run()
