"""JEPA + GC-IDM amortized planner as a LeRobot PreTrainedPolicy.

Two-stage pipeline:
  Stage 1 — train_lewm.py:    trains the JEPA world model offline.
  Stage 2 — train_gc_idm.py:  trains the GC-IDM on frozen encoder embeddings.

At deployment, ``select_action`` runs a single encoder + GC-IDM forward
pass per tick (~20-60 ms on GPU), eliminating CEM/MPPI search and enabling
closed-loop control at 10-30 Hz on SO-100.
"""

from __future__ import annotations

import logging
import sys
from collections import deque
from pathlib import Path
from typing import Unpack

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch import Tensor

from lerobot.policies.pretrained import ActionSelectKwargs, PreTrainedPolicy

# ── Make the leWorldRobot repo root importable (jepa.py, module.py, utils.py)
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import stable_pretraining as spt  # noqa: E402

from jepa import JEPA  # noqa: E402  (from leWorldRobot repo root)
from module import ARPredictor, Embedder, MLP  # noqa: E402
from utils import get_img_preprocessor  # noqa: E402

from lewm_robot.policies.jepa.configuration_jepa import JEPAConfig  # noqa: E402
from lewm_robot.policies.jepa.modeling_gc_idm import GCIDM  # noqa: E402

logger = logging.getLogger(__name__)


def build_world_model(config: JEPAConfig, device: torch.device) -> JEPA:
    """Instantiate a JEPA world model from a JEPAConfig.

    Architecture is identical to ``train_lewm.py`` so that checkpoints are
    fully compatible.
    """
    encoder = spt.backbone.utils.vit_hf(
        config.encoder_scale,
        patch_size=config.patch_size,
        image_size=config.img_size,
        pretrained=False,
        use_mask_token=False,
    )
    hidden_dim = encoder.config.hidden_size
    embed_dim = config.embed_dim

    predictor = ARPredictor(
        num_frames=config.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        depth=config.predictor_depth,
        heads=config.predictor_heads,
        mlp_dim=config.predictor_mlp_dim,
        dim_head=config.predictor_dim_head,
        dropout=config.predictor_dropout,
        emb_dropout=config.predictor_emb_dropout,
    )

    action_encoder = Embedder(
        input_dim=config.effective_action_dim,
        emb_dim=embed_dim,
    )
    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=nn.BatchNorm1d,
    )
    pred_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=nn.BatchNorm1d,
    )

    n_cams = len(config.image_keys)
    cam_fuser = (
        MLP(input_dim=n_cams * embed_dim, hidden_dim=2048, output_dim=embed_dim)
        if n_cams > 1
        else None
    )

    return JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=pred_proj,
        cam_fuser=cam_fuser,
    ).to(device)


class JEPAPolicy(PreTrainedPolicy):
    """JEPA world model + GC-IDM amortized planner.

    Deployment flow per tick:
        1. Pull camera frame(s) from ``batch``.
        2. Preprocess (ImageNet normalise + resize) per camera.
        3. Encode with frozen JEPA encoder → z_t (D,).
        4. GC-IDM(z_t, z_goal, h) → action chunk (frameskip × DOF).
        5. Decrement horizon h and return one action from the chunk.

    Goal embedding is set once at episode start via ``set_goal()``.
    """

    config_class = JEPAConfig
    name = "jepa"

    def __init__(self, config: JEPAConfig, **kwargs) -> None:
        super().__init__(config)
        self.config = config

        device = torch.device(config.device or "cpu")
        self._device = device

        # ── Build architecture ──────────────────────────────────────────────
        self.world_model = build_world_model(config, device)
        # GC-IDM predicts one native-step action (6 DOF), matching arXiv 2605.08732.
        # The deploy loop reshapes per-tick actions from the returned chunk.
        self.gc_idm = GCIDM(
            latent_dim=config.embed_dim,
            action_dim=config.action_dim,
            hidden_dim=config.gc_idm_hidden_dim,
            horizon_dim=config.gc_idm_horizon_dim,
            max_horizon=config.max_horizon,
        ).to(device)

        # ── Load checkpoints ────────────────────────────────────────────────
        if config.resolved_world_model_path is not None:
            self._load_world_model(config.resolved_world_model_path)
        if config.resolved_gc_idm_path is not None:
            self._load_gc_idm(config.resolved_gc_idm_path)

        # ── Action normalizers ──────────────────────────────────────────────
        self._action_mean = torch.zeros(config.action_dim)
        self._action_std = torch.ones(config.action_dim)
        if config.resolved_normalizers_path and config.resolved_normalizers_path.exists():
            stats = torch.load(config.resolved_normalizers_path, map_location="cpu")
            if "action" in stats:
                self._action_mean = stats["action"]["mean"]   # (action_dim,)
                self._action_std = stats["action"]["std"]     # (action_dim,)

        # ── Image preprocessors (one per camera) ───────────────────────────
        self._img_preprocessors = {
            key: get_img_preprocessor(source="pixels", target="pixels", img_size=config.img_size)
            for key in config.image_keys
        }

        # ── Goal state ──────────────────────────────────────────────────────
        self._goal_emb: Tensor | None = None
        self._horizon: int = config.max_horizon
        self._action_queue: deque[Tensor] = deque()

        if config.resolved_goal_image_path is not None:
            self._load_goal_from_file(config.resolved_goal_image_path)

    # ── Checkpoint loading ─────────────────────────────────────────────────

    def _load_world_model(self, path: Path) -> None:
        logger.info("Loading world model from %s", path)
        if path.suffix in (".pt", ".ckpt"):
            # Pickled JEPA object from ModelObjectCallBack
            loaded = torch.load(path, map_location=self._device, weights_only=False)
            if isinstance(loaded, JEPA):
                self.world_model.load_state_dict(loaded.state_dict())
            elif isinstance(loaded, dict):
                self.world_model.load_state_dict(loaded)
            else:
                logger.warning("Unexpected world model type %s; skipping.", type(loaded))
        elif path.suffix == ".safetensors":
            from safetensors.torch import load_file
            state = load_file(str(path))
            self.world_model.load_state_dict(state)
        self.world_model.eval()
        for p in self.world_model.parameters():
            p.requires_grad_(False)

    def _load_gc_idm(self, path: Path) -> None:
        logger.info("Loading GC-IDM from %s", path)
        state = torch.load(path, map_location=self._device, weights_only=True)
        self.gc_idm.load_state_dict(state)

    # ── Goal management ────────────────────────────────────────────────────

    def _load_goal_from_file(self, path: Path) -> None:
        img = Image.open(path).convert("RGB")
        chw = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).contiguous()
        self.set_goal({self.config.image_keys[0]: chw})

    def set_goal(self, goal_images: dict[str, Tensor]) -> None:
        """Encode a goal observation and cache its embedding.

        Args:
            goal_images: dict mapping image_key → (C, H, W) uint8 tensor.
                         Only the first (and optionally second) key is used.
        """
        self.world_model.eval()
        info = self._preprocess_obs(goal_images)
        with torch.no_grad():
            info = self.world_model.encode(info)
        self._goal_emb = info["emb"][:, 0].detach()    # (1, D)
        self._horizon = self.config.max_horizon

    # ── Internal helpers ───────────────────────────────────────────────────

    def _extract_pixels(self, batch: dict[str, Tensor], key: str) -> Tensor:
        """Pull one camera frame as (C, H, W) uint8 from the policy batch.

        Tries the full key (``observation.images.up``), then the plain suffix
        (``up``), to handle both our custom deploy loop and ``lerobot-rollout``
        (which strips the ``observation.images.`` prefix from robot camera keys).
        """
        cam = batch.get(key)
        if cam is None:
            # lerobot-rollout uses plain camera name (last component of the dot-path)
            plain = key.split(".")[-1]
            cam = batch.get(plain)
        if cam is None:
            raise KeyError(
                f"JEPAPolicy: image key {key!r} (or plain '{key.split('.')[-1]}') "
                f"not found in batch (keys: {list(batch.keys())})"
            )
        if cam.ndim == 4:   # (B, C, H, W) — drop batch dim
            cam = cam[0]
        if cam.dtype.is_floating_point:
            cam = (cam.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
        return cam  # (C, H, W) uint8

    def _preprocess_obs(self, obs: dict[str, Tensor]) -> dict[str, Tensor]:
        """Apply per-camera preprocessing and build the info dict for JEPA encode.

        Uses ``_extract_pixels`` so plain camera names (``up``, ``side``) are
        accepted in addition to full ``observation.images.*`` keys.
        """
        info: dict[str, Tensor] = {}
        for idx, key in enumerate(self.config.image_keys):
            try:
                cam = self._extract_pixels(obs, key)    # (C, H, W) uint8
            except KeyError:
                break   # second camera absent — single-cam mode
            if cam.ndim == 3:
                cam = cam.unsqueeze(0)  # (1, C, H, W)
            pre = self._img_preprocessors[key]({"pixels": cam})["pixels"]  # (1, C, H, W)
            pre = pre.to(self._device)
            slot = "pixels" if idx == 0 else "pixels2"
            info[slot] = pre.unsqueeze(1)   # (1, 1, C, H, W) — B=1, T=1
        return info

    @torch.no_grad()
    def _encode_obs(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode observation → z_t of shape (1, D)."""
        self.world_model.eval()
        obs = {k: self._extract_pixels(batch, k) for k in self.config.image_keys}
        info = self._preprocess_obs(obs)
        info = self.world_model.encode(info)
        return info["emb"][:, 0]    # (1, D)

    # ── PreTrainedPolicy API ───────────────────────────────────────────────

    def get_optim_params(self) -> dict:
        # World model is frozen; only GC-IDM parameters are trained.
        return {"params": list(self.gc_idm.parameters())}

    def reset(self) -> None:
        self._action_queue.clear()
        self._horizon = self.config.max_horizon

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        # No online training path — Stage 2 is done via train_gc_idm.py.
        zero = torch.zeros((), device=self._device, requires_grad=False)
        return zero, None

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        """Produce a chunk of per-step actions via a single GC-IDM forward pass.

        GC-IDM predicts one 6-DOF action per call (faithful to arXiv 2605.08732).
        With ``chunk_size=1`` (default, closed-loop) that single action is returned.
        With ``chunk_size>1`` the encoder runs once and GC-IDM is called
        ``chunk_size`` times with decreasing horizons — suitable for the RTC
        inference engine.

        Returns:
            Tensor of shape ``(1, chunk_size, action_dim)`` in
            denormalised robot joint-angle space.
        """
        self.eval()
        if self._goal_emb is None:
            raise RuntimeError(
                "JEPAPolicy has no goal. Call set_goal() or set "
                "config.goal_image_path before inference."
            )

        z_t = self._encode_obs(batch)               # (1, D)
        goal = self._goal_emb.to(z_t.device)        # (1, D)
        mean = self._action_mean.to(z_t.device)     # (action_dim,)
        std = self._action_std.to(z_t.device)       # (action_dim,)

        actions = []
        for i in range(self.config.chunk_size):
            h = torch.tensor(
                [max(1, self._horizon - i)], device=z_t.device, dtype=torch.long
            )
            a_norm = self.gc_idm(z_t, goal, h)      # (1, action_dim)
            actions.append(a_norm * std + mean)

        self._horizon = max(1, self._horizon - self.config.chunk_size)
        return torch.stack(actions, dim=1)          # (1, chunk_size, action_dim)

    @torch.no_grad()
    def select_action(
        self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        """Return one (1, action_dim) action; replan when queue empties."""
        self.eval()
        if not self._action_queue:
            chunk = self.predict_action_chunk(batch)[0]     # (chunk_size, action_dim)
            for i in range(chunk.shape[0]):
                self._action_queue.append(chunk[i])
        return self._action_queue.popleft().unsqueeze(0)    # (1, action_dim)

    # ── Serialisation ─────────────────────────────────────────────────────

    def _save_pretrained(self, save_directory: Path) -> None:  # type: ignore[override]
        """Save config + GC-IDM weights. World model saved separately by train_lewm.py."""
        self.config._save_pretrained(save_directory)
        torch.save(self.gc_idm.state_dict(), save_directory / "gc_idm.pt")
        if self.config.resolved_world_model_path:
            (save_directory / "WORLD_MODEL_PATH").write_text(
                str(self.config.resolved_world_model_path)
            )

    @classmethod
    def _load_as_safetensor(cls, model, model_file, map_location, strict):  # type: ignore[override]
        # GC-IDM is loaded in __init__ via _load_gc_idm(); skip safetensors loading here.
        return model
