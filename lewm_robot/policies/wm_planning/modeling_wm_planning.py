"""Inference-only PreTrainedPolicy that drives an SO-arm with random-shooting MPC.

Loads a pickled JEPA world model produced by lewm's ``train.py`` and exposes
``select_action``/``predict_action_chunk`` so the policy plugs into
``lerobot-rollout`` (with ``--inference.type=sync`` or ``rtc``).

Note on weight serialization
----------------------------
The world model is loaded via ``torch.load(world_model_path)`` rather than
through safetensors. The lewm ``ModelObjectCallBack`` already pickles the full
``JEPA`` module per-epoch; flattening it into safetensors would lose the
auxiliary ``ARPredictor`` / ``Embedder`` / ``MLP`` layout (and the policy's
``forward`` is unused — there's nothing to train here). ``_save_pretrained``
therefore writes only the configuration + a pointer file rather than weights.
"""

from __future__ import annotations

import logging
import sys
from collections import deque
from pathlib import Path
from typing import Unpack

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from lerobot.policies.pretrained import ActionSelectKwargs, PreTrainedPolicy

# Make the lewm repo root importable so `import jepa` resolves when the
# policy is loaded as a third-party plugin.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import lewm_robot  # noqa: F401, E402  (re-exports JEPA for torch.load)

from utils import get_img_preprocessor  # noqa: E402

from lewm_robot.planning import CEMPlanner, RandomShootingPlanner  # noqa: E402
from lewm_robot.policies.wm_planning.configuration_wm_planning import WMPlanningConfig  # noqa: E402

logger = logging.getLogger(__name__)


class WMPlanningPolicy(PreTrainedPolicy):
    """JEPA + random-shooting MPC, packaged as a LeRobot policy."""

    config_class = WMPlanningConfig
    name = "wm_planning"

    def __init__(self, config: WMPlanningConfig, **kwargs) -> None:
        super().__init__(config)
        self.config = config

        if config.resolved_world_model_path is None:
            raise ValueError(
                "WMPlanningConfig.world_model_path must point to a pickled JEPA "
                "checkpoint (e.g. lewm_epoch_*_object.ckpt)."
            )
        if not config.resolved_world_model_path.exists():
            raise FileNotFoundError(config.resolved_world_model_path)

        device = torch.device(config.device or "cpu")
        self._device = device

        logger.info("Loading world model from %s", config.resolved_world_model_path)
        self._world_model = torch.load(
            config.resolved_world_model_path, map_location=device, weights_only=False
        )
        self._world_model.eval().to(device)

        # Normalizers (action mean/std).
        self._action_mean = torch.zeros(config.action_dim)
        self._action_std = torch.ones(config.action_dim)
        if config.resolved_normalizers_path and config.resolved_normalizers_path.exists():
            stats = torch.load(config.resolved_normalizers_path, map_location="cpu")
            if "action" in stats:
                self._action_mean = stats["action"]["mean"]
                self._action_std = stats["action"]["std"]
                logger.info(
                    "Loaded action normalizer (mean shape=%s, std shape=%s)",
                    tuple(self._action_mean.shape),
                    tuple(self._action_std.shape),
                )
        else:
            logger.warning(
                "No normalizers loaded — actions will be sent in raw predictor space. "
                "This is almost certainly wrong; supply --policy.normalizers_path."
            )

        # Image preprocessor (matches training).
        self._img_preprocessor = get_img_preprocessor(
            source="pixels", target="pixels", img_size=config.img_size
        )

        # Goal embedding — encoded once, refreshed by reset().
        self._goal_pixels_pre: Tensor | None = None
        self._goal_emb: Tensor | None = None
        if config.resolved_goal_image_path is not None:
            self._refresh_goal()

        # Planner.
        if config.planner_type == "cem":
            self._planner = CEMPlanner(
                history_size=config.history_size,
                horizon=config.horizon,
                frameskip=config.frameskip,
                action_dim=config.action_dim,
                num_samples=config.num_samples,
                num_elites=config.cem_num_elites,
                num_iters=config.cem_num_iters,
                init_std=config.sample_std,
                device=device,
            )
        else:
            self._planner = RandomShootingPlanner(
                history_size=config.history_size,
                horizon=config.horizon,
                frameskip=config.frameskip,
                action_dim=config.action_dim,
                num_samples=config.num_samples,
                sample_std=config.sample_std,
                device=device,
            )

        # Per-episode rolling state.
        self._history_pixels: deque[Tensor] = deque(maxlen=config.history_size)
        self._history_actions: deque[Tensor] = deque(maxlen=config.history_size)
        self._action_queue: deque[Tensor] = deque(maxlen=config.frameskip)
        self._tick = 0

    # ──────────────────────────── helpers ────────────────────────────

    def _refresh_goal(self) -> None:
        path = self.config.resolved_goal_image_path
        if path is None:
            return
        img = Image.open(path).convert("RGB")
        chw = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).contiguous()
        pre = self._img_preprocessor({"pixels": chw.unsqueeze(0)})["pixels"]
        self._goal_pixels_pre = pre.to(self._device)  # (1, 3, h, w)
        with torch.no_grad():
            out = self._world_model.encode({"pixels": pre.unsqueeze(0).to(self._device)})
        self._goal_emb = out["emb"]

    def _extract_pixels(self, batch: dict[str, Tensor]) -> Tensor:
        """Pull the current camera frame from the policy batch as ``(3, H, W)`` uint8."""
        cam = batch.get(self.config.image_key)
        if cam is None:
            # Fallback: try any observation.images.* key.
            for k, v in batch.items():
                if k.startswith("observation.images."):
                    cam = v
                    break
        if cam is None:
            raise KeyError(
                f"WMPlanningPolicy requires an image input under "
                f"{self.config.image_key!r} (or any observation.images.* key); "
                f"got batch keys {list(batch.keys())}"
            )
        if cam.ndim == 4:  # (B, C, H, W) — drop batch dim
            cam = cam[0]
        # If dispatched through a normalizing processor it may already be float
        # in [0, 1]. Convert to uint8 so the same preprocessor used at training
        # time (ToImage + ImageNet normalize) gives matching outputs.
        if cam.dtype.is_floating_point:
            cam = (cam.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
        return cam

    def _build_info_dict(self) -> dict:
        """Stack history into the (1, 1, H, ...) tensors expected by ``get_cost``."""
        cfg = self.config
        # Pad with the most recent frame until history is full.
        while len(self._history_pixels) < cfg.history_size:
            self._history_pixels.appendleft(self._history_pixels[-1])
        while len(self._history_actions) < cfg.history_size:
            self._history_actions.appendleft(
                torch.zeros(cfg.frameskip * cfg.action_dim, dtype=torch.float32)
            )

        pixels_uint8 = torch.stack(list(self._history_pixels))  # (H, 3, h0, w0)
        pixels_pre = self._img_preprocessor({"pixels": pixels_uint8})["pixels"]
        pixels_pre = pixels_pre.to(self._device)  # (H, 3, h, w)
        action_hist = torch.stack(list(self._history_actions)).to(self._device)
        return {
            "pixels": pixels_pre.unsqueeze(0).unsqueeze(0),  # (1, 1, H, 3, h, w)
            "action": action_hist.unsqueeze(0).unsqueeze(0),  # (1, 1, H, F*ad)
            "goal": self._goal_pixels_pre.unsqueeze(0).unsqueeze(0),  # (1, 1, 1, 3, h, w)
        }

    # ──────────────────────────── PreTrainedPolicy API ────────────────

    def get_optim_params(self) -> dict:
        # Inference-only — no parameters to train.
        return {"params": []}

    def reset(self) -> None:
        self._history_pixels.clear()
        self._history_actions.clear()
        self._action_queue.clear()
        self._tick = 0
        if self._goal_pixels_pre is None:
            self._refresh_goal()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        # No training loss; return a zero loss for compatibility with code
        # paths that call ``forward()`` (e.g. processor pipelines).
        zero = torch.zeros((), device=self._device, requires_grad=False)
        return zero, None

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        """Plan a fresh action chunk and return ``(B=1, frameskip, action_dim)``."""
        self.eval()
        if self._goal_emb is None:
            raise RuntimeError(
                "WMPlanningPolicy has no goal image. Set "
                "`config.goal_image_path` or call _refresh_goal() before inference."
            )

        cam_uint8 = self._extract_pixels(batch).cpu()
        self._history_pixels.append(cam_uint8)

        info = self._build_info_dict()
        last = self._history_actions[-1] if self._history_actions else \
            torch.zeros(self.config.frameskip * self.config.action_dim)
        chunk_norm = self._planner.plan(self._world_model, info, last)
        # (frameskip, action_dim) in normalized space.
        self._history_actions.append(chunk_norm.reshape(-1))
        chunk_real = chunk_norm * self._action_std + self._action_mean
        return chunk_real.unsqueeze(0)  # (1, frameskip, action_dim)

    @torch.no_grad()
    def select_action(
        self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        """Return one action; replan when the per-chunk queue empties."""
        self.eval()

        if not self._action_queue:
            chunk = self.predict_action_chunk(batch)[0]  # (frameskip, action_dim)
            for i in range(chunk.shape[0]):
                self._action_queue.append(chunk[i])
        action = self._action_queue.popleft()
        self._tick += 1
        return action.unsqueeze(0)  # (1, action_dim)

    # safetensors round-trip is a no-op for this policy.

    def _save_pretrained(self, save_directory: Path) -> None:  # type: ignore[override]
        self.config._save_pretrained(save_directory)
        # Drop a pointer file so the world model can be located after reload.
        (save_directory / "WORLD_MODEL_PATH").write_text(
            str(self.config.resolved_world_model_path or "")
        )

    @classmethod
    def _load_as_safetensor(cls, model, model_file, map_location, strict):  # type: ignore[override]
        # Skip safetensors loading — the world model is loaded out-of-band in
        # ``__init__``. Used instead of the parent implementation to avoid the
        # missing-file error when a safetensors blob isn't present.
        return model
