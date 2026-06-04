"""Configuration for the JEPA + GC-IDM policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig


@PreTrainedConfig.register_subclass("jepa")
@dataclass
class JEPAConfig(PreTrainedConfig):
    """Configuration for :class:`JEPAPolicy`.

    Two-stage setup:
    - Stage 1 trains the JEPA world model via ``train_lewm.py``.
    - Stage 2 trains the GC-IDM amortized planner via ``train_gc_idm.py``.

    At deployment the policy is inference-only: it loads both checkpoints,
    encodes the current and goal observations with the frozen encoder, and
    predicts the next action with a single GC-IDM MLP forward pass.
    """

    # ── World-model checkpoint ──────────────────────────────────────────────
    # Path to a pickled JEPA module saved by ModelObjectCallBack in train_lewm.py
    # (e.g. ``~/.stable_worldmodel/<run_id>/lewm_epoch_100_object.ckpt``).
    world_model_path: str | None = None

    # ── GC-IDM checkpoint ───────────────────────────────────────────────────
    # Path to the GC-IDM state-dict saved by train_gc_idm.py (``gc_idm.pt``).
    gc_idm_path: str | None = None

    # ── Per-column normalizer stats ─────────────────────────────────────────
    # Path to ``lewm_normalizers.pt`` saved alongside the world-model checkpoint.
    normalizers_path: str | None = None

    # ── Goal image ──────────────────────────────────────────────────────────
    # Path to a PNG/JPG of the desired goal configuration.
    goal_image_path: str | None = None

    # ── Camera / image ──────────────────────────────────────────────────────
    # Ordered list of observation.images.* keys to consume (one per camera).
    # The first entry maps to ``pixels``, the second to ``pixels2`` inside the
    # JEPA model; any further entries are silently ignored.
    image_keys: list[str] = field(
        default_factory=lambda: ["observation.images.up", "observation.images.side"]
    )
    img_size: int = 224

    # ── World-model architecture (must match Stage 1 training) ──────────────
    encoder_scale: str = "tiny"
    patch_size: int = 14
    embed_dim: int = 192
    history_size: int = 3

    # Predictor architecture
    predictor_depth: int = 6
    predictor_heads: int = 16
    predictor_mlp_dim: int = 2048
    predictor_dim_head: int = 64
    predictor_dropout: float = 0.1
    predictor_emb_dropout: float = 0.0

    # ── Action space ────────────────────────────────────────────────────────
    action_dim: int = 6       # per-step DOF (e.g. 6 for SO-100)
    frameskip: int = 5        # native frames per predictor step

    # ── GC-IDM architecture ─────────────────────────────────────────────────
    gc_idm_hidden_dim: int = 512
    gc_idm_horizon_dim: int = 64
    max_horizon: int = 50     # H_max for horizon sampling / deployment reset

    # ── Inference ───────────────────────────────────────────────────────────
    # chunk_size=1 → per-step closed-loop (sync engine).
    # chunk_size>1 → RTC engine with cached chunk between re-encodes.
    chunk_size: int = 1
    # Horizon floor: never let the deployment horizon decay below this. At h=1
    # GC-IDM is trained to mean "reach the goal in a single step" → a large,
    # aggressive command. Holding a floor keeps motions measured for the whole
    # episode instead of collapsing to a 1-step jump after `max_horizon` ticks.
    horizon_floor: int = 1

    # ── LeRobot boilerplate ─────────────────────────────────────────────────
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=1e-4, weight_decay=1e-4)

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        return

    @property
    def observation_delta_indices(self) -> list[int] | None:
        return None

    @property
    def action_delta_indices(self) -> list[int] | None:
        return None

    @property
    def reward_delta_indices(self) -> None:
        return None

    # Resolved path helpers
    @property
    def resolved_world_model_path(self) -> Path | None:
        return Path(self.world_model_path).expanduser() if self.world_model_path else None

    @property
    def resolved_gc_idm_path(self) -> Path | None:
        return Path(self.gc_idm_path).expanduser() if self.gc_idm_path else None

    @property
    def resolved_normalizers_path(self) -> Path | None:
        return Path(self.normalizers_path).expanduser() if self.normalizers_path else None

    @property
    def resolved_goal_image_path(self) -> Path | None:
        return Path(self.goal_image_path).expanduser() if self.goal_image_path else None

    @property
    def effective_action_dim(self) -> int:
        """Dimension of one action chunk: frameskip × per-step DOF."""
        return self.frameskip * self.action_dim
