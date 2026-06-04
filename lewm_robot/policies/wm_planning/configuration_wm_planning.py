"""Config for the world-model planning policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim.optimizers import AdamWConfig


@PreTrainedConfig.register_subclass("wm_planning")
@dataclass
class WMPlanningConfig(PreTrainedConfig):
    """Configuration for :class:`WMPlanningPolicy`.

    The policy itself is inference-only — it loads a pre-trained JEPA world
    model from ``world_model_path`` and runs random-shooting MPC against a
    fixed goal image. Training-related fields (optimizer, scheduler) are
    stubbed out because the lewm world model is trained separately via
    ``train.py`` in the lewm repo.
    """

    # --- World model / planner ---------------------------------------------------

    # Absolute path to a pickled JEPA module (e.g.
    # ``~/.stable_worldmodel/<run_id>/lewm_epoch_100_object.ckpt``).
    world_model_path: str | None = None

    # Path to the per-column ``mean / std`` dict saved by ``train.py``
    # (``lewm_normalizers.pt``). Used to map raw robot actions ↔ normalized
    # action space the predictor was trained on.
    normalizers_path: str | None = None

    # Goal image (PNG/JPG). The policy encodes it once at __init__ and reuses
    # the latent across all subsequent calls until ``reset`` is invoked.
    goal_image_path: str | None = None

    # Dataset image key whose distribution we expect at deploy time.
    image_key: str = "observation.images.front"

    # Camera input size after resize.
    img_size: int = 224

    # Predictor history window (must match training-time wm.history_size).
    history_size: int = 3

    # Planning horizon, in chunked-action steps (each step covers ``frameskip``
    # robot ticks).
    horizon: int = 8

    # Number of native robot ticks per predictor step. Match training value.
    frameskip: int = 5

    # Action dimensionality (e.g. 6 for SO-100).
    action_dim: int = 6

    # Random-shooting / CEM knobs.
    planner_type: str = "random"  # "random" or "cem"
    num_samples: int = 256
    sample_std: float = 0.5
    cem_num_elites: int = 32
    cem_num_iters: int = 3

    # Number of robot ticks between re-plans. 1 = replan every tick (slow but
    # reactive); ``frameskip`` = replan once per chunk (recommended start).
    replan_every: int = 5

    # --- LeRobot-required boilerplate -------------------------------------------

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    def get_optimizer_preset(self) -> AdamWConfig:
        # World model is trained externally; this is just to satisfy the ABC.
        return AdamWConfig(lr=1e-4, weight_decay=0.0)

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        # We don't enforce feature presence here — the policy looks them up by
        # name from the input batch at select_action time. Validation happens
        # there to keep this config usable in standalone (non-training) flows.
        return

    @property
    def observation_delta_indices(self) -> list[int] | None:
        # We only need the current observation; history is maintained inside
        # the policy.
        return None

    @property
    def action_delta_indices(self) -> list[int] | None:
        return None

    @property
    def reward_delta_indices(self) -> None:
        return None

    # Helpful resolved paths.

    @property
    def resolved_world_model_path(self) -> Path | None:
        return Path(self.world_model_path).expanduser() if self.world_model_path else None

    @property
    def resolved_normalizers_path(self) -> Path | None:
        return Path(self.normalizers_path).expanduser() if self.normalizers_path else None

    @property
    def resolved_goal_image_path(self) -> Path | None:
        return Path(self.goal_image_path).expanduser() if self.goal_image_path else None
