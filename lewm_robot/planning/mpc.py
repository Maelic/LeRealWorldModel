"""Sampling-based planners for goal-conditioned MPC with a JEPA world model.

Both planners share the same interface:

    planner.plan(model, info_dict, last_action) -> action_chunk

where:

* ``model`` is a :class:`jepa.JEPA` instance whose ``get_cost`` returns a
  ``(B, S)`` cost tensor (batch ``B=1`` here, ``S = num_samples``).
* ``info_dict`` already contains ``"pixels"`` (the history images,
  ``(1, H, C, h, w)``) and ``"goal"`` (a single goal image, ``(1, C, h, w)``).
* ``last_action`` is the most recent normalized action chunk
  (``(frameskip * action_dim,)``), used to anchor sampling.

The returned ``action_chunk`` has shape ``(frameskip, action_dim)`` and is in
the *same normalized space* the predictor was trained on. The deploy code is
responsible for un-normalising before sending to the robot.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class _PlannerCfg:
    history_size: int
    horizon: int
    frameskip: int
    action_dim: int
    num_samples: int
    sample_std: float
    action_low: torch.Tensor | None
    action_high: torch.Tensor | None
    device: torch.device


class RandomShootingPlanner:
    """Vanilla random-shooting MPC.

    Samples ``num_samples`` action sequences of length ``history_size + horizon``
    from a Gaussian centred on ``last_action``, evaluates them all in one batched
    forward through ``model.get_cost``, and returns the first chunk-step of the
    best sequence.
    """

    def __init__(
        self,
        history_size: int,
        horizon: int,
        frameskip: int,
        action_dim: int,
        num_samples: int = 256,
        sample_std: float = 0.5,
        action_low: torch.Tensor | None = None,
        action_high: torch.Tensor | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.cfg = _PlannerCfg(
            history_size=history_size,
            horizon=horizon,
            frameskip=frameskip,
            action_dim=action_dim,
            num_samples=num_samples,
            sample_std=sample_std,
            action_low=action_low.to(device) if action_low is not None else None,
            action_high=action_high.to(device) if action_high is not None else None,
            device=torch.device(device),
        )

    def _sample(self, last_action: torch.Tensor) -> torch.Tensor:
        """Return ``(1, num_samples, T, frameskip*action_dim)`` candidates."""
        c = self.cfg
        T = c.history_size + c.horizon
        chunk_dim = c.frameskip * c.action_dim
        mean = last_action.to(c.device).reshape(1, 1, 1, chunk_dim)
        noise = torch.randn(
            (1, c.num_samples, T, chunk_dim),
            device=c.device,
            dtype=last_action.dtype,
        ) * c.sample_std
        candidates = mean + noise
        if c.action_low is not None and c.action_high is not None:
            low = c.action_low.repeat(c.frameskip).reshape(1, 1, 1, chunk_dim)
            high = c.action_high.repeat(c.frameskip).reshape(1, 1, 1, chunk_dim)
            candidates = torch.minimum(torch.maximum(candidates, low), high)
        return candidates

    @torch.no_grad()
    def plan(self, model, info_dict: dict, last_action: torch.Tensor) -> torch.Tensor:
        candidates = self._sample(last_action)
        cost = model.get_cost(info_dict, candidates)  # (1, S)
        best = cost.argmin(dim=1).item()
        # Take the first action *chunk* after the history window.
        chunk = candidates[0, best, self.cfg.history_size, :]
        return chunk.reshape(self.cfg.frameskip, self.cfg.action_dim).cpu()


class CEMPlanner:
    """Cross-entropy method MPC.

    Iteratively refits a Gaussian to the top-``num_elites`` lowest-cost
    candidates. Falls back to random shooting when ``num_iters == 1``.
    """

    def __init__(
        self,
        history_size: int,
        horizon: int,
        frameskip: int,
        action_dim: int,
        num_samples: int = 256,
        num_elites: int = 32,
        num_iters: int = 3,
        init_std: float = 0.5,
        min_std: float = 0.05,
        action_low: torch.Tensor | None = None,
        action_high: torch.Tensor | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        if num_elites > num_samples:
            raise ValueError("num_elites must be <= num_samples")
        self.cfg = _PlannerCfg(
            history_size=history_size,
            horizon=horizon,
            frameskip=frameskip,
            action_dim=action_dim,
            num_samples=num_samples,
            sample_std=init_std,
            action_low=action_low.to(device) if action_low is not None else None,
            action_high=action_high.to(device) if action_high is not None else None,
            device=torch.device(device),
        )
        self.num_elites = num_elites
        self.num_iters = num_iters
        self.min_std = min_std

    @torch.no_grad()
    def plan(self, model, info_dict: dict, last_action: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        T = c.history_size + c.horizon
        chunk_dim = c.frameskip * c.action_dim
        device = c.device

        # Per-timestep Gaussian, broadcast over the batch + sample dims.
        # Shapes: mean / std are (1, 1, T, chunk_dim); noise is (1, S, T, chunk_dim).
        anchor = last_action.to(device).reshape(1, 1, 1, chunk_dim)
        mean = anchor.expand(1, 1, T, chunk_dim).clone()
        std = torch.full_like(mean, c.sample_std)

        best_seq = None
        for _ in range(self.num_iters):
            noise = torch.randn(
                (1, c.num_samples, T, chunk_dim), device=device, dtype=mean.dtype
            )
            candidates = mean + noise * std
            if c.action_low is not None and c.action_high is not None:
                low = c.action_low.repeat(c.frameskip).reshape(1, 1, 1, chunk_dim)
                high = c.action_high.repeat(c.frameskip).reshape(1, 1, 1, chunk_dim)
                candidates = torch.minimum(torch.maximum(candidates, low), high)

            cost = model.get_cost(info_dict, candidates)  # (1, S)
            order = cost[0].argsort()
            elite_idx = order[: self.num_elites]
            elites = candidates[0, elite_idx]  # (K, T, chunk_dim)
            mean = elites.mean(dim=0).reshape(1, 1, T, chunk_dim)
            std = elites.std(dim=0).reshape(1, 1, T, chunk_dim).clamp_min(self.min_std)
            best_seq = elites[0]  # lowest-cost elite

        assert best_seq is not None
        chunk = best_seq[c.history_size]
        return chunk.reshape(c.frameskip, c.action_dim).cpu()
