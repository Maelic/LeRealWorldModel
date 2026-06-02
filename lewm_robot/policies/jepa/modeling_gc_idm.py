"""Goal-Conditioned Inverse Dynamics Model (GC-IDM).

Architecture from arXiv 2605.08732 "Latent Geometry Beyond Search:
Amortizing Planning in World Models". A 3-layer MLP with AdaLN-Zero
horizon conditioning that maps (z_t, z_goal, h) → a_t, replacing
expensive CEM/MPPI search with a single forward pass at inference.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding for scalar horizons.

    Encodes an integer horizon h ∈ [0, max_horizon] into a
    ``dim``-dimensional vector using sin/cos at varying frequencies.
    """

    def __init__(self, dim: int, max_horizon: int = 100) -> None:
        super().__init__()
        self.dim = dim
        # Build encoding table: (max_horizon+1, dim)
        position = torch.arange(max_horizon + 1, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
        )
        enc = torch.zeros(max_horizon + 1, dim)
        enc[:, 0::2] = torch.sin(position * div_term)
        enc[:, 1::2] = torch.cos(position * div_term[: dim // 2])
        self.register_buffer("enc", enc)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: (B,) long tensor of horizon values in [0, max_horizon]
        returns: (B, dim)
        """
        return self.enc[h.long()]


class GCIDM(nn.Module):
    """Goal-Conditioned Inverse Dynamics Model.

    Maps (z_t, z_goal, h) → a_t via a 3-layer MLP with AdaLN-Zero
    horizon conditioning (arXiv 2605.08732).

    Args:
        latent_dim:  Dimension D of encoder embeddings z_t and z_goal.
        action_dim:  Output action dimension (typically frameskip × DOF).
        hidden_dim:  Width of each MLP hidden layer (default 512).
        horizon_dim: Dimension of the sinusoidal horizon encoding (default 64).
        max_horizon: Maximum horizon value for the encoding table.
        dropout:     Dropout probability on hidden activations (default 0.1).
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        horizon_dim: int = 64,
        max_horizon: int = 100,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.horizon_embed = SinusoidalEncoding(horizon_dim, max_horizon)

        # AdaLN-Zero: produce shift/scale for both hidden layers from horizon
        # 4 × hidden_dim = shift1, scale1, shift2, scale2
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(horizon_dim, 4 * hidden_dim, bias=True),
        )
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

        self.fc1 = nn.Linear(2 * latent_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(
        self,
        z_t: torch.Tensor,
        z_goal: torch.Tensor,
        horizon: torch.Tensor,
    ) -> torch.Tensor:
        """
        z_t, z_goal: (B, D) encoder embeddings
        horizon:     (B,) integer steps remaining to goal
        returns:     (B, action_dim)
        """
        h_enc = self.horizon_embed(horizon)             # (B, horizon_dim)
        shifts_scales = self.adaLN(h_enc)               # (B, 4*hidden_dim)
        shift1, scale1, shift2, scale2 = shifts_scales.chunk(4, dim=-1)

        x = torch.cat([z_t, z_goal], dim=-1)            # (B, 2D)

        x = self.norm1(self.fc1(x))
        x = self.drop(self.act(x * (1.0 + scale1) + shift1))

        x = self.norm2(self.fc2(x))
        x = self.drop(self.act(x * (1.0 + scale2) + shift2))

        return self.fc3(x)                              # (B, action_dim)
