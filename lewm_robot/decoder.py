"""Lightweight image decoder for JEPA CLS embeddings.

Maps 192-dim CLS tokens back to images for representation quality inspection.
Architecture mirrors MAE's decoder: expand CLS to a patch grid, decode each
patch with a shared linear head, then fold patches into the full image.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class JEPADecoder(nn.Module):
    """CLS embedding → reconstructed image.

    Args:
        embed_dim:   Dimension of the JEPA CLS embedding (default: 192).
        img_size:    Output image size in pixels (default: 224).
        patch_size:  ViT patch size used during encoding (default: 14).
        decoder_dim: Width of the internal patch representation (default: 256).
    """

    def __init__(
        self,
        embed_dim: int = 192,
        img_size: int = 224,
        patch_size: int = 14,
        decoder_dim: int = 256,
        out_h: int | None = None,
        out_w: int | None = None,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.n = img_size // patch_size          # patches per side (16)
        self.num_patches = self.n * self.n       # total patches (256)
        # Output resolution — defaults to square img_size; bilinear upsample applied if different.
        self._out_h = out_h if out_h is not None else img_size
        self._out_w = out_w if out_w is not None else img_size

        # Expand the single CLS vector into per-patch embeddings.
        # The linear layer acts as a learned spatial expansion (no positional
        # encoding is added — the decoder must infer spatial structure from z).
        self.cls_to_patches = nn.Linear(embed_dim, self.num_patches * decoder_dim)

        # Shared per-patch decoder head.
        pixels_per_patch = 3 * patch_size * patch_size  # 3 × 14 × 14 = 588
        self.patch_head = nn.Sequential(
            nn.LayerNorm(decoder_dim),
            nn.Linear(decoder_dim, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, pixels_per_patch),
        )

        self._decoder_dim = decoder_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a batch of CLS embeddings to images.

        Args:
            z: ``(B, embed_dim)`` tensor of CLS embeddings.

        Returns:
            ``(B, 3, img_size, img_size)`` float tensor in ``[0, 1]``.
        """
        B = z.shape[0]
        # (B, num_patches × decoder_dim) → (B, num_patches, decoder_dim)
        patches = self.cls_to_patches(z).view(B, self.num_patches, self._decoder_dim)
        # (B, num_patches, 3 × patch_size²)
        pixels = self.patch_head(patches)
        # Fold patches into spatial image: (B, n, n, 3, p, p) → (B, 3, H, W)
        pixels = pixels.view(B, self.n, self.n, 3, self.patch_size, self.patch_size)
        img = pixels.permute(0, 3, 1, 4, 2, 5).reshape(B, 3, self.img_size, self.img_size)
        img = torch.sigmoid(img)
        if self._out_h != self.img_size or self._out_w != self.img_size:
            img = F.interpolate(img, size=(self._out_h, self._out_w),
                                mode="bilinear", align_corners=False)
        return img
