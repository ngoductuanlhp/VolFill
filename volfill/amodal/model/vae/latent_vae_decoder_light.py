"""
LightLatentTUDFDecoder: memory-efficient decoder for the 256³ TUDF VAE.

The standard LatentTUDFDecoder uses 3×3×3 Conv3d at every level.  At 128³
and 256³ those spatial convolutions require large intermediate activation
buffers (128³ × 64ch × 27-neighbor gather ≈ several GB fp32).

This decoder switches to pointwise Linear layers (equivalent to 1×1×1 conv)
starting from a configurable level, eliminating the spatial-neighborhood
gather cost at large resolutions.  Spatial coherence at those levels is
provided solely by the pixel-shuffle upsample that precedes each level.

Architecture for channels = [256, 128, 64, 32], pointwise_from_level = 2:

    32³  → ResBlock3d(256)  × N  → PixelShuffle → 64³
    64³  → ResBlock3d(128)  × N  → PixelShuffle → 128³
    128³ → LinearResBlock3d(64)  × N  → PixelShuffle → 256³   ← pointwise
    256³ → LinearResBlock3d(32)  × N                           ← pointwise
          → out_layer → (B, 1, 256, 256, 256)

API: identical to LatentTUDFDecoder — use as drop-in replacement.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from volfill.amodal.model.vae.latent_vae import (
    ResBlock3d,
    UpsampleBlock3d,
    _norm_layer,
    pixel_shuffle_3d,
)


# ---------------------------------------------------------------------------
# Pointwise building blocks
# ---------------------------------------------------------------------------

class LinearResBlock3d(nn.Module):
    """
    Pre-norm residual block using pointwise (1×1×1) convolutions.

    Identical layout to ResBlock3d but conv kernels are size 1, so there
    is no spatial mixing — each voxel is processed independently.  This
    cuts activation memory by ~27× vs. 3×3×3 Conv3d at the same resolution.

    Norm → SiLU → Linear → Norm → SiLU → zero-init Linear  +  skip Linear
    """

    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        norm_type: str = "layer",
    ):
        super().__init__()
        out_ch = out_channels or channels
        self.norm1 = _norm_layer(norm_type, channels)
        self.norm2 = _norm_layer(norm_type, out_ch)
        # 1×1×1 conv = per-voxel linear transform, no spatial gather
        self.fc1 = nn.Conv3d(channels, out_ch, 1)
        self.fc2 = nn.Conv3d(out_ch, out_ch, 1)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        self.skip = nn.Conv3d(channels, out_ch, 1) if channels != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(x))
        h = self.fc1(h)
        h = F.silu(self.norm2(h))
        h = self.fc2(h)
        return h + self.skip(x)


class LinearUpsampleBlock3d(nn.Module):
    """
    Pixel-shuffle upsampling with a pointwise (1×1×1) expansion conv.

    The pixel-shuffle rearrangement provides spatial structure; the preceding
    conv is pointwise to avoid a dense 3×3×3 gather on the large input map.

    (B, in_ch, D, H, W) → (B, out_ch, 2D, 2H, 2W)
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels * 8, 1)  # k=1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return pixel_shuffle_3d(self.conv(x), scale_factor=2)


# ---------------------------------------------------------------------------
# Light decoder
# ---------------------------------------------------------------------------

class LightLatentTUDFDecoder(nn.Module):
    """
    Memory-efficient decoder for 32³ latent → 256³ TUDF reconstruction.

    Drop-in replacement for LatentTUDFDecoder.  Levels 0..pointwise_from_level-1
    use standard 3×3×3 ResBlock3d; levels from pointwise_from_level onward use
    LinearResBlock3d (1×1×1 conv) and LinearUpsampleBlock3d.

    Args:
        out_channels:           Output channels (1 for TUDF).
        latent_channels:        VAE latent channels (must match encoder).
        channels:               Channel list, largest first: [256, 128, 64, 32].
                                len(channels)-1 upsample steps → 256³ output.
        num_res_blocks:         ResBlocks per level.
        num_res_blocks_middle:  ResBlocks in the 32³ bottleneck.
        norm_type:              "layer" or "group".
        pointwise_from_level:   First level index (0-based) to use pointwise
                                ops.  Default 2 → switch at 128³.
    """

    def __init__(
        self,
        out_channels: int = 1,
        latent_channels: int = 8,
        channels: List[int] = (256, 128, 64, 32),
        num_res_blocks: int = 2,
        num_res_blocks_middle: int = 2,
        norm_type: str = "layer",
        pointwise_from_level: int = 2,
    ):
        super().__init__()
        channels = list(channels)

        self.input_layer = nn.Conv3d(latent_channels, channels[0], 3, padding=1)

        self.middle_block = nn.Sequential(*[
            ResBlock3d(channels[0], norm_type=norm_type)
            for _ in range(num_res_blocks_middle)
        ])

        self.blocks = nn.ModuleList()
        for i, ch in enumerate(channels):
            pointwise = (i >= pointwise_from_level)
            res_cls  = LinearResBlock3d if pointwise else ResBlock3d
            ups_cls  = LinearUpsampleBlock3d if pointwise else UpsampleBlock3d

            for _ in range(num_res_blocks):
                self.blocks.append(res_cls(ch, norm_type=norm_type))
            if i < len(channels) - 1:
                self.blocks.append(ups_cls(ch, channels[i + 1]))

        # Output projection: use pointwise (k=1) when the last level is in the
        # pointwise range, otherwise use k=3 for a bit more spatial mixing.
        out_last_level = len(channels) - 1
        out_k = 1 if out_last_level >= pointwise_from_level else 3
        out_pad = 0 if out_k == 1 else 1
        self.out_layer = nn.Sequential(
            _norm_layer(norm_type, channels[-1]),
            nn.SiLU(),
            nn.Conv3d(channels[-1], out_channels, out_k, padding=out_pad),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        from torch.utils.checkpoint import checkpoint
        h = self.input_layer(z)
        if self.training:
            for block in self.middle_block:
                h = checkpoint(block, h, use_reentrant=False)
            for block in self.blocks:
                h = checkpoint(block, h, use_reentrant=False)
        else:
            h = self.middle_block(h)
            for block in self.blocks:
                h = block(h)
        return self.out_layer(h)


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dec_light = LightLatentTUDFDecoder(
        latent_channels=8,
        channels=[256, 128, 64, 32],
        num_res_blocks=2,
        num_res_blocks_middle=2,
        pointwise_from_level=2,
    ).to(device)

    from volfill.amodal.model.vae.latent_vae import LatentTUDFDecoder
    dec_dense = LatentTUDFDecoder(
        latent_channels=8,
        channels=[256, 128, 64, 32],
        num_res_blocks=2,
        num_res_blocks_middle=2,
    ).to(device)

    p_light = sum(p.numel() for p in dec_light.parameters()) / 1e6
    p_dense = sum(p.numel() for p in dec_dense.parameters()) / 1e6
    print(f"LightLatentTUDFDecoder params: {p_light:.1f}M")
    print(f"LatentTUDFDecoder params:      {p_dense:.1f}M")

    z = torch.randn(1, 8, 32, 32, 32, device=device)
    with torch.no_grad():
        out_light = dec_light(z)
        out_dense = dec_dense(z)
    print(f"Light output: {out_light.shape}")
    print(f"Dense output: {out_dense.shape}")
    print("Smoke test passed.")
