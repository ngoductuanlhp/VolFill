"""
LatentTUDFVAE: 3D convolutional KL-VAE for fine TUDF compression.

Compresses (B, 1, 256, 256, 256) fine TUDF → (B, latent_channels, 32, 32, 32)
latent cube via 3 stride-2 downsamples, then decodes back with 3 upsamples.

Architecture ported from TRELLIS SparseStructureEncoder/Decoder
(third_party/trellis/models/sparse_structure_vae.py) but:
  - Self-contained (no TRELLIS import deps)
  - Adapted for dense TUDF (not sparse occupancy)
  - Extended to 256³ → 32³ compression (vs TRELLIS 64³ → 8³)
  - Uses standard PyTorch BCDHW tensor convention throughout

Default encoder channels [32, 64, 128, 256] with 3 downsamples:
  256³ → 128³ → 64³ → 32³
Decoder mirrors encoder in reverse.

Usage:
    vae = LatentTUDFVAE(latent_channels=8)
    z, mean, logvar = vae.encode(x, sample_posterior=True, return_stats=True)
    recon = vae.decode(z)
    kl = vae.kl_loss(mean, logvar)
"""

from __future__ import annotations

import math
from typing import List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpointing


# ---------------------------------------------------------------------------
# fp16 conversion helpers (mirrors TRELLIS convert_module_to_f16/f32)
# Converts only Conv3d/Linear weights; norm layers stay in fp32 for stability.
# ---------------------------------------------------------------------------

def _convert_module_to_f16(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv3d, nn.Linear)):
        m.weight.data = m.weight.data.half()
        if m.bias is not None:
            m.bias.data = m.bias.data.half()


def _convert_module_to_f32(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv3d, nn.Linear)):
        m.weight.data = m.weight.data.float()
        if m.bias is not None:
            m.bias.data = m.bias.data.float()


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

class ChannelLayerNorm32(nn.LayerNorm):
    """LayerNorm over the channel dimension, computing in fp32."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W) — permute to (B, D, H, W, C) for LayerNorm over C
        x = x.permute(0, 2, 3, 4, 1)
        x = super().forward(x.float()).to(x.dtype)
        return x.permute(0, 4, 1, 2, 3).contiguous()


def _norm_layer(norm_type: str, channels: int) -> nn.Module:
    if norm_type == "group":
        groups = min(32, channels)
        while channels % groups != 0:
            groups //= 2
        return nn.GroupNorm(groups, channels)
    elif norm_type == "layer":
        return ChannelLayerNorm32(channels)
    raise ValueError(f"Unknown norm_type: {norm_type}")


# ---------------------------------------------------------------------------
# Pixel-shuffle upsampling (BCDHW convention)
# ---------------------------------------------------------------------------

def pixel_shuffle_3d(x: torch.Tensor, scale_factor: int) -> torch.Tensor:
    """
    3D sub-pixel convolution (pixel-shuffle) for BCDHW tensors.
    Input:  (B, C*sf³, D, H, W)
    Output: (B, C, D*sf, H*sf, W*sf)
    """
    B, C, D, H, W = x.shape
    sf = scale_factor
    C_ = C // (sf ** 3)
    x = x.reshape(B, C_, sf, sf, sf, D, H, W)
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4)   # B, C_, D, sf, H, sf, W, sf
    x = x.reshape(B, C_, D * sf, H * sf, W * sf)
    return x


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResBlock3d(nn.Module):
    """
    Pre-norm 3D residual block (no timestep conditioning — pure reconstruction).
    Matches TRELLIS SparseStructureVAE ResBlock3d style.
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
        self.conv1 = nn.Conv3d(channels, out_ch, 3, padding=1)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        # Zero-init output conv for stable residual training
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        self.skip = nn.Conv3d(channels, out_ch, 1) if channels != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return h + self.skip(x)


class DownsampleBlock3d(nn.Module):
    """Stride-2 convolution: (B, in_ch, D, H, W) → (B, out_ch, D/2, H/2, W/2)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, 2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpsampleBlock3d(nn.Module):
    """Pixel-shuffle upsampling: (B, in_ch, D, H, W) → (B, out_ch, 2D, 2H, 2W)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Output expansion: in_ch → out_ch * 8 then pixel-shuffle by 2
        self.conv = nn.Conv3d(in_channels, out_channels * 8, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return pixel_shuffle_3d(self.conv(x), scale_factor=2)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class LatentTUDFEncoder(nn.Module):
    """
    Encoder: (B, in_channels, R, R, R) → (B, latent_channels*2, R/8, R/8, R/8)

    Produces (mean, logvar) by chunking on dim=1.  For 256³ input with 3
    downsamples this yields 32³ latents.

    Args:
        in_channels:         Input channels (1 for TUDF).
        latent_channels:     VAE latent channels (8 by default).
        channels:            Feature channel list [32, 64, 128, 256].
                             len(channels)-1 downsampling steps.
        num_res_blocks:      ResBlocks per resolution level.
        num_res_blocks_middle: Middle block ResBlocks.
        norm_type:           "layer" or "group".
    """

    def __init__(
        self,
        in_channels: int = 1,
        latent_channels: int = 8,
        channels: List[int] = (32, 64, 128, 256),
        num_res_blocks: int = 2,
        num_res_blocks_middle: int = 2,
        norm_type: str = "layer",
        use_fp16: bool = False,
    ):
        super().__init__()
        channels = list(channels)
        self.use_fp16 = use_fp16
        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.input_layer = nn.Conv3d(in_channels, channels[0], 3, padding=1)

        self.blocks = nn.ModuleList()
        for i, ch in enumerate(channels):
            for _ in range(num_res_blocks):
                self.blocks.append(ResBlock3d(ch, norm_type=norm_type))
            if i < len(channels) - 1:
                self.blocks.append(DownsampleBlock3d(ch, channels[i + 1]))

        self.middle_block = nn.Sequential(*[
            ResBlock3d(channels[-1], norm_type=norm_type)
            for _ in range(num_res_blocks_middle)
        ])

        self.out_layer = nn.Sequential(
            _norm_layer(norm_type, channels[-1]),
            nn.SiLU(),
            nn.Conv3d(channels[-1], latent_channels * 2, 3, padding=1),
        )

        if use_fp16:
            self.convert_to_fp16()

    def convert_to_fp16(self) -> None:
        """Convert internal blocks to fp16; input/output projections stay fp32."""
        self.use_fp16 = True
        self.dtype = torch.float16
        self.blocks.apply(_convert_module_to_f16)
        self.middle_block.apply(_convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """Restore internal blocks to fp32."""
        self.use_fp16 = False
        self.dtype = torch.float32
        self.blocks.apply(_convert_module_to_f32)
        self.middle_block.apply(_convert_module_to_f32)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (mean, logvar), each (B, latent_channels, D, H, W)."""
        h = self.input_layer(x)
        h = h.type(self.dtype)          # cast to fp16 for internal blocks
        for block in self.blocks:
            h = block(h)
        h = self.middle_block(h)
        h = h.type(x.dtype)            # restore to input dtype for out_layer
        h = self.out_layer(h)
        mean, logvar = h.chunk(2, dim=1)
        return mean, logvar


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class LatentTUDFDecoder(nn.Module):
    """
    Decoder: (B, latent_channels, R/8, R/8, R/8) → (B, out_channels, R, R, R)

    Mirror of encoder with channels reversed.

    Args:
        out_channels:        Output channels (1 for TUDF).
        latent_channels:     VAE latent channels.
        channels:            Feature channel list [256, 128, 64, 32] (encoder reversed).
        num_res_blocks:      ResBlocks per resolution level.
        num_res_blocks_middle: Middle block ResBlocks.
        norm_type:           "layer" or "group".
    """

    def __init__(
        self,
        out_channels: int = 1,
        latent_channels: int = 8,
        channels: List[int] = (256, 128, 64, 32),
        num_res_blocks: int = 2,
        num_res_blocks_middle: int = 2,
        norm_type: str = "layer",
        use_fp16: bool = False,
        crop_level: int = 2,
    ):
        super().__init__()
        channels = list(channels)
        self.use_fp16 = use_fp16
        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.input_layer = nn.Conv3d(latent_channels, channels[0], 3, padding=1)

        self.middle_block = nn.Sequential(*[
            ResBlock3d(channels[0], norm_type=norm_type)
            for _ in range(num_res_blocks_middle)
        ])

        self.blocks = nn.ModuleList()
        for i, ch in enumerate(channels):
            for _ in range(num_res_blocks):
                self.blocks.append(ResBlock3d(ch, norm_type=norm_type))
            if i < len(channels) - 1:
                self.blocks.append(UpsampleBlock3d(ch, channels[i + 1]))

        self.out_layer = nn.Sequential(
            _norm_layer(norm_type, channels[-1]),
            nn.SiLU(),
            nn.Conv3d(channels[-1], out_channels, 3, padding=1),
        )

        # Patch-based training split point.
        # After crop_level decoder levels (each = num_res_blocks ResBlocks + 1 Upsample),
        # the feature map is at an intermediate resolution suitable for random cropping.
        # _crop_scale: scale from cropped feature spatial size to the final 256³ output.
        # Example (5-channel config, crop_level=2): 16³→32³→64³ | crop | 64³→128³→256³
        #   _crop_split_idx = 2*(2+1) = 6,  _crop_scale = 2^(5-1-2) = 4
        self._crop_split_idx = crop_level * (num_res_blocks + 1)
        self._crop_scale = 2 ** (len(channels) - 1 - crop_level)

        if use_fp16:
            self.convert_to_fp16()

    def convert_to_fp16(self) -> None:
        """Convert internal blocks to fp16; input/output projections stay fp32."""
        self.use_fp16 = True
        self.dtype = torch.float16
        self.blocks.apply(_convert_module_to_f16)
        self.middle_block.apply(_convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """Restore internal blocks to fp32."""
        self.use_fp16 = False
        self.dtype = torch.float32
        self.blocks.apply(_convert_module_to_f32)
        self.middle_block.apply(_convert_module_to_f32)

    def forward_to_crop(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.dtype]:
        """Run decoder from latent z through the cheap low-res levels, stopping at
        the crop split point (e.g. 64³ for a 5-channel / 16³-latent config).

        Returns:
            h:          Intermediate feature map in self.dtype.
            orig_dtype: z.dtype — pass to forward_from_crop to restore before out_layer.
        """
        h = self.input_layer(z)
        h = h.type(self.dtype)
        if self.training:
            h = checkpointing.checkpoint(self.middle_block, h, use_reentrant=False)
            for block in self.blocks[:self._crop_split_idx]:
                h = checkpointing.checkpoint(block, h, use_reentrant=False)
        else:
            h = self.middle_block(h)
            for block in self.blocks[:self._crop_split_idx]:
                h = block(h)
        return h, z.dtype

    def forward_from_crop(self, h: torch.Tensor, orig_dtype: torch.dtype) -> torch.Tensor:
        """Run the expensive high-res decoder levels on a (possibly cropped) feature map.

        Args:
            h:          Feature map output from forward_to_crop, optionally spatially
                        cropped to a P³ sub-cube before calling this.
            orig_dtype: z.dtype returned by forward_to_crop.
        Returns:
            (B, out_channels, P*_crop_scale, P*_crop_scale, P*_crop_scale) tensor.
        """
        if self.training:
            for block in self.blocks[self._crop_split_idx:]:
                h = checkpointing.checkpoint(block, h, use_reentrant=False)
        else:
            for block in self.blocks[self._crop_split_idx:]:
                h = block(h)
        h = h.type(orig_dtype)
        return self.out_layer(h)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h, orig_dtype = self.forward_to_crop(z)
        return self.forward_from_crop(h, orig_dtype)


# ---------------------------------------------------------------------------
# VAE wrapper
# ---------------------------------------------------------------------------

class LatentTUDFVAE(nn.Module):
    """
    KL-regularized VAE for fine TUDF compression.

    Compresses (B, 1, 256, 256, 256) → (B, latent_channels, 32, 32, 32).

    The encoder and decoder are symmetric: encoder channels [32,64,128,256]
    with 3 stride-2 downsamples; decoder channels [256,128,64,32] with 3
    pixel-shuffle upsamples.

    Args:
        in_channels:           TUDF input channels (1).
        out_channels:          Reconstruction output channels (1).
        latent_channels:       Latent width (default 8, following TRELLIS).
        encoder_channels:      Channel list for encoder levels.
        num_res_blocks:        ResBlocks per level.
        num_res_blocks_middle: ResBlocks in the middle bottleneck.
        norm_type:             "layer" (default) or "group".
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        latent_channels: int = 8,
        encoder_channels: List[int] = (32, 64, 128, 256),
        num_res_blocks: int = 2,
        num_res_blocks_middle: int = 2,
        norm_type: str = "layer",
        use_fp16: bool = False,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        decoder_channels = list(reversed(encoder_channels))

        self.encoder = LatentTUDFEncoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
            channels=encoder_channels,
            num_res_blocks=num_res_blocks,
            num_res_blocks_middle=num_res_blocks_middle,
            norm_type=norm_type,
            use_fp16=use_fp16,
        )
        self.decoder = LatentTUDFDecoder(
            out_channels=out_channels,
            latent_channels=latent_channels,
            channels=decoder_channels,
            num_res_blocks=num_res_blocks,
            num_res_blocks_middle=num_res_blocks_middle,
            norm_type=norm_type,
            use_fp16=use_fp16,
        )

    def convert_to_fp16(self) -> None:
        """Convert encoder and decoder internal blocks to fp16."""
        self.encoder.convert_to_fp16()
        self.decoder.convert_to_fp16()

    def convert_to_fp32(self) -> None:
        """Restore encoder and decoder internal blocks to fp32."""
        self.encoder.convert_to_fp32()
        self.decoder.convert_to_fp32()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(
        self,
        x: torch.Tensor,
        sample_posterior: bool = True,
        return_stats: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode x to latent z.

        Args:
            x:                 (B, 1, 256, 256, 256) normalized TUDF.
            sample_posterior:  If True, sample z ~ N(mean, std); else z = mean.
            return_stats:      If True, return (z, mean, logvar).

        Returns:
            z or (z, mean, logvar).
        """
        mean, logvar = self.encoder(x)
        logvar = logvar.clamp(-30.0, 20.0)
        if sample_posterior:
            std = torch.exp(0.5 * logvar)
            z = mean + std * torch.randn_like(std)
        else:
            z = mean
        if return_stats:
            return z, mean, logvar
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent z → reconstructed TUDF (B, 1, 256, 256, 256)."""
        return self.decoder(z)

    def forward(
        self,
        x: torch.Tensor,
        sample_posterior: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full encode → decode pass.

        Returns:
            (recon, mean, logvar)
            recon: (B, 1, 256, 256, 256)
            mean, logvar: (B, latent_channels, 32, 32, 32)
        """
        z, mean, logvar = self.encode(x, sample_posterior=sample_posterior, return_stats=True)
        recon = self.decode(z)
        return recon, mean, logvar

    @staticmethod
    def kl_loss(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        KL divergence from N(mean, exp(logvar)) to N(0, 1), averaged over batch.
        KL = -0.5 * mean(1 + logvar - mean² - exp(logvar))
        """
        return -0.5 * (1.0 + logvar - mean.pow(2) - logvar.exp()).mean()
