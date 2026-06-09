"""
SparseTUDFEncoderVAE: Asymmetric VAE — sparse encoder + dense decoder.

Encoder: TRELLIS sparse convolutions operate only on the surface band
         (|tudf_norm| < tau), typically <5% of 256³ voxels.  Three
         SparseDownsample(2) steps compress 256³ → 128³ → 64³ → 32³,
         then the bottleneck SparseTensor is converted to a dense
         (B, 2·latent_ch, 32, 32, 32) map for KL sampling.

Decoder: Reuses LatentTUDFDecoder (dense Conv3d) from latent_vae.py.

API: encode / decode / forward / kl_loss — identical to LatentTUDFVAE
     so it can be used as a drop-in in train_latent_vae.py.

Memory note:
  The dense encoder (LatentTUDFVAE) holds a 256³ × 32-ch feature map
  ≈ 512 MB fp32 at the first level.  The sparse encoder avoids that
  because it only allocates features for surface voxels.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from volfill.amodal.model.vae.latent_vae import LatentTUDFDecoder
from volfill.amodal.model.vae.latent_vae_decoder_light import LightLatentTUDFDecoder

        
        
from volfill.utils.timer import CUDATimer
# ---------------------------------------------------------------------------
# TRELLIS sparse module loader
# Bypasses third_party/trellis/__init__.py which requires 'easydict' for
# pipeline classes we don't need.
# ---------------------------------------------------------------------------

def _load_trellis_sparse():
    """
    Import third_party/trellis/modules/sparse without triggering the broken
    trellis top-level __init__.py.

    Returns the `third_party.trellis.modules.sparse` package, which exposes:
        SparseTensor, SparseConv3d, SparseDownsample, SparseLinear,
        SparseSiLU, SparseLayerNorm32
    """
    repo_root    = Path(__file__).parents[4]
    trellis_root = repo_root / "third_party" / "trellis"
    sparse_dir   = trellis_root / "modules" / "sparse"

    # Ensure repo root is on sys.path so `third_party` resolves as an implicit
    # namespace package (PEP 420 — third_party/__init__.py does not exist).
    # This keeps OTHER third_party.X subpackages (third_party.moge, .vggt, .pi3)
    # importable; we only stub the broken trellis subtree below.
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if "third_party" not in sys.modules:
        import third_party  # noqa: F401  -- populates sys.modules with the namespace pkg

    # Register stub ancestors for the broken trellis subtree only, so that
    # relative imports inside sparse submodules (`from .. import BACKEND`)
    # resolve correctly without triggering trellis/__init__.py.
    for mod_name in [
        "third_party.trellis",
        "third_party.trellis.modules",
    ]:
        if mod_name not in sys.modules:
            stub = types.ModuleType(mod_name)
            stub.__path__ = []         # mark as package
            stub.__package__ = mod_name
            sys.modules[mod_name] = stub

    pkg_name = "third_party.trellis.modules.sparse"
    if pkg_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            pkg_name,
            str(sparse_dir / "__init__.py"),
            submodule_search_locations=[str(sparse_dir)],
        )
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = pkg_name
        mod.__path__    = [str(sparse_dir)]
        sys.modules[pkg_name] = mod
        spec.loader.exec_module(mod)

    return sys.modules[pkg_name]


_sparse = _load_trellis_sparse()

# Bind names locally — trigger the lazy __getattr__ in sparse/__init__.py
SparseTensor     = _sparse.SparseTensor
SparseConv3d     = _sparse.SparseConv3d
SparseDownsample = _sparse.SparseDownsample
SparseLinear     = _sparse.SparseLinear
SparseSiLU       = _sparse.SparseSiLU

# TRELLIS SparseLayerNorm permutes feats to (1, C, n_voxels) before LayerNorm,
# which mis-matches normalized_shape=[C] (last dim would be n_voxels).
# We define a corrected version that normalizes the channel dim of (N, C) feats.
class _SparseLayerNorm32(nn.LayerNorm):
    """Per-voxel channel LayerNorm in fp32. Applied to feats (N, C) directly."""

    def forward(self, x: "SparseTensor") -> "SparseTensor":
        new_feats = super().forward(x.feats.float()).to(x.feats.dtype)
        return x.replace(new_feats)


# ---------------------------------------------------------------------------
# Surface-band densification helper
# ---------------------------------------------------------------------------
def dense_to_sparse(
    tudf: torch.Tensor,
    tau: float = 0.9,
    dilate: int = 1,
    min_voxels: int = 64,
) -> "SparseTensor":
    """
    Convert a dense TUDF volume to a SparseTensor retaining only surface-band voxels.

    Args:
        tudf:        (B, 1, D, H, W) normalized TUDF in [-1, 1].
                     Convention: surface voxels -> -1, empty space -> +1.
        tau:         Threshold: keep voxels where tudf_norm < tau.
                     E.g. tau=0.5 keeps the surface band and near-surface
                     voxels while dropping truly empty space (values near +1).
        dilate:      Morphological dilation radius in voxels (0 = none).
                     Implemented as max-pool with kernel=2*dilate+1.
        min_voxels:  Minimum active voxels per sample; if fewer survive the
                     threshold, widen tau until at least min_voxels are kept.

    Returns:
        SparseTensor with feats (N, 1) and coords (N, 4) as [batch, d, h, w]
        in int32, ordered by batch index (required by spconv).
    """
    # spconv does not support bf16 — cast to fp32 here so this function is safe
    # regardless of the surrounding autocast context (training or eval).
    tudf = tudf.float()

    B, _, D, H, W = tudf.shape
    tudf_vol = tudf[:, 0]          # (B, D, H, W)

    # Build surface-band mask.
    # Normalized TUDF: surface=-1, empty=+1.  Keep voxels below tau to
    # include the surface and near-surface band while dropping empty space.
    mask = tudf_vol < tau          # (B, D, H, W)

    # Optional morphological dilation to close thin surface gaps
    if dilate > 0:
        k = 2 * dilate + 1
        mask = (
            F.max_pool3d(
                mask.float().unsqueeze(1),
                kernel_size=k, stride=1, padding=dilate,
            ).squeeze(1)
            > 0.5
        )

    # Guarantee minimum occupancy per sample (avoids empty SparseTensors).
    # Widen tau to the min_voxels-th smallest value when too few voxels pass.
    for b in range(B):
        n_active = mask[b].sum().item()
        if n_active < min_voxels:
            sorted_vals = tudf_vol[b].reshape(-1).sort().values
            fallback_tau = sorted_vals[min(min_voxels, sorted_vals.numel() - 1)].item()
            mask[b] = tudf_vol[b] < fallback_tau

    # Build (N, 4) int32 coords [batch, d, h, w] — must be contiguous per batch
    coords_list: List[torch.Tensor] = []
    feats_list:  List[torch.Tensor] = []
    for b in range(B):
        ijk = mask[b].nonzero(as_tuple=False).int()   # (n_b, 3)
        b_col = torch.full(
            (ijk.shape[0], 1), b, dtype=torch.int32, device=tudf.device
        )
        coords_list.append(torch.cat([b_col, ijk], dim=1))  # (n_b, 4)
        feats_list.append(tudf_vol[b][mask[b]].unsqueeze(1))  # (n_b, 1)

    coords = torch.cat(coords_list, dim=0)   # (N, 4)
    feats  = torch.cat(feats_list,  dim=0)   # (N, 1)

    return SparseTensor(feats=feats, coords=coords)


# ---------------------------------------------------------------------------
# Sparse residual block (no timestep conditioning)
# ---------------------------------------------------------------------------

class SparseConvResBlock(nn.Module):
    """
    Sparse pre-norm residual block for the encoder pyramid.

    Uses SubMConv3d (stride=1) which preserves the active voxel pattern —
    the output coords are identical to the input coords, so residual add works.

    Layout: norm → SiLU → conv1 → norm → SiLU → zero-init conv2 + skip
    """

    def __init__(self, channels: int, out_channels: Optional[int] = None):
        super().__init__()
        out_ch = out_channels or channels

        self.norm1 = _SparseLayerNorm32(channels)
        self.act1  = SparseSiLU()
        self.conv1 = SparseConv3d(channels, out_ch, kernel_size=3)

        self.norm2 = _SparseLayerNorm32(out_ch)
        self.act2  = SparseSiLU()
        self.conv2 = SparseConv3d(out_ch, out_ch, kernel_size=3)

        # Zero-init output conv for stable residual training
        nn.init.zeros_(self.conv2.conv.weight)
        nn.init.zeros_(self.conv2.conv.bias)

        # Skip: SparseLinear (point-wise) when channels differ, else identity
        self.skip = SparseLinear(channels, out_ch) if channels != out_ch else nn.Identity()

    def forward(self, x: "SparseTensor") -> "SparseTensor":
        h = self.conv1(self.act1(self.norm1(x)))
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.skip(x)


# ---------------------------------------------------------------------------
# Sparse encoder
# ---------------------------------------------------------------------------

class SparseTUDFEncoder(nn.Module):
    """
    Sparse encoder: SparseTensor(in_ch, 256³) → (B, 2·latent_ch, 32³) dense.

    3-level pyramid with SparseDownsample(2):
        level 0: 256³, channels[0]  (ResBlocks → Downsample → Linear proj)
        level 1: 128³, channels[1]
        level 2:  64³, channels[2]
        middle:   32³, channels[3]  (ResBlocks at bottleneck)
        out:      32³, 2·latent_ch  (mean + logvar concatenated on dim=1)

    The SparseTensor at 32³ is converted to a dense tensor with `.dense()`
    and zero-padded if the surface band didn't reach the volume boundary.

    Args:
        in_channels:            Input feature channels (1 for raw TUDF).
        latent_channels:        VAE latent channels (8).
        channels:               Channel list, len = num_levels + 1.
                                [32, 64, 128, 256] → 3 downsamples.
        num_res_blocks:         ResBlocks per pyramid level.
        num_res_blocks_middle:  ResBlocks at the bottleneck.
        target_resolution:      Expected latent spatial size (32).
    """

    def __init__(
        self,
        in_channels: int = 1,
        latent_channels: int = 8,
        channels: List[int] = (32, 64, 128, 256),
        num_res_blocks: int = 2,
        num_res_blocks_middle: int = 2,
        target_resolution: int = 32,
    ):
        super().__init__()
        channels = list(channels)
        self.target_resolution = target_resolution

        # Point-wise input projection
        self.input_proj = SparseLinear(in_channels, channels[0])

        # Encoder: ResBlocks → SparseDownsample(2) → channel proj per level
        self.enc_blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            for _ in range(num_res_blocks):
                self.enc_blocks.append(SparseConvResBlock(channels[i]))
            self.enc_blocks.append(SparseDownsample(2))
            self.enc_blocks.append(SparseLinear(channels[i], channels[i + 1]))

        # Middle blocks at bottleneck (32³)
        self.middle = nn.ModuleList([
            SparseConvResBlock(channels[-1])
            for _ in range(num_res_blocks_middle)
        ])

        # Output projection → mean + logvar
        self.out_norm = _SparseLayerNorm32(channels[-1])
        self.out_act  = SparseSiLU()
        self.out_proj = SparseLinear(channels[-1], latent_channels * 2)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def forward(
        self, x_sparse: "SparseTensor"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_sparse: SparseTensor with feats (N, in_channels), coords (N, 4).
        Returns:
            mean, logvar: each (B, latent_channels, R, R, R)
                          where R = target_resolution.
        """
        # original_dtype = x_sparse.feats.dtype
        # with torch.autocast(device_type=self.device.type, dtype=torch.float32, enabled=(original_dtype == torch.bfloat16)):
        h = self.input_proj(x_sparse)

        for block in self.enc_blocks:
            h = block(h)

        for block in self.middle:
            h = block(h)

        h = self.out_proj(self.out_act(self.out_norm(h)))

        # --- Bridge to dense ---
        # SparseTensor.dense() → (B, 2·latent_ch, D', H', W')
        # D'/H'/W' = max_coord+1, which may be < target_resolution if the
        # surface band doesn't reach the volume boundary.
        dense = h.dense()

        # print(f"encoder dense dtype: {dense.dtype}")

        R = self.target_resolution
        D, H, W = dense.shape[-3], dense.shape[-2], dense.shape[-1]
        if D < R or H < R or W < R:
            dense = F.pad(dense, (0, max(0, R - W),
                                  0, max(0, R - H),
                                  0, max(0, R - D)))
        # Trim in case the sparse volume somehow exceeded target_resolution
        dense = dense[..., :R, :R, :R]

        mean, logvar = dense.chunk(2, dim=1)
        return mean, logvar


# ---------------------------------------------------------------------------
# Asymmetric VAE: sparse encoder + dense decoder
# ---------------------------------------------------------------------------

class SparseTUDFEncoderVAE(nn.Module):
    """
    KL-VAE with a sparse 3D convolutional encoder and a dense decoder.

    The encoder uses TRELLIS SubMConv3d blocks on the surface band of the
    256³ fine TUDF, avoiding expensive dense 256³ convolutions at the first
    encoder level.  The decoder is the standard LatentTUDFDecoder (dense).

    Args:
        in_channels:            TUDF input channels (1).
        out_channels:           Reconstruction output channels (1).
        latent_channels:        Latent width (8).
        encoder_channels:       Channel list [32, 64, 128, 256].
        num_res_blocks:         ResBlocks per encoder/decoder level.
        num_res_blocks_middle:  Middle bottleneck ResBlocks.
        norm_type:              Decoder normalization ("layer" or "group").
        sparse_band_tau:        Keep voxels where tudf_norm < tau (surface=-1, empty=+1).
        sparse_dilate:          Morphological dilation radius (0 = no dilation).
        sparse_min_voxels:      Min active voxels per sample (safety floor).
        light_decoder:          Use LightLatentTUDFDecoder when decoder_type="light".
                                Ignored when decoder_type is set explicitly.
        pointwise_from_level:   First decoder level (0-based) to use pointwise ops
                                (only for "light" decoder).
        decoder_type:           "light" | "dense" | "sparse_gated".
                                "light"        → LightLatentTUDFDecoder (default).
                                "dense"        → LatentTUDFDecoder (full 3×3×3 Conv3d).
                                "sparse_gated" → SparseLatentTUDFDecoder (occ-gated
                                                 sparse 128→256 upsample).
                                When set, overrides the light_decoder flag.
        # sparse_gated-only decoder params:
        sparse_dec_channels:    Feature width in the sparse 256³ stage.
        sparse_dec_num_res_blocks: Refinement blocks at 256³.
        tau_surface:            Voxels with tudf < tau_surface → occupied at 256³.
        occ_threshold:          Sigmoid threshold for inference mask at 128³.
        gt_mask_fixed_active_128: If >0 (sparse_gated only), cap sparse-path active
                                voxels per sample (memory). Full coarse GT is still
                                used for occupancy loss.
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
        sparse_band_tau: float = 0.9,
        sparse_dilate: int = 1,
        sparse_min_voxels: int = 64,
        light_decoder: bool = True,
        pointwise_from_level: int = 2,
        decoder_type: str = "light",
        sparse_dec_channels: int = 32,
        sparse_dec_num_res_blocks: int = 1,
        tau_surface: float = 0.0,
        occ_threshold: float = 0.5,
        gt_mask_fixed_active_128: int = 0,
        gt_mask_fixed_active: int = 0,
        occ_resolution: int = 128,
        crop_level: int = 2,
        # sparse_two_stage extras
        gt_mask_fixed_active_64: int = 0,
        mid_occ_resolution: int = 64,
    ):
        super().__init__()
        self.latent_channels   = latent_channels
        self.sparse_band_tau   = sparse_band_tau
        self.sparse_dilate     = sparse_dilate
        self.sparse_min_voxels = sparse_min_voxels

        encoder_channels = list(encoder_channels)
        decoder_channels = list(reversed(encoder_channels))

        # Derive latent resolution from channel count: each level is stride-2.
        # 4 channels → 3 downsamples → 256/8 = 32; 5 channels → 256/16 = 16.
        target_resolution = 256 // (2 ** (len(encoder_channels) - 1))

        self.encoder = SparseTUDFEncoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
            channels=encoder_channels,
            num_res_blocks=num_res_blocks,
            num_res_blocks_middle=num_res_blocks_middle,
            target_resolution=target_resolution,
        )

        if decoder_type == "sparse_gated":
            # Lazy import to avoid circular dependency

            from volfill.amodal.model.vae.latent_vae_sparse_decoder import SparseLatentTUDFDecoder
            # (SparseLatentTUDFDecoder imports from this module for sparse ops)
            self.decoder = SparseLatentTUDFDecoder(
                out_channels=out_channels,
                latent_channels=latent_channels,
                channels=decoder_channels,
                num_res_blocks=num_res_blocks,
                num_res_blocks_middle=num_res_blocks_middle,
                norm_type=norm_type,
                sparse_channels=sparse_dec_channels,
                sparse_num_res_blocks=sparse_dec_num_res_blocks,
                tau_surface=tau_surface,
                occ_threshold=occ_threshold,
                gt_mask_fixed_active_128=gt_mask_fixed_active_128,
                gt_mask_fixed_active=gt_mask_fixed_active,
                occ_resolution=occ_resolution,
                latent_resolution=target_resolution,
            )
        elif decoder_type == "sparse_full":
            from volfill.amodal.model.vae.latent_vae_sparse_decoder import FullSparseLatentTUDFDecoder
            self.decoder = FullSparseLatentTUDFDecoder(
                out_channels=out_channels,
                latent_channels=latent_channels,
                channels=decoder_channels,
                num_res_blocks=num_res_blocks,
                num_res_blocks_middle=num_res_blocks_middle,
                norm_type=norm_type,
                sparse_channels=sparse_dec_channels,
                sparse_num_res_blocks=sparse_dec_num_res_blocks,
                tau_surface=tau_surface,
                occ_threshold=occ_threshold,
                gt_mask_fixed_active=gt_mask_fixed_active or gt_mask_fixed_active_128,
                latent_resolution=target_resolution,
            )
        elif decoder_type == "sparse_two_stage":
            from volfill.amodal.model.vae.latent_vae_sparse_decoder import TwoStageSparseLatentTUDFDecoder
            self.decoder = TwoStageSparseLatentTUDFDecoder(
                out_channels=out_channels,
                latent_channels=latent_channels,
                channels=decoder_channels,
                num_res_blocks=num_res_blocks,
                num_res_blocks_middle=num_res_blocks_middle,
                norm_type=norm_type,
                sparse_channels=sparse_dec_channels,
                sparse_num_res_blocks=sparse_dec_num_res_blocks,
                tau_surface=tau_surface,
                occ_threshold=occ_threshold,
                gt_mask_fixed_active=gt_mask_fixed_active or gt_mask_fixed_active_128,
                occ_threshold_16=occ_threshold,
                occ_threshold_64=occ_threshold,
                gt_mask_fixed_active_64=gt_mask_fixed_active_64,
                latent_resolution=target_resolution,
                mid_occ_resolution=mid_occ_resolution,
            )
        elif decoder_type == "dense" or (decoder_type == "light" and not light_decoder):
            self.decoder = LatentTUDFDecoder(
                out_channels=out_channels,
                latent_channels=latent_channels,
                channels=decoder_channels,
                num_res_blocks=num_res_blocks,
                num_res_blocks_middle=num_res_blocks_middle,
                norm_type=norm_type,
                crop_level=crop_level,
            )
        else:  # "light" (default)
            self.decoder = LightLatentTUDFDecoder(
                out_channels=out_channels,
                latent_channels=latent_channels,
                channels=decoder_channels,
                num_res_blocks=num_res_blocks,
                num_res_blocks_middle=num_res_blocks_middle,
                norm_type=norm_type,
                pointwise_from_level=pointwise_from_level,
            )

    # ------------------------------------------------------------------
    # Public API — same as LatentTUDFVAE
    # ------------------------------------------------------------------

    def encode(
        self,
        x,
        sample_posterior: bool = True,
        return_stats: bool = False,
    ):
        """
        Encode x to latent z using the sparse encoder.

        Args:
            x:                 Either a dense (B, 1, 256, 256, 256) TUDF tensor
                               *or* a pre-computed SparseTensor (from the
                               dataloader's SparseTUDFMixin.collate_fn path).
                               Passing a SparseTensor skips the GPU-side
                               dense_to_sparse conversion.
            sample_posterior:  Sample z ~ N(mean, std) if True; else z = mean.
            return_stats:      Return (z, mean, logvar) if True.
        """
        if isinstance(x, SparseTensor):
            # Pre-computed in the dataloader — use directly
            x_sparse = x
        else:
            x_sparse = dense_to_sparse(x, self.sparse_band_tau,
                                       self.sparse_dilate, self.sparse_min_voxels)
        mean, logvar    = self.encoder(x_sparse)
        logvar          = logvar.clamp(-30.0, 20.0)

        if sample_posterior:
            std = torch.exp(0.5 * logvar)
            z   = mean + std * torch.randn_like(std)
        else:
            z = mean

        if return_stats:
            return z, mean, logvar
        return z

    def decode(self, z: torch.Tensor, **decoder_kwargs) -> torch.Tensor:
        """Decode latent z → reconstructed TUDF (B, 1, 256, 256, 256)."""

        dec_out = self.decoder(z, **decoder_kwargs)
        return dec_out

    def forward(
        self,
        x,
        sample_posterior: bool = True,
        **decoder_kwargs,
    ):
        """
        Full encode → decode pass.

        x may be a dense (B,1,256³) tensor or a pre-computed SparseTensor
        (from SparseTUDFMixin.collate_fn).  Extra keyword arguments are
        forwarded to the decoder.  For SparseLatentTUDFDecoder this includes
        tudf_gt, use_gt_mask, return_dense_256.  For dense/light decoders,
        pass no kwargs.

        Returns:
            (dec_out, mean, logvar)

            Dense/light decoder:
              dec_out = (B, 1, 256, 256, 256) tensor

            SparseLatentTUDFDecoder:
              dec_out = (pred_sparse, coords_256, logits_occ_128)
                     or (pred_sparse, coords_256, logits_occ_128, dense_256)
                        when decoder_kwargs contains return_dense_256=True
        """
        z, mean, logvar = self.encode(x, sample_posterior=sample_posterior,
                                      return_stats=True)

        # with CUDATimer(measure_memory=True):
        dec_out = self.decoder(z, **decoder_kwargs)
        return dec_out, mean, logvar

    @staticmethod
    def kl_loss(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL(N(mean, exp(logvar)) ‖ N(0,1)), averaged over all elements."""
        return -0.5 * (1.0 + logvar - mean.pow(2) - logvar.exp()).mean()


# ---------------------------------------------------------------------------
# Quick smoke test (run with: python -m volfill.amodal.model.vae.latent_vae_sparse_encoder)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    os.environ.setdefault("SPARSE_BACKEND", "spconv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Small resolution for fast test
    R = 64   # use 64³ instead of 256³ so the test finishes quickly
    B = 1

    vae = SparseTUDFEncoderVAE(
        latent_channels=8,
        encoder_channels=[16, 32, 64, 128],
        num_res_blocks=1,
        num_res_blocks_middle=1,
        sparse_band_tau=0.9,
        sparse_dilate=1,
    ).to(device)

    total = sum(p.numel() for p in vae.parameters()) / 1e6
    print(f"SparseTUDFEncoderVAE params: {total:.1f}M")

    x = torch.randn(B, 1, R, R, R, device=device)
    print(f"Input shape: {x.shape}")

    with torch.no_grad():
        recon, mean, logvar = vae(x)
    print(f"Recon:   {recon.shape}")
    print(f"Mean:    {mean.shape}")
    print(f"Logvar:  {logvar.shape}")
    print(f"KL loss: {SparseTUDFEncoderVAE.kl_loss(mean, logvar).item():.4f}")
    print("Smoke test passed.")
