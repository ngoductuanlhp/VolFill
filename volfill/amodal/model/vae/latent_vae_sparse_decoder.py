"""
SparseLatentTUDFDecoder: occupancy-gated sparse upsample decoder.

Inspired by Seen2Scene (arXiv:2603.28548) dual-head masked sparse VAE —
adapted as a hybrid dense-then-sparse decoder with a configurable split point:

  Dense trunk: z (latent_resolution³) → ... → occ_resolution³
  Occ head at occ_resolution³: predicts which voxels contain surface
  Sparse occ_resolution→256:
    - Training: GT occupancy mask (teacher forcing)
    - Inference: predicted occupancy mask
    → gather dense features at active voxels → SparseTensor
    → SparseSubdivide × log2(256/occ_resolution) (NN-copy, 2x each)
    → SparseConv refinement blocks
    → SparseLinear → TUDF per active 256-voxel

occ_resolution is configurable (e.g. 64 or 128):
  occ_resolution=128, latent_resolution=32: dense 32→64→128, 1 subdivide to 256
  occ_resolution=64,  latent_resolution=32: dense 32→64, 2 subdivides to 256
  occ_resolution=128, latent_resolution=16: dense 16→32→64→128, 1 subdivide to 256

The 256³ stage never allocates a full dense feature map:
  - Training loss: SmoothL1 on sparse pred vs gathered GT at the same indices
  - Occupancy loss: BCE on logits_occ vs GT occupancy at occ_resolution
  - Optional: scatter to dense only for visualization (return_dense_256=True)

API:
    # training
    pred_sparse, coords_256, logits_occ = dec(z, tudf_gt=gt, use_gt_mask=True)
    # inference
    pred_sparse, coords_256, logits_occ = dec(z)
    # val / viz (materializes full 256³ dense pred)
    pred_sparse, coords_256, logits_occ, dense_256 = dec(z, return_dense_256=True)
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpointing

from volfill.amodal.model.vae.latent_vae import ResBlock3d, UpsampleBlock3d, _norm_layer
from volfill.amodal.model.vae.latent_vae_sparse_encoder import (
    _load_trellis_sparse,
    _SparseLayerNorm32,
    SparseConvResBlock,
)

# Load TRELLIS sparse ops (cached — fast if already loaded by encoder)
_sparse = _load_trellis_sparse()

SparseTensor    = _sparse.SparseTensor
SparseLinear    = _sparse.SparseLinear
SparseSiLU      = _sparse.SparseSiLU
SparseSubdivide = _sparse.SparseSubdivide


# ---------------------------------------------------------------------------
# GT-mask and gather utilities (used by both decoder and training loop)
# ---------------------------------------------------------------------------

def get_gt_occ_coarse(
    tudf_gt: torch.Tensor,
    tau_surface: float = 0.0,
    occ_resolution: int = 128,
) -> torch.Tensor:
    """
    Derive GT occupancy at occ_resolution³ from 256³ GT TUDF.

    Any surface voxel in a (stride)³ block makes that coarse block occupied
    (max-pool = "any-pool" philosophy, same as Seen2Scene Sec. 3.2).

    Args:
        tudf_gt:        (B, 1, 256, 256, 256) normalized TUDF (surface=-1, empty=+1).
        tau_surface:    voxels with tudf_gt < tau_surface → occupied.
        occ_resolution: target coarse resolution (must divide 256). Default 128.
    Returns:
        (B, 1, occ_resolution, occ_resolution, occ_resolution) float in {0, 1}.
    """
    stride = 256 // occ_resolution
    occ_256 = (tudf_gt < tau_surface).float()
    return F.max_pool3d(occ_256, kernel_size=stride, stride=stride)


def get_gt_occ_128(tudf_gt: torch.Tensor, tau_surface: float = 0.0) -> torch.Tensor:
    """Backward-compatible alias for get_gt_occ_coarse(..., occ_resolution=128)."""
    return get_gt_occ_coarse(tudf_gt, tau_surface, occ_resolution=128)


def subsample_gt_mask_to_exact_k(
    mask_128: torch.Tensor,
    logits_occ_128: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """
    Per batch item: build a bool mask with exactly ``k`` True voxels at 128³.

    Starts from ``mask_128`` (typically GT occupancy). If there are more than ``k``
    True voxels, randomly keeps ``k``. If fewer, fills the remainder with the
    highest ``logits_occ_128`` among voxels that are still False (same idea as
    ``min_active_voxels`` on the predicted path).

    Args:
        mask_128:       (B, D, H, W) bool — candidate occupied voxels.
        logits_occ_128: (B, 1, D, H, W) — used only when count < k.
        k:              Target number of True voxels per batch item.

    Returns:
        (B, D, H, W) bool with exactly ``k`` True per row (unless D*H*W < k).
    """
    if k <= 0:
        return mask_128
    B, D, H, W = mask_128.shape
    vol = D * H * W
    k = min(k, vol)
    out = torch.zeros_like(mask_128)
    flat_logits = logits_occ_128[:, 0].reshape(B, -1)

    for b in range(B):
        pos = mask_128[b].reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        n = int(pos.numel())

        if n >= k:
            perm = torch.randperm(n, device=mask_128.device)
            sel = pos[perm[:k]]
            out[b].view(-1)[sel] = True
        else:
            if n > 0:
                out[b].view(-1)[pos] = True
            need = k - n
            if need <= 0:
                continue
            neg = (~out[b]).reshape(-1)
            neg_idx = neg.nonzero(as_tuple=False).squeeze(-1)
            if neg_idx.numel() == 0:
                continue
            take = min(need, int(neg_idx.numel()))
            vals = flat_logits[b, neg_idx]
            _, topi = vals.topk(take)
            sel_flat = neg_idx[topi]
            out[b].view(-1)[sel_flat] = True

    return out


def subsample_gt_mask_less_than_k(mask_128: torch.Tensor, k: int) -> torch.Tensor:
    """
    Per batch item: at most ``k`` True voxels; no logit fill when count is low.

    If a sample has more than ``k`` True voxels, randomly keeps ``k``. If it has
    ``k`` or fewer, returns that sample's mask unchanged.

    Args:
        mask_128: (B, D, H, W) bool — candidate occupied voxels.
        k:        Max True voxels per batch item (no-op when ``k <= 0``).

    Returns:
        (B, D, H, W) bool.
    """
    if k <= 0:
        return mask_128
    B, D, H, W = mask_128.shape
    vol = D * H * W
    k = min(k, vol)
    out = torch.zeros_like(mask_128)

    for b in range(B):
        pos = mask_128[b].reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        n = int(pos.numel())

        if n > k:
            perm = torch.randperm(n, device=mask_128.device)
            sel = pos[perm[:k]]
            out[b].view(-1)[sel] = True
        else:
            out[b] = mask_128[b]

    return out


def gather_gt_at_coords(
    tudf_gt: torch.Tensor,
    coords_256: torch.Tensor,
) -> torch.Tensor:
    """
    Gather GT TUDF at sparse 256³ coordinates.

    Avoids materializing a second dense (B,1,256³) prediction tensor during
    training.  Uses advanced indexing: O(N_sparse) memory, not O(256³).

    Args:
        tudf_gt:    (B, 1, 256, 256, 256).
        coords_256: (N, 4) int32 [batch, d, h, w].
    Returns:
        (N,) gathered GT TUDF values.
    """
    b = coords_256[:, 0].long()
    d = coords_256[:, 1].long()
    h = coords_256[:, 2].long()
    w = coords_256[:, 3].long()
    return tudf_gt[b, 0, d, h, w]


# ---------------------------------------------------------------------------
# Sparse decoder
# ---------------------------------------------------------------------------

class SparseLatentTUDFDecoder(nn.Module):
    """
    Hybrid dense-then-sparse TUDF decoder.

    Dense trunk upsample from latent_resolution to occ_resolution, then an
    occupancy head gates a sparse path that subdivides up to 256³.

    The split point (occ_resolution) is configurable:
      - occ_resolution=128, latent_resolution=32 (8x VAE):
          dense 32→64→128, sparse 128→256 (1 subdivide)
      - occ_resolution=64, latent_resolution=32 (8x VAE):
          dense 32→64, sparse 64→128→256 (2 subdivides)
      - occ_resolution=128, latent_resolution=16 (16x VAE):
          dense 16→32→64→128, sparse 128→256 (1 subdivide)

    Args:
        out_channels:           Output channels (1 for TUDF).
        latent_channels:        Must match encoder latent_channels.
        channels:               Channel list largest-first (reversed encoder).
                                channels[i] = feature width at the i-th decoder level.
                                len(channels)-1 determines how many decoder levels exist.
        num_res_blocks:         ResBlocks per dense trunk level.
        num_res_blocks_middle:  ResBlocks at the latent bottleneck.
        norm_type:              Dense trunk normalization ("layer" or "group").
        sparse_channels:        Feature width in the sparse path.
        sparse_num_res_blocks:  SparseConvResBlock count in the sparse path.
        tau_surface:            Training: voxels with tudf_norm < tau_surface
                                are labelled occupied → max-pooled to occ_resolution.
        occ_threshold:          Sigmoid threshold for the inference occupancy mask.
        min_active_voxels:      Safety floor to avoid empty SparseTensors.
        max_active_voxels:      Inference cap on active voxels per sample (0 = uncapped).
        gt_mask_fixed_active:   Training (GT mask): if >0, cap sparse-path ``mask``
                                active voxels per item (memory). Occ loss still uses
                                full coarse GT from TUDF.
        occ_resolution:         Coarse resolution where occupancy is predicted and
                                sparsification begins. Must be a power-of-2 divisor of 256.
                                Default 128. Example: 64 → sparse path covers 64→256.
        latent_resolution:      Spatial size of the input latent z. Default 32.
                                Must satisfy occ_resolution >= latent_resolution.
    """

    is_sparse_gated = True   # sentinel used by train loop to detect decoder type

    def __init__(
        self,
        out_channels: int = 1,
        latent_channels: int = 8,
        channels: List[int] = (256, 128, 64, 32),
        num_res_blocks: int = 2,
        num_res_blocks_middle: int = 2,
        norm_type: str = "layer",
        sparse_channels: int = 32,
        sparse_num_res_blocks: int = 1,
        tau_surface: float = 0.0,
        occ_threshold: float = 0.5,
        min_active_voxels: int = 8,
        max_active_voxels: int = 0,
        gt_mask_fixed_active: int = 0,
        occ_resolution: int = 128,
        latent_resolution: int = 32,
        # Legacy aliases (older configs may use these names)
        max_active_voxels_128: int = 0,
        gt_mask_fixed_active_128: int = 0,
    ):
        super().__init__()
        channels = list(channels)

        # Legacy aliases
        if max_active_voxels_128 and not max_active_voxels:
            max_active_voxels = max_active_voxels_128
        if gt_mask_fixed_active_128 and not gt_mask_fixed_active:
            gt_mask_fixed_active = gt_mask_fixed_active_128


        # How many dense 2x upsample steps to reach occ_resolution from latent_resolution
        num_dense_levels = int(round(math.log2(occ_resolution / latent_resolution)))
        # How many SparseSubdivide (2x each) steps to reach 256 from occ_resolution
        num_sparse_subdivides = int(round(math.log2(256 / occ_resolution)))

        assert occ_resolution == latent_resolution * (2 ** num_dense_levels), (
            f"occ_resolution={occ_resolution} must be a power-of-2 multiple of "
            f"latent_resolution={latent_resolution}"
        )
        assert 256 == occ_resolution * (2 ** num_sparse_subdivides), (
            f"occ_resolution={occ_resolution} must be a power-of-2 divisor of 256"
        )
        assert num_dense_levels < len(channels), (
            f"Need at least {num_dense_levels + 1} channel entries for "
            f"latent_resolution={latent_resolution} → occ_resolution={occ_resolution}, "
            f"got {len(channels)}"
        )

        # Feature width at occ_resolution (trunk output)
        trunk_ch = channels[num_dense_levels]

        self.occ_resolution       = occ_resolution
        self.num_sparse_subdivides = num_sparse_subdivides
        self.tau_surface           = tau_surface
        self.occ_threshold         = occ_threshold
        self.min_active_voxels     = min_active_voxels
        self.max_active_voxels     = max_active_voxels
        self.gt_mask_fixed_active  = gt_mask_fixed_active

        # ------------------------------------------------------------------
        # Dense trunk: latent_resolution → ... → occ_resolution
        # ------------------------------------------------------------------
        self.input_layer = nn.Conv3d(latent_channels, channels[0], 3, padding=1)

        self.middle_block = nn.Sequential(*[
            ResBlock3d(channels[0], norm_type=norm_type)
            for _ in range(num_res_blocks_middle)
        ])

        self.trunk = nn.ModuleList()
        for i in range(num_dense_levels):
            for _ in range(num_res_blocks):
                self.trunk.append(ResBlock3d(channels[i], norm_type=norm_type))
            self.trunk.append(UpsampleBlock3d(channels[i], channels[i + 1]))
        for _ in range(num_res_blocks):             # final level at occ_resolution, no upsample
            self.trunk.append(ResBlock3d(trunk_ch, norm_type=norm_type))

        # ------------------------------------------------------------------
        # Occupancy head at occ_resolution³
        # ------------------------------------------------------------------
        self.occ_head = nn.Sequential(
            _norm_layer(norm_type, trunk_ch),
            nn.SiLU(),
            nn.Conv3d(trunk_ch, 1, 1),
        )

        # ------------------------------------------------------------------
        # Sparse path: occ_resolution → 256 (num_sparse_subdivides × 2x)
        #
        # Channel convention: sparse_channels is the FINAL width (before tudf_head).
        # Each subdivide level uses twice the channels of the next level, following
        # the trunk_ch // 2 rule: sparse_channels = trunk_ch // 2 at the first level.
        #
        # num_sparse_subdivides == 1  (occ_resolution=128, trunk_ch=64):
        #   sparse_in_proj (64→32) → subdivide → sparse_blocks(32) → tudf_head
        #
        # num_sparse_subdivides == 2  (occ_resolution=64, trunk_ch=128):
        #   sparse_in_proj (128→64) → subdivide → sparse_blocks_mid(64)
        #   → sparse_mid_proj (64→32) → subdivide → sparse_blocks(32) → tudf_head
        #
        # Module names for num_sparse_subdivides==1 are identical to the original
        # hardcoded-128 design, preserving checkpoint key compatibility.
        # ------------------------------------------------------------------
        self.subdivide = SparseSubdivide()

        if num_sparse_subdivides == 1:
            self.sparse_in_proj = SparseLinear(trunk_ch, sparse_channels)
            self.sparse_blocks  = nn.ModuleList([
                SparseConvResBlock(sparse_channels)
                for _ in range(sparse_num_res_blocks)
            ])
        else:
            # num_sparse_subdivides == 2
            sparse_ch_mid = sparse_channels * 2        # trunk_ch // 2 = 64
            self.sparse_in_proj    = SparseLinear(trunk_ch, sparse_ch_mid)
            self.sparse_blocks_mid = nn.ModuleList([
                SparseConvResBlock(sparse_ch_mid)
                for _ in range(sparse_num_res_blocks)
            ])
            self.sparse_mid_proj   = SparseLinear(sparse_ch_mid, sparse_channels)
            self.sparse_blocks     = nn.ModuleList([
                SparseConvResBlock(sparse_channels)
                for _ in range(sparse_num_res_blocks)
            ])

        self.tudf_head = nn.Sequential(
            _SparseLayerNorm32(sparse_channels),
            SparseSiLU(),
            SparseLinear(sparse_channels, out_channels),
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_mask(
        self,
        tudf_gt: Optional[torch.Tensor],
        use_gt_mask: bool,
        logits_occ: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Return (mask, gt_occ_float).

        mask         : (B, R, R, R) bool — which voxels enter the sparse path.
        gt_occ_float : (B, 1, R, R, R) float in {0,1} — GT occ for BCE/Dice loss.
                       Returned only during GT-mask mode; None otherwise.

        Training (use_gt_mask=True, tudf_gt provided):
            GT occupancy = max-pool(tudf_gt < tau_surface) from 256 → occ_resolution.
            ``gt_occ_float`` is always that full coarse GT (for occ BCE/Dice). If
            ``gt_mask_fixed_active`` > 0, ``mask`` is subsampled (e.g. cap active
            voxels for memory); the occupancy target is not changed to match ``mask``.
        Inference:
            Predicted occupancy = sigmoid(logits_occ) > occ_threshold.
        Fallback: guarantee at least min_active_voxels per batch item.
        """
        if use_gt_mask and tudf_gt is not None:
            gt_occ_float = get_gt_occ_coarse(
                tudf_gt, self.tau_surface, occ_resolution=self.occ_resolution
            )                                            # (B, 1, R, R, R) float {0,1}
            mask = gt_occ_float.squeeze(1) > 0.5        # (B, R, R, R) bool
            # print(f"gt_occ_float: {gt_occ_float.sum()}")
            if self.gt_mask_fixed_active > 0:
                # Sparse path only: cap active voxels. Keep gt_occ_float as full coarse GT
                # for occupancy loss (do not train occ against the subsampled mask).
                # mask = subsample_gt_mask_to_exact_k(
                #     mask, logits_occ, self.gt_mask_fixed_active
                # )
                mask = subsample_gt_mask_less_than_k(mask, self.gt_mask_fixed_active)
            return mask, gt_occ_float
        else:
            mask = logits_occ.sigmoid().squeeze(1) > self.occ_threshold

            need_clamp = self.min_active_voxels > 0 or self.max_active_voxels > 0

            if need_clamp:
                mask = mask.clone()
                for b in range(mask.shape[0]):
                    flat = logits_occ[b, 0].reshape(-1)
                    n_active = int(mask[b].sum().item())
                    D, H, W = mask.shape[1], mask.shape[2], mask.shape[3]

                    if n_active < self.min_active_voxels:
                        top_k = min(self.min_active_voxels, flat.numel())
                        _, top_idx = flat.topk(top_k)
                        ds = top_idx // (H * W)
                        hs = (top_idx // W) % H
                        ws = top_idx % W
                        mask[b, ds, hs, ws] = True
                    elif self.max_active_voxels > 0 and n_active > self.max_active_voxels:

                        top_k = self.max_active_voxels
                        _, top_idx = flat.topk(top_k)
                        new_mask = torch.zeros_like(mask[b])
                        ds = top_idx // (H * W)
                        hs = (top_idx // W) % H
                        ws = top_idx % W
                        new_mask[ds, hs, ws] = True
                        mask[b] = new_mask

            return mask, None

    @staticmethod
    def _dense_to_sparse(
        h: torch.Tensor,
        mask: torch.Tensor,
    ) -> "SparseTensor":
        """
        Gather dense features at occupied voxels into a SparseTensor.

        Args:
            h:    (B, C, D, H, W) — dense features at occ_resolution
            mask: (B, D, H, W) bool
        Returns:
            SparseTensor with feats (N, C) and coords (N, 4) [batch, d, h, w].
        """
        B, C, D, H, W = h.shape
        coords_list: List[torch.Tensor] = []
        feats_list:  List[torch.Tensor] = []
        for b in range(B):
            ijk = mask[b].nonzero(as_tuple=False).int()          # (n_b, 3)
            b_col = torch.full(
                (ijk.shape[0], 1), b, dtype=torch.int32, device=h.device
            )
            coords_list.append(torch.cat([b_col, ijk], dim=1))   # (n_b, 4)
            feats_list.append(h[b].permute(1, 2, 3, 0)[mask[b]])  # (n_b, C)
        coords = torch.cat(coords_list, dim=0)
        feats  = torch.cat(feats_list,  dim=0)
        return SparseTensor(feats=feats, coords=coords)

    @staticmethod
    def _scatter_to_dense(
        sp: "SparseTensor",
        batch_size: int,
        resolution: int = 256,
        fill_value: float = 1.0,
    ) -> torch.Tensor:
        """
        Scatter sparse 256³ predictions to a dense tensor.

        Inactive voxels receive fill_value (1.0 = empty in normalized TUDF).
        Intended only for visualization / eval; not used during training.
        """
        device = sp.feats.device
        dtype  = sp.feats.dtype
        out = torch.full(
            (batch_size, 1, resolution, resolution, resolution),
            fill_value, device=device, dtype=dtype,
        )
        coords = sp.coords
        b = coords[:, 0].long().clamp(0, batch_size - 1)
        d = coords[:, 1].long().clamp(0, resolution - 1)
        h = coords[:, 2].long().clamp(0, resolution - 1)
        w = coords[:, 3].long().clamp(0, resolution - 1)
        out[b, 0, d, h, w] = sp.feats[:, 0]
        return out

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        z: torch.Tensor,
        *,
        tudf_gt: Optional[torch.Tensor] = None,
        use_gt_mask: bool = True,
        return_dense_256: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """
        Args:
            z:               (B, latent_channels, latent_resolution³)
            tudf_gt:         (B, 1, 256, 256, 256) GT TUDF — required for
                             teacher-forced training mask and gather-based recon.
            use_gt_mask:     True  → build occ_resolution³ mask from GT (training).
                             False → build from predicted logits (inference).
                             Forced False if tudf_gt is None.
            return_dense_256: Scatter sparse pred to dense (B,1,256³) for
                              visualization / evaluation.

        Returns (always):
            pred_tudf_sparse : (N,)           — TUDF at active 256-voxels
            coords_256       : (N, 4)         — int32 [batch, d, h, w]
            logits_occ       : (B,1,R,R,R)    — raw occupancy logits at occ_resolution R
        Appended when return_dense_256=True:
            dense_256        : (B,1,256³)     — scatter with fill=1.0 for empty
        """
        B = z.shape[0]

        # ---- Dense trunk: latent_resolution → occ_resolution ----
        if self.training:
            h = checkpointing.checkpoint(self.input_layer, z, use_reentrant=False)
            h = checkpointing.checkpoint(self.middle_block, h, use_reentrant=False)
            for block in self.trunk:
                h = checkpointing.checkpoint(block, h, use_reentrant=False)
        else:
            h = self.input_layer(z)
            h = self.middle_block(h)
            for block in self.trunk:
                h = block(h)
        h_occ = h                                   # (B, trunk_ch, occ_resolution³)

        # ---- Occupancy head ----
        logits_occ = self.occ_head(h_occ)           # (B, 1, occ_resolution³)

        # ---- Build occupancy mask ----
        # gt_occ_float is non-None only during GT-mask training; reuse it in the
        # return tuple so the training loop can skip a redundant get_gt_occ_coarse call.
        mask_occ, gt_occ_float = self._build_mask(
            tudf_gt, use_gt_mask and tudf_gt is not None, logits_occ
        )


        # ---- Sparse bridge: gather h_occ at active occ_resolution voxels ----
        sp = self._dense_to_sparse(h_occ, mask_occ)  # SparseTensor(trunk_ch)

        # ---- Channel proj then interleaved subdivide + refinement ----
        sp = self.sparse_in_proj(sp)
        if self.num_sparse_subdivides == 1:
            # occ=128: [trunk→32] → subdivide(128→256) → blocks(32) → tudf_head
            sp = self.subdivide(sp)
            for block in self.sparse_blocks:
                sp = block(sp)
        else:
            # occ=64: [trunk→64] → subdivide(64→128) → blocks_mid(64)
            #         → [64→32] → subdivide(128→256) → blocks(32) → tudf_head
            sp = self.subdivide(sp)
            for block in self.sparse_blocks_mid:
                sp = block(sp)
            sp = self.sparse_mid_proj(sp)
            sp = self.subdivide(sp)
            for block in self.sparse_blocks:
                sp = block(sp)

        # ---- TUDF head ----
        sp = self.tudf_head(sp)
        pred_tudf_sparse = sp.feats[:, 0]           # (N,)
        coords_256       = sp.coords                # (N, 4)

        if return_dense_256:
            dense_256 = self._scatter_to_dense(sp, B)
            return pred_tudf_sparse, coords_256, logits_occ, dense_256

        if gt_occ_float is not None:
            # Training with GT mask: include gt_occ so the caller avoids recomputing it.
            return pred_tudf_sparse, coords_256, logits_occ, gt_occ_float

        return pred_tudf_sparse, coords_256, logits_occ


# ---------------------------------------------------------------------------
# Full-sparse decoder: gate at latent_resolution³, subdivide 4× to 256³
# ---------------------------------------------------------------------------

class FullSparseLatentTUDFDecoder(SparseLatentTUDFDecoder):
    """
    Full sparse TUDF decoder that gates at latent_resolution³ (e.g. 16³).

    Unlike SparseLatentTUDFDecoder (which first upsample densely to occ_resolution
    then sparsifies), this decoder:
      1. Runs all dense processing at latent_resolution³ only (no dense upsampling).
      2. Predicts occupancy logits at latent_resolution³.
      3. Gates a sparse path that subdivides log2(256/latent_resolution) times.

    Sparse channel ladder (mapped from decoder_channels):
      channels[0] — dense stem width (e.g. 512 at 16³)
      channels[1] — after first subdivide (e.g. 256 at 32³)
      channels[2] — after second subdivide (e.g. 128 at 64³)
      ...
      sparse_channels — final width at 256³ (e.g. 32)

    Inherits _build_mask, _dense_to_sparse, _scatter_to_dense from
    SparseLatentTUDFDecoder without running the parent __init__.
    """

    is_sparse_gated = True

    def __init__(
        self,
        out_channels: int = 1,
        latent_channels: int = 16,
        channels: List[int] = (512, 256, 128, 64, 32),
        num_res_blocks: int = 2,
        num_res_blocks_middle: int = 2,
        norm_type: str = "layer",
        sparse_channels: int = 32,
        sparse_num_res_blocks: int = 1,
        tau_surface: float = 0.0,
        occ_threshold: float = 0.5,
        min_active_voxels: int = 8,
        max_active_voxels: int = 0,
        gt_mask_fixed_active: int = 0,
        latent_resolution: int = 16,
    ):
        # Bypass SparseLatentTUDFDecoder.__init__ — only inherit its helper methods.
        nn.Module.__init__(self)
        channels = list(channels)

        num_sparse_subdivides = int(round(math.log2(256 / latent_resolution)))
        assert 256 == latent_resolution * (2 ** num_sparse_subdivides), (
            f"latent_resolution={latent_resolution} must be a power-of-2 divisor of 256"
        )

        self.occ_resolution        = latent_resolution   # gate at latent resolution
        self.num_sparse_subdivides = num_sparse_subdivides
        self.tau_surface           = tau_surface
        self.occ_threshold         = occ_threshold
        self.min_active_voxels     = min_active_voxels
        self.max_active_voxels     = max_active_voxels
        self.gt_mask_fixed_active  = gt_mask_fixed_active

        # Dense stem at latent_resolution³ (no upsampling)
        self.input_layer = nn.Conv3d(latent_channels, channels[0], 3, padding=1)
        self.middle_block = nn.Sequential(*[
            ResBlock3d(channels[0], norm_type=norm_type)
            for _ in range(num_res_blocks_middle)
        ])
        self.dense_blocks = nn.Sequential(*[
            ResBlock3d(channels[0], norm_type=norm_type)
            for _ in range(num_res_blocks)
        ])

        # Occupancy head at latent_resolution³
        self.occ_head = nn.Sequential(
            _norm_layer(norm_type, channels[0]),
            nn.SiLU(),
            nn.Conv3d(channels[0], 1, 1),
        )

        # Sparse path: num_sparse_subdivides levels with channel projections.
        # Order: proj (reduce channels) → subdivide (expand spatially) → res blocks.
        # This keeps fewer channels during the spatial expansion, saving memory.
        self.subdivide = SparseSubdivide()
        self.sparse_projs      = nn.ModuleList()
        self.sparse_res_blocks = nn.ModuleList()
        in_ch = channels[0]
        for i in range(num_sparse_subdivides):
            out_ch = channels[i + 1] if (i + 1) < len(channels) else sparse_channels
            self.sparse_projs.append(SparseLinear(in_ch, out_ch))
            self.sparse_res_blocks.append(nn.ModuleList([
                SparseConvResBlock(out_ch)
                for _ in range(sparse_num_res_blocks)
            ]))
            in_ch = out_ch

        self.tudf_head = nn.Sequential(
            _SparseLayerNorm32(in_ch),
            SparseSiLU(),
            SparseLinear(in_ch, out_channels),
        )

    def forward(
        self,
        z: torch.Tensor,
        *,
        tudf_gt: Optional[torch.Tensor] = None,
        use_gt_mask: bool = True,
        return_dense_256: bool = False,
    ):
        B = z.shape[0]

        # Dense stem at latent_resolution³ (no spatial upsample)
        if self.training:
            h = checkpointing.checkpoint(self.input_layer, z, use_reentrant=False)
            h = checkpointing.checkpoint(self.middle_block, h, use_reentrant=False)
            h = checkpointing.checkpoint(self.dense_blocks, h, use_reentrant=False)
        else:
            h = self.input_layer(z)
            h = self.middle_block(h)
            h = self.dense_blocks(h)
        h_occ = h                                   # (B, channels[0], latent_resolution³)

        # Occupancy head at latent_resolution³
        logits_occ = self.occ_head(h_occ)           # (B, 1, latent_resolution³)

        # Build occupancy mask
        mask_occ, gt_occ_float = self._build_mask(
            tudf_gt, use_gt_mask and tudf_gt is not None, logits_occ
        )

        # Sparse bridge: gather latent_resolution³ features at active voxels
        sp = self._dense_to_sparse(h_occ, mask_occ)

        # Sparse path: proj → subdivide → res blocks, repeated num_sparse_subdivides times
        for i in range(self.num_sparse_subdivides):
            sp = self.sparse_projs[i](sp)
            sp = self.subdivide(sp)
            for block in self.sparse_res_blocks[i]:
                sp = block(sp)

        # TUDF head at 256³
        sp = self.tudf_head(sp)
        pred_tudf_sparse = sp.feats[:, 0]           # (N,)
        coords_256       = sp.coords                # (N, 4)

        if return_dense_256:
            dense_256 = self._scatter_to_dense(sp, B)
            return pred_tudf_sparse, coords_256, logits_occ, dense_256

        if gt_occ_float is not None:
            return pred_tudf_sparse, coords_256, logits_occ, gt_occ_float

        return pred_tudf_sparse, coords_256, logits_occ


# ---------------------------------------------------------------------------
# Two-stage sparse decoder: gate at latent_resolution³, prune at mid_occ_resolution³
# ---------------------------------------------------------------------------

class TwoStageSparseLatentTUDFDecoder(SparseLatentTUDFDecoder):
    """
    Two-stage sparse TUDF decoder.

    Stage 1 (coarse recall gate):
      Dense stem at latent_resolution³ → occ16 head (dense) → sparsify at 16³
      → sparse proj/subdivide/resblocks  16 → 32 → 64

    Stage 2 (fine precision prune):
      Sparse occ64 head at mid_occ_resolution³ → prune SparseTensor at 64³
      → sparse proj/subdivide/resblocks  64 → 128 → 256 → TUDF head

    Channel ladder (same convention as FullSparseLatentTUDFDecoder):
      channels[0]                    — dense stem width at latent_resolution³
      channels[1..num_stage1]        — sparse widths at 32³, 64³ (stage 1)
      channels[num_stage1+1..total]  — sparse widths at 128³, 256³ (stage 2)

    Return contract (train / eval via new train_latent_vae_two_stage_sparse.py):
      (pred_tudf_sparse, coords_256, occ_dict)
      where occ_dict = {
          "logits16": (B,1,R16,R16,R16) dense logits at latent_resolution,
          "logits64": SparseTensor with feats (N,1) at mid_occ_resolution,
          "gt16":     (B,1,R16,R16,R16) float {0,1} — present when tudf_gt provided,
          "gt64":     (N,)              float {0,1} — GT occ at sparse 64³ coords,
      }
      With return_dense_256=True, appends dense_256 as the 4th element.

    Inherits _build_mask, _dense_to_sparse, _scatter_to_dense from
    SparseLatentTUDFDecoder without calling parent __init__.
    """

    is_sparse_gated     = True
    is_two_stage_sparse = True

    def __init__(
        self,
        out_channels: int = 1,
        latent_channels: int = 16,
        channels: List[int] = (512, 256, 128, 64, 32),
        num_res_blocks: int = 2,
        num_res_blocks_middle: int = 2,
        norm_type: str = "layer",
        sparse_channels: int = 32,
        sparse_num_res_blocks: int = 1,
        tau_surface: float = 0.0,
        occ_threshold: float = 0.5,
        min_active_voxels: int = 8,
        max_active_voxels: int = 0,
        # Stage-1 gate (dense, at latent_resolution)
        gt_mask_fixed_active: int = 0,
        occ_threshold_16: float = 0.5,
        # Stage-2 prune (sparse, at mid_occ_resolution)
        gt_mask_fixed_active_64: int = 0,
        occ_threshold_64: float = 0.5,
        # Resolutions
        latent_resolution: int = 16,
        mid_occ_resolution: int = 64,
    ):
        # Skip SparseLatentTUDFDecoder.__init__ — inherit helper methods only.
        nn.Module.__init__(self)
        channels = list(channels)

        assert mid_occ_resolution > latent_resolution, (
            f"mid_occ_resolution ({mid_occ_resolution}) must be > latent_resolution ({latent_resolution})"
        )
        num_stage1 = int(round(math.log2(mid_occ_resolution / latent_resolution)))
        num_stage2 = int(round(math.log2(256 / mid_occ_resolution)))
        assert latent_resolution * (2 ** num_stage1) == mid_occ_resolution
        assert mid_occ_resolution * (2 ** num_stage2) == 256

        # Public attributes used by training/eval/val_step probes
        self.occ_resolution        = latent_resolution      # primary gate resolution (compat)
        self.mid_occ_resolution    = mid_occ_resolution
        self.num_stage1            = num_stage1
        self.num_stage2            = num_stage2
        self.tau_surface           = tau_surface
        self.occ_threshold         = occ_threshold_16       # used by _build_mask stage-1
        self.occ_threshold_16      = occ_threshold_16
        self.occ_threshold_64      = occ_threshold_64
        self.min_active_voxels     = min_active_voxels
        self.max_active_voxels     = max_active_voxels
        self.gt_mask_fixed_active  = gt_mask_fixed_active
        self.gt_mask_fixed_active_64 = gt_mask_fixed_active_64

        # ---- Dense stem at latent_resolution³ ----
        self.input_layer  = nn.Conv3d(latent_channels, channels[0], 3, padding=1)
        self.middle_block = nn.Sequential(*[
            ResBlock3d(channels[0], norm_type=norm_type)
            for _ in range(num_res_blocks_middle)
        ])
        self.dense_blocks = nn.Sequential(*[
            ResBlock3d(channels[0], norm_type=norm_type)
            for _ in range(num_res_blocks)
        ])

        # ---- Occupancy head at latent_resolution³ (dense) ----
        self.occ_head = nn.Sequential(
            _norm_layer(norm_type, channels[0]),
            nn.SiLU(),
            nn.Conv3d(channels[0], 1, 1),
        )

        self.subdivide = SparseSubdivide()

        # ---- Unified sparse path: latent_resolution → 256 (all levels) ----
        # Keys match FullSparseLatentTUDFDecoder exactly (sparse_projs.i,
        # sparse_res_blocks.i) so pretrained FullSparse weights load via strict=False
        # without any remapping.  num_stage1 marks where the mid-prune gate sits:
        #   indices 0 .. num_stage1-1   → stage-1  (latent_resolution → mid_occ_resolution)
        #   indices num_stage1 .. total → stage-2  (mid_occ_resolution → 256)
        self.sparse_projs      = nn.ModuleList()
        self.sparse_res_blocks = nn.ModuleList()
        in_ch = channels[0]
        for i in range(num_stage1 + num_stage2):
            out_ch = channels[i + 1] if (i + 1) < len(channels) else sparse_channels
            self.sparse_projs.append(SparseLinear(in_ch, out_ch))
            self.sparse_res_blocks.append(nn.ModuleList([
                SparseConvResBlock(out_ch) for _ in range(sparse_num_res_blocks)
            ]))
            in_ch = out_ch
        # in_ch at index num_stage1 = channel width at mid_occ_resolution
        mid_ch = channels[num_stage1] if num_stage1 < len(channels) else sparse_channels

        # ---- Occupancy head at mid_occ_resolution³ (sparse) ----
        # occ_head_64 has no equivalent in FullSparse — always randomly initialised.
        self.occ_head_64 = nn.Sequential(
            _SparseLayerNorm32(mid_ch),
            SparseSiLU(),
            SparseLinear(mid_ch, 1),
        )

        self.tudf_head = nn.Sequential(
            _SparseLayerNorm32(in_ch),
            SparseSiLU(),
            SparseLinear(in_ch, out_channels),
        )

    # ------------------------------------------------------------------
    # Sparse pruning helper
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Pretrain loading helper
    # ------------------------------------------------------------------

    @classmethod
    def remap_full_sparse_state_dict(cls, src: dict) -> dict:
        """
        Prepare a FullSparseLatentTUDFDecoder state dict for loading into
        TwoStageSparseLatentTUDFDecoder.

        Both classes share identical module names
        (sparse_projs.i, sparse_res_blocks.i, input_layer, middle_block,
        dense_blocks, occ_head, tudf_head), so no key renaming is required.
        The only difference is that TwoStage adds occ_head_64, which has no
        counterpart in FullSparse.  load_state_dict(strict=False) will leave
        occ_head_64 at its random initialisation and ignore nothing extra.

        This method is provided for explicitness; it returns a shallow copy of
        src unchanged.  Call it as:

            ckpt = torch.load(path)["decoder"]
            sd   = TwoStageSparseLatentTUDFDecoder.remap_full_sparse_state_dict(ckpt)
            model.decoder.load_state_dict(sd, strict=False)
        """
        return dict(src)

    # ------------------------------------------------------------------
    # Sparse pruning helper
    # ------------------------------------------------------------------

    @staticmethod
    def _prune_sparse(
        sp: "SparseTensor",
        keep_mask: torch.Tensor,
    ) -> "SparseTensor":
        """Filter active voxels in a SparseTensor by a boolean keep_mask (N,)."""
        return SparseTensor(feats=sp.feats[keep_mask], coords=sp.coords[keep_mask])

    def _build_sparse_mask_64(
        self,
        sp_logits_occ64: "SparseTensor",
        sp_coords: torch.Tensor,
        tudf_gt: Optional[torch.Tensor],
        use_gt_mask: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Build boolean keep_mask for active voxels at mid_occ_resolution.

        Args:
            sp_logits_occ64: SparseTensor from occ_head_64 — feats (N, 1).
            sp_coords:       (N, 4) int32 coords [batch, d, h, w] at mid_occ_resolution.
            tudf_gt:         (B, 1, 256, 256, 256) or None.
            use_gt_mask:     Whether to use GT teacher forcing.

        Returns:
            keep_mask:   (N,) bool — which voxels survive the prune.
            gt64_at_sp:  (N,) float {0,1} GT occupancy at sparse coords — for BCE loss.
                         None when not in GT-mask mode.
            gt_occ_64:   (B,1,R,R,R) dense GT occ at mid_occ_resolution — for BCE loss.
                         None when not in GT-mask mode.
        """
        logit_vals = sp_logits_occ64.feats[:, 0]     # (N,)

        if use_gt_mask and tudf_gt is not None:
            gt_occ_64 = get_gt_occ_coarse(
                tudf_gt, self.tau_surface, occ_resolution=self.mid_occ_resolution
            )                                         # (B, 1, R64, R64, R64)
            b = sp_coords[:, 0].long()
            d = sp_coords[:, 1].long()
            h = sp_coords[:, 2].long()
            w = sp_coords[:, 3].long()
            gt64_at_sp = gt_occ_64[b, 0, d, h, w]   # (N,) float {0,1}
            keep_mask  = gt64_at_sp > 0.5

            if self.gt_mask_fixed_active_64 > 0:
                n_active = int(keep_mask.sum().item())
                if n_active > self.gt_mask_fixed_active_64:
                    pos  = keep_mask.nonzero(as_tuple=False).squeeze(-1)
                    perm = torch.randperm(n_active, device=keep_mask.device)
                    keep_mask_new = torch.zeros_like(keep_mask)
                    keep_mask_new[pos[perm[:self.gt_mask_fixed_active_64]]] = True
                    keep_mask = keep_mask_new

            return keep_mask, gt64_at_sp, gt_occ_64

        # Inference: threshold sparse logits
        keep_mask = logit_vals.sigmoid() > self.occ_threshold_64
        n_active  = int(keep_mask.sum().item())
        if n_active < self.min_active_voxels:
            top_k = min(self.min_active_voxels, logit_vals.numel())
            _, top_idx = logit_vals.topk(top_k)
            keep_mask_new = torch.zeros_like(keep_mask)
            keep_mask_new[top_idx] = True
            keep_mask = keep_mask_new
        elif self.max_active_voxels > 0 and n_active > self.max_active_voxels:
            _, top_idx = logit_vals.topk(self.max_active_voxels)
            keep_mask_new = torch.zeros_like(keep_mask)
            keep_mask_new[top_idx] = True
            keep_mask = keep_mask_new

        return keep_mask, None, None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        z: torch.Tensor,
        *,
        tudf_gt: Optional[torch.Tensor] = None,
        use_gt_mask: bool = True,
        return_dense_256: bool = False,
    ):
        """
        Args:
            z:               (B, latent_channels, latent_resolution³)
            tudf_gt:         (B, 1, 256³) GT TUDF — needed for teacher-forcing and loss.
            use_gt_mask:     True  → teacher-force both gates with GT (training).
                             False → use predicted logits at both gates (inference).
            return_dense_256: Scatter final sparse pred to dense (B,1,256³).

        Returns (3-tuple):
            pred_tudf_sparse : (N,)      TUDF at active 256-voxels
            coords_256       : (N, 4)    int32 [batch, d, h, w]
            occ_dict         : dict {
                "logits16"   : (B,1,R16,R16,R16) dense logits at latent_resolution,
                "logits64"   : SparseTensor, feats (N64,1) at mid_occ_resolution,
                "gt16"       : (B,1,R16,R16,R16) float {0,1}  — only when GT provided,
                "gt64_sparse": (N64,) float {0,1} GT occ at sparse 64³ coords — only GT,
                "gt64_dense" : (B,1,R64,R64,R64) float {0,1}                  — only GT,
            }
        Appended when return_dense_256=True:
            dense_256        : (B,1,256³) scatter with fill=1.0 for empty voxels
        """
        B = z.shape[0]
        do_gt = use_gt_mask and tudf_gt is not None

        # ---- Dense stem ----
        if self.training:
            h = checkpointing.checkpoint(self.input_layer,  z, use_reentrant=False)
            h = checkpointing.checkpoint(self.middle_block, h, use_reentrant=False)
            h = checkpointing.checkpoint(self.dense_blocks, h, use_reentrant=False)
        else:
            h = self.input_layer(z)
            h = self.middle_block(h)
            h = self.dense_blocks(h)
        h_occ = h                                     # (B, C0, latent_resolution³)

        # ---- Stage-1 occupancy head (dense, at latent_resolution) ----
        logits16 = self.occ_head(h_occ)              # (B, 1, R16, R16, R16)

        mask16, gt_occ16 = self._build_mask(
            tudf_gt, do_gt, logits16
        )                                             # mask16: (B, R16, R16, R16) bool

        # ---- Dense → sparse at latent_resolution ----
        sp = self._dense_to_sparse(h_occ, mask16)    # SparseTensor(N16, C0)

        # ---- Stage-1 sparse path: latent_resolution → mid_occ_resolution ----
        for i in range(self.num_stage1):
            sp = self.sparse_projs[i](sp)
            sp = self.subdivide(sp)
            for blk in self.sparse_res_blocks[i]:
                sp = blk(sp)
        # sp is now at mid_occ_resolution (e.g. 64³)

        # ---- Stage-2 occupancy head (sparse, at mid_occ_resolution) ----
        sp_logits64  = self.occ_head_64(sp)           # SparseTensor(N64, 1) — ALL active voxels
        sp_coords_64 = sp.coords                      # (N64, 4)

        keep_mask, gt64_at_sp, gt_occ_64_dense = self._build_sparse_mask_64(
            sp_logits64, sp_coords_64, tudf_gt, do_gt
        )
        # gt64_at_sp: (N64,) float {0,1} for ALL active 64³ voxels — has both 0s and 1s.
        # keep_mask is used only to prune the feature SparseTensor for stage-2.
        # Loss is computed against the full N64 logits vs full N64 GT (not just kept voxels).

        # ---- Prune sparse tensor at mid_occ_resolution ----
        sp = self._prune_sparse(sp, keep_mask)        # prune features for stage-2
        # sp_logits64 (pre-prune) is kept for loss and IoU; no pruned copy needed.

        # ---- Stage-2 sparse path: mid_occ_resolution → 256 ----
        for i in range(self.num_stage2):
            j = self.num_stage1 + i
            sp = self.sparse_projs[j](sp)
            sp = self.subdivide(sp)
            for blk in self.sparse_res_blocks[j]:
                sp = blk(sp)

        # ---- TUDF head ----
        sp              = self.tudf_head(sp)
        pred_tudf_sparse = sp.feats[:, 0]             # (N,)
        coords_256       = sp.coords                  # (N, 4)

        # ---- Assemble occupancy dict ----
        occ_dict: dict = {
            "logits16": logits16,
            # Full pre-prune logits at mid_occ_resolution — used for loss and IoU.
            # Contains logits for ALL descendants of active 16³ voxels (0s and 1s in GT).
            "logits64": sp_logits64,
        }
        if do_gt:
            occ_dict["gt16"]         = gt_occ16       # (B,1,R16,R16,R16) float
            # Full GT for ALL N64 sparse voxels — has both 0s (bg inside active 16³ block)
            # and 1s (surface voxels). Correct target for BCE/Dice at the prune gate.
            occ_dict["gt64_sparse"]  = gt64_at_sp
            occ_dict["gt64_dense"]   = gt_occ_64_dense
        # Include active-voxel counts for monitoring (detached)
        occ_dict["n_active_16"] = int(mask16.sum().item())
        occ_dict["n_active_64"] = int(keep_mask.sum().item())

        if return_dense_256:
            dense_256 = self._scatter_to_dense(sp, B)
            return pred_tudf_sparse, coords_256, occ_dict, dense_256

        return pred_tudf_sparse, coords_256, occ_dict


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Tiny channels to avoid OOM on a shared GPU; architecture logic is unchanged.
    # Production config: channels=[256,128,64,32], latent_channels=8.
    dec = SparseLatentTUDFDecoder(
        latent_channels=8,
        channels=reversed([32, 64, 128, 256]),      # tiny — keeps ResBlock / Upsample / Sparse logic
        num_res_blocks=1,
        num_res_blocks_middle=1,
        sparse_channels=4,
        sparse_num_res_blocks=1,
        max_active_voxels_128=10000, # cap for smoke test: untrained model produces random logits
    ).to(device)

    total = sum(p.numel() for p in dec.parameters()) / (float(1e6))
    print(f"SparseLatentTUDFDecoder params: {total:.1f}M")

    # z and tudf_gt at correct resolutions (32³ and 256³)
    # Use a realistic sparse tudf_gt: mostly empty (+1), thin spherical shell as surface.
    # Random uniform in [-1,1] would activate ~50% of 256³ → OOM in SparseConv at 256³.
    z       = torch.randn(1, 8, 32, 32, 32, device=device)
    tudf_gt = torch.ones(1, 1, 256, 256, 256, device=device)  # all empty
    c = 128
    r_inner, r_outer = 50, 55  # thin shell ~5-voxel thick
    lin = torch.arange(256, device=device).float() - c
    d_sq = (lin[:, None, None] ** 2 + lin[None, :, None] ** 2 + lin[None, None, :] ** 2)
    shell = (d_sq >= r_inner ** 2) & (d_sq <= r_outer ** 2)
    tudf_gt[0, 0][shell] = -1.0  # mark shell as surface

    with torch.no_grad():
        # --- Training mode: GT mask, indexed SmoothL1 + occ BCE ---
        pred, coords, logits = dec(z, tudf_gt=tudf_gt, use_gt_mask=True)
        print(f"pred_tudf_sparse : {pred.shape}")
        print(f"coords_256       : {coords.shape}  (active 256-voxels)")
        print(f"logits_occ_128   : {logits.shape}")

        gathered   = gather_gt_at_coords(tudf_gt, coords)
        recon_loss = F.smooth_l1_loss(pred, gathered)
        gt_occ     = get_gt_occ_128(tudf_gt, tau_surface=0.0)
        occ_loss   = F.binary_cross_entropy_with_logits(logits, gt_occ)
        print(f"recon_loss={recon_loss.item():.4f}  occ_loss={occ_loss.item():.4f}")

        # --- Inference mode: predicted mask ---
        pred2, coords2, logits2 = dec(z)
        print(f"inference coords : {coords2.shape}")

    print("SparseLatentTUDFDecoder smoke test passed.")

    # ---- TwoStageSparseLatentTUDFDecoder smoke test ----
    print("\n--- TwoStageSparseLatentTUDFDecoder ---")
    # Tiny channels: 16³ latent, 64³ mid prune, then 256³.
    # channels = [C0=64, C1=32, C2=16, C3=8, C4=8] — 4 subdivides (16→32→64→128→256)
    two_stage = TwoStageSparseLatentTUDFDecoder(
        latent_channels=4,
        channels=[64, 32, 16, 8, 8],
        num_res_blocks=1,
        num_res_blocks_middle=1,
        sparse_channels=8,
        sparse_num_res_blocks=1,
        tau_surface=0.0,
        occ_threshold_16=0.5,
        occ_threshold_64=0.5,
        max_active_voxels=5000,   # cap inference for smoke test
        latent_resolution=16,
        mid_occ_resolution=64,
    ).to(device)

    total2 = sum(p.numel() for p in two_stage.parameters()) / 1e6
    print(f"TwoStageSparseLatentTUDFDecoder params: {total2:.2f}M")

    # Realistic sparse GT: thin spherical shell surface at 256³.
    z2      = torch.randn(1, 4, 16, 16, 16, device=device)
    gt2     = torch.ones(1, 1, 256, 256, 256, device=device)   # all empty
    lin2    = torch.arange(256, device=device).float() - 128.0
    dsq2    = lin2[:, None, None]**2 + lin2[None, :, None]**2 + lin2[None, None, :]**2
    gt2[0, 0][(dsq2 >= 50**2) & (dsq2 <= 55**2)] = -1.0        # thin shell = surface

    with torch.no_grad():
        # --- Training: GT mask at both gates ---
        pred_s, coords_s, occ_d = two_stage(z2, tudf_gt=gt2, use_gt_mask=True)
        print(f"pred_tudf_sparse : {pred_s.shape}")
        print(f"coords_256       : {coords_s.shape}")
        print(f"logits16 shape   : {occ_d['logits16'].shape}")
        print(f"logits64 feats   : {occ_d['logits64'].feats.shape}")
        print(f"n_active_16      : {occ_d['n_active_16']}")
        print(f"n_active_64      : {occ_d['n_active_64']}")

        # Verify GT keys present
        assert "gt16"         in occ_d, "gt16 missing"
        assert "gt64_sparse"  in occ_d, "gt64_sparse missing"
        assert "gt64_dense"   in occ_d, "gt64_dense missing"

        # Loss computation (mirrors train_latent_vae_two_stage_sparse.py)
        gathered2   = gather_gt_at_coords(gt2, coords_s)
        recon2      = F.smooth_l1_loss(pred_s, gathered2)
        occ16_loss  = F.binary_cross_entropy_with_logits(occ_d["logits16"], occ_d["gt16"])
        occ64_loss  = F.binary_cross_entropy_with_logits(
            occ_d["logits64"].feats[:, 0], occ_d["gt64_sparse"].float()
        )
        print(f"recon={recon2.item():.4f}  occ16={occ16_loss.item():.4f}  occ64={occ64_loss.item():.4f}")

        # --- return_dense_256 path ---
        pred_d, coords_d, occ_d2, dense_256 = two_stage(z2, tudf_gt=gt2, return_dense_256=True)
        assert dense_256.shape == (1, 1, 256, 256, 256)
        print(f"dense_256 shape  : {dense_256.shape}")

        # --- Inference (no GT mask) ---
        pred_i, coords_i, occ_i = two_stage(z2, use_gt_mask=False)
        print(f"inference coords : {coords_i.shape}")
        assert "gt16" not in occ_i, "GT keys should not appear in inference mode"

    print("TwoStageSparseLatentTUDFDecoder smoke test passed.")
