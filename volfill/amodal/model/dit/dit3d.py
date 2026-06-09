"""
CoarseTUDFDiT: Self-contained DiT for 3D TUDF generation via flow matching.

Ported from TRELLIS SparseStructureFlowModel with the following key components:
  TimestepEmbedder, AbsolutePositionEmbedder, LayerNorm32,
  MultiHeadAttention (self + cross), FeedForwardNet,
  ModulatedTransformerCrossBlock, patchify / unpatchify.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Attention backend: use PyTorch native SDPA (no xformers/flash-attn dep)
# ---------------------------------------------------------------------------

def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Scaled dot-product attention. Inputs: [B, L, H, C]. Output: [B, L, H, C]."""
    q = q.permute(0, 2, 1, 3)   # [B, H, L, C]
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)
    out = F.scaled_dot_product_attention(q, k, v)
    return out.permute(0, 2, 1, 3)  # [B, L, H, C]


def scaled_dot_product_attention(*args, **kwargs) -> torch.Tensor:
    """
    Multi-signature attention dispatch, matching TRELLIS's interface:
      (qkv,)  — qkv: [B, L, 3, H, C]
      (q, kv) — q:   [B, L, H, C],  kv: [B, Lk, 2, H, C]
      (q, k, v)      each [B, L, H, C]
    """
    n = len(args) + len(kwargs)
    if n == 1:
        qkv = args[0] if args else kwargs["qkv"]
        q, k, v = qkv.unbind(dim=2)
    elif n == 2:
        q = args[0] if len(args) > 0 else kwargs["q"]
        kv = args[1] if len(args) > 1 else kwargs["kv"]
        k, v = kv.unbind(dim=2)
    else:
        q = args[0] if len(args) > 0 else kwargs["q"]
        k = args[1] if len(args) > 1 else kwargs["k"]
        v = args[2] if len(args) > 2 else kwargs["v"]
    return _sdpa(q, k, v)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

class LayerNorm32(nn.LayerNorm):
    """LayerNorm that upcasts to fp32 before computing, then casts back."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).to(x.dtype)


# ---------------------------------------------------------------------------
# Position embedder
# ---------------------------------------------------------------------------

class AbsolutePositionEmbedder(nn.Module):
    """Sinusoidal absolute position embedder for multi-dimensional grids."""

    def __init__(self, channels: int, in_channels: int = 3):
        super().__init__()
        self.channels = channels
        self.in_channels = in_channels
        self.freq_dim = channels // in_channels // 2
        freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        self.register_buffer("freqs", 1.0 / (10000 ** freqs), persistent=False)

    def _sin_cos(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.outer(x, self.freqs)
        return torch.cat([torch.sin(out), torch.cos(out)], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, D) grid coordinates → (N, channels)."""
        N, D = x.shape
        embed = self._sin_cos(x.reshape(-1))
        embed = embed.reshape(N, -1)
        if embed.shape[1] < self.channels:
            pad = torch.zeros(N, self.channels - embed.shape[1], device=embed.device)
            embed = torch.cat([embed, pad], dim=-1)
        return embed


# ---------------------------------------------------------------------------
# Timestep embedder
# ---------------------------------------------------------------------------

class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps (already scaled by 1000) into vectors."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -np.log(max_period) * torch.arange(half, dtype=torch.float32) / half
        ).to(t.device)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


# ---------------------------------------------------------------------------
# Attention components
# ---------------------------------------------------------------------------

class MultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (F.normalize(x.float(), dim=-1) * self.gamma * self.scale).to(x.dtype)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int] = None,
        type: Literal["self", "cross"] = "self",
        qkv_bias: bool = True,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.head_dim = channels // num_heads
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self.num_heads = num_heads
        self._type = type
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm

        if type == "self":
            self.to_qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        else:
            self.to_q = nn.Linear(channels, channels, bias=qkv_bias)
            self.to_kv = nn.Linear(self.ctx_channels, channels * 2, bias=qkv_bias)

        if qk_rms_norm:
            self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)

        self.to_out = nn.Linear(channels, channels)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, C = x.shape
        if self._type == "self":
            qkv = self.to_qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
            if self.qk_rms_norm:
                q, k, v = qkv.unbind(dim=2)
                q = self.q_rms_norm(q)
                k = self.k_rms_norm(k)
                h = scaled_dot_product_attention(q, k, v)
            else:
                h = scaled_dot_product_attention(qkv)
        else:
            assert context is not None
            Lkv = context.shape[1]
            q = self.to_q(x).reshape(B, L, self.num_heads, self.head_dim)
            kv = self.to_kv(context).reshape(B, Lkv, 2, self.num_heads, self.head_dim)
            if self.qk_rms_norm:
                k, v = kv.unbind(dim=2)
                q = self.q_rms_norm(q)
                k = self.k_rms_norm(k)
                h = scaled_dot_product_attention(q, k, v)
            else:
                h = scaled_dot_product_attention(q, kv)
        h = h.reshape(B, L, C)
        return self.to_out(h)


# ---------------------------------------------------------------------------
# FFN
# ---------------------------------------------------------------------------

class FeedForwardNet(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# ---------------------------------------------------------------------------
# Modulated transformer cross-attention block (adaLN-Zero + self + cross + FFN)
# ---------------------------------------------------------------------------

class ModulatedTransformerCrossBlock(nn.Module):
    """
    adaLN-Zero modulation + self-attention + cross-attention + MLP.
    Matches TRELLIS modules/transformer/modulated.py:76-157.
    """

    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod

        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)

        self.self_attn = MultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.cross_attn = MultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
        )
        self.mlp = FeedForwardNet(channels, mlp_ratio=mlp_ratio)

        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True),
            )

    def _forward(
        self, x: torch.Tensor, mod: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(mod).chunk(6, dim=1)
            )

        # Self-attention with adaLN-Zero modulation
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h = self.self_attn(h)
        h = h * gate_msa.unsqueeze(1)
        x = x + h

        # Cross-attention
        h = self.norm2(x)
        h = self.cross_attn(h, context)
        x = x + h

        # FFN with adaLN-Zero modulation
        h = self.norm3(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.mlp(h)
        h = h * gate_mlp.unsqueeze(1)
        x = x + h
        return x

    def forward(
        self, x: torch.Tensor, mod: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, mod, context, use_reentrant=False
            )
        return self._forward(x, mod, context)


# ---------------------------------------------------------------------------
# Patchify / unpatchify (3-D)
# ---------------------------------------------------------------------------

def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """(B, C, D, H, W) → (B, C*ps³, D/ps, H/ps, W/ps)."""
    DIM = x.dim() - 2
    for d in range(2, DIM + 2):
        assert x.shape[d] % patch_size == 0

    x = x.reshape(
        *x.shape[:2],
        *sum([[x.shape[d] // patch_size, patch_size] for d in range(2, DIM + 2)], []),
    )
    x = x.permute(
        0, 1,
        *([2 * i + 3 for i in range(DIM)] + [2 * i + 2 for i in range(DIM)]),
    )
    x = x.reshape(x.shape[0], x.shape[1] * (patch_size ** DIM), *(x.shape[-DIM:]))
    return x


def unpatchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """(B, C*ps³, D/ps, H/ps, W/ps) → (B, C, D, H, W)."""
    DIM = x.dim() - 2
    assert x.shape[1] % (patch_size ** DIM) == 0

    x = x.reshape(
        x.shape[0],
        x.shape[1] // (patch_size ** DIM),
        *([patch_size] * DIM),
        *(x.shape[-DIM:]),
    )
    x = x.permute(
        0, 1,
        *(sum([[2 + DIM + i, 2 + i] for i in range(DIM)], [])),
    )
    x = x.reshape(
        x.shape[0], x.shape[1],
        *[x.shape[2 + 2 * i] * patch_size for i in range(DIM)],
    )
    return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class CoarseTUDFDiT(nn.Module):
    """
    DiT for coarse 64³ TUDF generation via flow matching.

    Direct port of TRELLIS SparseStructureFlowModel adapted for dense TUDF.

    Input  x: (B, in_channels, res, res, res) — noisy TUDF + frustum mask
    Input  t: (B,) — timestep (scaled by 1000)
    Input  cond: (B, N_tokens, cond_channels) — MOGe patch tokens
    Output: (B, out_channels, res, res, res) — predicted velocity
    """

    def __init__(
        self,
        resolution: int = 64,
        in_channels: int = 2,          # noisy TUDF (1) + frustum mask (1)
        model_channels: int = 768,
        cond_channels: int = 1024,     # MOGe token dim (cross-attn context)
        out_channels: int = 1,         # predicted velocity
        num_blocks: int = 12,
        num_heads: Optional[int] = None,
        num_head_channels: int = 64,
        mlp_ratio: float = 4.0,
        patch_size: int = 2,
        pe_mode: Literal["ape"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or (model_channels // num_head_channels)
        self.patch_size = patch_size
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.dtype = torch.float16 if use_fp16 else torch.float32

        # Timestep embedding
        self.t_embedder = TimestepEmbedder(model_channels)

        # Shared adaLN modulation (optional)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias=True),
            )
        self.share_mod = share_mod

        # Absolute position embeddings for the patched 3-D grid
        if pe_mode == "ape":
            pos_embedder = AbsolutePositionEmbedder(model_channels, 3)
            res_p = resolution // patch_size  # 32 when resolution=64 patch_size=2
            coords = torch.meshgrid(
                *[torch.arange(res_p, dtype=torch.float32)] * 3, indexing="ij"
            )
            coords = torch.stack(coords, dim=-1).reshape(-1, 3)
            pos_emb = pos_embedder(coords)   # (res_p³, model_channels)
            self.register_buffer("pos_emb", pos_emb)

        # Input projection: patch tokens → model_channels
        self.input_layer = nn.Linear(in_channels * patch_size ** 3, model_channels)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            ModulatedTransformerCrossBlock(
                model_channels,
                cond_channels,
                num_heads=self.num_heads,
                mlp_ratio=mlp_ratio,
                use_checkpoint=use_checkpoint,
                use_rope=False,
                share_mod=share_mod,
                qk_rms_norm=qk_rms_norm,
                qk_rms_norm_cross=qk_rms_norm_cross,
            )
            for _ in range(num_blocks)
        ])

        # Output projection: model_channels → patch tokens
        self.out_layer = nn.Linear(model_channels, out_channels * patch_size ** 3)

        self._initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _initialize_weights(self) -> None:
        def _basic_init(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(_basic_init)

        # Re-init timestep embedder MLP with small normal
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-init adaLN final layers
        if self.share_mod:
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        else:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-init output layer
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    # ------------------------------------------------------------------
    # fp16 helpers
    # ------------------------------------------------------------------

    def convert_to_fp16(self) -> None:
        for block in self.blocks:
            block.half()

    def convert_to_fp32(self) -> None:
        for block in self.blocks:
            block.float()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:    (B, in_channels, res, res, res) noisy TUDF + frustum mask
            t:    (B,) timestep, already scaled by 1000
            cond: (B, N_tokens, cond_channels) MOGe patch tokens

        Returns:
            (B, out_channels, res, res, res) predicted velocity field
        """
        # Patchify: (B, C, res, res, res) → (B, C*ps³, res/ps, res/ps, res/ps)
        h = patchify(x, self.patch_size)
        res_p = self.resolution // self.patch_size
        # Flatten spatial dims: (B, C*ps³, S, S, S) → (B, S³, C*ps³)
        h = h.view(h.shape[0], h.shape[1], -1).permute(0, 2, 1).contiguous()

        # Project to model channels and add position embeddings
        h = self.input_layer(h)
        h = h + self.pos_emb[None]

        # Timestep conditioning
        t_emb = self.t_embedder(t)                     # (B, model_channels)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)      # (B, 6*model_channels)

        # Cast to working dtype
        t_emb = t_emb.to(self.dtype)
        h = h.to(self.dtype)
        cond = cond.to(self.dtype)

        # Transformer blocks
        for block in self.blocks:
            h = block(h, t_emb, cond)

        # Cast back and final norm + projection
        h = h.to(x.dtype)
        h = F.layer_norm(h, h.shape[-1:])
        h = self.out_layer(h)

        # Reshape: (B, S³, C_out*ps³) → (B, C_out*ps³, res/ps, res/ps, res/ps)
        h = h.permute(0, 2, 1).view(
            h.shape[0], h.shape[2], *[res_p] * 3
        )

        # Unpatchify: → (B, out_channels, res, res, res)
        h = unpatchify(h, self.patch_size).contiguous()
        return h
