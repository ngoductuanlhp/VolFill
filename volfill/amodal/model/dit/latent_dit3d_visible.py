"""
LatentTUDFDiTVisible: Flow-matching DiT with visible-latent dual conditioning.

Supports two visible-conditioning modes (set via vis_mode):

  "concat" (default):
    dit_input = cat([x_t, z_vis], dim=1)   # (B, latent_ch + visible_ch, R, R, R)
    v_pred    = dit(dit_input, t, tokens)   # (B, latent_ch, R, R, R)
    The widened input projection sees both latents jointly.

  "add":
    dit_input = x_t + vis_proj(z_vis)       # (B, latent_ch, R, R, R)
    v_pred    = dit(dit_input, t, tokens)   # (B, latent_ch, R, R, R)
    vis_proj is a 1×1×1 Conv3d zero-initialized so training starts from the
    unconditional baseline and the visible path grows in gradually.

External API is identical in both modes: the caller always provides
cat([x_t, z_vis], dim=1) — the model splits and routes internally.

Forward interface:
    forward(x, t, cond) → velocity
    x:    (B, latent_channels + visible_channels, R, R, R)
    t:    (B,) timestep in [0, 1000]
    cond: (B, N, cond_channels) image patch tokens
    returns: (B, latent_channels, R, R, R) predicted velocity

`R` is the latent grid resolution controlled by the `resolution` parameter:
    resolution=32  → 32³ latent space (default, matches the 256³→32³ VAE)
    resolution=16  → 16³ latent space (occupancy VAE: 64³→16³)
"""

from __future__ import annotations

import torch.nn as nn

from volfill.amodal.model.dit.dit3d import CoarseTUDFDiT


class LatentTUDFDiTVisible(CoarseTUDFDiT):
    """
    DiT for visible-latent conditioned flow matching in an R³ latent space.

    The caller always provides x = cat([x_t, z_vis], dim=1). Internally the
    model routes this according to vis_mode:
      "concat": pass x directly through the (latent_ch+visible_ch)-channel input proj.
      "add":    split x → x_t, z_vis; add zero-init vis_proj(z_vis) to x_t; run DiT
                on latent_ch-channel input. Training starts from the unconditional
                baseline and visible conditioning grows in from zero.

    During CFG the unconditional visible branch uses zeros in place of z_vis.

    Args:
        latent_channels:  Number of VAE latent channels (must match trained VAE).
        visible_channels: Number of visible-latent channels (default = latent_channels).
        vis_mode:         "concat" (default) or "add".
        resolution:       Latent grid side length (default 32; use 16 for occ VAE).
        model_channels:   Transformer hidden dim (default 1024).
        cond_channels:    Image token dim from MoGeConditioner (default 768).
        num_blocks:       Number of transformer blocks (default 24).
        num_heads:        Attention heads (default 16).
        patch_size:       3D patch size for patchify (default 2).
        use_fp16:         Use fp16 weights (Accelerate handles autocast — leave False).
        use_checkpoint:   Gradient checkpointing.
    """

    def __init__(
        self,
        latent_channels: int = 8,
        visible_channels: int = 8,
        vis_mode: str = "concat",
        resolution: int = 32,
        model_channels: int = 1024,
        cond_channels: int = 768,
        num_blocks: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        patch_size: int = 2,
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
    ):
        assert vis_mode in ("concat", "add"), f"vis_mode must be 'concat' or 'add', got {vis_mode!r}"
        in_channels = latent_channels + visible_channels if vis_mode == "concat" else latent_channels
        super().__init__(
            resolution=resolution,
            in_channels=in_channels,
            model_channels=model_channels,
            cond_channels=cond_channels,
            out_channels=latent_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            patch_size=patch_size,
            pe_mode="ape",
            use_fp16=use_fp16,
            use_checkpoint=use_checkpoint,
            share_mod=share_mod,
            qk_rms_norm=qk_rms_norm,
            qk_rms_norm_cross=qk_rms_norm_cross,
        )
        self.latent_channels = latent_channels
        self.visible_channels = visible_channels
        self.vis_mode = vis_mode
        self.resolution = resolution

        if vis_mode == "add":
            self.vis_proj = nn.Conv3d(visible_channels, latent_channels, kernel_size=1, bias=True)
            nn.init.zeros_(self.vis_proj.weight)
            nn.init.zeros_(self.vis_proj.bias)

    def forward(self, x, t, cond):
        if self.vis_mode == "add":
            x_t   = x[:, :self.latent_channels]
            z_vis = x[:, self.latent_channels:]
            x = x_t + self.vis_proj(z_vis)
        elif self.vis_mode == "concat":
            pass # x is already concatenated
        else:
            raise ValueError(f"Invalid vis_mode: {self.vis_mode!r}")

        return super().forward(x, t, cond)
