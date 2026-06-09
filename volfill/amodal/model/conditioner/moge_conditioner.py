"""
MoGeConditioner: Frozen MoGe-v2 image encoder + trainable token projection.

Extracts DINOv2 patch tokens from MoGe's encoder using forward_semantics(),
projects them to the cross-attention token dimension, and applies
classifier-free guidance (CFG) dropout during training.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from volfill.utils.timer import CUDATimer

# Add third_party/ so that `import moge` resolves as a proper package,
# which allows moge's own relative imports (e.g. `from ..utils.geometry_torch`)
# to work correctly.
_REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(_REPO_ROOT)


from third_party.moge.model.v2 import MoGeModel


class MoGeConditioner(nn.Module):
    """
    Frozen MoGe-v2 image encoder + trainable token projection.

    Args:
        moge_model_name: HuggingFace repo id or local path for MoGe-v2.
        token_proj_dim:  Output dimension for cross-attention tokens.
        encoder_dim:     DINOv2 encoder output dimension (1024 for ViT-L/14).
        p_uncond:        Probability of zeroing conditioning during training (CFG).
        cfg_dropout_in_forward: If True, apply CFG token dropout here. If False, caller
            (e.g. visible-latent training) applies joint cond + visible dropout instead.
    """

    def __init__(
        self,
        moge_model_name: str = "Ruicheng/moge-2-vitl",
        token_proj_dim: int = 1024,
        encoder_dim: int = 1024,
        p_uncond: float = 0.1,
        cfg_dropout_in_forward: bool = True,
    ):
        super().__init__()
        self.p_uncond = p_uncond
        self.cfg_dropout_in_forward = cfg_dropout_in_forward
        self.encoder_dim = encoder_dim
        self.token_proj_dim = token_proj_dim

        # Load frozen MoGe model
        self.moge = MoGeModel.from_pretrained(moge_model_name)
        for p in self.moge.parameters():
            p.requires_grad_(False)
        self.moge.eval()

        # Trainable projection: LayerNorm + Linear
        self.token_proj = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, token_proj_dim),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.moge.eval()
        return self

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _extract_tokens(self, image: torch.Tensor) -> torch.Tensor:
        """Run frozen MoGe encoder and return patch tokens (B, N, encoder_dim)."""
        img_tokens, cls_token = self.moge.forward_semantics(image)
        return img_tokens, cls_token

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        image: torch.Tensor,
        force_uncond: bool = False,
        return_geometry: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            image:        (B, 3, H, W) RGB image in [0, 1]
            force_uncond: if True, return zeros (for CFG at inference)

        Returns:
            (B, N_tokens, token_proj_dim) conditioning tokens
        """
                # tokens, cls_token = self._extract_tokens(image)           # (B, N, encoder_dim)
                # cond_tokens = self.token_proj(tokens)           # (B, N, token_proj_dim)

                # if force_uncond:
                #     return torch.zeros_like(cond_tokens)

                # # CFG dropout during training (optional: disabled when caller applies joint drop)
                # if self.cfg_dropout_in_forward and self.training and self.p_uncond > 0.0:
                #     B = cond_tokens.shape[0]
                #     drop_mask = torch.rand(B, device=cond_tokens.device) < self.p_uncond
                #     # Zero out entire samples selected for unconditional training
                #     cond_tokens = cond_tokens * (~drop_mask).view(B, 1, 1).float()

                # if not return_geometry:
                #     return cond_tokens

                # # with CUDATimer(name="MoGeConditioner.forward_with_geometry", enabled=False):
                # with torch.no_grad():
                #     moge_output = self.moge.infer(image, resolution_level=9)

                # geometry = {
                #     'points': moge_output['points'],
                #     'mask': moge_output['mask'].bool(),
                # }
                # return cond_tokens, geometry

        if not return_geometry:
            tokens, cls_token = self._extract_tokens(image)           # (B, N, encoder_dim)
            cond_tokens = self.token_proj(tokens)           # (B, N, token_proj_dim)

            if force_uncond:
                return torch.zeros_like(cond_tokens)

            # CFG dropout during training
            if self.cfg_dropout_in_forward and self.training and self.p_uncond > 0.0:
                B = cond_tokens.shape[0]
                drop_mask = torch.rand(B, device=cond_tokens.device) < self.p_uncond
                # Zero out entire samples selected for unconditional training
                cond_tokens = cond_tokens * (~drop_mask).view(B, 1, 1).float()

            return cond_tokens
        else:
            with torch.no_grad():
                out = self.moge.forward_semantics_and_geometry(image)

            cond_tokens = self.token_proj(out['semantics'])  # (B, N, token_proj_dim)
            if force_uncond:
                cond_tokens = torch.zeros_like(cond_tokens)

            # CFG dropout during training
            if self.cfg_dropout_in_forward and self.training and self.p_uncond > 0.0:
                B = cond_tokens.shape[0]
                drop_mask = torch.rand(B, device=cond_tokens.device) < self.p_uncond
                # Zero out entire samples selected for unconditional training
                cond_tokens = cond_tokens * (~drop_mask).view(B, 1, 1).float()

            geometry = {k: out[k] for k in ('points', 'mask') if k in out}
            return cond_tokens, geometry

    def forward_with_geometry(
        self,
        image: torch.Tensor,
        force_uncond: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Single MoGe encoder pass returning conditioning tokens AND geometry.

        Runs the frozen MoGe encoder once at native patch resolution (img_h // 14)
        and fans out to both the projected conditioning tokens (for the DiT) and
        fully post-processed geometry outputs (points, mask — same format as
        MoGeModel.infer()).  Avoids a second separate MoGe forward for visible TUDF.

        Args:
            image:        (B, 3, H, W) RGB in [0, 1].
            force_uncond: if True, cond_tokens are zeroed for CFG.

        Returns:
            cond_tokens: (B, N, token_proj_dim)
            geometry:    dict with 'points' (B, H, W, 3) and 'mask' (B, H, W).
                         Empty dict if the model has no geometry heads.
        """
        # batch_size, _, img_h, img_w = image.shape
        # aspect_ratio = img_w / img_h
        # # dtype, device = image.dtype, image.device
        # # base_h, base_w = img_h // 14, img_w // 14
        # num_tokens = 3600
        # base_h, base_w = (num_tokens / aspect_ratio) ** 0.5, (num_tokens * aspect_ratio) ** 0.5
        # new_h, new_w = int(base_h * 14), int(base_w * 14)
        # image = F.interpolate(image, (new_h, new_w), mode='bilinear', align_corners=False, antialias=False)
        # with CUDATimer(name="MoGeConditioner.forward_with_geometry", enabled=True):
                # out = self.moge.forward_semantics_and_geometry(image)
                # cond_tokens = self.token_proj(out['semantics'])  # (B, N, token_proj_dim)
                # geometry = {k: out[k] for k in ('points', 'mask') if k in out}
                # if force_uncond:
                #     cond_tokens = torch.zeros_like(cond_tokens)
                # return cond_tokens, geometry
        return self.forward(image, force_uncond=force_uncond, return_geometry=True)

    def get_uncond_tokens(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return zero tokens for unconditional sampling (CFG inference)."""
        # We need to know the token count; use a dummy forward or cache it.
        # The number of tokens depends on the image resolution; caller should
        # pass neg_cond = zeros_like(cond_tokens) for simplicity.
        raise NotImplementedError(
            "Pass neg_cond=torch.zeros_like(cond_tokens) for CFG inference."
        )
