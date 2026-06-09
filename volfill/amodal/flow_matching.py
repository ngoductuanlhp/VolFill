"""
Flow matching utilities and Euler sampler.

Ported from:
  TRELLIS trainers/flow_matching/flow_matching.py  (FlowMatching)
  TRELLIS pipelines/samplers/flow_euler.py          (FlowEulerSampler)
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm


class FlowMatching:
    """
    Stateless flow-matching helper (not an nn.Module).

    Uses the linear interpolation path:
        x_t = (1 - t) * x_0 + (sigma_min + (1 - sigma_min) * t) * noise
    Velocity target:
        v = (1 - sigma_min) * noise - x_0
    Time is sampled with LogitNormal (mean=0, std=1) and passed to the network
    scaled by 1000.
    """

    def __init__(self, sigma_min: float = 1e-5):
        self.sigma_min = sigma_min

    # ------------------------------------------------------------------
    # Timestep sampling
    # ------------------------------------------------------------------

    def sample_t(self, batch_size: int, mean: float = 0.0, std: float = 1.0, scheduler: str = "logitnormal") -> torch.Tensor:
        """LogitNormal timestep sampling → values in (0, 1)."""
        if scheduler == "logitnormal":
            t = torch.sigmoid(torch.randn(batch_size) * std + mean)
            # t1 = t = torch.sigmoid(torch.randn(batch_size) * std + mean)
            return t
            
        elif scheduler == "uniform":
            return torch.rand(batch_size)

    # ------------------------------------------------------------------
    # Forward process
    # ------------------------------------------------------------------

    def diffuse(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample x_t | x_0 at timestep t.

        Args:
            x_0:   (B, C, ...) clean data
            t:     (B,) timestep in [0, 1]
            noise: optional (B, C, ...) Gaussian noise; sampled if None
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        t_view = t.view(-1, *([1] * (x_0.dim() - 1)))
        return (1 - t_view) * x_0 + (self.sigma_min + (1 - self.sigma_min) * t_view) * noise

    # ------------------------------------------------------------------
    # Velocity target
    # ------------------------------------------------------------------

    def get_velocity_target(
        self,
        x_0: torch.Tensor,
        noise: torch.Tensor,
        t: Optional[torch.Tensor] = None,  # unused, kept for API symmetry
    ) -> torch.Tensor:
        """v = (1 - sigma_min) * noise - x_0."""
        return (1 - self.sigma_min) * noise - x_0


class FlowEulerSampler:
    """
    Euler sampler for flow matching with optional classifier-free guidance.

    Ported from TRELLIS FlowEulerSampler + FlowEulerCfgSampler.
    """

    def __init__(self, sigma_min: float = 1e-5):
        self.sigma_min = sigma_min

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _v_to_xstart(self, x_t: torch.Tensor, t: float, v: torch.Tensor) -> torch.Tensor:
        """Recover x_0 estimate from predicted velocity at time t."""
        return (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t) * v

    def _call_model(
        self,
        model: Any,
        x_t: torch.Tensor,
        t: float,
        cond: Optional[torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        t_tensor = torch.full(
            (x_t.shape[0],), 1000.0 * t, device=x_t.device, dtype=torch.float32
        )

        # Use fp16 to match training mixed_precision; falls back to fp32 on CPU
        autocast_dtype = torch.float16 if x_t.device.type == "cuda" else torch.bfloat16
        with torch.autocast(device_type=x_t.device.type, dtype=autocast_dtype):
            return model(x_t, t_tensor, cond, **kwargs)

    def _cfg_velocity(
        self,
        model: Any,
        x_t: torch.Tensor,
        t: float,
        cond: torch.Tensor,
        neg_cond: torch.Tensor,
        cfg_strength: float,
        vis_cond: Optional[torch.Tensor] = None,
        neg_vis_cond: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """(1 + cfg) * v_cond - cfg * v_uncond.

        If vis_cond is provided (visible-latent conditioning), the positive
        branch uses ``cat([x_t, vis_cond], dim=1)`` and the negative branch
        uses ``cat([x_t, neg_vis_cond], dim=1)`` (zeros by default).  The
        model must accept the wider input (in_channels = latent_ch + visible_ch)
        and output only the latent-channel velocity.
        """
        original_batch_size = x_t.shape[0]

        if vis_cond is not None:
            if neg_vis_cond is None:
                neg_vis_cond = torch.zeros_like(vis_cond)
            x_pos = torch.cat([x_t, vis_cond], dim=1)
            x_neg = torch.cat([x_t, neg_vis_cond], dim=1)
            # x_neg = torch.cat([x_t, vis_cond], dim=1)
            x_cat = torch.cat([x_pos, x_neg], dim=0)
        else:
            x_cat = torch.cat([x_t, x_t], dim=0)

        v_cat = self._call_model(
            model,
            x_cat,
            t,
            torch.cat([cond, neg_cond], dim=0),
            **kwargs,
        )

        v_cond = v_cat[:original_batch_size]
        v_uncond = v_cat[original_batch_size:]
        return (1 + cfg_strength) * v_cond - cfg_strength * v_uncond

    # ------------------------------------------------------------------
    # Single Euler step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_once(
        self,
        model: Any,
        x_t: torch.Tensor,
        t: float,
        t_prev: float,
        cond: Optional[torch.Tensor] = None,
        neg_cond: Optional[torch.Tensor] = None,
        cfg_strength: float = 0.0,
        vis_cond: Optional[torch.Tensor] = None,
        neg_vis_cond: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One Euler step from t → t_prev.

        Args:
            vis_cond:     (B, C_vis, D, H, W) visible-latent conditioning — pre-concatenated
                          with x_t before the model forward pass.
            neg_vis_cond: (B, C_vis, D, H, W) negative visible conditioning for CFG
                          (zeros used if None and vis_cond is provided).

        Returns:
            x_{t_prev}: denoised one step
            pred_x_0:   predicted clean sample
        """
        if neg_cond is not None and cfg_strength > 0.0:
            v_pred = self._cfg_velocity(
                model, x_t, t, cond, neg_cond, cfg_strength,
                vis_cond=vis_cond, neg_vis_cond=neg_vis_cond, **kwargs,
            )
        else:
            if vis_cond is not None:
                x_in = torch.cat([x_t, vis_cond], dim=1)
            else:
                x_in = x_t
            v_pred = self._call_model(model, x_in, t, cond, **kwargs)

        x_prev = x_t - (t - t_prev) * v_pred
        pred_x_0 = self._v_to_xstart(x_t, t, v_pred)
        return x_prev, pred_x_0

    # ------------------------------------------------------------------
    # Full sampling loop
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        model: Any,
        noise: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        neg_cond: Optional[torch.Tensor] = None,
        steps: int = 50,
        cfg_strength: float = 0.0,
        rescale_t: float = 1.0,
        verbose: bool = True,
        vis_cond: Optional[torch.Tensor] = None,
        neg_vis_cond: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Full Euler integration from noise → predicted TUDF.

        Args:
            model:        callable with signature model(x_t, t_scaled, cond)
            noise:        (B, C, D, H, W) initial Gaussian noise
            cond:         (B, N, D) conditioning tokens; None for unconditional
            neg_cond:     (B, N, D) negative conditioning tokens for CFG;
                          if None, uses zeros when cfg_strength > 0
            steps:        number of Euler steps
            cfg_strength: CFG scale (0 = disabled)
            rescale_t:    optional timestep rescaling factor (1.0 = linear)
            verbose:      show tqdm progress bar
            vis_cond:     (B, C_vis, D, H, W) visible-latent tensor pre-concatenated
                          with x_t at each step (dual conditioning).
            neg_vis_cond: (B, C_vis, D, H, W) negative visible conditioning for CFG;
                          zeros used if None and vis_cond is provided.

        Returns:
            (B, C, D, H, W) predicted TUDF at t=0
        """
        # Build time schedule: 1 → 0 in `steps` steps
        t_seq = np.linspace(1.0, 0.0, steps + 1)
        if rescale_t != 1.0:
            t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list(zip(t_seq[:-1], t_seq[1:]))

        # Build negative conditioning for CFG
        if cfg_strength > 0.0 and neg_cond is None and cond is not None:
            neg_cond = torch.zeros_like(cond)

        x = noise
        for t, t_prev in tqdm(t_pairs, desc="Sampling", disable=not verbose):
            x, _ = self.sample_once(
                model, x, t, t_prev,
                cond=cond,
                neg_cond=neg_cond,
                cfg_strength=cfg_strength,
                vis_cond=vis_cond,
                neg_vis_cond=neg_vis_cond,
                **kwargs,
            )
        return x


class FlowEulerGuidanceIntervalSampler(FlowEulerSampler):
    """
    Euler sampler that applies classifier-free guidance only within a
    timestep interval ``cfg_interval = (t_low, t_high)`` (both in [0, 1],
    *higher* t = noisier).  Outside the interval the conditional branch is
    used without guidance, which can improve sample quality by avoiding
    over-guidance at very clean or very noisy steps.

    API is identical to ``FlowEulerSampler`` with two extra kwargs:
        cfg_interval: Tuple[float, float]  — default (0.0, 1.0) (always on)
    """

    @torch.no_grad()
    def sample_once(
        self,
        model: Any,
        x_t: torch.Tensor,
        t: float,
        t_prev: float,
        cond: Optional[torch.Tensor] = None,
        neg_cond: Optional[torch.Tensor] = None,
        cfg_strength: float = 0.0,
        vis_cond: Optional[torch.Tensor] = None,
        neg_vis_cond: Optional[torch.Tensor] = None,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        t_lo, t_hi = cfg_interval
        apply_cfg = (
            neg_cond is not None
            and cfg_strength > 0.0
            and t_lo <= t <= t_hi
        )
        if apply_cfg:
            # print(f"At timestep {t}, applying CFG")
            v_pred = self._cfg_velocity(
                model, x_t, t, cond, neg_cond, cfg_strength,
                vis_cond=vis_cond, neg_vis_cond=neg_vis_cond, **kwargs,
            )
        else:
            # print(f"At timestep {t}, not applying CFG")
            x_in = torch.cat([x_t, vis_cond], dim=1) if vis_cond is not None else x_t
            v_pred = self._call_model(model, x_in, t, cond, **kwargs)

        x_prev = x_t - (t - t_prev) * v_pred
        pred_x_0 = self._v_to_xstart(x_t, t, v_pred)
        return x_prev, pred_x_0

    @torch.no_grad()
    def sample(
        self,
        model: Any,
        noise: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        neg_cond: Optional[torch.Tensor] = None,
        steps: int = 50,
        cfg_strength: float = 0.0,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        rescale_t: float = 1.0,
        verbose: bool = True,
        vis_cond: Optional[torch.Tensor] = None,
        neg_vis_cond: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Full Euler integration with interval-gated CFG.

        Args:
            cfg_interval: ``(t_low, t_high)`` — CFG is applied only when
                          the current timestep ``t`` satisfies
                          ``t_low <= t <= t_high``.  Default ``(0.0, 1.0)``
                          matches the behaviour of ``FlowEulerSampler``.
                          Example: ``(0.0, 0.7)`` disables guidance for the
                          noisiest 30 % of the trajectory.
            All other args: identical to ``FlowEulerSampler.sample``.
        """
        t_seq = np.linspace(1.0, 0.0, steps + 1)
        if rescale_t != 1.0:
            t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list(zip(t_seq[:-1], t_seq[1:]))

        if cfg_strength > 0.0 and neg_cond is None and cond is not None:
            neg_cond = torch.zeros_like(cond)

        x = noise
        for t, t_prev in tqdm(t_pairs, desc="Sampling", disable=not verbose):
            x, _ = self.sample_once(
                model, x, t, t_prev,
                cond=cond,
                neg_cond=neg_cond,
                cfg_strength=cfg_strength,
                vis_cond=vis_cond,
                neg_vis_cond=neg_vis_cond,
                cfg_interval=cfg_interval,
                **kwargs,
            )
        return x
