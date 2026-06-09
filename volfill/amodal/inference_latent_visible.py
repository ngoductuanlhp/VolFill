"""
LatentTUDFVisibleInference: end-to-end latent pipeline with visible-latent conditioning.

Extends the token-only inference (inference_latent.py) by adding a second
conditioning path: MoGe visible-geometry latent concatenated with the noisy
target latent before each DiT step.

Pipeline (online):
    image → MoGe visible pointmap → visible TUDF (in estimated canonical frame)
          → VAE encoder → z_vis
    image → MoGe tokens → cross-attention conditioning
    → Euler sampling with dual conditioning → VAE decode → predicted TUDF

NOTE on canonical frame
    The bbox is always estimated from the MoGe visible pointmap using the
    same ``estimate_isotropic_bounds`` call as training preprocessing.  This
    introduces a mild distribution shift relative to training (which used the
    GT-derived canonical frame), but requires no external metadata file.

Usage:
    # Online inference on a single image (bbox estimated from MoGe — no metadata needed)
    python -m volfill.amodal.inference_latent_visible \
        --config configs/inference.yaml \
        --dit_checkpoint checkpoints/volfill_dit.pth \
        --vae_checkpoint checkpoints/volfill_vae.pth \
        --input_path image.jpg \
        --output ./results/

    # Batch inference over a directory of samples listed in a JSON split
    python -m volfill.amodal.inference_latent_visible \
        --config configs/inference.yaml \
        --dit_checkpoint checkpoints/volfill_dit.pth \
        --vae_checkpoint checkpoints/volfill_vae.pth \
        --input_path <data_root/> \
        --split <split.json> \
        --output ./results/
"""

from __future__ import annotations

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Dict, Optional, Union

from tqdm import tqdm
import numpy as np
import torch
from PIL import Image

# Repo root (contains the ``third_party`` namespace package) on sys.path so the
# TRELLIS sparse importer and MoGe resolve when run as a script.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(_REPO_ROOT))

from third_party.moge.model.v2 import MoGeModel

from volfill.amodal.config_utils import load_config
from volfill.amodal.datasets.scannetpp_tudf import compute_moge_image_size
from volfill.amodal.flow_matching import FlowEulerSampler
from volfill.amodal.model.conditioner.moge_conditioner import MoGeConditioner
from volfill.amodal.model.dit.latent_dit3d_visible import LatentTUDFDiTVisible
from volfill.amodal.model.vae.latent_vae import LatentTUDFVAE
from volfill.amodal.model.vae.latent_vae_sparse_encoder import SparseTUDFEncoderVAE
from volfill.amodal.checkpoint_utils import load_vae_checkpoint
from volfill.preprocess.visible_tudf_prep import (
    compute_truncated_distance_field,
    estimate_isotropic_bounds,
    voxelize_points,
)

try:
    import open3d as o3d
    _OPEN3D_AVAILABLE = True
except ImportError:
    _OPEN3D_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers for online visible TUDF computation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Inference class
# ---------------------------------------------------------------------------

class LatentTUDFVisibleInference:
    """
    End-to-end latent TUDF inference with visible-latent dual conditioning.

    Args:
        dit_checkpoint:     Path to visible-conditioning DiT .pth checkpoint.
        vae_checkpoint:     Path to VAE checkpoint.  Optional when ``config_path``
                            is provided (falls back to ``vae_checkpoint`` in config).
        config_path:        Path to the YAML training config
                            (e.g. configs/inference.yaml).
                            When given, model architecture is built from the config
                            rather than relying solely on the checkpoint's saved args.
                            Config values take precedence over checkpoint args.
        device:             Torch device.
        cfg_strength:       CFG scale (default 3.0).
        steps:              Euler steps (default 50).
        truncation_voxels:  TUDF truncation matching training (default 3.0).
        tudf_threshold:     Surface extraction threshold (default 0.0).
        latent_stats_path:  Path to latent stats .npy for normalization.
                            If None and ``config_path`` is given, read from config.
        vis_stats_path:     Path to visible latent stats .npy; falls back to
                            ``latent_stats_path`` if not provided.
    """

    def __init__(
        self,
        dit_checkpoint: Union[str, Path],
        vae_checkpoint: Optional[Union[str, Path]] = None,
        config_path: Optional[Union[str, Path]] = None,
        device: str = "cuda",
        cfg_strength: float = 3.0,
        steps: int = 50,
        truncation_voxels: float = 3.0,
        tudf_threshold: float = 0.0,
        latent_stats_path: Optional[str] = None,
        vis_stats_path: Optional[str] = None,
        occ_threshold: Optional[float] = None,
        occ_threshold_16: Optional[float] = None,
        occ_threshold_64: Optional[float] = None,
    ):
        self.device = torch.device(device)
        self.cfg_strength = cfg_strength
        self.steps = steps
        self.truncation_voxels = truncation_voxels
        self.tudf_threshold = tudf_threshold

        # ---- Load training config (optional) ----
        # flow_args is the flat config dict that takes precedence over the
        # checkpoint's own saved "args".
        flow_args: Optional[dict] = None
        if config_path is not None:
            _cfg = load_config(str(config_path))
            flow_args = vars(_cfg)

        # ---- Resolve VAE checkpoint path ----
        if vae_checkpoint is None:
            if flow_args is not None and flow_args.get("vae_checkpoint"):
                vae_checkpoint = flow_args["vae_checkpoint"]
            else:
                raise ValueError(
                    "--vae_checkpoint must be provided (or set vae.vae_checkpoint in config)"
                )

        # ---- Resolve latent stats paths ----
        if latent_stats_path is None and flow_args is not None:
            latent_stats_path = flow_args.get("latent_stats")
        if vis_stats_path is None and flow_args is not None:
            vis_stats_path = flow_args.get("visible_latent_stats") or flow_args.get("latent_stats")

        self._load_dit(Path(dit_checkpoint), flow_args=flow_args)
        self._load_vae(Path(vae_checkpoint), flow_args=flow_args)
        self._apply_occ_overrides(occ_threshold, occ_threshold_16, occ_threshold_64)
        self._load_stats(latent_stats_path, vis_stats_path)
        self._load_moge_for_visible()

    # ------------------------------------------------------------------
    # Hugging Face Hub
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        *,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        device: str = "cuda",
        dit_filename: str = "volfill_dit.pth",
        vae_filename: str = "volfill_vae.pth",
        config_filename: str = "inference.yaml",
        latent_stats_filename: str = "latent_stats_16x.npy",
        cfg_strength: Optional[float] = None,
        steps: Optional[int] = None,
        **kwargs,
    ) -> "LatentTUDFVisibleInference":
        """Build the inference pipeline from a Hugging Face Hub model repo.

        The repo is expected to contain the DiT and VAE checkpoints, the
        inference config, and the latent-normalization stats (default file
        names above).  Each file is downloaded and cached via
        ``huggingface_hub.hf_hub_download``; weights download on first use.

        Example::

            from volfill.amodal.inference_latent_visible import LatentTUDFVisibleInference
            from PIL import Image

            infer  = LatentTUDFVisibleInference.from_pretrained("your-org/volfill")
            result = infer(Image.open("image.jpg").convert("RGB"))

        Args:
            repo_id:   Hub repo id, e.g. ``"your-org/volfill"``.
            revision:  Optional branch / tag / commit.
            cache_dir: Optional download cache directory.
            device:    Torch device.
            *_filename: Override the expected file names in the repo.
            cfg_strength / steps: Override the sampler settings; default to the
                ``val_cfg`` / ``val_steps`` values from the downloaded config.
            **kwargs:  Forwarded to ``__init__`` (e.g. ``tudf_threshold``,
                ``occ_threshold``).
        """
        from huggingface_hub import hf_hub_download

        def _dl(filename: str) -> str:
            return hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                cache_dir=cache_dir,
            )

        dit_path    = _dl(dit_filename)
        vae_path    = _dl(vae_filename)
        config_path = _dl(config_filename)
        stats_path  = _dl(latent_stats_filename)

        # Sampler settings come from the downloaded config unless overridden.
        cfg_vars = vars(load_config(config_path))
        if cfg_strength is None:
            cfg_strength = float(cfg_vars.get("val_cfg", 3.0))
        if steps is None:
            steps = int(cfg_vars.get("val_steps", 50))

        return cls(
            dit_checkpoint=dit_path,
            vae_checkpoint=vae_path,
            config_path=config_path,
            latent_stats_path=stats_path,
            device=device,
            cfg_strength=cfg_strength,
            steps=steps,
            **kwargs,
        )

    def _apply_occ_overrides(
        self,
        occ_threshold: Optional[float],
        occ_threshold_16: Optional[float],
        occ_threshold_64: Optional[float],
    ) -> None:
        """Override decoder occupancy thresholds to match a tuned eval config.

        The two-stage sparse decoder builds with occ_threshold_16/64 = 0.5 by
        default; tuned quali results use lower values (see eval_newdatasets_best.sh).
        """
        dec = getattr(self.vae, "decoder", None)
        if dec is None:
            return
        if occ_threshold is not None and hasattr(dec, "occ_threshold"):
            dec.occ_threshold = occ_threshold
        if occ_threshold_16 is not None and hasattr(dec, "occ_threshold_16"):
            dec.occ_threshold_16 = occ_threshold_16
            # stage-1 _build_mask reads occ_threshold (aliased to the 16³ level)
            dec.occ_threshold = occ_threshold_16
        if occ_threshold_64 is not None and hasattr(dec, "occ_threshold_64"):
            dec.occ_threshold_64 = occ_threshold_64
        msg = [f"occ={getattr(dec, 'occ_threshold', None)}"]
        if hasattr(dec, "occ_threshold_16"):
            msg.append(f"occ16={dec.occ_threshold_16}")
        if hasattr(dec, "occ_threshold_64"):
            msg.append(f"occ64={dec.occ_threshold_64}")
        print("[infer] decoder thresholds: " + " ".join(msg))

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    def _load_dit(self, ckpt_path: Path, flow_args: Optional[dict] = None) -> None:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        # Merge: checkpoint's saved args are the base; flow_args (from config) override.
        train_args: dict = {**ckpt.get("args", {}), **(flow_args or {})}

        latent_channels  = train_args.get("latent_channels", 8)
        visible_channels = train_args.get("visible_channels", latent_channels)
        self.latent_channels  = latent_channels
        self.visible_channels = visible_channels

        # Derive latent spatial resolution from VAE encoder depth (same as build_flow_models_visible)
        vae_enc_channels = train_args.get("vae_encoder_channels", [32, 64, 128, 256])
        latent_resolution = 256 // (2 ** (len(vae_enc_channels) - 1))

        self.dit = LatentTUDFDiTVisible(
            latent_channels=latent_channels,
            visible_channels=visible_channels,
            vis_mode=train_args.get("vis_cond_mode", "concat"),
            resolution=latent_resolution,
            model_channels=train_args.get("model_channels", 1024),
            cond_channels=train_args.get("cond_channels", 768),
            num_blocks=train_args.get("num_blocks", 24),
            num_heads=train_args.get("num_heads", 16),
            mlp_ratio=train_args.get("mlp_ratio", 4.0),
            patch_size=train_args.get("patch_size", 2),
            use_fp16=False,
            use_checkpoint=False,
            share_mod=train_args.get("share_mod", False),
            qk_rms_norm=train_args.get("qk_rms_norm", False),
            qk_rms_norm_cross=train_args.get("qk_rms_norm_cross", False),
        )
        self.dit.load_state_dict(ckpt["dit"])
        self.dit.to(self.device).eval()

        self.conditioner = MoGeConditioner(
            moge_model_name=train_args.get("moge_model_name", "Ruicheng/moge-2-vitl"),
            token_proj_dim=train_args.get("cond_channels", 768),
        )
        self.conditioner.token_proj.load_state_dict(ckpt["conditioner_proj"])
        self.conditioner.to(self.device).eval()

        sigma_min = train_args.get("sigma_min", 1e-5)
        self.sampler = FlowEulerSampler(sigma_min=sigma_min)

    def _load_vae(self, ckpt_path: Path, flow_args: Optional[dict]) -> None:
        """Build and load the VAE, mirroring build_flow_models_visible() in the trainer.

        When ``flow_args`` is provided (i.e. ``--config`` was given or the DiT
        checkpoint's saved args contain vae_* keys), architecture parameters are
        read from it using the same ``vae_*``-prefixed key names as the training
        config.  Otherwise the VAE checkpoint's own saved ``args`` (bare key names)
        are used as a fallback.
        """
        # ---- Path 1: use flow config args directly (mirrors training script) ----
        fa = flow_args
        vae_type = fa.get("vae_type", "dense")
        if vae_type == "sparse_encoder":
            self.vae = SparseTUDFEncoderVAE(
                in_channels=1,
                out_channels=1,
                latent_channels=fa["vae_latent_channels"],
                encoder_channels=list(fa["vae_encoder_channels"]),
                num_res_blocks=fa["vae_num_res_blocks"],
                num_res_blocks_middle=fa["vae_num_res_blocks_middle"],
                norm_type=fa["vae_norm_type"],
                sparse_band_tau=fa.get("vae_sparse_band_tau", 0.9),
                sparse_dilate=fa.get("vae_sparse_dilate", 1),
                sparse_min_voxels=fa.get("vae_sparse_min_voxels", 64),
                light_decoder=fa.get("vae_light_decoder", True),
                pointwise_from_level=fa.get("vae_pointwise_from_level", 2),
                decoder_type=fa["vae_decoder_type"],
                sparse_dec_channels=fa["vae_sparse_dec_channels"],
                sparse_dec_num_res_blocks=fa["vae_sparse_dec_num_res_blocks"],
                tau_surface=fa["vae_tau_surface"],
                occ_threshold=fa["vae_occ_threshold"],
                occ_resolution=fa.get("vae_occ_resolution", 128),
                gt_mask_fixed_active_128=fa.get("vae_gt_mask_fixed_active_128", 0),
                gt_mask_fixed_active=fa.get("vae_gt_mask_fixed_active", 0),
            )
        else:
            self.vae = LatentTUDFVAE(
                in_channels=1,
                out_channels=1,
                latent_channels=fa["vae_latent_channels"],
                encoder_channels=list(fa["vae_encoder_channels"]),
                num_res_blocks=fa["vae_num_res_blocks"],
                num_res_blocks_middle=fa["vae_num_res_blocks_middle"],
                norm_type=fa["vae_norm_type"],
            )

        self._vae_is_sparse = isinstance(self.vae, SparseTUDFEncoderVAE)
        load_vae_checkpoint(str(ckpt_path), self.vae, key="vae")
        self.vae.to(self.device).eval()

    def _load_moge_for_visible(self) -> None:
        """Load the MoGe-normal model used for online visible TUDF computation.

        Uses the same model variant as ``infer_moge_visible_pointmap`` in the
        preprocessing pipeline (``moge-2-vitl-normal``).  The model is kept alive
        as an instance attribute so it is not re-loaded on every inference call.
        """
        self.moge_for_visible = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal")
        self.moge_for_visible.to(self.device).eval()

    def _load_stats(self, latent_stats_path: Optional[str], vis_stats_path: Optional[str]) -> None:
        def _read(path):
            if path and os.path.isfile(path):
                s = np.load(path, allow_pickle=True).item()
                m = torch.tensor(s["mean"], dtype=torch.float32).to(self.device).view(1, -1, 1, 1, 1)
                s = torch.tensor(s["std"],  dtype=torch.float32).to(self.device).view(1, -1, 1, 1, 1).clamp(min=1e-6)
                return m, s
            return None, None

        self.latent_mean, self.latent_std = _read(latent_stats_path)
        vis_path = vis_stats_path or latent_stats_path
        self.vis_mean, self.vis_std = _read(vis_path)

    # ------------------------------------------------------------------
    # Visible latent helpers
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _compute_visible_tudf_online(
        self,
        image: Image.Image,
        fine_resolution: int = 256,
        moge_resolution_level: int = 9,
        bounds_margin: float = 0.1,
        robust_percentile: float = 1.0,
        max_depth: Optional[float] = None,
    ) -> tuple:
        """
        Run MoGe inference on *image* and compute a visible TUDF volume.

        Mirrors the FM-visible branch of ``process_target_view`` in
        the dataset preprocessing pipeline:
          1. Run MoGe → camera-frame visible pointmap + mask.
          2. Estimate isotropic canonical bbox from visible points.
          3. Clip visible points to bbox.
          4. Voxelize → binary occupancy.
          5. EDT → TUDF in [0, truncation_voxels].

        Returns:
            tudf:     (fine_resolution,) * 3  float32 TUDF in [0, truncation_voxels]
            bbox_min: (3,) float32  canonical bbox origin
            extent:   (3,) float32  canonical bbox extent (bbox_max - bbox_min)
        """
        img_np = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
        img_t  = torch.tensor(img_np, dtype=torch.float32, device=self.device).permute(2, 0, 1)
        out = self.moge_for_visible.infer(img_t, resolution_level=moge_resolution_level)

        pointmap = out["points"].squeeze(0).cpu().numpy().astype(np.float32)   # (H, W, 3)
        mask     = out["mask"].squeeze(0).cpu().numpy().astype(bool)            # (H, W)

        visible_pts = pointmap[mask]   # (N, 3) — camera-frame points

        _fallback = (
            np.full((fine_resolution,) * 3, self.truncation_voxels, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            np.ones(3, dtype=np.float32),
        )
        if visible_pts.shape[0] == 0:
            return _fallback

        # ---- Estimate isotropic bbox (mirrors preprocessing) ----
        canon_bbox_min, canon_bbox_max, _, _, _ = estimate_isotropic_bounds(
            visible_pts,
            margin_ratio=bounds_margin,
            robust_percentile=robust_percentile,
            max_depth=max_depth,
        )
        canon_extent = (canon_bbox_max - canon_bbox_min).astype(np.float32)

        # ---- Clip visible points to bbox ----
        inside = (
            np.all(visible_pts >= canon_bbox_min[None, :], axis=-1) &
            np.all(visible_pts <= canon_bbox_max[None, :], axis=-1)
        )
        visible_pts_clipped = visible_pts[inside]
        if visible_pts_clipped.shape[0] == 0:
            return np.full((fine_resolution,) * 3, self.truncation_voxels, dtype=np.float32), canon_bbox_min, canon_extent

        # ---- Voxelize and compute TUDF ----
        occ, _, _ = voxelize_points(
            visible_pts_clipped, canon_bbox_min, canon_bbox_max, resolution=fine_resolution
        )
        tudf = compute_truncated_distance_field(occ, truncation_voxels=self.truncation_voxels)
        return tudf, canon_bbox_min, canon_extent

    @torch.inference_mode()
    def _encode_visible_tudf(self, tudf_np: np.ndarray) -> torch.Tensor:
        """Encode a (256, 256, 256) float32 TUDF [0, trunc] → (1, C, 32, 32, 32) latent mean."""
        tudf_norm = 2.0 * tudf_np / self.truncation_voxels - 1.0   # [-1, 1]
        tudf_t = torch.from_numpy(tudf_norm).float().unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, 256, 256, 256)
        # Use the public encode() API which handles both dense and sparse VAE encoder types
        _, mean, _ = self.vae.encode(tudf_t, sample_posterior=False, return_stats=True)
        return mean   # (1, C, 32, 32, 32)

    def _normalize_vis_latent(self, z_vis: torch.Tensor) -> torch.Tensor:
        if self.vis_mean is not None:
            return (z_vis - self.vis_mean) / self.vis_std.clamp(min=1e-6)
        return z_vis

    def _normalize_latent(self, z: torch.Tensor) -> torch.Tensor:
        if self.latent_mean is not None:
            return (z - self.latent_mean) / self.latent_std.clamp(min=1e-6)
        return z

    def _denormalize_latent(self, z_norm: torch.Tensor) -> torch.Tensor:
        if self.latent_mean is not None:
            return z_norm * self.latent_std + self.latent_mean
        return z_norm

    @torch.inference_mode()
    def _vae_decode_to_dense(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent → dense (B, 1, 256, 256, 256) TUDF, handling both VAE types."""
        if self._vae_is_sparse:
            # SparseLatentTUDFDecoder returns a tuple; request dense scatter
            _, _, _, dense = self.vae.decode(latent, return_dense_256=True)
            return dense   # (B, 1, 256, 256, 256)
        else:
            return self.vae.decode(latent)   # (B, 1, 256, 256, 256)

    # ------------------------------------------------------------------
    # Main inference entry point
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def __call__(
        self,
        image: Union[Image.Image, torch.Tensor],
        max_size: int = 518,
    ) -> Dict:
        """
        Run visible-latent conditioned TUDF generation for a single image.

        The canonical bbox is always estimated from the MoGe visible pointmap so
        no external metadata file is needed.  The same bbox is stored in the
        returned dict and used for metric point cloud extraction.

        Args:
            image:    PIL Image or (3, H, W) float [0,1] tensor.
            max_size: Longest image dim for MoGe resize.

        Returns dict:
            "tudf":      (1, 1, 256, 256, 256) predicted TUDF in [-1, 1].
            "latent":    (1, C, 32, 32, 32) sampled latent.
            "vis_latent":(1, C, 32, 32, 32) visible conditioning latent.
            "bbox_min":  (3,) float32 canonical bbox origin (metres).
            "extent":    (3,) float32 canonical bbox extent (metres).
            "points":    (N, 3) metric point cloud.
        """
        # ---- Preprocess image ----
        if isinstance(image, Image.Image):
            orig_w, orig_h = image.size
            target_h, target_w = compute_moge_image_size(orig_h, orig_w, max_size)
            image_resized = image.resize((target_w, target_h), Image.BICUBIC)
            image_t = torch.from_numpy(
                np.array(image_resized, dtype=np.float32) / 255.0
            ).permute(2, 0, 1)
        else:
            image_t = image.float()
            image_resized = image

        image_t = image_t.unsqueeze(0).to(self.device)   # (1, 3, H, W)

        # ---- Image token conditioning ----
        cond_tokens = self.conditioner(image_t, force_uncond=False)   # (1, N, cond_ch)
        neg_cond    = torch.zeros_like(cond_tokens)

        # ---- Visible-latent conditioning ----
        # Online: MoGe → visible TUDF + canonical bbox → encode latent
        pil_image = image if isinstance(image, Image.Image) else Image.fromarray(
            (image_t[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        )
        vis_tudf_np, bbox_min_np, extent_np = self._compute_visible_tudf_online(
            pil_image,
            fine_resolution=256,
            moge_resolution_level=9,
            bounds_margin=0.1,
            robust_percentile=1.0,
            max_depth=None,
        )
        z_vis = self._encode_visible_tudf(vis_tudf_np)   # (1, C, 32, 32, 32)

        z_vis_norm = self._normalize_vis_latent(z_vis)

        # ---- Latent Euler sampling with dual conditioning ----
        R = z_vis.shape[-1]   # latent spatial resolution (16 for 16x VAE, 32 for 8x)
        noise = torch.randn(1, self.latent_channels, R, R, R, device=self.device)

        pred_latent_norm = self.sampler.sample(
            lambda x, t, c: self.dit(x, t, c),
            noise,
            cond=cond_tokens,
            neg_cond=neg_cond,
            steps=self.steps,
            cfg_strength=self.cfg_strength,
            vis_cond=z_vis_norm,
            neg_vis_cond=torch.zeros_like(z_vis_norm),
            verbose=False,
        )   # (1, C, 32, 32, 32) — normalized space

        pred_latent = self._denormalize_latent(pred_latent_norm)   # (1, C, 32, 32, 32)

        # ---- VAE decode ----
        pred_tudf = self._vae_decode_to_dense(pred_latent)   # (1, 1, 256, 256, 256)

        bbox_min_t = torch.from_numpy(bbox_min_np)
        extent_t   = torch.from_numpy(extent_np)

        result: Dict = {
            "tudf":      pred_tudf.cpu(),     # (1, 1, 256, 256, 256) in [-1, 1]
            "latent":    pred_latent.cpu(),   # (1, C, 32, 32, 32)
            "vis_latent":z_vis.cpu(),         # (1, C, 32, 32, 32)
            "bbox_min":  bbox_min_t,          # (3,) float32
            "extent":    extent_t,            # (3,) float32
            # "points":    self._tudf_to_pointcloud(pred_tudf[0, 0], bbox_min_t, extent_t),
        }

        return result

    def _tudf_to_pointcloud(
        self,
        tudf: torch.Tensor,
        bbox_min: Union[torch.Tensor, np.ndarray],
        extent: Union[torch.Tensor, np.ndarray],
    ) -> np.ndarray:
        if isinstance(bbox_min, np.ndarray):
            bbox_min = torch.from_numpy(bbox_min)
        if isinstance(extent, np.ndarray):
            extent = torch.from_numpy(extent)
        R = tudf.shape[0]
        below = tudf < self.tudf_threshold
        iz, iy, ix = torch.where(below)
        if iz.numel() == 0:
            return np.zeros((0, 3), dtype=np.float32)
        voxel_size = extent.to(self.device) / R
        x_m = bbox_min[0].to(self.device) + (ix.float() + 0.5) * voxel_size[0]
        y_m = bbox_min[1].to(self.device) + (iy.float() + 0.5) * voxel_size[1]
        z_m = bbox_min[2].to(self.device) + (iz.float() + 0.5) * voxel_size[2]
        return torch.stack([x_m, y_m, z_m], dim=-1).cpu().numpy()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _save_result(
    result: Dict,
    image: Image.Image,
    out_dir: Path,
    truncation_voxels: float,
) -> None:
    """Save inference outputs for one sample into *out_dir*."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Denormalize TUDF from [-1, 1] → [0, truncation_voxels]
    trunc = truncation_voxels
    tudf_norm = result["tudf"].squeeze().numpy()   # (256, 256, 256) in [-1, 1]
    tudf_raw  = np.clip((tudf_norm + 1.0) * trunc / 2.0, 0.0, trunc).astype(np.float32)
    res = tudf_raw.shape[0]

    np.savez_compressed(out_dir / f"pred_tudf_{res}.npz", tudf=tudf_raw)
    print(f"  pred_tudf_{res}.npz  shape={tudf_raw.shape}  "
          f"range=[{tudf_raw.min():.3f}, {tudf_raw.max():.3f}]")

    out_meta = {
        "representation":    "tudf",
        "truncation_voxels": trunc,
        "field_range":       [0.0, trunc],
        "field_units":       "voxel_units",
        "bbox_min":          result["bbox_min"].tolist(),
        "extent_xyz":        result["extent"].tolist(),
        "pred_resolution":   [res, res, res],
    }
    with (out_dir / "metadata.json").open("w") as f:
        json.dump(out_meta, f, indent=2)

    dest_img = out_dir / "image.jpg"
    if not dest_img.exists():
        image.save(str(dest_img), quality=95)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visible-latent conditioned TUDF inference")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to the YAML training config "
             "(e.g. configs/inference.yaml). "
             "When given, model architecture is built from the config, "
             "and vae_checkpoint / latent_stats are read from it if not "
             "explicitly overridden on the command line.",
    )
    parser.add_argument(
        "--hf_repo", default=None,
        help="Hugging Face Hub repo id (e.g. 'your-org/volfill'). When set, the DiT/VAE "
             "checkpoints, config, and latent stats are auto-downloaded from the Hub; "
             "--config / --dit_checkpoint / --vae_checkpoint are then not needed.",
    )
    parser.add_argument("--hf_revision", default=None,
                        help="Optional HF repo revision (branch / tag / commit).")
    parser.add_argument(
        "--dit_checkpoint", default=None,
        help="Path to the DiT .pth checkpoint. Required unless --hf_repo is given.",
    )
    parser.add_argument(
        "--vae_checkpoint",
        default=None,
        help="Path to VAE checkpoint. Optional when --config provides vae_checkpoint.",
    )
    parser.add_argument(
        "--input_path", required=True,
        help="Path to a single image file (*.png / *.jpg / *.jpeg) OR a dataset root "
             "directory.  When a directory is given, --split must also be provided.",
    )
    parser.add_argument(
        "--split", default=None,
        help="Path to a JSON list of relative sample paths (e.g. data_lists/scrream_tudf_list.json). "
             "Required when --input_path is a directory.  Each entry is joined with "
             "--input_path and 'image.jpg' to form the full image path.",
    )
    parser.add_argument("--output",          default="./visible_inference_output/")
    parser.add_argument("--cfg_strength",    type=float, default=None,
                        help="CFG scale. Defaults to 3.0 (or val_cfg from config).")
    parser.add_argument("--steps",          type=int,   default=None,
                        help="Euler steps. Defaults to 50 (or val_steps from config).")
    parser.add_argument("--device",         default="cuda")
    parser.add_argument("--tudf_threshold", type=float, default=0.0)
    parser.add_argument("--latent_stats",   default=None,
                        help="Path to latent stats .npy. Falls back to config if not given.")
    parser.add_argument("--vis_stats",      default=None,
                        help="Path to visible latent stats .npy. Defaults to latent_stats.")
    parser.add_argument("--occ_threshold",    type=float, default=None,
                        help="Override decoder occ_threshold (default: keep config/decoder value).")
    parser.add_argument("--occ_threshold_16", type=float, default=None,
                        help="Override two-stage decoder occ_threshold_16 (stage-1 16³ mask).")
    parser.add_argument("--occ_threshold_64", type=float, default=None,
                        help="Override two-stage decoder occ_threshold_64 (stage-2 64³ mask).")
    args = parser.parse_args()

    if args.hf_repo is not None:
        # Auto-download checkpoints + config + stats from the Hugging Face Hub.
        infer = LatentTUDFVisibleInference.from_pretrained(
            args.hf_repo,
            revision=args.hf_revision,
            device=args.device,
            cfg_strength=args.cfg_strength,
            steps=args.steps,
            tudf_threshold=args.tudf_threshold,
            occ_threshold=args.occ_threshold,
            occ_threshold_16=args.occ_threshold_16,
            occ_threshold_64=args.occ_threshold_64,
        )
    else:
        if args.dit_checkpoint is None:
            parser.error("--dit_checkpoint is required unless --hf_repo is given")

        # Resolve cfg_strength / steps from config if not overridden on CLI
        cfg_strength = args.cfg_strength
        steps        = args.steps
        if args.config is not None and (cfg_strength is None or steps is None):
            _cfg_vars = vars(load_config(args.config))
            if cfg_strength is None:
                cfg_strength = float(_cfg_vars.get("val_cfg", 3.0))
            if steps is None:
                steps = int(_cfg_vars.get("val_steps", 50))
        cfg_strength = cfg_strength or 3.0
        steps        = steps        or 50

        infer = LatentTUDFVisibleInference(
            dit_checkpoint=args.dit_checkpoint,
            vae_checkpoint=args.vae_checkpoint,
            config_path=args.config,
            device=args.device,
            cfg_strength=cfg_strength,
            steps=steps,
            tudf_threshold=args.tudf_threshold,
            latent_stats_path=args.latent_stats,
            vis_stats_path=args.vis_stats,
            occ_threshold=args.occ_threshold,
            occ_threshold_16=args.occ_threshold_16,
            occ_threshold_64=args.occ_threshold_64,
        )

    input_path = Path(args.input_path)
    out_root   = Path(args.output)

    # ---- Determine mode: single image or dataset directory ----
    if input_path.suffix.lower() in (".png", ".jpg", ".jpeg"):
        # Single-image mode
        image_paths = [(input_path, out_root)]
    else:
        with open(args.split) as f:
            samples = json.load(f)
        image_paths = [
            (input_path / sample / "image.jpg", out_root / sample)
            for sample in samples
        ]

    # ---- Run inference ----
    for i, (img_path, out_dir) in tqdm(enumerate(image_paths), total=len(image_paths), desc="Inference"):
        if not img_path.exists():
            print(f"[{i+1}/{len(image_paths)}] SKIP (not found): {img_path}")
            continue
        print(f"[{i+1}/{len(image_paths)}] {img_path}")
        image = Image.open(img_path).convert("RGB")
        result = infer(image)
        _save_result(result, image, out_dir, infer.truncation_voxels)

    print("\nDone.")
    if len(image_paths) == 1:
        print(f"Visualize with:\n  python -m volfill.visualize "
              f"--sample_dir {image_paths[0][1]} --field pred --threshold 0.8")


if __name__ == "__main__":
    main()
