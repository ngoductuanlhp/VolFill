"""Checkpoint loading utilities shared across amodal training/eval scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
from safetensors.torch import load_file as _load_safetensors


def load_vae_checkpoint(
    checkpoint_path: str,
    vae: torch.nn.Module,
    key: str = "vae",
    strict: bool = True,
) -> Dict[str, Any]:
    """Load VAE weights from either a .pth file or an Accelerate safetensors directory.

    For .pth files: extracts state dict under `key`, falls back to the full dict.
    For directories: loads model.safetensors; metadata (args etc.) is read from meta.pt if present.

    Returns:
        metadata dict (may be empty) — contains "args", "epoch", etc. when available.
    """
    path = Path(checkpoint_path)
    meta: Dict[str, Any] = {}

    if path.is_dir():
        state_dict = _load_safetensors(path / "model.safetensors")
        meta_path = path / "meta.pt"
        if meta_path.is_file():
            meta = torch.load(meta_path, map_location="cpu")
    else:
        ckpt = torch.load(path, map_location="cpu")
        state_dict = ckpt.get(key, ckpt)
        # Remaining keys (args, epoch, …) are the metadata
        meta = {k: v for k, v in ckpt.items() if k != key}

    vae.load_state_dict(state_dict, strict=strict)
    return meta

def load_checlpoint_with_filter(checkpoint: Dict[str, Any], model_state_dict: Dict[str, Any]) -> Dict[str, Any]:

    filtered_checkpoint = dict()
    for k, v in checkpoint.items():
        if k not in model_state_dict:
            print(f"Skipping {k} because it is not in the model state dict")
            continue
        if v.shape != model_state_dict[k].shape:
            print(f"Skipping {k} because the shape does not match, src: {v.shape}, model: {model_state_dict[k].shape}")
            continue
        filtered_checkpoint[k] = v
     
    return filtered_checkpoint


def _remap_moge_only_to_combined(state_dict: Dict[str, Any], conditioner: Any) -> Dict[str, Any]:
    """If the conditioner is a DinoMoGe-style module (no ``token_proj``, has
    ``moge_token_proj``), remap MoGe-only keys ``token_proj.*`` to
    ``moge_token_proj.*`` so a MoGe-only pretrain warm-starts the MoGe branch
    of the combined conditioner.  Untouched if the target conditioner has its
    own ``token_proj`` attribute (i.e. is the MoGe-only conditioner).
    """
    if hasattr(conditioner, "token_proj") or not hasattr(conditioner, "moge_token_proj"):
        return state_dict
    remapped: Dict[str, Any] = {}
    for k, v in state_dict.items():
        if k.startswith("token_proj."):
            remapped["moge_" + k] = v
        else:
            remapped[k] = v
    return remapped


def load_dit_flow_pretrain(
    checkpoint_path: str | Path,
    dit: torch.nn.Module,
    conditioner: Any,
) -> None:
    """
    Load DiT + trainable conditioner params for flow pretrain (partial / shape-matched load).

    Supports:
    - A single ``.pth`` from ``save_model_only`` (keys ``dit``, ``conditioner_trainable``,
      and legacy ``conditioner_proj``).
    - An Accelerate ``save_state`` directory (e.g. ``.../checkpoint-last``) with
      ``model.safetensors`` (DiT) and ``model_1.safetensors`` (full conditioner;
      only trainable-param keys are extracted).

    When the target conditioner is a combined DinoMoGe-style module (has
    ``moge_token_proj``), keys named ``token_proj.*`` in the pretrain are
    automatically remapped to ``moge_token_proj.*`` so a MoGe-only checkpoint
    warm-starts the MoGe branch.  The DINO branch and DiT cross-attn cond
    projection are left at their construction-time init.
    """
    p = Path(checkpoint_path)

    if p.is_dir():
        dit_sf = p / "model.safetensors"
        if not dit_sf.is_file():
            raise FileNotFoundError(
                f"Expected {dit_sf} (Accelerate save_state). "
                "For a single-file checkpoint, use save_model_only output (.pth)."
            )
        dit_sd = _load_safetensors(dit_sf)
        filtered = load_checlpoint_with_filter(dit_sd, dit.state_dict())
        dit.load_state_dict(filtered, strict=False)

        cond_sf = p / "model_1.safetensors"
        if cond_sf.is_file():
            full_cond = _load_safetensors(cond_sf)
            full_cond = _remap_moge_only_to_combined(full_cond, conditioner)
            trainable_names = {n for n, p_ in conditioner.named_parameters() if p_.requires_grad}
            trainable_sd = {k: v for k, v in full_cond.items() if k in trainable_names}
            if trainable_sd:
                f2 = load_checlpoint_with_filter(trainable_sd, conditioner.state_dict())
                conditioner.load_state_dict(f2, strict=False)
            elif hasattr(conditioner, "token_proj"):
                # Legacy: single MoGeConditioner/DinoConditioner with token_proj
                prefix = "token_proj."
                proj_sd = {k[len(prefix):]: v for k, v in full_cond.items() if k.startswith(prefix)}
                if proj_sd:
                    f2 = load_checlpoint_with_filter(proj_sd, conditioner.token_proj.state_dict())
                    conditioner.token_proj.load_state_dict(f2, strict=False)
    else:
        ckpt = torch.load(p, map_location="cpu")
        if "dit" in ckpt:
            filtered = load_checlpoint_with_filter(ckpt["dit"], dit.state_dict())
            dit.load_state_dict(filtered, strict=False)
        if "conditioner_trainable" in ckpt:
            remapped = _remap_moge_only_to_combined(ckpt["conditioner_trainable"], conditioner)
            f2 = load_checlpoint_with_filter(remapped, conditioner.state_dict())
            conditioner.load_state_dict(f2, strict=False)
        elif "conditioner_proj" in ckpt:
            # Legacy: single token_proj conditioner. Two cases:
            #   (a) target has token_proj → load directly (MoGe-only path).
            #   (b) target has moge_token_proj (combined) → load into MoGe branch.
            if hasattr(conditioner, "token_proj"):
                f2 = load_checlpoint_with_filter(
                    ckpt["conditioner_proj"], conditioner.token_proj.state_dict()
                )
                conditioner.token_proj.load_state_dict(f2, strict=False)
            elif hasattr(conditioner, "moge_token_proj"):
                f2 = load_checlpoint_with_filter(
                    ckpt["conditioner_proj"], conditioner.moge_token_proj.state_dict()
                )
                conditioner.moge_token_proj.load_state_dict(f2, strict=False)

