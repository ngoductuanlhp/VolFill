"""Config loading for inference.

Loads a YAML config (OmegaConf) and flattens its top-level sections
(``experiment``, ``data``, ``model``, ``vae``, ``training`` …) into a single
flat ``SimpleNamespace``, so callers can read ``cfg.latent_channels``,
``cfg.vae_latent_channels``, etc. directly.

This is the inference-side counterpart of the ``load_config`` helper used by
the (non-public) training entry points; it depends only on ``omegaconf``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf


def load_config(config_path: str, overrides: Optional[List[str]] = None) -> SimpleNamespace:
    cfg = OmegaConf.load(config_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    raw: Dict = OmegaConf.to_container(cfg, resolve=True)
    flat: Dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            flat.update(value)
        else:
            flat[key] = value
    return SimpleNamespace(**flat)
