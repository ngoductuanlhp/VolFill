"""Image-sizing helper shared with the MoGe preprocessing pipeline.

Only the ``compute_moge_image_size`` function is needed for inference; the full
dataset classes that originally lived here are training/eval-only and are not
part of the public release.
"""

from __future__ import annotations

# MoGe / DINOv2 patch size — image dims must be divisible by this.
_PATCH_SIZE = 14


def compute_moge_image_size(orig_h: int, orig_w: int, max_size: int = 518) -> tuple[int, int]:
    """
    Return (target_h, target_w) such that:
      - the largest dimension equals ``max_size``
      - aspect ratio is preserved
      - both dims are divisible by ``_PATCH_SIZE`` (14)

    The final dims are snapped down to the nearest multiple of 14, so the
    actual largest dimension may be slightly less than ``max_size``.
    """
    scale = max_size / max(orig_h, orig_w)
    new_h = int(orig_h * scale)
    new_w = int(orig_w * scale)
    # Snap to nearest multiple of 14 (floor to avoid exceeding max_size)
    new_h = max(_PATCH_SIZE, (new_h // _PATCH_SIZE) * _PATCH_SIZE)
    new_w = max(_PATCH_SIZE, (new_w // _PATCH_SIZE) * _PATCH_SIZE)
    return new_h, new_w
