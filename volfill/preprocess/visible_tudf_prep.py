"""Visible-geometry → TUDF preprocessing helpers.

These three functions turn a MoGe visible point cloud into the 256³ visible
TUDF volume (in an estimated canonical frame) that conditions the DiT:

  1. ``estimate_isotropic_bounds`` — isotropic canonical bounding box.
  2. ``voxelize_points``          — point cloud → binary occupancy grid.
  3. ``compute_truncated_distance_field`` — occupancy → truncated unsigned
     distance field (EDT, clamped to ``truncation_voxels``).

They are extracted verbatim (logic unchanged) from the dataset preprocessing
pipeline so the inference path has no dependency on the training-only
``data_preprocess`` package. Only NumPy + SciPy are required; CuPy is used
automatically for the EDT when available.
"""

from __future__ import annotations

import numpy as np

try:  # optional GPU EDT — falls back to SciPy on CPU
    import cupy as cp
    from cupyx.scipy.ndimage import distance_transform_edt as _edt_cupy
except Exception:  # pragma: no cover - cupy is optional
    cp = None
    _edt_cupy = None


# ---------------------------------------------------------------------------
# Bounding box
# ---------------------------------------------------------------------------

def _apply_depth_filters(points: np.ndarray, max_depth: float | None = None) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    valid = np.all(np.isfinite(points), axis=-1)
    if max_depth is not None:
        valid = valid & (points[:, 2] <= max_depth)
    filtered = points[valid]
    if filtered.shape[0] == 0:
        raise ValueError("No valid points remain after filtering.")
    return filtered


def estimate_isotropic_bounds(
    points: np.ndarray,
    margin_ratio: float = 0.1,
    robust_percentile: float | None = None,
    max_depth: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    points = _apply_depth_filters(points, max_depth=max_depth)

    if robust_percentile is not None:
        if not (0.0 <= robust_percentile < 50.0):
            raise ValueError("robust_percentile must be in [0, 50).")
        low = robust_percentile
        high = 100.0 - robust_percentile
        bbox_min = np.percentile(points, low, axis=0).astype(np.float32)
        bbox_max = np.percentile(points, high, axis=0).astype(np.float32)
    else:
        bbox_min = points.min(axis=0).astype(np.float32)
        bbox_max = points.max(axis=0).astype(np.float32)

    center = 0.5 * (bbox_min + bbox_max)
    half_extent = 0.5 * (bbox_max - bbox_min)
    half_scale = float(np.maximum(half_extent.max(), 1e-4))
    half_scale *= 1.0 + float(margin_ratio)
    extent = np.full((3,), 2.0 * half_scale, dtype=np.float32)
    bbox_min = (center - half_scale).astype(np.float32)
    bbox_max = (center + half_scale).astype(np.float32)
    return bbox_min, bbox_max, extent, center.astype(np.float32), float(half_scale)


# ---------------------------------------------------------------------------
# Voxelization
# ---------------------------------------------------------------------------

def _normalize_resolution(resolution: int | tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(resolution, int):
        return (resolution, resolution, resolution)
    if len(resolution) != 3:
        raise ValueError(f"Expected a 3-tuple resolution, got {resolution}")
    return tuple(int(v) for v in resolution)


def voxel_size_xyz_from_bbox(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    resolution: int | tuple[int, int, int],
) -> np.ndarray:
    """Per-axis voxel edge length (x, y, z) for the grid covering ``bbox_min``..``bbox_max``."""
    res_z, res_y, res_x = _normalize_resolution(resolution)
    voxel_resolution_xyz = np.array([res_x, res_y, res_z], dtype=np.float32)
    extent = np.asarray(bbox_max - bbox_min, dtype=np.float32)
    return extent / voxel_resolution_xyz


def voxelize_points(
    points: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    resolution: int | tuple[int, int, int],
    voxel_size_xyz: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    res_z, res_y, res_x = _normalize_resolution(resolution)
    voxel_resolution_xyz = np.array([res_x, res_y, res_z], dtype=np.int64)

    if voxel_size_xyz is None:
        voxel_size_xyz = voxel_size_xyz_from_bbox(bbox_min, bbox_max, resolution)
    else:
        voxel_size_xyz = np.asarray(voxel_size_xyz, dtype=np.float32)
        extent = np.asarray(bbox_max - bbox_min, dtype=np.float32)
        expected_extent = voxel_size_xyz * voxel_resolution_xyz.astype(np.float32)
        if not np.allclose(expected_extent, extent, rtol=1e-5, atol=1e-5):
            raise ValueError(
                "voxel_size_xyz is inconsistent with bbox extent and resolution: "
                f"got extent {extent}, expected {expected_extent} from voxel_size_xyz."
            )

    points = np.asarray(points, dtype=np.float32)
    idx_xyz = np.floor((points - bbox_min[None, :]) / voxel_size_xyz[None, :]).astype(np.int64)
    idx_xyz = np.clip(idx_xyz, 0, voxel_resolution_xyz[None, :] - 1)

    lin = (
        idx_xyz[:, 0]
        + idx_xyz[:, 1] * voxel_resolution_xyz[0]
        + idx_xyz[:, 2] * voxel_resolution_xyz[0] * voxel_resolution_xyz[1]
    )
    lin_unique = np.unique(lin)

    occ = np.zeros((res_z, res_y, res_x), dtype=bool)
    xy_plane = voxel_resolution_xyz[0] * voxel_resolution_xyz[1]
    iz = lin_unique // xy_plane
    rem = lin_unique % xy_plane
    iy = rem // voxel_resolution_xyz[0]
    ix = rem % voxel_resolution_xyz[0]
    occ[iz, iy, ix] = True

    voxel_size_zyx = np.array([voxel_size_xyz[2], voxel_size_xyz[1], voxel_size_xyz[0]], dtype=np.float32)
    return occ, idx_xyz, voxel_size_zyx


# ---------------------------------------------------------------------------
# Truncated unsigned distance field
# ---------------------------------------------------------------------------

def compute_truncated_distance_field(
    occupancy: np.ndarray,
    truncation_voxels: float,
    backend: str = "auto",
) -> np.ndarray:
    """Compute truncated unsigned distance field from a bool occupancy grid.

    Args:
        occupancy: Bool array of any shape.
        truncation_voxels: Clamp distance to this value (in voxel units).
        backend: ``"auto"`` uses cupy if available, else scipy.
                 ``"cupy"`` requires cupy/cupyx.
                 ``"scipy"`` always uses scipy (CPU).
    """
    if truncation_voxels <= 0:
        raise ValueError("truncation_voxels must be positive.")

    occupancy = np.asarray(occupancy, dtype=bool)
    trunc = float(truncation_voxels)

    use_cupy = (backend == "cupy") or (backend == "auto" and cp is not None and _edt_cupy is not None)

    if use_cupy:
        occ_gpu = cp.asarray(~occupancy)
        dist_gpu = _edt_cupy(occ_gpu, float64_distances=False)
        cp.clip(dist_gpu, 0.0, trunc, out=dist_gpu)
        return cp.asnumpy(dist_gpu).astype(np.float32)

    from scipy import ndimage
    dist = ndimage.distance_transform_edt(~occupancy).astype(np.float32)
    np.clip(dist, 0.0, trunc, out=dist)
    return dist
