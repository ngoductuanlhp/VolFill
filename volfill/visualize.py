"""Visualize a predicted TUDF as a point cloud in an interactive viser viewer.

Reads the inference outputs written by ``inference_latent_visible`` — a
``pred_tudf_*.npz`` volume (key ``"tudf"``, values in ``[0, truncation_voxels]``)
and a ``metadata.json`` (``bbox_min``, ``extent_xyz``) — converts the field to a
metric point cloud, and serves it with `viser <https://viser.studio>`_.

Example:
    # Run inference first, then point this at the output sample dir
    python -m volfill.visualize --sample_dir results/<sample> --threshold 0.8

    # Save a .ply alongside (no viewer / headless)
    python -m volfill.visualize --sample_dir results/<sample> --save_ply --no_viewer

Extra (optional) dependencies — installed only if you use this script:
    pip install viser            # interactive viewer (omit with --no_viewer)
    pip install matplotlib       # nicer height colormap (falls back otherwise)
    pip install scikit-image trimesh   # only for --method marching_cubes
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

_FIELD_GLOBS = {
    "pred":    ("pred_tudf_*.npz", "tudf"),
    "visible": ("visible_tudf_fine_*.npz", "tudf"),
    "gt":      ("gt_*.npz", "tudf"),
}


def load_sample(sample_dir: Path, field: str) -> tuple[np.ndarray, dict]:
    if field not in _FIELD_GLOBS:
        raise ValueError(f"--field must be one of {list(_FIELD_GLOBS)}, got {field!r}")
    pattern, npz_key = _FIELD_GLOBS[field]

    candidates = sorted(sample_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No {pattern} found in {sample_dir}")
    volume = np.load(candidates[0])
    if npz_key not in volume:
        raise KeyError(f"Expected key '{npz_key}' in {candidates[0]}")

    metadata_path = sample_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    return np.asarray(volume[npz_key], dtype=np.float32), metadata


# ---------------------------------------------------------------------------
# TUDF -> point cloud
# ---------------------------------------------------------------------------

def grid_centers_from_mask(mask: np.ndarray, bbox_min: np.ndarray, extent: np.ndarray) -> np.ndarray:
    res_z, res_y, res_x = mask.shape
    voxel_size = extent / np.array([res_x, res_y, res_z], dtype=np.float32)
    iz, iy, ix = np.where(mask)
    if iz.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return np.stack(
        [
            bbox_min[0] + (ix.astype(np.float32) + 0.5) * voxel_size[0],
            bbox_min[1] + (iy.astype(np.float32) + 0.5) * voxel_size[1],
            bbox_min[2] + (iz.astype(np.float32) + 0.5) * voxel_size[2],
        ],
        axis=-1,
    ).astype(np.float32)


def field_to_pointcloud(
    field_grid: np.ndarray,
    bbox_min: np.ndarray,
    extent: np.ndarray,
    method: str = "threshold",
    threshold: float = 0.5,
) -> np.ndarray:
    """Convert a TUDF grid (values in voxel units) into a metric point cloud.

    ``threshold``      : keep voxels with ``tudf <= threshold`` (near surface).
    ``marching_cubes`` : extract the ``level=threshold`` isosurface vertices
                         (requires scikit-image).
    """
    field_grid = np.asarray(field_grid, dtype=np.float32)
    if field_grid.ndim != 3:
        raise ValueError(f"Expected a 3D field grid, got shape {field_grid.shape}")

    if method == "threshold":
        return grid_centers_from_mask(field_grid <= float(threshold), bbox_min, extent)

    if method != "marching_cubes":
        raise ValueError(f"Unknown method: {method}")

    try:
        from skimage.measure import marching_cubes
    except ImportError as exc:  # pragma: no cover
        raise ImportError("--method marching_cubes requires scikit-image (pip install scikit-image).") from exc

    level = float(threshold)
    fmin, fmax = float(field_grid.min()), float(field_grid.max())
    if not (fmin <= level <= fmax):
        raise ValueError(f"level {level} outside field range [{fmin}, {fmax}]; adjust --threshold.")

    res_z, res_y, res_x = field_grid.shape
    voxel_size = extent / np.array([res_x, res_y, res_z], dtype=np.float32)
    verts_zyx, _, _, _ = marching_cubes(field_grid, level=level)
    points = np.empty((verts_zyx.shape[0], 3), dtype=np.float32)
    points[:, 0] = bbox_min[0] + verts_zyx[:, 2] * voxel_size[0]
    points[:, 1] = bbox_min[1] + verts_zyx[:, 1] * voxel_size[1]
    points[:, 2] = bbox_min[2] + verts_zyx[:, 0] * voxel_size[2]
    return points


# ---------------------------------------------------------------------------
# Colors / IO
# ---------------------------------------------------------------------------

def height_colors(points: np.ndarray, bbox_min: np.ndarray, extent: np.ndarray) -> np.ndarray:
    """Color points by height (z) through a colormap. Uses matplotlib's 'magma'
    if available, otherwise a simple blue->yellow numpy ramp."""
    if points.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    z = points[:, 2].astype(np.float64)
    t = np.clip((z - float(bbox_min[2])) / (float(extent[2]) + 1e-8), 0.0, 1.0)
    try:
        from matplotlib import colormaps
        rgb = colormaps["magma"](t)[:, :3]
    except Exception:
        # fallback: blue (low) -> yellow (high)
        rgb = np.stack([t, t, 1.0 - t], axis=-1)
    return (rgb * 255.0).astype(np.uint8)


def maybe_subsample(points: np.ndarray, colors: np.ndarray, max_points: int):
    if max_points <= 0 or points.shape[0] <= max_points:
        return points, colors
    keep = np.random.permutation(points.shape[0])[:max_points]
    return points[keep], colors[keep]


def save_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Write a binary PLY (xyz + rgb) with no external dependencies."""
    n = points.shape[0]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    verts = np.empty(
        n,
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
               ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    verts["x"], verts["y"], verts["z"] = points[:, 0], points[:, 1], points[:, 2]
    verts["red"], verts["green"], verts["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        f.write(verts.tobytes())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a predicted TUDF point cloud (viser).")
    parser.add_argument("--sample_dir", required=True, type=Path,
                        help="An inference output dir containing pred_tudf_*.npz + metadata.json.")
    parser.add_argument("--field", choices=list(_FIELD_GLOBS), default="pred")
    parser.add_argument("--method", choices=["threshold", "marching_cubes"], default="threshold")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="TUDF distance threshold in voxel units (near-surface / isosurface level).")
    parser.add_argument("--point_size", type=float, default=0.005)
    parser.add_argument("--max_points", type=int, default=300000)
    parser.add_argument("--save_ply", nargs="?", const="pred_points.ply", default=None,
                        help="Save the point cloud as a .ply (default name pred_points.ply in --sample_dir).")
    parser.add_argument("--no_viewer", action="store_true", help="Skip launching the viser viewer.")
    parser.add_argument("--port", type=int, default=7891)
    args = parser.parse_args()

    field_grid, metadata = load_sample(args.sample_dir, args.field)
    bbox_min = np.asarray(metadata["bbox_min"], dtype=np.float32)
    extent = np.asarray(metadata["extent_xyz"], dtype=np.float32)

    points = field_to_pointcloud(field_grid, bbox_min, extent, args.method, args.threshold)
    colors = height_colors(points, bbox_min, extent)
    points, colors = maybe_subsample(points, colors, args.max_points)

    print(f"Field {args.field} {field_grid.shape} -> {points.shape[0]} points "
          f"(method={args.method}, threshold={args.threshold})")

    if args.save_ply is not None:
        out = Path(args.save_ply)
        if not out.is_absolute() and out.parent == Path("."):
            out = args.sample_dir / out
        save_ply(out, points, colors)
        print(f"Saved point cloud -> {out}")

    if args.no_viewer:
        return

    try:
        import viser
        import viser.transforms as tf
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("The viewer needs viser (pip install viser), or pass --no_viewer.") from exc

    server = viser.ViserServer(port=args.port)
    server.scene.add_frame("/world", wxyz=tf.SO3.identity().wxyz, position=(0, 0, 0), show_axes=False)
    if points.shape[0] > 0:
        # Points keep their native +Y-down (camera) convention; tell the viewer -Y is up.
        server.scene.add_point_cloud(
            name="/world/pred_pointcloud",
            points=points,
            colors=colors,
            point_size=args.point_size,
            point_shape="rounded",
        )
    print(f"viser running at http://localhost:{args.port}  (Ctrl+C to exit)")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
