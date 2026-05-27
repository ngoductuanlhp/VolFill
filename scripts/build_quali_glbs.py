"""Convert VolFill / baseline qualitative outputs into web-ready .glb files.

Scans `VolFill/results/quali_results/` for paired samples across six methods
and writes one `.glb` per (sample, method) into the project-page `models/`
directory. The .glb files are GLTF point primitives that Google's
`<model-viewer>` renders natively, so they drop into the Qualitative Results
section without further processing.

Input layout (per sample, identified by `{dataset}/{scene_id}/{frame_id}`):

    quali_results/
      visible_flow_trellis_.../<dataset>/<scene>/<frame>/   # Ours (TUDF)
          pred_tudf_256.npz, metadata.json, image.jpg
      eval_da3/<dataset>/<scene>/<frame>/                   # PLY
          pred_points.ply, metadata.json, image.jpg
      eval_moge2/...                                        # PLY
      eval_vggt/...                                         # PLY
      eval_lari_pointmap/...                                # PLY
      eval_nova3r_single_view/...                           # PLY

Output layout:

    VolFill_ghpage/models/<scene_id>/
        input.jpg          (copied from any available method)
        ours.glb
        da3.glb            (optional — only if pred_points.ply exists)
        moge2.glb          (optional)
        vggt.glb           (optional)
        lari.glb           (optional)
        nova3r.glb         (optional)

`<scene_id>` defaults to `{dataset}_{scene}_{frame}`. After running you can
rename or symlink directories to short names (`scene01`, …) and update the
`data-stem` attributes in `index.html`.

Usage:
    cd VolFill_ghpage
    python scripts/build_quali_glbs.py                       # convert everything
    python scripts/build_quali_glbs.py --samples list.txt    # only these
    python scripts/build_quali_glbs.py --dry_run             # list samples, no writes

Requirements (volfill env or a fresh one): numpy, trimesh, open3d, matplotlib.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Method registry — (slug, source-dir, label, loader)
# ---------------------------------------------------------------------------

OURS_DIR = "visible_flow_trellis_scannetpp_3dfront_hypersim_16x_add_finetune_50steps_align_rotation"

METHOD_DIRS: dict[str, str] = {
    "ours":   OURS_DIR,
    "da3":    "eval_da3",
    "moge2":  "eval_moge2",
    "vggt":   "eval_vggt",
    "lari":   "eval_lari_pointmap",
    "nova3r": "eval_nova3r_single_view",
}

DEFAULT_MAX_POINTS = 150_000
DEFAULT_THRESHOLD = 0.5

# Inline plasma colormap (matplotlib's plasma, 11 evenly-spaced control points).
# Keeps the script free of a matplotlib dependency.
_PLASMA_LUT = np.array([
    [0.050, 0.030, 0.528],
    [0.215, 0.018, 0.599],
    [0.354, 0.006, 0.626],
    [0.483, 0.015, 0.616],
    [0.602, 0.085, 0.564],
    [0.712, 0.216, 0.466],
    [0.815, 0.368, 0.355],
    [0.910, 0.540, 0.234],
    [0.944, 0.633, 0.151],
    [0.987, 0.812, 0.124],
    [0.940, 0.975, 0.131],
], dtype=np.float64)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_ply(path: Path, max_points: int, remove_outliers: bool) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(str(path))
    if remove_outliers:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pts = np.asarray(pcd.points, dtype=np.float32)
    if pts.shape[0] == 0:
        return None
    if max_points > 0 and pts.shape[0] > max_points:
        idx = np.random.permutation(pts.shape[0])[:max_points]
        pts = pts[idx]
    return pts


def _load_tudf_pointcloud(
    sample_dir: Path, threshold: float, max_points: int
) -> Optional[np.ndarray]:
    """Convert pred_tudf_*.npz → point cloud via thresholding."""
    candidates = sorted(sample_dir.glob("pred_tudf_*.npz"))
    meta_path = sample_dir / "metadata.json"
    if not candidates or not meta_path.exists():
        return None

    field = np.asarray(np.load(candidates[0])["tudf"], dtype=np.float32)
    meta = json.loads(meta_path.read_text())
    bbox_min = np.asarray(meta["bbox_min"], dtype=np.float32)
    extent   = np.asarray(meta["extent_xyz"], dtype=np.float32)

    mask = field <= float(threshold)
    res_z, res_y, res_x = field.shape
    voxel = extent / np.array([res_x, res_y, res_z], dtype=np.float32)
    iz, iy, ix = np.where(mask)
    if iz.size == 0:
        return None
    pts = np.stack([
        bbox_min[0] + (ix.astype(np.float32) + 0.5) * voxel[0],
        bbox_min[1] + (iy.astype(np.float32) + 0.5) * voxel[1],
        bbox_min[2] + (iz.astype(np.float32) + 0.5) * voxel[2],
    ], axis=-1).astype(np.float32)
    if max_points > 0 and pts.shape[0] > max_points:
        idx = np.random.permutation(pts.shape[0])[:max_points]
        pts = pts[idx]
    return pts


# ---------------------------------------------------------------------------
# Transform + colorize
# ---------------------------------------------------------------------------

def _normalize_and_orient(pts: np.ndarray) -> np.ndarray:
    """Center + scale to fit [-0.7, 0.7]^3, then rotate 180° about X so the
    point cloud sits in front of GLTF's default camera with the correct
    handedness. OpenCV uses +X right / +Y down / +Z into scene; GLTF uses
    +X right / +Y up / +Z toward viewer — so we negate Y *and* Z. Negating
    only Y reflects the scene (left-handed), causing the apparent horizontal
    mirror in `<model-viewer>`.
    """
    centroid = pts.mean(axis=0)
    scale = float(np.abs(pts - centroid).max()) + 1e-8
    out = (pts - centroid) / scale * 0.7
    out[:, 1] *= -1.0  # OpenCV +Y-down → GLTF +Y-up
    out[:, 2] *= -1.0  # OpenCV +Z-into-scene → GLTF +Z-toward-viewer
    return out


def _height_colors(pts: np.ndarray) -> np.ndarray:
    """Plasma colormap over the (post-orientation) y-axis = scene up-axis in GLTF.

    Uses matplotlib's plasma when available; falls back to a small inline LUT.
    """
    if pts.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.uint8)
    y = pts[:, 1].astype(np.float64)
    t = np.clip((y - y.min()) / (y.max() - y.min() + 1e-8), 0.0, 1.0)
    try:
        from matplotlib import colormaps
        rgba = colormaps["plasma"](t)
    except ImportError:
        n = _PLASMA_LUT.shape[0] - 1
        f = t * n
        i0 = np.floor(f).astype(np.int64)
        i1 = np.minimum(i0 + 1, n)
        w = (f - i0)[:, None]
        rgb = _PLASMA_LUT[i0] * (1.0 - w) + _PLASMA_LUT[i1] * w
        rgba = np.concatenate([rgb, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    return (rgba * 255.0).astype(np.uint8)  # (N, 4) RGBA


def _write_glb(pts: np.ndarray, colors_rgba: np.ndarray, out_path: Path) -> None:
    import trimesh
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pc = trimesh.PointCloud(vertices=pts.astype(np.float32), colors=colors_rgba)
    scene = trimesh.Scene([pc])
    scene.export(out_path, file_type="glb")


# ---------------------------------------------------------------------------
# Sample discovery
# ---------------------------------------------------------------------------

def _discover_samples(quali_root: Path) -> list[str]:
    """Scan the Ours directory for any leaf that has metadata.json + a
    pred_tudf_*.npz. Sample IDs are the relative path from the Ours root,
    so this works for both `<dataset>/<scene>/<frame>` (e.g. `scrream/...`)
    and `<group>/<scene>` (e.g. `custom/eth1`) layouts.
    """
    ours_root = quali_root / METHOD_DIRS["ours"]
    if not ours_root.exists():
        print(f"[discover] Ours root not found: {ours_root}", file=sys.stderr)
        return []
    samples: list[str] = []
    for meta_path in sorted(ours_root.rglob("metadata.json")):
        sample_dir = meta_path.parent
        has_pred = (any(sample_dir.glob("pred_tudf_*.npz"))
                    or (sample_dir / "pred_points.ply").exists())
        if has_pred:
            samples.append(sample_dir.relative_to(ours_root).as_posix())
    return samples


def _scene_id(sample: str) -> str:
    """Flatten 'scrream/scene01_full_00/000140' → 'scrream_scene01_full_00_000140'."""
    return sample.replace("/", "_")


# ---------------------------------------------------------------------------
# Per-sample conversion
# ---------------------------------------------------------------------------

def convert_sample(
    sample: str,
    quali_root: Path,
    out_root: Path,
    threshold: float,
    max_points: int,
    remove_outliers: bool,
) -> dict[str, str]:
    """Convert one sample across all methods. Returns a dict of method → status string."""
    scene_id = _scene_id(sample)
    out_dir = out_root / scene_id
    out_dir.mkdir(parents=True, exist_ok=True)
    statuses: dict[str, str] = {}

    # Locate any image.jpg / metadata.json once for the input thumbnail copy.
    input_jpg_src: Optional[Path] = None

    for method, source_dir in METHOD_DIRS.items():
        sample_dir = quali_root / source_dir / sample
        if not sample_dir.exists():
            statuses[method] = "missing dir"
            continue

        if method == "ours":
            pts = _load_tudf_pointcloud(sample_dir, threshold, max_points)
        else:
            pts = _load_ply(sample_dir / "pred_points.ply", max_points, remove_outliers)

        if pts is None:
            statuses[method] = "no points"
            continue

        if input_jpg_src is None:
            img_path = sample_dir / "image.jpg"
            if img_path.exists():
                input_jpg_src = img_path

        pts_oriented = _normalize_and_orient(pts)
        colors = _height_colors(pts_oriented)
        _write_glb(pts_oriented, colors, out_dir / f"{method}.glb")
        statuses[method] = f"{pts_oriented.shape[0]:,} pts"

    if input_jpg_src is not None:
        shutil.copy(input_jpg_src, out_dir / "input.jpg")
        statuses["_input"] = "copied"

    return statuses


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[Iterable[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    default_quali = (Path(__file__).resolve().parents[2] / "VolFill" / "results" / "quali_results")
    default_out   = (Path(__file__).resolve().parents[1] / "models")
    ap.add_argument("--quali_root", type=Path, default=default_quali,
                    help=f"Root dir of method-named subfolders (default: {default_quali}).")
    ap.add_argument("--out_dir", type=Path, default=default_out,
                    help=f"Output models directory (default: {default_out}).")
    ap.add_argument("--samples", type=Path, default=None,
                    help="Optional manifest: text file with one 'dataset/scene/frame' per line.")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help="TUDF surface threshold for the ours pointcloud (default 0.5).")
    ap.add_argument("--max_points", type=int, default=DEFAULT_MAX_POINTS,
                    help="Subsample each cloud to at most this many points (default 150k).")
    ap.add_argument("--no_outlier_removal", action="store_true",
                    help="Skip statistical outlier removal on .ply clouds.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry_run", action="store_true",
                    help="List samples that would be converted and exit.")
    args = ap.parse_args(argv)

    np.random.seed(args.seed)

    if args.samples is not None:
        samples = [ln.strip() for ln in args.samples.read_text().splitlines() if ln.strip()]
    else:
        samples = _discover_samples(args.quali_root)

    if not samples:
        print("No samples found.", file=sys.stderr)
        sys.exit(1)

    print(f"Source : {args.quali_root}")
    print(f"Output : {args.out_dir}")
    print(f"Samples: {len(samples)}\n")
    for s in samples:
        print(f"  • {s}  →  {_scene_id(s)}")
    if args.dry_run:
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    remove_outliers = not args.no_outlier_removal

    for s in samples:
        print(f"\n[{_scene_id(s)}]")
        statuses = convert_sample(
            s, args.quali_root, args.out_dir,
            threshold=args.threshold,
            max_points=args.max_points,
            remove_outliers=remove_outliers,
        )
        for method, status in statuses.items():
            print(f"  {method:<8} {status}")

    print(f"\nDone. Wrote into {args.out_dir}.")
    print("Now edit VolFill_ghpage/index.html: update the `data-stem` of each scene "
          "thumbnail in Row A to the new scene IDs, and add `da3`/`moge2`/`vggt` thumbs "
          "to Row B if you want those baselines visible.")


if __name__ == "__main__":
    main()
