"""Convert VolFill inference outputs (.ply / .obj) into web-ready .glb files.

Each scene becomes a folder under VolFill_ghpage/models/<scene_id>/ containing:
    input.jpg     # input RGB image
    ours.glb      # VolFill prediction
    gt.glb        # ground-truth (optional)
    nova3r.glb    # NOVA3R baseline (optional)
    trellis.glb   # TRELLIS baseline (optional)
    thumb.png     # off-axis render for the thumbnail row

Usage (pass a manifest JSON):
    python scripts/build_glb_assets.py --manifest scenes.json --out models/

scenes.json layout:
    [
      {
        "id": "scene01",
        "input": "/abs/path/to/image.jpg",
        "ours":  "/abs/path/to/volfill_output.ply",
        "gt":    "/abs/path/to/gt_pointcloud.ply",
        "nova3r":"/abs/path/to/nova3r_output.ply",
        "trellis":"/abs/path/to/trellis_output.obj"
      },
      ...
    ]

Point clouds are subsampled to MAX_POINTS (default 150k) so a GLB stays
under ~10 MB. Meshes are passed through trimesh unchanged.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import trimesh

MAX_POINTS = 150_000
THUMB_AZIMUTH_DEG = 35.0
THUMB_ELEVATION_DEG = 20.0
THUMB_SIZE = (320, 240)


def load_geometry(path: Path):
    obj = trimesh.load(path, force="scene")
    geometries = list(obj.geometry.values()) if isinstance(obj, trimesh.Scene) else [obj]
    return geometries[0] if len(geometries) == 1 else trimesh.util.concatenate(geometries)


def subsample_point_cloud(pc: trimesh.PointCloud, max_pts: int) -> trimesh.PointCloud:
    n = len(pc.vertices)
    if n <= max_pts:
        return pc
    idx = np.random.default_rng(42).choice(n, size=max_pts, replace=False)
    colors = pc.colors[idx] if pc.colors is not None and len(pc.colors) == n else None
    return trimesh.PointCloud(vertices=pc.vertices[idx], colors=colors)


def export_glb(src: Path, dst: Path):
    geom = load_geometry(src)
    if isinstance(geom, trimesh.PointCloud):
        geom = subsample_point_cloud(geom, MAX_POINTS)
    scene = geom if isinstance(geom, trimesh.Scene) else trimesh.Scene(geom)
    dst.parent.mkdir(parents=True, exist_ok=True)
    scene.export(dst, file_type="glb")
    print(f"  wrote {dst} ({dst.stat().st_size / 1024:.1f} KB)")


def render_thumb(src: Path, dst: Path):
    """Off-axis render via trimesh's pyglet viewer. Falls back to a blank PNG if rendering fails."""
    try:
        geom = load_geometry(src)
        scene = geom if isinstance(geom, trimesh.Scene) else trimesh.Scene(geom)
        png_bytes = scene.save_image(
            resolution=THUMB_SIZE,
            visible=False,
        )
        if png_bytes:
            dst.write_bytes(png_bytes)
            print(f"  wrote {dst}")
            return
    except Exception as exc:
        print(f"  thumb render failed for {src}: {exc}")
    blank = np.full((*THUMB_SIZE[::-1], 3), 220, dtype=np.uint8)
    trimesh.util.PIL_enabled and __import__("PIL.Image").Image.fromarray(blank).save(dst)
    print(f"  wrote placeholder {dst}")


def process_scene(entry: dict, out_root: Path):
    scene_id = entry["id"]
    scene_dir = out_root / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[{scene_id}] -> {scene_dir}")

    if "input" in entry:
        in_src = Path(entry["input"])
        shutil.copy(in_src, scene_dir / "input.jpg")
        print(f"  copied input image")

    for variant in ("ours", "gt", "nova3r", "trellis"):
        src = entry.get(variant)
        if not src:
            continue
        export_glb(Path(src), scene_dir / f"{variant}.glb")

    primary = entry.get("ours") or entry.get("gt")
    if primary:
        render_thumb(Path(primary), scene_dir / "thumb.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="JSON list of scene entries")
    ap.add_argument("--out", default="models", help="Output directory (relative to site root)")
    args = ap.parse_args()

    site_root = Path(__file__).resolve().parents[1]
    out_root = site_root / args.out
    out_root.mkdir(parents=True, exist_ok=True)

    with open(args.manifest) as f:
        manifest = json.load(f)

    for entry in manifest:
        process_scene(entry, out_root)

    print(f"\nDone. {len(manifest)} scenes written under {out_root}.")


if __name__ == "__main__":
    main()
