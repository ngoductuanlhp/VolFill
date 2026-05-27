"""Render 360° turntable videos of VolFill reconstructions.

For each scene listed in the manifest, renders a fixed-camera-elevation orbit
of the point cloud / mesh into videos/orbit_<scene_id>.mp4 and extracts the
first frame as posters/orbit_<scene_id>.jpg.

Usage:
    python scripts/render_orbits.py --manifest scenes.json

scenes.json layout (re-uses the same manifest as build_glb_assets.py; only
the "id" and "ours" fields are required here):
    [
      {"id": "scene01", "ours": "/abs/path/to/volfill_output.ply"},
      ...
    ]

Dependencies: open3d (headless), imageio[ffmpeg].
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    import open3d as o3d
except ImportError:
    print("open3d is required: pip install open3d", file=sys.stderr)
    sys.exit(1)

try:
    import imageio.v2 as imageio
except ImportError:
    print("imageio is required: pip install 'imageio[ffmpeg]'", file=sys.stderr)
    sys.exit(1)


FRAMES = 240          # 10 s @ 24 fps
FPS = 24
SIZE = (1280, 720)
ELEVATION = 25.0      # degrees above ground plane
RADIUS_MULT = 2.4     # camera distance = RADIUS_MULT × scene radius


def load_geometry(path: Path):
    if path.suffix.lower() in {".obj", ".ply"} and "mesh" in path.stem.lower():
        m = o3d.io.read_triangle_mesh(str(path))
        m.compute_vertex_normals()
        return m
    if path.suffix.lower() == ".obj":
        return o3d.io.read_triangle_mesh(str(path))
    return o3d.io.read_point_cloud(str(path))


def render_orbit(geom, out_mp4: Path, poster: Path):
    aabb = geom.get_axis_aligned_bounding_box()
    center = aabb.get_center()
    radius = np.linalg.norm(aabb.get_extent()) * 0.5
    cam_dist = radius * RADIUS_MULT

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=SIZE[0], height=SIZE[1], visible=False)
    vis.add_geometry(geom)
    opt = vis.get_render_option()
    opt.background_color = np.array([1.0, 1.0, 1.0])
    opt.point_size = 2.0

    ctr = vis.get_view_control()
    frames = []
    for i in range(FRAMES):
        theta = 2 * math.pi * i / FRAMES
        eye = center + np.array([
            cam_dist * math.cos(theta) * math.cos(math.radians(ELEVATION)),
            cam_dist * math.sin(theta) * math.cos(math.radians(ELEVATION)),
            cam_dist * math.sin(math.radians(ELEVATION)),
        ])
        ctr.set_lookat(center)
        ctr.set_front((center - eye) / np.linalg.norm(center - eye))
        ctr.set_up([0, 0, 1])
        ctr.set_zoom(0.7)
        vis.poll_events()
        vis.update_renderer()
        img = (np.asarray(vis.capture_screen_float_buffer(do_render=True)) * 255).astype(np.uint8)
        frames.append(img)

    vis.destroy_window()

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(out_mp4, fps=FPS, codec="libx264", quality=8, macro_block_size=1)
    for f in frames:
        writer.append_data(f)
    writer.close()
    print(f"  wrote {out_mp4}")

    poster.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(poster, frames[0])
    print(f"  wrote {poster}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()

    site_root = Path(__file__).resolve().parents[1]
    with open(args.manifest) as f:
        manifest = json.load(f)

    for entry in manifest:
        scene_id = entry["id"]
        src = entry.get("ours")
        if not src:
            print(f"[{scene_id}] no 'ours' entry — skipping")
            continue
        print(f"[{scene_id}]")
        geom = load_geometry(Path(src))
        out_mp4 = site_root / "videos" / f"orbit_{scene_id}.mp4"
        poster = site_root / "posters" / f"orbit_{scene_id}.jpg"
        render_orbit(geom, out_mp4, poster)

    print(f"\nDone. {len(manifest)} scenes rendered.")


if __name__ == "__main__":
    main()
