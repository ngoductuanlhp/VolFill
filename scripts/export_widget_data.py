"""Export data for the two interactive widgets.

W1  TUDF slice   -> media/widgets/tudf-slice.png (8-bit distance, voxels) + tudf-slice.json
    python scripts/export_widget_data.py tudf --glb models/<id>/ours.glb [--height 0.32]
    python scripts/export_widget_data.py tudf --npz <sample>/tudf_fine_256.npz --up 1   (GT grid, on AWS)

W2  Denoising trajectory -> media/widgets/flow/<name>/step_XX.webp + flow.json
    python scripts/export_widget_data.py flow --traj_dir runs/web_traj/out/<scene>/<frame> --name scene09

Teaser poster -> media/teaser-poster.webp (shown until <model-viewer> loads)
    python scripts/export_widget_data.py poster --glb models/$(head -1 models/valid_samples_teaser.txt)/ours.glb

Points live in the OpenCV camera frame (+x right, +y down, +z forward), matching build_quali_glbs.py.
"""
import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "media" / "widgets"
DMAX = 24.0  # voxels; stored range of the untruncated distance

PLASMA = np.array([
    [0.050, 0.030, 0.528], [0.215, 0.018, 0.599], [0.354, 0.006, 0.626], [0.483, 0.015, 0.616],
    [0.602, 0.085, 0.564], [0.712, 0.216, 0.466], [0.815, 0.368, 0.355], [0.910, 0.540, 0.234],
    [0.944, 0.633, 0.151], [0.987, 0.812, 0.124], [0.940, 0.975, 0.131],
])


def plasma(t):
    t = np.clip(t, 0, 1) * (len(PLASMA) - 1)
    i = np.minimum(t.astype(int), len(PLASMA) - 2)
    f = (t - i)[:, None]
    return PLASMA[i] * (1 - f) + PLASMA[i + 1] * f


def load_glb_points(path, with_colors=False):
    data = Path(path).read_bytes()
    _, _, _ = struct.unpack_from("<III", data, 0)
    jlen, _ = struct.unpack_from("<II", data, 12)
    gltf = json.loads(data[20:20 + jlen])
    blen, _ = struct.unpack_from("<II", data, 20 + jlen)
    binbuf = data[28 + jlen:28 + jlen + blen]
    acc = gltf["accessors"][gltf["meshes"][0]["primitives"][0]["attributes"]["POSITION"]]
    view = gltf["bufferViews"][acc["bufferView"]]
    off = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    pts = np.frombuffer(binbuf, np.float32, acc["count"] * 3, off).reshape(-1, 3).copy()
    pts[:, 1] *= -1  # GLTF -> OpenCV
    pts[:, 2] *= -1
    if not with_colors:
        return pts
    cacc = gltf["accessors"][gltf["meshes"][0]["primitives"][0]["attributes"]["COLOR_0"]]
    cview = gltf["bufferViews"][cacc["bufferView"]]
    coff = cview.get("byteOffset", 0) + cacc.get("byteOffset", 0)
    col = np.frombuffer(binbuf, np.uint8, cacc["count"] * 4, coff).reshape(-1, 4)[:, :3] / 255.0
    return pts, col


# ----------------------------------------------------------------------------- W1

def slice_from_points(pts, height, max_px):
    tree = cKDTree(pts)
    voxel = float(np.median(tree.query(pts[::50], k=2)[0][:, 1]))
    y_floor, y_ceil = np.percentile(pts[:, 1], [99.5, 0.5])  # +y is down
    y0 = y_floor - height * (y_floor - y_ceil)
    (x0, z0), (x1, z1) = pts[:, [0, 2]].min(0), pts[:, [0, 2]].max(0)
    pad = 4 * voxel
    x0, z0, x1, z1 = x0 - pad, z0 - pad, x1 + pad, z1 + pad
    step = max(voxel, max(x1 - x0, z1 - z0) / max_px)
    xs = np.arange(x0, x1, step)
    zs = np.arange(z1, z0, -step)  # far side at the top, camera at the bottom
    gx, gz = np.meshgrid(xs, zs)
    q = np.stack([gx.ravel(), np.full(gx.size, y0), gz.ravel()], 1)
    d, _ = tree.query(q, distance_upper_bound=DMAX * voxel * 1.5)
    d = np.minimum(np.nan_to_num(d / voxel, posinf=DMAX), DMAX).reshape(gx.shape)
    return d, {"voxel_size": voxel, "px_per_voxel": voxel / step, "height_frac": height}


def slice_from_grid(tudf, up, height):
    surf = tudf < 0.5
    d = distance_transform_edt(~surf)
    occ = np.nonzero(surf.any(axis=tuple(a for a in range(3) if a != up)))[0]
    k = int(round(occ.max() - height * (occ.max() - occ.min())))
    sl = np.take(d, k, axis=up)
    return np.minimum(sl, DMAX), {"voxel_size": 1.0, "px_per_voxel": 1.0, "height_frac": height, "slice_index": k}


def export_tudf(args):
    if args.glb:
        d, meta = slice_from_points(load_glb_points(args.glb), args.height, args.max_px)
        meta["source"] = Path(args.glb).parent.name
    else:
        tudf = np.load(args.npz)["tudf"].astype(np.float32)
        d, meta = slice_from_grid(tudf, args.up, args.height)
        meta["source"] = Path(args.npz).name
    OUT.mkdir(parents=True, exist_ok=True)
    q = np.round(d / DMAX * 255).astype(np.uint8)
    Image.fromarray(q, "L").save(OUT / "tudf-slice.png", optimize=True)
    meta.update({"dmax": DMAX, "tau": 3, "width": int(q.shape[1]), "height": int(q.shape[0]),
                 "camera": "bottom", "surface_fraction_tau3": float((d < 3).mean())})
    (OUT / "tudf-slice.json").write_text(json.dumps(meta, indent=1))
    print(f"tudf-slice.png {q.shape[1]}x{q.shape[0]}  band(<3 vox) = {meta['surface_fraction_tau3']:.1%}")


# ----------------------------------------------------------------------------- W2

def view_basis(yaw_deg, elev_deg):
    """Orthographic look-at from behind/above the input camera (OpenCV frame: -y is up)."""
    yaw, el = np.radians(yaw_deg), np.radians(elev_deg)
    f = -np.array([np.sin(yaw) * np.cos(el), -np.sin(el), -np.cos(yaw) * np.cos(el)])
    right = np.cross(f, [0.0, -1.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, f)
    return right, up, f


def project(pts, basis):
    right, up, f = basis
    return pts @ right, -(pts @ up), pts @ f


def render_points(pts, colors, size, basis, fit, splat=2):
    W, H = size
    u, v, depth = project(pts, basis)
    (u0, v0), scale = fit
    u = ((u - u0) * scale + W / 2).astype(int)
    v = ((v - v0) * scale + H / 2).astype(int)
    order = np.argsort(-depth)  # far first, near overwrite
    img = np.zeros((H, W, 4), np.uint8)
    rgb = (colors * 255).astype(np.uint8)[order]
    u, v = u[order], v[order]
    for du in range(splat):
        for dv in range(splat):
            uu, vv = u + du, v + dv
            ok = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            img[vv[ok], uu[ok], :3] = rgb[ok]
            img[vv[ok], uu[ok], 3] = 255
    return img


def tudf_to_points(tudf, meta, thr):
    """Voxel centres with TUDF < thr. VolFill grids are indexed (z, y, x), bbox is (x, y, z)."""
    idx = np.argwhere(tudf < thr).astype(np.float32)[:, ::-1]
    if len(idx) == 0:
        return np.zeros((0, 3), np.float32)
    lo, hi = np.array(meta["bbox_min"]), np.array(meta["bbox_max"])
    return lo + (idx + 0.5) / np.array(tudf.shape[::-1], np.float32) * (hi - lo)


def export_flow(args):
    traj = Path(args.traj_dir)
    steps = sorted(traj.glob("step_*"))
    assert steps, f"no step_* dirs in {traj}"
    final_meta = json.loads((steps[-1] / "metadata.json").read_text())
    final = tudf_to_points(np.load(steps[-1] / "pred_tudf_256.npz")["tudf"], final_meta, args.thr)
    y_lo, y_hi = np.percentile(final[:, 1], [0.5, 99.5])
    W, H = args.size
    basis = view_basis(args.yaw, args.elev)
    fu, fv, _ = project(final, basis)
    lo_u, hi_u = np.percentile(fu, [0.2, 99.8])
    lo_v, hi_v = np.percentile(fv, [0.2, 99.8])
    fit = (((lo_u + hi_u) / 2, (lo_v + hi_v) / 2), 0.9 * min(W / (hi_u - lo_u), H / (hi_v - lo_v)))

    out = OUT / "flow" / args.name
    out.mkdir(parents=True, exist_ok=True)
    img_src = traj / "image.jpg"
    if args.color == "visibility":
        # Paper colouring (Fig. 1): first hit along each input-camera ray = photo colour, rest slate.
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from build_quali_glbs import _camera_K, _visibility_colors
        photo = np.asarray(Image.open(img_src).convert("RGB"))
        if args.intrinsics is not None:
            Kn = np.load(args.intrinsics)["intrinsics"].astype(np.float64)
            Hp, Wp = photo.shape[:2]
            K = np.array([[Kn[0, 0] * Wp, 0, Kn[0, 2] * Wp], [0, Kn[1, 1] * Hp, Kn[1, 2] * Hp], [0, 0, 1]])
        else:
            K = _camera_K(traj, "", None, photo.shape[:2])          # pixel intrinsics from metadata.json
        assert K is not None, "--color visibility needs --intrinsics <moge.npz> (or intrinsics in metadata.json)"
        voxel = float(np.asarray(final_meta["extent_xyz"])[0]) / 256
    keep = sorted(set(np.linspace(0, len(steps) - 1, min(args.frames, len(steps))).round().astype(int)))
    frames = []
    for k in keep:
        s = steps[k]
        meta = json.loads((s / "metadata.json").read_text())
        pts = tudf_to_points(np.load(s / "pred_tudf_256.npz")["tudf"], meta, args.thr)
        if len(pts) > args.max_points:
            pts = pts[np.random.default_rng(0).choice(len(pts), args.max_points, replace=False)]
        if args.color == "visibility":
            col = _visibility_colors(pts, photo, K, voxel)[:, :3] / 255.0
        else:
            col = plasma(1 - (pts[:, 1] - y_lo) / (y_hi - y_lo + 1e-8))  # +y down -> invert
        img = render_points(pts, col, (W, H), basis, fit)
        name = f"step_{k:02d}.webp"
        Image.fromarray(img, "RGBA").save(out / name, "WEBP", quality=82, method=6)
        # Code integrates t from 1 (noise) to 0; the page uses the paper's t (0 = noise, 1 = data).
        frames.append({"src": name, "step": int(meta["step"]) + 1, "t": round(1 - float(meta["t_prev"]), 4),
                       "points": int(len(pts))})
        print(f"  {name}  t={frames[-1]['t']:.2f}  pts={len(pts)}")
    if img_src.exists():
        Image.open(img_src).convert("RGB").save(out / "input.jpg", quality=85)
    (out / "flow.json").write_text(json.dumps({"total_steps": final_meta["total_steps"], "frames": frames}, indent=1))
    total = sum(f.stat().st_size for f in out.glob("*.webp"))
    print(f"{len(frames)} frames -> {out}  ({total / 1e6:.2f} MB)")


def export_poster(args):
    """Static stand-in for the hero <model-viewer> (same default orbit: front, 15 deg up)."""
    pts, col = load_glb_points(args.glb, with_colors=True)
    W, H = args.size
    basis = view_basis(args.yaw, args.elev)
    u, v, _ = project(pts, basis)
    lo_u, hi_u = np.percentile(u, [0.2, 99.8])
    lo_v, hi_v = np.percentile(v, [0.2, 99.8])
    fit = (((lo_u + hi_u) / 2, (lo_v + hi_v) / 2), args.fill * min(W / (hi_u - lo_u), H / (hi_v - lo_v)))
    img = render_points(pts, col, (W, H), basis, fit, splat=3)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img, "RGBA").save(out, "WEBP", quality=80, method=6)
    print(f"{out.relative_to(ROOT)}  {W}x{H}  {out.stat().st_size // 1024} KB")


# ----------------------------------------------------------------------------- S3 (real VAE stages)

def gravity_frame(P, vox, iters=400, seed=0):
    """Rotation Rg (p' = Rg p) that levels the floor (up = -y, OpenCV). Floor = dominant near-horizontal
    plane among the lowest points (RANSAC + SVD refit); same procedure as the paper's figure renders."""
    rng = np.random.default_rng(seed)
    low = P[P[:, 1] > np.percentile(P[:, 1], 60)]
    low = low[rng.choice(len(low), min(len(low), 40000), replace=False)]
    best, bn, bd = 0, None, None
    for _ in range(iters):
        a, b, c = low[rng.choice(len(low), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        if abs(n[1]) < np.cos(np.radians(45)):
            continue
        cnt = int((np.abs(low @ n - n @ a) < 2 * vox).sum())
        if cnt > best:
            best, bn, bd = cnt, n, -n @ a
    inl = low[np.abs(low @ bn + bd) < 2 * vox]
    n = np.linalg.svd(inl - inl.mean(0))[2][2]
    up = -n if n[1] > 0 else n
    v, cth = np.cross(up, [0.0, -1.0, 0.0]), float(up @ [0.0, -1.0, 0.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + cth)


def export_vae(args):
    """One horizontal, gravity-aligned slice through the real VAE stages for S3: distance to the GT surface,
    distance to the VAE reconstruction, and the decoder's predicted 64^3 occupancy cells cut by the plane."""
    meta = json.loads(Path(args.meta).read_text())
    gt, rec, occ = np.load(args.gt)["tudf"], np.load(args.recon)["tudf"], np.load(args.occ)
    lo, hi = np.array(meta["bbox_min"]), np.array(meta["bbox_max"])
    vox = float(hi[0] - lo[0]) / gt.shape[0]
    Rg = gravity_frame(tudf_to_points(gt, meta, 1.0), vox)
    G, R = tudf_to_points(gt, meta, 1.0) @ Rg.T, tudf_to_points(rec, meta, 1.0) @ Rg.T
    y_floor, y_ceil = np.percentile(G[:, 1], [99.5, 0.5])          # +y is down
    y0 = y_floor - args.height * (y_floor - y_ceil)
    (x0, z0), (x1, z1) = G[:, [0, 2]].min(0) - 4 * vox, G[:, [0, 2]].max(0) + 4 * vox
    xs, zs = np.arange(x0, x1, vox), np.arange(z1, z0, -vox)      # far side at the top, camera at the bottom
    gx, gz = np.meshgrid(xs, zs)
    q = np.stack([gx.ravel(), np.full(gx.size, y0), gz.ravel()], 1)

    def dist(P):
        d, _ = cKDTree(P).query(q, distance_upper_bound=DMAX * vox * 1.5)
        return np.minimum(np.nan_to_num(d / vox, posinf=DMAX), DMAX).reshape(gx.shape)

    # occupancy on the plane: sample each 2D cell (2x2 sub-samples), map back to the grid frame, look up occ
    cell = (hi[0] - lo[0]) / occ.shape[0]
    nr, nc = int(np.ceil(len(zs) * vox / cell)), int(np.ceil(len(xs) * vox / cell))
    sub = (np.arange(2) + 0.5) / 2
    cx = x0 + (np.arange(nc)[:, None] + sub[None]).ravel() * cell
    cz = z1 - (np.arange(nr)[:, None] + sub[None]).ravel() * cell
    qx, qz = np.meshgrid(cx, cz)
    P = np.stack([qx.ravel(), np.full(qx.size, y0), qz.ravel()], 1) @ Rg     # gravity frame -> camera frame
    ijk = np.floor((P - lo) / cell).astype(int)                                 # (x, y, z) cell index
    inside = ((ijk >= 0) & (ijk < occ.shape[0])).all(1)
    hit = np.zeros(len(P), bool)
    hit[inside] = occ[ijk[inside, 2], ijk[inside, 1], ijk[inside, 0]] > args.occ_thr
    m = hit.reshape(nr, 2, nc, 2).any(axis=(1, 3))
    OUT.mkdir(parents=True, exist_ok=True)
    for name, d in (("gt", dist(G)), ("recon", dist(R))):
        Image.fromarray(np.round(d / DMAX * 255).astype(np.uint8), "L").save(OUT / f"vae-slice-{name}.png", optimize=True)
    Image.fromarray(m.astype(np.uint8) * 255, "L").save(OUT / "vae-slice-occ.png", optimize=True)
    info = {"source": args.source, "height_frac": args.height, "voxel_size": vox, "cell_px": float(cell / vox),
            "dmax": DMAX, "width": len(xs), "height": len(zs), "occ_cells": [int(m.shape[1]), int(m.shape[0])],
            "occ_kept": int(m.sum())}
    (OUT / "vae-slice.json").write_text(json.dumps(info, indent=1))
    print(f"vae-slice {len(xs)}x{len(zs)} px, occupancy {m.shape[1]}x{m.shape[0]} cells ({m.sum()} kept)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tudf")
    src = t.add_mutually_exclusive_group(required=True)
    src.add_argument("--glb")
    src.add_argument("--npz")
    t.add_argument("--up", type=int, default=1, help="grid axis pointing down/up (npz only)")
    t.add_argument("--height", type=float, default=0.4, help="slice height as a fraction floor->ceiling")
    t.add_argument("--max_px", type=int, default=320)
    f = sub.add_parser("flow")
    f.add_argument("--traj_dir", required=True)
    f.add_argument("--name", required=True)
    f.add_argument("--thr", type=float, default=1.0, help="TUDF threshold (voxels) for surface points")
    f.add_argument("--frames", type=int, default=50)
    f.add_argument("--size", type=int, nargs=2, default=[800, 600])
    f.add_argument("--yaw", type=float, default=-30)
    f.add_argument("--elev", type=float, default=38)
    f.add_argument("--max_points", type=int, default=400_000)
    f.add_argument("--color", choices=["visibility", "height"], default="visibility",
                   help="visibility = paper colouring (photo / slate hidden); height = old plasma")
    f.add_argument("--intrinsics", default=None, help="moge.npz with normalised intrinsics of the input photo")
    va = sub.add_parser("vae")
    va.add_argument("--gt", required=True, help="GT tudf npz (key tudf, voxel units)")
    va.add_argument("--recon", required=True, help="VAE reconstruction of --gt (same format)")
    va.add_argument("--occ", required=True, help="decoder occupancy probabilities, 64^3 .npy")
    va.add_argument("--meta", required=True, help="metadata.json with bbox_min / bbox_max")
    va.add_argument("--height", type=float, default=0.4, help="slice height, fraction floor -> top")
    va.add_argument("--occ_thr", type=float, default=0.5)
    va.add_argument("--source", default="nrgbd_kitchen_001435")
    po = sub.add_parser("poster")
    po.add_argument("--glb", required=True)
    po.add_argument("--out", default="media/teaser-poster.webp")
    po.add_argument("--size", type=int, nargs=2, default=[1280, 656])
    po.add_argument("--fill", type=float, default=0.62)
    po.add_argument("--yaw", type=float, default=0.0)
    po.add_argument("--elev", type=float, default=15.0, help="15 deg matches the <model-viewer> default orbit")
    args = ap.parse_args()
    {"tudf": export_tudf, "flow": export_flow, "poster": export_poster, "vae": export_vae}[args.cmd](args)


if __name__ == "__main__":
    main()
