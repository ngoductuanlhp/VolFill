"""Shared helpers for the page animations."""
import json
from pathlib import Path

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, ImageMobject, Scene, Tex, MathTex, VGroup, config)
from manim.constants import RESAMPLING_ALGORITHMS
from PIL import Image

from palette import P, THEME, rgb

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "out"
SLICE_PNG = ROOT / "media" / "widgets" / "tudf-slice.png"
SLICE_JSON = ROOT / "media" / "widgets" / "tudf-slice.json"

config.background_color = P["bg"]
Tex.set_default(color=P["text"])
MathTex.set_default(color=P["text"])


class PageScene(Scene):
    """Records the start time of each scroll step via self.mark()."""

    def setup(self):
        self.camera.background_color = P["bg"]
        self._marks = []

    def mark(self):
        self._marks.append(round(self.renderer.time, 3))

    def tear_down(self):
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"{type(self).__name__}.json").write_text(
            json.dumps({"marks": self._marks, "duration": round(self.renderer.time, 3)}))


def label(text, color=None, size=40, math=False):
    if math:
        return MathTex(text, color=color or P["muted"], font_size=size)
    return Tex(text, color=color or P["muted"], font_size=size, tex_environment="flushleft")


# ------------------------------------------------------------------ real slice

def load_vae_slice():
    """Real VAE stages on one kitchen slice (scripts/export_widget_data.py vae): distance to the GT surface and
    to the VAE reconstruction (voxels), and the decoder's predicted 64^3 occupancy cells cut by the plane."""
    meta = json.loads((SLICE_JSON.parent / "vae-slice.json").read_text())
    d = lambda n: np.asarray(Image.open(SLICE_PNG.parent / f"vae-slice-{n}.png"), np.float32) / 255.0 * meta["dmax"]
    occ = np.asarray(Image.open(SLICE_PNG.parent / "vae-slice-occ.png")) > 0
    return d("gt"), d("recon"), occ, meta


def load_slice():
    d = np.asarray(Image.open(SLICE_PNG), dtype=np.float32) / 255.0
    meta = json.loads(SLICE_JSON.read_text())
    return d * meta["dmax"], meta


def raster(rgba, height):
    """Nearest-neighbour ImageMobject so voxels stay crisp at 1080p."""
    im = ImageMobject(rgba.astype(np.uint8))
    im.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    im.height = height
    return im


def mask_rgba(mask, color, alpha=255):
    out = np.zeros((*mask.shape, 4), np.uint8)
    out[mask] = (*rgb(color), alpha)
    return out


def px_to_point(img_mob, shape, col, row):
    """Pixel centre (col, row) of the slice image -> scene coordinates."""
    h, w = shape
    left, top = img_mob.get_corner(UP + LEFT)[:2]
    s = img_mob.width / w
    return np.array([left + (col + 0.5) * s, top - (row + 0.5) * s, 0.0])


def visibility(surface, cam, n_rays=None, step=0.35):
    """First hits along rays from `cam` (col,row) and which surface pixels are occluded."""
    h, w = surface.shape
    ys, xs = np.nonzero(surface)
    pts = np.stack([xs + 0.5, ys + 0.5], 1)
    vis = np.zeros(len(pts), bool)
    for i, p in enumerate(pts):
        d = p - cam
        L = np.linalg.norm(d)
        n = int(L / step)
        ts = np.arange(1, n) * step / L
        q = cam + ts[:, None] * d
        c = q[:, 0].astype(int)
        r = q[:, 1].astype(int)
        ok = (c >= 0) & (c < w) & (r >= 0) & (r < h)
        hit = surface[r[ok], c[ok]]
        dist_left = (1 - ts[ok]) * L
        vis[i] = not np.any(hit & (dist_left > 1.6))
    visible = np.zeros_like(surface)
    visible[ys[vis], xs[vis]] = True
    return visible, surface & ~visible


def first_hits(surface, cam, angles, step=0.35, max_len=600):
    h, w = surface.shape
    hits = []
    for a in angles:
        d = np.array([np.sin(a), -np.cos(a)])
        for k in range(1, int(max_len / step)):
            q = cam + d * k * step
            c, r = int(q[0]), int(q[1])
            if 0 <= c < w and 0 <= r < h and surface[r, c]:
                hits.append(q)
                break
        else:
            hits.append(None)
    return hits


def stack_right(*mobs, buff=0.25):
    g = VGroup(*mobs)
    g.arrange(DOWN, aligned_edge=LEFT, buff=buff)
    return g


__all__ = ["PageScene", "label", "load_slice", "raster", "mask_rgba", "px_to_point", "visibility",
           "first_hits", "stack_right", "P", "THEME", "rgb", "ROOT", "LEFT", "RIGHT", "UP", "DOWN"]
