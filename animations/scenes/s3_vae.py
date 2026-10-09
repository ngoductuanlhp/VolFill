import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, Arrow, FadeIn, FadeOut, GrowArrow, Group, LaggedStart,
                   Write, smooth)

from common import P, PageScene, label, load_slice, raster, rgb


def pool(mask, f):
    h, w = mask.shape
    H, W = -(-h // f), -(-w // f)
    pad = np.zeros((H * f, W * f), bool)
    pad[:h, :w] = mask
    return pad.reshape(H, f, W, f).any(axis=(1, 3))


def cells(mask, color, cell=6, gap=1, off_color=None, off_alpha=0):
    """Draw a coarse grid as visible cells with gaps (one cell = one voxel)."""
    h, w = mask.shape
    img = np.zeros((h * cell, w * cell, 4), np.uint8)
    on = (*rgb(color), 255)
    off = (*rgb(off_color or color), off_alpha)
    for r in range(h):
        for c in range(w):
            img[r * cell:(r + 1) * cell - gap, c * cell:(c + 1) * cell - gap] = on if mask[r, c] else off
    return img


class S3VAE(PageScene):
    def construct(self):
        d, _ = load_slice()
        band = d < 3.0
        PW = 2.9
        y = 0.55
        xs = [-5.35, -1.75, 1.85, 5.35]

        def panel(img, x):
            m = raster(img, 1)
            m.width = PW
            return m.move_to(np.array([x, y, 0]))

        def caption(m, top, sub, color):
            t = label(top, color, 38).next_to(m, DOWN, buff=0.3)
            s = label(sub, P["muted"], 32).next_to(t, DOWN, buff=0.12)
            return t, s

        full = np.zeros((*band.shape, 4), np.uint8)
        full[band] = (*rgb(P["blue"]), 255)
        a = panel(full, xs[0])

        # ---------------------------------------------------------------- step 0: sparse input
        self.mark()
        a_t, a_s = caption(a, r"$256^3$ TUDF", r"3--5\% of voxels active", P["blue"])
        self.play(FadeIn(a, scale=0.96), run_time=0.9)
        self.play(FadeIn(a_t, shift=UP * 0.1), FadeIn(a_s, shift=UP * 0.1), run_time=0.7)
        sparse_tag = label(r"sparse tensor", P["faint"], 30).next_to(a, UP, buff=0.2)
        self.play(FadeIn(sparse_tag), run_time=0.4)
        self.wait(0.9)

        # ---------------------------------------------------------------- step 1: sparse encoder -> 16^3 x 16
        self.mark()
        stages = [pool(band, f) for f in (2, 4, 8)]
        ghosts = [panel(cells(m, P["blue"], cell=max(1, 16 // (2 ** (i + 1))) + 1, gap=1 if i else 0), xs[1]) for i, m in enumerate(stages)]
        lat_mask = np.ones_like(pool(band, 16))
        layers = Group(*[panel(cells(lat_mask, P["teal"], cell=10, gap=2), xs[1]).set_opacity(0.18 + 0.06 * k)
                         .shift((UP + RIGHT) * 0.05 * (6 - k)) for k in range(6)])
        lat = panel(cells(lat_mask, P["teal"], cell=10, gap=2), xs[1])
        latent = Group(layers, lat)
        e_arrow = Arrow(a.get_right(), lat.get_left(), buff=0.15, color=P["muted"], stroke_width=4)
        e_lbl = label(r"\mathcal E", P["text"], 46, math=True).next_to(e_arrow, UP, buff=0.08)
        self.play(GrowArrow(e_arrow), Write(e_lbl), run_time=0.6)
        cur = ghosts[0]
        self.play(FadeIn(cur), run_time=0.4)
        for g in ghosts[1:]:
            self.play(FadeOut(cur), FadeIn(g), run_time=0.45)
            cur = g
        self.play(FadeOut(cur, scale=0.9), FadeIn(latent, scale=1.1), run_time=0.8, rate_func=smooth)
        l_t, l_s = caption(lat, r"$16^3 \times 16$ latent", r"dense, KL-regularised", P["teal"])
        self.play(FadeIn(l_t, shift=UP * 0.1), FadeIn(l_s, shift=UP * 0.1), run_time=0.6)
        self.wait(0.9)

        # ---------------------------------------------------------------- step 2: dense upsampling to 64^3
        self.mark()
        m64 = pool(band, 4)
        dense64 = panel(cells(np.ones_like(m64), P["teal"], cell=6, gap=1), xs[2]).set_opacity(0.55)
        d_arrow = Arrow(lat.get_right(), dense64.get_left(), buff=0.15, color=P["muted"], stroke_width=4)
        d_lbl = label(r"\mathcal D", P["text"], 46, math=True).next_to(d_arrow, UP, buff=0.08)
        self.play(GrowArrow(d_arrow), Write(d_lbl), run_time=0.6)
        self.play(FadeIn(dense64, scale=0.6), run_time=1.0)
        d_t, d_s = caption(dense64, r"dense $\uparrow\ 64^3$", r"$\approx$262K voxels, still cheap", P["teal"])
        self.play(FadeIn(d_t, shift=UP * 0.1), FadeIn(d_s, shift=UP * 0.1), run_time=0.6)
        self.wait(0.9)

        # ---------------------------------------------------------------- step 3: occupancy head prunes
        self.mark()
        kept = panel(cells(m64, P["teal"], cell=6, gap=1), xs[2])
        pruned = panel(cells(~m64, P["red"], cell=6, gap=1), xs[2]).set_opacity(0.7)
        o_t, o_s = caption(kept, r"occupancy head", r"keeps surface-adjacent cells", P["text"])
        self.play(FadeIn(pruned), run_time=0.6)
        self.play(FadeOut(pruned), FadeOut(dense64), FadeIn(kept), FadeOut(d_t), FadeOut(d_s),
                  FadeIn(o_t), FadeIn(o_s), run_time=1.2)
        self.wait(0.9)

        # ---------------------------------------------------------------- step 4: sparse upsampling to 256^3
        self.mark()
        out = np.zeros((*band.shape, 4), np.uint8)
        out[band] = (*rgb(P["yellow"]), 255)
        b = panel(out, xs[3])
        s_arrow = Arrow(kept.get_right(), b.get_left(), buff=0.15, color=P["muted"], stroke_width=4)
        self.play(GrowArrow(s_arrow), run_time=0.5)
        self.play(FadeIn(b, scale=1.08), run_time=1.0)
        b_t, b_s = caption(b, r"sparse $\uparrow\ 256^3$", r"cost $\propto$ surface area", P["yellow"])
        self.play(LaggedStart(FadeIn(b_t, shift=UP * 0.1), FadeIn(b_s, shift=UP * 0.1), lag_ratio=0.3), run_time=0.8)
        self.wait(1.4)
