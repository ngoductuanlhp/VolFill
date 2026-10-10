import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from manim import DOWN, LEFT, RIGHT, UP, Arrow, FadeIn, FadeOut, GrowArrow, Group

from common import P, PageScene, label, load_vae_slice, raster, rgb


def cells(mask, color, cell=6, gap=1, alpha=255):
    h, w = mask.shape
    img = np.zeros((h * cell, w * cell, 4), np.uint8)
    for r, c in zip(*np.nonzero(mask)):
        img[r * cell:(r + 1) * cell - gap, c * cell:(c + 1) * cell - gap] = (*rgb(color), alpha)
    return img


def fine(mask, color):
    img = np.zeros((*mask.shape, 4), np.uint8)
    img[mask] = (*rgb(color), 255)
    return img


class S3VAE(PageScene):
    """256^3 TUDF -> latent -> dense to 64^3, keep surface cells -> sparse to 256^3. Kept minimal on purpose.
    Real data from the paper's VAE (NRGB-D kitchen, as in Fig. 2): GT slice, predicted occupancy, reconstruction."""

    def construct(self):
        gt, rec, m64, _ = load_vae_slice()
        band = gt < 3.0
        PW, Y = 4.0, 0.3
        XL, XC, XR = -4.7, 0.0, 4.7

        def panel(img, x, width=PW):
            m = raster(img, 1)
            m.width = width
            return m.move_to(np.array([x, Y, 0]))

        def tag(tex, m, color):
            return label(tex, color, 40).next_to(m, DOWN, buff=0.4)

        inp = panel(fine(band, P["blue"]), XL)
        inp_tag = tag(r"$256^3$ TUDF", inp, P["blue"])
        code = Group(*[panel(cells(np.ones((8, 8), bool), P["teal"], cell=10, gap=2), XC, 1.3)
                       .set_opacity(0.2 + 0.2 * k).shift((UP + RIGHT) * 0.06 * (3 - k)) for k in range(4)])
        code_tag = tag(r"latent $16^3$", code, P["teal"])
        a1 = Arrow(inp.get_right(), code.get_left(), buff=0.3, color=P["muted"], stroke_width=5)
        a1_lbl = label(r"encode", P["muted"], 34).next_to(a1, UP, buff=0.12)

        # ---------------------------------------------------------------- beat 0: encode
        self.mark()
        self.play(FadeIn(inp), FadeIn(inp_tag), run_time=0.8)
        self.play(GrowArrow(a1), FadeIn(a1_lbl), FadeIn(code, shift=LEFT * 0.3), FadeIn(code_tag), run_time=1.0)
        self.wait(2.0)

        # ---------------------------------------------------------------- beat 1: dense to 64^3, keep surface cells
        self.mark()
        dense = panel(cells(np.ones_like(m64), P["teal"], cell=6, gap=1, alpha=90), XR)
        kept = panel(cells(m64, P["teal"], cell=6, gap=1), XR)
        a2 = Arrow(code.get_right(), dense.get_left(), buff=0.3, color=P["muted"], stroke_width=5)
        a2_lbl = label(r"decode", P["muted"], 34).next_to(a2, UP, buff=0.12)
        dense_tag = tag(r"dense $\to 64^3$", dense, P["teal"])
        kept_tag = tag(r"keep surface cells", kept, P["teal"])
        self.play(GrowArrow(a2), FadeIn(a2_lbl), FadeIn(dense, shift=LEFT * 0.3), FadeIn(dense_tag), run_time=1.0)
        self.wait(0.8)
        self.play(FadeIn(kept), dense.animate.set_opacity(0), FadeOut(dense_tag), FadeIn(kept_tag), run_time=1.0)
        self.remove(dense)
        self.wait(1.4)

        # ---------------------------------------------------------------- beat 2: sparse to 256^3
        self.mark()
        out = panel(fine(rec < 3.0, P["yellow"]), XR)
        out_tag = tag(r"sparse $\to 256^3$", out, P["yellow"])
        self.play(FadeOut(kept), FadeIn(out), FadeOut(kept_tag), FadeIn(out_tag), run_time=1.2)
        self.wait(2.4)
