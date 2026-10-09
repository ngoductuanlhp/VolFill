import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, Arrow, Circle, Create, CurvedArrow, FadeIn, FadeOut, GrowArrow,
                   ImageMobject, LaggedStart, MathTex, RoundedRectangle, Square, VGroup, Write, smooth)

from common import P, PageScene, ROOT, label, load_slice, mask_rgba, raster, visibility

INPUT = ROOT / "models" / "scrream_scene09_full_00_000275" / "input.jpg"


def box(w, h, title, color, sub=None):
    r = RoundedRectangle(width=w, height=h, corner_radius=0.14, stroke_color=color, stroke_width=3,
                         fill_color=P["surface"], fill_opacity=1)
    t = label(title, P["text"], 36).move_to(r)
    g = VGroup(r, t)
    if sub:
        s = label(sub, P["faint"], 28).next_to(r, DOWN, buff=0.1)
        g.add(s)
    return g


def tokens(n, color, size=0.26, opacity=1.0):
    return VGroup(*[Square(size, stroke_width=0, fill_color=color, fill_opacity=opacity) for _ in range(n)]).arrange(RIGHT, buff=0.06)


class S5Conditioning(PageScene):
    def construct(self):
        d, _ = load_slice()
        surface = d < 1.0
        h, w = surface.shape
        visible, _ = visibility(surface, np.array([w / 2, h * 1.42]))
        grown = visible.copy()
        for dy, dx in ((0, 1), (1, 0), (1, 1)):
            grown[dy:, dx:] |= visible[:h - dy, :w - dx]
        visible = grown

        img = ImageMobject(str(INPUT))
        img.height = 1.75
        img.move_to([-5.55, 2.35, 0])
        moge = box(2.3, 0.9, r"MoGe-2", P["border_strong"], sub=r"frozen").move_to([-5.55, 0.15, 0])
        toks = tokens(6, P["blue"]).move_to([-2.2, 2.75, 0])
        tok_lbl = label(r"image tokens", P["blue"], 30).next_to(toks, UP, buff=0.12)

        dit = VGroup(*[box(2.7, 0.62, rf"DiT block {k}", P["muted"]) for k in (3, 2, 1)]).arrange(DOWN, buff=0.22)
        dit.move_to([1.85, 1.55, 0])
        dit_lbl = label(r"\Phi", P["text"], 46, math=True).next_to(dit[-1], RIGHT, buff=0.25)

        # ---------------------------------------------------------------- step 0: frozen encoder -> tokens
        self.mark()
        a0 = Arrow(img.get_bottom(), moge.get_top(), buff=0.08, color=P["muted"], stroke_width=4)
        a1 = Arrow(moge[0].get_right(), toks.get_left() + DOWN * 0.05, buff=0.15, color=P["blue"], stroke_width=4)
        self.play(FadeIn(img), run_time=0.6)
        self.play(GrowArrow(a0), FadeIn(moge), run_time=0.7)
        self.play(GrowArrow(a1), LaggedStart(*[FadeIn(t, scale=0.4) for t in toks], lag_ratio=0.1), FadeIn(tok_lbl), run_time=1.1)
        self.wait(0.8)

        # ---------------------------------------------------------------- step 1: cross-attention into every block
        self.mark()
        self.play(FadeIn(dit, lag_ratio=0.15), FadeIn(dit_lbl), run_time=0.9)
        xattn = VGroup(*[CurvedArrow(toks.get_right() + RIGHT * 0.05, blk[0].get_left() + LEFT * 0.02 + UP * 0.12,
                                     angle=-0.5, color=P["blue"], stroke_width=3) for blk in dit])
        xa_lbl = label(r"cross-attention", P["blue"], 32).move_to([-0.05, 3.35, 0])
        self.play(LaggedStart(*[Create(a) for a in xattn], lag_ratio=0.2), FadeIn(xa_lbl), run_time=1.3)
        self.wait(0.9)

        # ---------------------------------------------------------------- step 2: visible point map -> z_vis
        self.mark()
        vis = raster(mask_rgba(visible, P["teal"]), 1.0)
        vis.width = 2.0
        vis.move_to([-5.55, -2.4, 0])
        vis_frame = RoundedRectangle(width=2.25, height=vis.height + 0.25, corner_radius=0.1,
                                     stroke_color=P["border_strong"], stroke_width=2).move_to(vis)
        vis_lbl = label(r"visible TUDF", P["teal"], 30).next_to(vis_frame, DOWN, buff=0.1)
        a2 = Arrow(moge.get_bottom(), vis_frame.get_top(), buff=0.08, color=P["teal"], stroke_width=4)
        enc = box(0.9, 0.7, r"$\mathcal E$", P["border_strong"]).move_to([-3.05, -2.4, 0])
        zvis = VGroup(*[Square(0.22, stroke_width=0, fill_color=P["teal"], fill_opacity=0.9) for _ in range(16)]).arrange_in_grid(4, 4, buff=0.04)
        zvis.move_to([-1.55, -2.4, 0])
        zvis_lbl = label(r"\mathbf z_{\text{vis}}", P["teal"], 40, math=True).next_to(zvis, DOWN, buff=0.12)
        a3 = Arrow(vis_frame.get_right(), enc.get_left(), buff=0.06, color=P["teal"], stroke_width=4)
        a4 = Arrow(enc.get_right(), zvis.get_left(), buff=0.06, color=P["teal"], stroke_width=4)
        self.play(GrowArrow(a2), FadeIn(vis_frame), FadeIn(vis), FadeIn(vis_lbl), run_time=1.0)
        self.play(GrowArrow(a3), FadeIn(enc), GrowArrow(a4), FadeIn(zvis, lag_ratio=0.05), FadeIn(zvis_lbl), run_time=1.2)
        self.wait(0.8)

        # ---------------------------------------------------------------- step 3: zero-init MLP, additive fusion
        self.mark()
        mlp = box(1.25, 0.7, r"$\mathrm{MLP}_0$", P["teal"]).move_to([0.3, -1.05, 0])
        plus = VGroup(Circle(0.26, color=P["text"], stroke_width=3), MathTex(r"+", color=P["text"], font_size=40))
        plus.move_to([1.85, -1.05, 0])
        zt = label(r"\mathbf z_t", P["text"], 42, math=True).move_to([4.3, -1.05, 0])
        a5 = Arrow(zvis.get_right(), mlp[0].get_left(), buff=0.1, color=P["teal"], stroke_width=4)
        a6 = Arrow(mlp[0].get_right(), plus.get_left(), buff=0.06, color=P["teal"], stroke_width=4)
        a7 = Arrow(zt.get_left(), plus.get_right(), buff=0.12, color=P["muted"], stroke_width=4)
        a8 = Arrow(plus.get_top(), dit[-1][0].get_bottom(), buff=0.06, color=P["text"], stroke_width=4)
        zero = label(r"\text{starts at } 0", P["faint"], 30, math=True).next_to(mlp, DOWN, buff=0.1)
        eq = label(r"\tilde{\mathbf z}_t = \mathbf z_t + \mathrm{MLP}_0(\mathbf z_{\text{vis}})", P["text"], 46, math=True).move_to([3.4, -3.15, 0])
        self.play(GrowArrow(a5), FadeIn(mlp), FadeIn(zero), run_time=0.8)
        self.play(GrowArrow(a6), FadeIn(plus), FadeIn(zt), GrowArrow(a7), run_time=0.8)
        self.play(GrowArrow(a8), Write(eq), run_time=1.1)
        self.wait(0.9)

        # ---------------------------------------------------------------- step 4: classifier-free guidance
        self.mark()
        vc = label(r"\mathbf v_c", P["text"], 42, math=True).move_to([4.75, 2.75, 0])
        out = Arrow(dit[0][0].get_right(), vc.get_left(), buff=0.12, color=P["text"], stroke_width=4)
        self.play(GrowArrow(out), FadeIn(vc), run_time=0.6)
        null_toks = tokens(6, P["border_strong"]).move_to(toks)
        null_z = zvis.copy().set_fill(P["border_strong"])
        empty1 = label(r"\varnothing", P["muted"], 40, math=True).next_to(null_toks, RIGHT, buff=0.12)
        empty2 = label(r"\mathbf 0", P["muted"], 40, math=True).next_to(null_z, UP, buff=0.12)
        self.play(toks.animate.become(null_toks), zvis.animate.become(null_z), FadeIn(empty1), FadeIn(empty2),
                  xattn.animate.set_color(P["border_strong"]), a5.animate.set_color(P["border_strong"]), run_time=0.9)
        vn = label(r"\mathbf v_\varnothing", P["muted"], 42, math=True).move_to([4.75, 1.85, 0])
        out2 = Arrow(dit[0][0].get_right() + DOWN * 0.12, vn.get_left(), buff=0.12, color=P["muted"], stroke_width=3)
        self.play(GrowArrow(out2), FadeIn(vn), run_time=0.6)
        cfg = label(r"\mathbf v = \mathbf v_\varnothing + w\,(\mathbf v_c - \mathbf v_\varnothing),\quad w = 3",
                    P["yellow"], 44, math=True).move_to([3.4, -3.15, 0])
        self.play(FadeOut(eq, shift=DOWN * 0.1), FadeIn(cfg, shift=UP * 0.1), run_time=0.9)
        self.wait(1.4)
