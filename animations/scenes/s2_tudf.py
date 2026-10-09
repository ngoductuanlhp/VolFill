import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, Axes, Circle, Create, DashedLine, Dot, FadeIn, FadeOut,
                   Line, Transform, ValueTracker, VGroup, Write, always_redraw, smooth)

from common import P, PageScene, label, load_slice, px_to_point, raster, rgb


def heat(d, tau, eps):
    near, far = np.array(rgb(P["blue"]), float), np.array(rgb(P["surface2"]), float)
    empty, surf = np.array(rgb(P["bg"]), float), np.array(rgb(P["yellow"]), float)
    t = np.clip(d / max(tau, 1e-6), 0, 1)[..., None] ** 0.6
    out = near * (1 - t) + far * t
    out = np.where((d >= tau)[..., None], empty, out)
    out = np.where((d < eps)[..., None], surf, out)
    return np.dstack([out, np.full(d.shape, 255.0)])


class S2TUDF(PageScene):
    def construct(self):
        d, meta = load_slice()
        h, w = d.shape
        H = 4.6
        center = LEFT * 3.0 + UP * 0.6
        tau = ValueTracker(meta["dmax"])
        eps = ValueTracker(1.0)
        img = always_redraw(lambda: raster(heat(d, tau.get_value(), eps.get_value()), H).move_to(center))
        surf_only = raster(heat(d, 1e-3, 1.0), H).move_to(center)
        to_pt = lambda c, r: px_to_point(surf_only, (h, w), c, r)

        col_x = 1.0
        place = lambda m, y: m.move_to(np.array([col_x, y, 0]), aligned_edge=LEFT)

        # ---------------------------------------------------------------- step 0: distance to nearest surface
        self.mark()
        formula = place(label(r"V(\mathbf p) = \min\big(\operatorname{dist}(\mathbf p, \mathcal S),\ \tau\big)", P["text"], 44, math=True), 2.6)
        self.play(FadeIn(surf_only), run_time=0.6)
        self.play(Write(formula), run_time=1.2)
        self.add(img)
        self.remove(surf_only)
        self.play(FadeIn(img), run_time=0.01)
        ys, xs = np.nonzero(d < 1)
        free = np.argwhere((d > 9) & (d < 14))
        r0, c0 = free[len(free) // 2]
        k = np.argmin((xs - c0) ** 2 + (ys - r0) ** 2)
        p, q = to_pt(c0, r0), to_pt(xs[k], ys[k])
        probe = Dot(p, radius=0.06, color=P["text"])
        p_lbl = label(r"\mathbf p", P["text"], 40, math=True).next_to(probe, UP + LEFT, buff=0.05)
        ring = Circle(radius=np.linalg.norm(q - p), color=P["text"], stroke_width=2).move_to(p)
        seg = Line(p, q, color=P["text"], stroke_width=3)
        self.play(FadeIn(probe, scale=0.5), FadeIn(p_lbl), run_time=0.5)
        self.play(Create(ring), Create(seg), run_time=1.2)
        t0 = place(label(r"every voxel stores how far\\ the nearest surface is", size=38), 1.55)
        self.play(FadeIn(t0, shift=UP * 0.1), run_time=0.6)
        self.wait(0.8)

        # ---------------------------------------------------------------- step 1: truncation
        self.mark()
        tau_lbl = always_redraw(lambda: place(label(rf"\tau = {tau.get_value():.1f}\ \text{{voxels}}", P["blue"], 44, math=True), 0.35))
        self.play(FadeOut(ring), FadeOut(seg), FadeOut(probe), FadeOut(p_lbl), FadeOut(t0), FadeIn(tau_lbl), run_time=0.6)
        self.play(tau.animate.set_value(3.0), run_time=2.4, rate_func=smooth)
        t1 = place(label(r"only a thin band survives.\\ In 3D: 3--5\% of voxels", size=38), -0.65)
        self.play(FadeIn(t1, shift=UP * 0.1), run_time=0.6)
        self.wait(0.8)

        # ---------------------------------------------------------------- step 2: unsigned vs signed
        self.mark()
        self.play(FadeOut(t1), FadeOut(tau_lbl), formula.animate.set_opacity(0.35), run_time=0.5)
        ax = Axes(x_range=[-4, 4, 1], y_range=[-3.5, 3.5, 1], x_length=5.2, y_length=3.4,
                  axis_config={"color": P["border_strong"], "stroke_width": 2, "include_ticks": False})
        ax.move_to(np.array([3.9, 0.1, 0]))
        wall = DashedLine(ax.c2p(0, -3.3), ax.c2p(0, 3.3), color=P["yellow"], stroke_width=4, dash_length=0.12)
        wall_lbl = label(r"open wall", P["yellow"], 32).next_to(wall, UP, buff=0.05)
        udf = ax.plot(lambda x: min(abs(x), 3), x_range=[-4, 4], color=P["blue"], stroke_width=5)
        sdf = ax.plot(lambda x: max(min(x, 3), -3), x_range=[-4, 4], color=P["red"], stroke_width=4)
        q1 = label(r"inside?", P["red"], 32).move_to(ax.c2p(-2.4, -2.6))
        q2 = label(r"outside?", P["red"], 32).move_to(ax.c2p(2.4, 2.6) + DOWN * 0.1)
        s_lbl = label(r"signed: which side is inside?", P["red"], 38).next_to(ax, DOWN, buff=0.25)
        self.play(Create(ax), FadeIn(wall), FadeIn(wall_lbl), run_time=0.8)
        self.play(Create(sdf), FadeIn(q1), FadeIn(q2), FadeIn(s_lbl), run_time=1.0)
        self.wait(0.6)
        u_lbl = label(r"unsigned: just \emph{how far}", P["blue"], 38).next_to(ax, DOWN, buff=0.25)
        self.play(FadeOut(q1), FadeOut(q2), Transform(sdf, udf), Transform(s_lbl, u_lbl), run_time=1.2)
        self.wait(1.0)

        # ---------------------------------------------------------------- step 3: extract surfaces
        self.mark()
        self.play(FadeOut(VGroup(ax, wall, wall_lbl, sdf, s_lbl)), formula.animate.set_opacity(1), run_time=0.6)
        eps.set_value(0.01)
        eps_lbl = always_redraw(lambda: place(label(rf"V(\mathbf p) < \epsilon = {eps.get_value():.2f}", P["yellow"], 44, math=True), 0.35))
        self.play(FadeIn(eps_lbl), run_time=0.3)
        self.play(eps.animate.set_value(1.0), run_time=1.8, rate_func=smooth)
        t3 = place(label(r"$\to$ point cloud, or a mesh\\ via MeshUDF", size=38), -0.65)
        self.play(FadeIn(t3, shift=UP * 0.1), run_time=0.6)
        self.wait(1.2)
