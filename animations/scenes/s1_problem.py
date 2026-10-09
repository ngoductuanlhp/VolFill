import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, AnimationGroup, Create, Dot, FadeIn, FadeOut, Intersection, LaggedStart, Line,
                   Polygon, Rectangle, VGroup, Write, linear, rate_functions, smooth)

from common import P, PageScene, label, load_slice, mask_rgba, px_to_point, raster, visibility, first_hits


class S1Problem(PageScene):
    def construct(self):
        d, _ = load_slice()
        surface = d < 1.0
        h, w = surface.shape
        cam = np.array([w / 2, h * 1.42])
        visible, occluded = visibility(surface, cam)

        H = 4.6
        base = raster(mask_rgba(np.ones_like(surface), P["bg"], 0), H).shift(LEFT * 3.0 + UP * 1.0)
        to_pt = lambda c, r: px_to_point(base, (h, w), c, r)
        cam_pt = to_pt(cam[0] - 0.5, cam[1] - 0.5)

        sy, sx = np.nonzero(surface)
        half_fov = np.abs(np.arctan2(sx + 0.5 - cam[0], cam[1] - sy - 0.5)).max() * 1.04
        tri = Polygon(cam_pt + np.array([-0.22, -0.32, 0]), cam_pt + np.array([0.22, -0.32, 0]), cam_pt,
                      color=P["text"], stroke_width=3, fill_color=P["surface2"], fill_opacity=1)
        cam_lbl = label(r"camera", size=32).next_to(tri, DOWN, buff=0.12)

        vis_img = raster(mask_rgba(visible, P["blue"]), H).move_to(base)
        occ_img = raster(mask_rgba(occluded, P["occluded"], 210), H).move_to(base)
        gen_img = raster(mask_rgba(occluded, P["yellow"]), H).move_to(base)

        col_x = 1.0
        def place(m, y):
            return m.move_to(np.array([col_x, y, 0]), aligned_edge=LEFT)

        # ---------------------------------------------------------------- step 0: rays to first hits
        self.mark()
        self.play(FadeIn(tri, shift=UP * 0.1), FadeIn(cam_lbl), run_time=0.6)
        angles = np.linspace(-half_fov * 0.97, half_fov * 0.97, 40)
        hits = first_hits(surface, cam, angles)
        rays, dots = VGroup(), VGroup()
        for a, hpt in zip(angles, hits):
            if hpt is None:
                continue
            end = to_pt(hpt[0] - 0.5, hpt[1] - 0.5)
            rays.add(Line(cam_pt, end, color=P["muted"], stroke_width=1.6, stroke_opacity=0.7))
            dots.add(Dot(end, radius=0.045, color=P["blue"]))
        t0 = place(label(r"one ray per pixel", P["text"], 46), 2.2)
        t1 = place(label(r"stops at the \emph{first} hit", size=38), 1.6)
        self.play(LaggedStart(*[Create(r) for r in rays], lag_ratio=0.06), Write(t0), run_time=2.2, rate_func=linear)
        self.play(LaggedStart(*[FadeIn(dt, scale=0.5) for dt in dots], lag_ratio=0.03), FadeIn(t1), run_time=1.0)
        self.wait(0.6)

        # ---------------------------------------------------------------- step 1: the visible shell
        self.mark()
        t2 = place(label(r"pixel-aligned depth", P["blue"], 46), 2.2)
        t3 = place(label(r"MoGe-2, Depth Anything 3:\\ a 2.5D shell", size=38), 1.45)
        self.play(FadeOut(rays), FadeIn(vis_img), FadeOut(dots), FadeOut(t0), FadeOut(t1), run_time=1.2)
        self.play(FadeIn(t2, shift=UP * 0.1), FadeIn(t3, shift=UP * 0.1), run_time=0.8)
        self.wait(0.8)

        # ---------------------------------------------------------------- step 2: occluded
        self.mark()
        t4 = place(label(r"occluded", P["occluded"], 46), 0.4)
        t5 = place(label(r"never observed,\\ left empty", size=38), -0.3)
        self.play(FadeIn(occ_img), run_time=1.4, rate_func=smooth)
        self.play(FadeIn(t4, shift=UP * 0.1), FadeIn(t5, shift=UP * 0.1), run_time=0.8)
        self.wait(0.8)

        # ---------------------------------------------------------------- step 3: amodal = whole frustum
        self.mark()
        tl, tr = to_pt(-1, -1), to_pt(w, -1)
        far = max(np.linalg.norm(tl - cam_pt), np.linalg.norm(tr - cam_pt))
        dl = np.array([np.sin(-half_fov), np.cos(half_fov), 0]) * far
        dr = np.array([np.sin(half_fov), np.cos(half_fov), 0]) * far
        cone = Polygon(cam_pt, cam_pt + dl, cam_pt + dr)
        box = Rectangle(width=base.width + 0.4, height=base.get_top()[1] - cam_pt[1] + 0.15).move_to(
            np.array([base.get_center()[0], (base.get_top()[1] + 0.15 + cam_pt[1]) / 2, 0]))
        frustum = Intersection(cone, box, stroke_color=P["teal"], stroke_width=2, fill_color=P["teal"], fill_opacity=0.06)
        frustum.set_z_index(-1)
        t6 = place(label(r"amodal", P["teal"], 46), -1.45)
        t7 = place(label(r"\mathbf{I} \;\mapsto\; S \text{ (all of it)}", P["text"], 42, math=True), -2.1)
        self.play(Create(frustum), run_time=1.2)
        self.play(FadeIn(t6, shift=UP * 0.1), Write(t7), run_time=1.0)
        self.wait(0.8)

        # ---------------------------------------------------------------- step 4: generate the hidden part
        self.mark()
        rng = np.random.default_rng(3)
        oy, ox = np.nonzero(occluded)
        pick = rng.choice(len(ox), size=min(260, len(ox)), replace=False)
        targets = [to_pt(ox[i], oy[i]) for i in pick]
        starts = []
        for _ in targets:
            a = rng.uniform(-half_fov, half_fov)
            r = rng.uniform(0.25, 1.0) * far * 0.9
            starts.append(cam_pt + np.array([np.sin(a) * r, np.cos(a) * r, 0]))
        sparks = VGroup(*[Dot(s, radius=0.03, color=P["yellow"], fill_opacity=0.9) for s in starts])
        t8 = place(label(r"V \sim P(V \mid \mathbf{I})", P["yellow"], 48, math=True), -3.05)
        self.play(FadeOut(t4), FadeOut(t5), FadeIn(sparks, lag_ratio=0.01), run_time=0.8)
        self.play(AnimationGroup(*[s.animate.move_to(t) for s, t in zip(sparks, targets)]),
                  Write(t8), run_time=1.8, rate_func=rate_functions.ease_in_out_cubic)
        self.play(FadeOut(occ_img), FadeIn(gen_img), FadeOut(sparks), run_time=0.9)
        self.wait(1.2)
