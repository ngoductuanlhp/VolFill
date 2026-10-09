import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, Arrow, Create, Dot, FadeIn, FadeOut, LaggedStart, Line,
                   ValueTracker, VGroup, Write, always_redraw, linear, smooth)

from common import P, PageScene, label, load_slice


class S4Flow(PageScene):
    def construct(self):
        d, _ = load_slice()
        ys, xs = np.nonzero(d < 1.0)
        h, w = d.shape
        rng = np.random.default_rng(7)
        n = 420
        pick = rng.choice(len(xs), n, replace=False)
        # Data: real slice surface points, placed on the right. Noise: N(0, I), on the left.
        sc = 4.4 / w
        data = np.stack([(xs[pick] - w / 2) * sc + 3.1, -(ys[pick] - h / 2) * sc + 0.1, np.zeros(n)], 1)
        noise = np.stack([rng.normal(-3.9, 0.85, n), rng.normal(0.1, 0.85, n), np.zeros(n)], 1)

        e_lbl = label(r"\boldsymbol\epsilon \sim \mathcal N(0, I)", P["blue"], 44, math=True).move_to([-3.9, 2.85, 0])
        z_lbl = label(r"\mathbf z_1 \ \text{(data)}", P["yellow"], 44, math=True).move_to([3.1, 2.85, 0])

        # ---------------------------------------------------------------- step 0: noise and data
        self.mark()
        nd = VGroup(*[Dot(p, radius=0.032, color=P["blue"], fill_opacity=0.85) for p in noise])
        dd = VGroup(*[Dot(p, radius=0.032, color=P["yellow"], fill_opacity=0.9) for p in data])
        self.play(FadeIn(nd, lag_ratio=0.002), Write(e_lbl), run_time=1.2)
        self.play(FadeIn(dd, lag_ratio=0.002), Write(z_lbl), run_time=1.2)
        self.wait(0.8)

        # ---------------------------------------------------------------- step 1: straight paths
        self.mark()
        sel = rng.choice(n, 18, replace=False)
        paths = VGroup(*[Line(noise[i], data[i], color=P["muted"], stroke_width=1.6, stroke_opacity=0.55) for i in sel])
        t = ValueTracker(0.0)
        movers = VGroup(*[Dot(noise[i], radius=0.06, color=P["text"]) for i in sel])
        for m, i in zip(movers, sel):
            m.add_updater(lambda mob, i=i: mob.move_to((1 - t.get_value()) * noise[i] + t.get_value() * data[i]))
        eq = label(r"\mathbf z_t = (1 - t)\,\boldsymbol\epsilon + t\,\mathbf z_1", P["text"], 46, math=True).move_to([0, -2.85, 0])
        t_read = always_redraw(lambda: label(rf"t = {t.get_value():.2f}", P["muted"], 40, math=True).next_to(eq, DOWN, buff=0.2))
        self.play(LaggedStart(*[Create(p) for p in paths], lag_ratio=0.05), Write(eq), run_time=1.4)
        self.add(movers, t_read)
        self.play(t.animate.set_value(1.0), run_time=2.0, rate_func=smooth)
        self.play(t.animate.set_value(0.35), run_time=0.8, rate_func=smooth)
        self.wait(0.4)

        # ---------------------------------------------------------------- step 2: constant velocity target
        self.mark()
        arrows = VGroup()
        for i in sel[:9]:
            for s in (0.2, 0.5, 0.8):
                p0 = (1 - s) * noise[i] + s * data[i]
                v = (data[i] - noise[i]) * 0.12
                arrows.add(Arrow(p0, p0 + v, buff=0, color=P["teal"], stroke_width=4, max_tip_length_to_length_ratio=0.35))
        u_eq = label(r"\mathbf u^\star = \mathbf z_1 - \boldsymbol\epsilon", P["teal"], 46, math=True).move_to([0, -2.85, 0])
        self.play(FadeOut(t_read), FadeOut(movers), run_time=0.3)
        self.play(LaggedStart(*[FadeIn(a, shift=0.05 * RIGHT) for a in arrows], lag_ratio=0.03),
                  eq.animate.shift(UP * 0.0).set_opacity(0.0), FadeIn(u_eq), run_time=1.3)
        phi = label(r"\Phi(\mathbf z_t, t, c) \approx \mathbf u^\star", P["muted"], 40, math=True).next_to(u_eq, DOWN, buff=0.2)
        self.play(Write(phi), run_time=0.9)
        self.wait(1.0)

        # ---------------------------------------------------------------- step 3: Euler integration from fresh noise
        self.mark()
        self.play(FadeOut(paths), FadeOut(arrows), FadeOut(u_eq), FadeOut(phi), FadeOut(dd), run_time=0.6)
        steps = 10
        cur = noise.copy()
        e_txt = label(r"\text{50 Euler steps + CFG}", P["text"], 42, math=True).move_to([0, -2.85, 0])
        k_read = label(r"\text{step } 0", P["muted"], 38, math=True).next_to(e_txt, DOWN, buff=0.2)
        self.play(FadeIn(e_txt), FadeIn(k_read), nd.animate.set_color(P["text"]), run_time=0.5)
        for k in range(1, steps + 1):
            s = k / steps
            nxt = (1 - s) * noise + s * data
            new_read = label(rf"\text{{step }} {k * 5}", P["muted"], 38, math=True).move_to(k_read)
            self.play(*[dot.animate.move_to(p) for dot, p in zip(nd, nxt)], k_read.animate.become(new_read),
                      run_time=0.28, rate_func=linear)
        self.play(nd.animate.set_color(P["yellow"]), run_time=0.6)
        decode = label(r"\to \mathcal D \to V", P["yellow"], 44, math=True).next_to(z_lbl, DOWN, buff=0.15)
        self.play(FadeIn(decode, shift=UP * 0.1), run_time=0.6)
        self.wait(1.2)
