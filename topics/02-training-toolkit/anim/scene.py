"""
Manim scene for Module 02 — the training toolkit.

Two beats in one video:
  1. Optimizers: three of the *real* nanograd optimizers (SGD, momentum, Adam)
     descend the Beale loss surface from the same start. Plain SGD stalls in the
     curved valley; momentum builds speed; Adam adapts per-coordinate and drives
     straight for the minimum. The trajectories are computed by the same optimizer
     objects that train the MLP (viz-style principle: show the real thing).
  2. Initialization: a signal is pushed through a deep tanh stack. With the
     Module-01 "small" uniform init the per-layer activation std decays toward
     zero (the vanishing signal); with Xavier init it holds steady.

Render (Windows, from repo root, with the conda ffmpeg on PATH):
    python -m manim -qm --media_dir topics/02-training-toolkit/anim/media \
      topics/02-training-toolkit/anim/scene.py ToolkitStory

No LaTeX required — all labels use Text (Pango), not MathTex.
"""

import os
import sys

import numpy as np
from manim import (
    Scene, Rectangle, Line, Text, VGroup, Dot, ImageMobject, TracedPath,
    FadeIn, FadeOut, Create, GrowFromEdge,
    BLUE, ORANGE, GREEN, GREY, WHITE, UP, DOWN, LEFT, RIGHT, ORIGIN,
)

# Reuse the real nanograd optimizers so the trajectories match the code.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lib", "python"))
import nanograd as ng  # noqa: E402

ACCENT = "#6d4aff"   # Adam
ACTION = "#d81b60"   # SGD (the one that struggles)
MOMENT = "#f2a900"   # momentum


# ---- Beale function: a curved valley that punishes plain gradient descent ----
def beale(x, y):
    return ((1.5 - x + x * y) ** 2 + (2.25 - x + x * y ** 2) ** 2
            + (2.625 - x + x * y ** 3) ** 2)


def beale_grad(p):
    x, y = p
    a, b, c = 1.5 - x + x * y, 2.25 - x + x * y ** 2, 2.625 - x + x * y ** 3
    dx = 2 * a * (y - 1) + 2 * b * (y ** 2 - 1) + 2 * c * (y ** 3 - 1)
    dy = 2 * a * x + 2 * b * (2 * x * y) + 2 * c * (3 * x * y ** 2)
    return np.array([dx, dy])


class ToolkitStory(Scene):
    def construct(self):
        title = Text("The training toolkit", font_size=44).to_edge(UP)
        self.play(FadeIn(title))
        self._beat_optimizers()
        self.play(FadeOut(title))
        self._beat_init()

    # ── Beat 1: optimizers descend the loss surface ─────────────────────────
    def _beat_optimizers(self):
        x0, x1, y0, y1 = -1.0, 4.0, -1.0, 3.0
        sx = 9.0 / (x1 - x0)
        sy = 5.2 / (y1 - y0)

        def d2s(px, py):
            return np.array([(px - (x0 + x1) / 2) * sx,
                             (py - (y0 + y1) / 2) * sy, 0.0])

        # Loss surface as a lightened heatmap image (rows top->bottom = ymax->ymin).
        W, H = 320, int(320 * (y1 - y0) / (x1 - x0))
        gx = np.linspace(x0, x1, W)
        gy = np.linspace(y1, y0, H)
        mx, my = np.meshgrid(gx, gy)
        Z = np.log1p(beale(mx, my))
        Z = (Z - Z.min()) / (Z.max() - Z.min())
        base = np.array([0.15, 0.35, 0.75])          # deep = low loss (bluish)
        rgb = (1 - Z)[..., None] * base + Z[..., None] * np.array([0.97, 0.97, 0.97])
        img = ImageMobject((rgb * 255).astype(np.uint8))
        img.height = (y1 - y0) * sy
        img.move_to(ORIGIN)

        minimum = Dot(d2s(3.0, 0.5), radius=0.08, color=WHITE).set_stroke(GREY, 2)
        min_lbl = Text("minimum", font_size=20, color=GREY).next_to(minimum, DOWN, buff=0.15)
        cap = Text("Same start, three optimizers", font_size=26, color=GREY).to_edge(DOWN)
        self.add(img)
        self.play(FadeIn(img), FadeIn(minimum), FadeIn(min_lbl), FadeIn(cap))

        # Precompute each optimizer's path using the real nanograd objects.
        start = (-0.5, 2.0)
        specs = [("SGD", ng.SGD(lr=1e-3), ACTION),
                 ("momentum", ng.SGD(lr=1e-3, momentum=0.9), MOMENT),
                 ("Adam", ng.Adam(lr=2e-2), ACCENT)]
        dots, traces, labels = [], [], []
        for name, opt, col in specs:
            p = np.array(start, dtype=np.float64)
            pts = [p.copy()]
            for _ in range(220):
                opt.step([p], [beale_grad(p)])
                pts.append(p.copy())
            dot = Dot(d2s(*pts[0]), radius=0.09, color=col)
            trace = TracedPath(dot.get_center, stroke_color=col, stroke_width=3)
            dot._path = pts
            dots.append(dot); traces.append(trace)
            labels.append(Text(name, font_size=22, color=col))

        legend = VGroup(*labels).arrange(DOWN, aligned_edge=LEFT, buff=0.15) \
            .to_corner(UP + RIGHT).shift(DOWN * 0.6)
        self.add(*traces)
        self.play(*[FadeIn(d) for d in dots], FadeIn(legend))

        # Animate all three descending together, sampling their precomputed paths.
        FRAMES = 40
        for f in range(1, FRAMES + 1):
            anims = []
            for dot in dots:
                idx = min(len(dot._path) - 1, int(f / FRAMES * (len(dot._path) - 1)))
                anims.append(dot.animate.move_to(d2s(*dot._path[idx])))
            self.play(*anims, run_time=0.06)
        punch = Text("Adam adapts its step and steers straight in",
                     font_size=26, color=ACCENT).to_edge(DOWN)
        self.play(FadeOut(cap), FadeIn(punch))
        self.wait(1.0)
        self.play(*[FadeOut(m) for m in
                    (img, minimum, min_lbl, legend, punch, *dots, *traces)])

    # ── Beat 2: initialization keeps the signal alive ───────────────────────
    def _beat_init(self):
        cap = Text("Push a signal through a deep tanh stack",
                   font_size=26, color=GREY).to_edge(DOWN)
        self.play(FadeIn(cap))

        def stds(init):
            net = ng.mlp([128] * 8, activation="tanh", init=init, seed=0)
            x = np.random.default_rng(0).standard_normal((256, 128))
            out = []
            for layer in net.layers:
                x = layer.forward(x)
                if not isinstance(layer, ng.Linear):
                    out.append(float(x.std()))
            return out

        # Two rows of bars: bar height is proportional to activation std.
        rows = [("small uniform init", stds("small"), ACTION, 1.4),
                ("Xavier init", stds("xavier"), BLUE, -1.4)]
        groups = []
        for name, s, col, yshift in rows:
            bars = VGroup()
            for i, v in enumerate(s):
                h = max(0.04, min(2.2, v)) * 1.1
                bar = Rectangle(width=0.5, height=h, color=col,
                                fill_opacity=0.85, stroke_width=0)
                bar.move_to(np.array([-4.2 + i * 1.15, yshift, 0.0]), aligned_edge=DOWN)
                bars.add(bar)
            lbl = Text(name, font_size=24, color=col)
            lbl.next_to(bars, LEFT, buff=0.3).shift(UP * 0.3)
            groups.append((bars, lbl))

        for bars, lbl in groups:
            self.play(FadeIn(lbl),
                      *[GrowFromEdge(b, DOWN) for b in bars], run_time=1.2)
        punch = Text("Bad init lets the signal decay; scaled init holds it  →  Module 03",
                     font_size=25, color=WHITE).to_edge(DOWN)
        self.play(FadeOut(cap), FadeIn(punch))
        self.wait(1.5)
