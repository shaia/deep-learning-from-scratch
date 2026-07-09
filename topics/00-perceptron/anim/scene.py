"""
Manim scene for Module 00 — the perceptron.

Two beats in one video:
  1. Blobs: run Rosenblatt's rule and watch the decision line swing into place.
  2. XOR: the same rule can never separate four points — the wall.

The line is not hand-animated: we actually run the perceptron and animate the
boundary through the real weight snapshots, so the picture matches the code.

Render (Windows, from repo root, with the conda ffmpeg on PATH):
    python -m manim -qm topics/00-perceptron/anim/scene.py PerceptronStory

No LaTeX required — all labels use Text (Pango), not MathTex.
"""

from manim import (
    Scene, Axes, Dot, Line, Text, VGroup, FadeIn, FadeOut, Create, Transform,
    BLUE, ORANGE, GREY, WHITE, RED, UP, DOWN, LEFT, RIGHT,
)
import numpy as np

# Semantic colors, echoing the site's Okabe–Ito palette (docs/conventions/viz-style.md).
CLASS_A = BLUE
CLASS_B = ORANGE
ACCENT = "#6d4aff"   # decision boundary
ACTION = "#d81b60"   # the point driving the current update


def make_blobs(seed=7, n=10, spread=1.2):
    rng = np.random.default_rng(seed)
    pts, labels = [], []
    for c, center in enumerate([(-2.0, -2.0), (2.0, 2.0)]):
        for _ in range(n):
            pts.append((center[0] + spread * rng.uniform(-1, 1),
                        center[1] + spread * rng.uniform(-1, 1)))
            labels.append(c)
    return np.array(pts), np.array(labels)


def predict(w, b, x, y):
    return 1 if (w[0] * x + w[1] * y + b) >= 0 else 0


def run_perceptron(pts, labels, w0, lr=0.1, max_updates=40):
    """Run Rosenblatt's rule; record (w, b) each time the line actually moves."""
    w = list(w0[:2])
    b = w0[2]
    snaps = [(tuple(w), b, -1)]  # (weights, bias, index of driving point)
    updates = 0
    i = 0
    while updates < max_updates:
        p = pts[i % len(pts)]
        yhat = predict(w, b, p[0], p[1])
        err = int(labels[i % len(pts)]) - yhat
        if err != 0:
            w[0] += lr * err * p[0]
            w[1] += lr * err * p[1]
            b += lr * err
            snaps.append((tuple(w), b, i % len(pts)))
            updates += 1
        i += 1
        # stop once everything is classified correctly
        if all(predict(w, b, q[0], q[1]) == int(l) for q, l in zip(pts, labels)):
            break
    return snaps


class PerceptronStory(Scene):
    def construct(self):
        axes = Axes(x_range=[-5, 5, 1], y_range=[-5, 5, 1],
                    x_length=7, y_length=7, tips=False,
                    axis_config={"stroke_color": GREY, "stroke_width": 2})
        title = Text("The Perceptron", font_size=40).to_edge(UP)
        self.play(FadeIn(title), Create(axes))

        # ── Beat 1: blobs ──────────────────────────────────────────────────
        pts, labels = make_blobs()
        dots = VGroup(*[
            Dot(axes.c2p(x, y), radius=0.08, color=CLASS_A if l == 0 else CLASS_B)
            for (x, y), l in zip(pts, labels)
        ])
        self.play(FadeIn(dots, lag_ratio=0.03))

        caption = Text("Learn by nudging the line toward mistakes",
                       font_size=24, color=GREY).next_to(axes, DOWN)
        self.play(FadeIn(caption))

        def boundary_line(w, b):
            # w0*x + w1*y + b = 0  ->  a Line clipped to the visible x-range.
            xs = np.array([-5.0, 5.0])
            if abs(w[1]) > 1e-6:
                ys = -(w[0] * xs + b) / w[1]
            else:  # near-vertical
                xs = np.array([-b / w[0], -b / w[0]])
                ys = np.array([-5.0, 5.0])
            ys = np.clip(ys, -5, 5)
            return Line(axes.c2p(xs[0], ys[0]), axes.c2p(xs[1], ys[1]),
                        color=ACCENT, stroke_width=6)

        snaps = run_perceptron(pts, labels, w0=(1.5, -1.2, 0.5), lr=0.1)
        line = boundary_line(*snaps[0][:2])
        self.play(Create(line))

        for w, b, drive in snaps[1:]:
            highlight = None
            if 0 <= drive < len(dots):
                highlight = dots[drive].copy().set_color(ACTION).scale(1.6)
                self.add(highlight)
            self.play(Transform(line, boundary_line(w, b)), run_time=0.5)
            if highlight:
                self.play(FadeOut(highlight), run_time=0.15)

        done = Text("Separated — 100% accuracy", font_size=24, color=CLASS_A)
        done.next_to(axes, DOWN)
        self.play(Transform(caption, done))
        self.wait(1.0)

        # ── Beat 2: the XOR wall ───────────────────────────────────────────
        self.play(FadeOut(dots), FadeOut(line), FadeOut(caption))
        xor_pts = [(-2, -2, 0), (2, 2, 0), (-2, 2, 1), (2, -2, 1)]
        xor_dots = VGroup(*[
            Dot(axes.c2p(x, y), radius=0.1, color=CLASS_A if l == 0 else CLASS_B)
            for x, y, l in xor_pts
        ])
        xcap = Text("XOR: no straight line can separate these",
                    font_size=24, color=RED).next_to(axes, DOWN)
        self.play(FadeIn(xor_dots), FadeIn(xcap))

        # Sweep a line through several angles to show every one fails.
        wall = Line(axes.c2p(-5, -5), axes.c2p(5, 5), color=ACCENT, stroke_width=6)
        self.play(Create(wall))
        for w, b in [((1, -1), 0), ((1, 0.4), 0), ((-0.4, 1), 0), ((1, 1), 0)]:
            self.play(Transform(wall, boundary_line(w, b)), run_time=0.6)
        self.wait(0.8)

        punch = Text("One line is not enough → stack them (Module 01)",
                     font_size=24, color=WHITE).next_to(axes, DOWN)
        self.play(Transform(xcap, punch))
        self.wait(1.5)
