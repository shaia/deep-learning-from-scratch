"""
Manim scene for Module 01 — MLP + backpropagation.

Two beats in one video:
  1. Backprop: a signal flows forward to a prediction, the error appears at the
     output, and that error flows *backward* through every edge — the chain rule
     made visible.
  2. Two moons: the decision boundary is not hand-drawn — we train the real MLP
     from python/mlp.py and show its learned regions bending into place epoch by
     epoch, until the two moons are cleanly split.

Render (Windows, from repo root, with the conda ffmpeg on PATH):
    python -m manim -qm topics/01-mlp-backprop/anim/scene.py MLPStory

No LaTeX required — all labels use Text (Pango), not MathTex.
"""

import os
import sys

import numpy as np
from manim import (
    Scene, Circle, Line, Text, VGroup, Dot, ImageMobject,
    FadeIn, FadeOut, Create, FadeTransform, LaggedStart, ShowPassingFlash,
    BLUE, ORANGE, GREY, WHITE, UP, DOWN, LEFT, RIGHT, ORIGIN,
)

# Reuse the real network so the boundary matches the code (viz-style principle).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import mlp  # noqa: E402

# Semantic colors, echoing the site's Okabe–Ito palette (docs/conventions/viz-style.md).
CLASS_A = BLUE
CLASS_B = ORANGE
ACCENT = "#6d4aff"   # forward signal / model
ACTION = "#d81b60"   # the error flowing backward (change/motion)

# Okabe–Ito blue/orange as RGB, for tinting the decision regions.
_BLUE = np.array([0.0, 0.45, 0.70])
_ORANGE = np.array([0.90, 0.60, 0.0])
_WHITE = np.array([1.0, 1.0, 1.0])


class MLPStory(Scene):
    def construct(self):
        title = Text("Backpropagation", font_size=44).to_edge(UP)
        self.play(FadeIn(title))
        self._beat_backprop()
        self.play(FadeOut(title))
        self._beat_two_moons()

    # ── Beat 1: the error flows backward ────────────────────────────────────
    def _beat_backprop(self):
        in_y = [1.2, -1.2]
        hid_y = [2.0, 0.7, -0.7, -2.0]
        in_pos = [np.array([-5.0, y, 0.0]) for y in in_y]
        hid_pos = [np.array([-1.0, y, 0.0]) for y in hid_y]
        out_pos = np.array([3.2, 0.0, 0.0])

        def node(pos, color):
            return Circle(radius=0.28, color=color, fill_opacity=0.85,
                          stroke_color=WHITE, stroke_width=2).move_to(pos)

        in_nodes = VGroup(*[node(p, CLASS_A) for p in in_pos])
        hid_nodes = VGroup(*[node(p, ACCENT) for p in hid_pos])
        out_node = node(out_pos, ORANGE)

        edges1 = VGroup(*[Line(p, q, stroke_color=GREY, stroke_width=2)
                          for p in in_pos for q in hid_pos])   # input -> hidden
        edges2 = VGroup(*[Line(p, out_pos, stroke_color=GREY, stroke_width=2)
                          for p in hid_pos])                    # hidden -> output

        labels = VGroup(
            Text("x", font_size=24).next_to(in_pos[0], LEFT, buff=0.2),
            Text("x", font_size=24).next_to(in_pos[1], LEFT, buff=0.2),
            Text("ŷ", font_size=26).next_to(out_pos, RIGHT, buff=0.25),
        )

        self.play(Create(edges1), Create(edges2), run_time=1.0)
        self.play(FadeIn(in_nodes), FadeIn(hid_nodes), FadeIn(out_node), FadeIn(labels))

        cap = Text("Forward: inputs flow to a prediction",
                   font_size=26, color=GREY).to_edge(DOWN)
        self.play(FadeIn(cap))

        # Forward: an accent pulse travels left -> right along every edge.
        self.play(LaggedStart(*[ShowPassingFlash(e.copy().set_color(ACCENT).set_stroke(width=6),
                                                 time_width=0.5) for e in edges1],
                              lag_ratio=0.05, run_time=1.2))
        self.play(LaggedStart(*[ShowPassingFlash(e.copy().set_color(ACCENT).set_stroke(width=6),
                                                 time_width=0.5) for e in edges2],
                              lag_ratio=0.1, run_time=0.8))
        self.play(out_node.animate.set_fill(ACTION), run_time=0.4)

        err = Text("error = ŷ − y", font_size=26, color=ACTION).next_to(out_pos, UP, buff=0.5)
        self.play(FadeIn(err))
        back_cap = Text("Backward: push the error back, nudge every weight",
                        font_size=26, color=ACTION).to_edge(DOWN)
        self.play(FadeTransform(cap, back_cap))

        # Backward: an action pulse travels right -> left along reversed edges.
        rev2 = [Line(e.get_end(), e.get_start()) for e in edges2]
        rev1 = [Line(e.get_end(), e.get_start()) for e in edges1]
        self.play(LaggedStart(*[ShowPassingFlash(r.set_color(ACTION).set_stroke(width=6),
                                                 time_width=0.5) for r in rev2],
                              lag_ratio=0.1, run_time=0.8))
        self.play(LaggedStart(*[ShowPassingFlash(r.set_color(ACTION).set_stroke(width=6),
                                                 time_width=0.5) for r in rev1],
                              lag_ratio=0.05, run_time=1.2))
        # The weights "move": edges briefly thicken in the model color.
        self.play(edges1.animate.set_stroke(color=ACCENT, width=3.5),
                  edges2.animate.set_stroke(color=ACCENT, width=3.5), run_time=0.6)
        self.wait(0.8)

        self.play(*[FadeOut(m) for m in
                    [in_nodes, hid_nodes, out_node, edges1, edges2, labels, err, back_cap]])

    # ── Beat 2: the two-moons boundary bends into place ─────────────────────
    def _beat_two_moons(self):
        X, y = mlp.make_two_moons(n=300, noise=0.18, seed=0)
        model = mlp.MLPVec(2, 16, 2, seed=0)

        # Data box -> scene transform (equal scale, centered at the origin).
        dx0, dx1 = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        dy0, dy1 = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        s = min(10.0 / (dx1 - dx0), 5.6 / (dy1 - dy0))

        def d2s(px, py):
            return np.array([(px - (dx0 + dx1) / 2) * s, (py - (dy0 + dy1) / 2) * s, 0.0])

        # Pixel grid for the decision regions (rows top->bottom = ymax->ymin).
        W = 240
        H = int(W * (dy1 - dy0) / (dx1 - dx0))
        gx = np.linspace(dx0, dx1, W)
        gy = np.linspace(dy1, dy0, H)
        mesh = np.stack(np.meshgrid(gx, gy), axis=-1).reshape(-1, 2)

        def region_image():
            p = model.forward(mesh)[:, 1].reshape(H, W)          # P(class 1)
            base = (1 - p)[..., None] * _BLUE + p[..., None] * _ORANGE
            rgb = 0.5 * _WHITE + 0.5 * base                       # lighten so dots read
            img = ImageMobject((rgb * 255).astype(np.uint8))
            img.height = (dy1 - dy0) * s
            img.move_to(ORIGIN)
            return img

        dots = VGroup(*[Dot(d2s(px, py), radius=0.085,
                            color=CLASS_A if lbl == 0 else CLASS_B)
                        .set_stroke(WHITE, width=1.2)
                        for (px, py), lbl in zip(X, y)])

        snaps = [0, 6, 25, 150]  # epochs at which we photograph the boundary
        img = region_image()
        epoch_lbl = Text("epoch 0", font_size=26, color=GREY).to_edge(UP)
        cap = Text("An MLP bends a boundary between the two moons",
                   font_size=26, color=GREY).to_edge(DOWN)
        self.add(img)
        self.play(FadeIn(img), FadeIn(dots), FadeIn(epoch_lbl), FadeIn(cap))
        self.bring_to_front(dots)

        done = 0
        for target in snaps[1:]:
            mlp.train_minibatch(model, X, y, lr=0.5, epochs=target - done,
                                batch_size=32, seed=0)
            done = target
            new_img = region_image()
            new_lbl = Text(f"epoch {target}", font_size=26, color=GREY).to_edge(UP)
            self.play(FadeTransform(img, new_img), FadeTransform(epoch_lbl, new_lbl),
                      run_time=1.1)
            img = new_img
            epoch_lbl = new_lbl
            self.bring_to_front(dots)  # keep points on top of the new region image

        punch = Text("One line couldn't — a learned boundary can  →  Module 02",
                     font_size=26, color=WHITE).to_edge(DOWN)
        self.play(FadeTransform(cap, punch))
        self.bring_to_front(dots)
        self.wait(1.5)
