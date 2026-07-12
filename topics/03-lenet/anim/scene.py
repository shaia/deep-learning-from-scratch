"""
Manim scene for Module 03 — convolutions / LeNet-5.

One story in five beats:
  1. An image is not a list of numbers: a real MNIST digit as a pixel grid,
     and the price the MLP paid for flattening it (101,770 weights) vs the
     CNN's 44,426.
  2. The kernel slides: a 5x5 window sweeps the digit and the feature map
     fills in, one dot product per position — the same 25 numbers everywhere.
  3. Pooling: keep the strongest response per 2x2 window; 24x24 -> 12x12.
  4. The full LeNet stack, shape by shape, down to 10 logits.
  5. The learned filters: conv1's six real trained kernels (from
     lenetFilters.json — the same data the site widget uses) and their
     response maps. Gradient descent invented edge detectors on its own.

Render (Windows, from repo root, with the conda ffmpeg on PATH):
    python -m manim -qm --media_dir topics/03-lenet/anim/media \
      topics/03-lenet/anim/scene.py LeNetStory

No LaTeX required — all labels use Text (Pango), not MathTex.
"""

import json
import os
import sys

import numpy as np
from manim import (
    Scene, Rectangle, Square, Arrow, Text, VGroup, Group, ImageMobject,
    FadeIn, FadeOut, GrowFromEdge,
    BLACK, GREY, WHITE, UP, DOWN, LEFT, RIGHT, ORIGIN,
)

# Reuse the module's real conv (the naive definition) so the maps match the code.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, os.path.join(_HERE, "..", "python"))
from lenet import naive_conv2d_forward  # noqa: E402

ACCENT = "#6d4aff"   # the sliding window / positive responses
ACTION = "#d81b60"   # negative responses
FILTERS_JSON = os.path.join(_REPO, "site", "src", "components", "viz",
                            "lenetFilters.json")


def _load_assets():
    """The trained conv1 filters + a digit — one source of truth with the widget."""
    with open(FILTERS_JSON, encoding="utf-8") as fh:
        data = json.load(fh)
    filters = np.array(data["filters"], dtype=np.float64)        # (6, 5, 5)
    digit = np.array(data["digits"][0], dtype=np.float64) / 255  # (28, 28)
    return filters, digit


def _gray_img(arr, height, scale=12):
    """Blocky grayscale ImageMobject from a [0,1] array (white on black)."""
    big = np.kron(arr, np.ones((scale, scale)))
    img = ImageMobject((big * 255).astype(np.uint8))
    img.height = height
    return img


def _signed_img(arr, height, scale=12):
    """Diverging colormap image for signed values: purple +, pink -."""
    m = max(1e-9, np.abs(arr).max())
    t = arr / m
    pos = np.array([109, 74, 255]) / 255.0
    neg = np.array([216, 27, 96]) / 255.0
    rgb = np.where(t[..., None] >= 0, t[..., None] * pos, -t[..., None] * neg)
    big = np.kron(rgb, np.ones((scale, scale, 1)))
    img = ImageMobject((big * 255).astype(np.uint8))
    img.height = height
    return img


# A hand-designed vertical-edge kernel for the sliding beat (same preset as
# the site widget); the learned ones appear in the final beat.
VEDGE = np.tile(np.array([-1.0, -2.0, 0.0, 2.0, 1.0]) / 8.0, (5, 1))


class LeNetStory(Scene):
    def construct(self):
        filters, digit = _load_assets()
        title = Text("Convolutions: teaching networks to see", font_size=40).to_edge(UP)
        self.play(FadeIn(title))
        self._beat_waste(digit)
        self._beat_slide(digit)
        self.play(FadeOut(title))
        self._beat_pool(digit)
        self._beat_stack()
        self._beat_learned(filters, digit)

    # ── Beat 1: what flattening throws away ─────────────────────────────────
    def _beat_waste(self, digit):
        img = _gray_img(digit, 3.4).shift(LEFT * 3.5 + DOWN * 0.4)
        cap = Text("A 28x28 digit. Nearby pixels belong together.",
                   font_size=26, color=GREY).to_edge(DOWN)
        self.play(FadeIn(img), FadeIn(cap))
        self.wait(0.8)

        mlp = Text("MLP (Module 01):\nevery unit reads all 784 pixels\n101,770 weights",
                   font_size=26, line_spacing=1.1).shift(RIGHT * 3.2 + UP * 0.9)
        self.play(FadeIn(mlp))
        self.wait(1.0)

        win = Square(3.4 * 5 / 28, color=ACCENT, stroke_width=5)
        win.move_to(img.get_corner(UP + LEFT)
                    + np.array([win.width / 2, -win.width / 2, 0]))
        cnn = Text("CNN: one unit reads a 5x5 patch,\nand the SAME 25 weights\n"
                   "are reused everywhere\n44,426 weights total",
                   font_size=26, line_spacing=1.1, color=WHITE)
        cnn.shift(RIGHT * 3.2 + DOWN * 1.6)
        self.play(FadeIn(win), FadeIn(cnn))
        self.wait(1.6)
        self.play(FadeOut(mlp), FadeOut(cnn), FadeOut(cap),
                  FadeOut(img), FadeOut(win))

    # ── Beat 2: the kernel slides, the feature map fills ────────────────────
    def _beat_slide(self, digit):
        # Left: digit. Right: its vertical-edge response, revealed row by row.
        fmap = naive_conv2d_forward(digit[None, None], VEDGE[None, None],
                                    np.zeros(1))[0, 0]           # (24, 24)
        img = _gray_img(digit, 3.6).shift(LEFT * 3.2 + DOWN * 0.4)
        out = _signed_img(fmap, 3.6 * 24 / 28).shift(RIGHT * 3.2 + DOWN * 0.4)
        out_h = out.height

        # Mask that hides the not-yet-computed part of the feature map.
        mask = Rectangle(width=out.width, height=out_h, color=BLACK,
                         fill_color=BLACK, fill_opacity=1.0, stroke_width=0)
        mask.move_to(out.get_center())

        cap = Text("Slide the kernel: one dot product per position",
                   font_size=26, color=GREY).to_edge(DOWN)
        in_lbl = Text("input", font_size=22, color=GREY).next_to(img, UP, buff=0.15)
        out_lbl = Text("feature map", font_size=22, color=GREY).next_to(out, UP, buff=0.15)
        self.add(out, mask)
        self.play(FadeIn(img), FadeIn(in_lbl), FadeIn(out_lbl), FadeIn(cap))

        cell = 3.6 / 28                       # one input pixel, in scene units
        win = Square(cell * 5, color=ACCENT, stroke_width=5)

        def win_pos(u, v):
            """Scene position of the window covering input rows u..u+4, cols v..v+4."""
            tl = img.get_corner(UP + LEFT)
            return tl + np.array([(v + 2.5) * cell, -(u + 2.5) * cell, 0.0])

        win.move_to(win_pos(0, 0))
        self.play(FadeIn(win))

        # Row 0 slowly: the window slides, the first feature-map row appears.
        row_h = out_h / 24
        self.play(win.animate.move_to(win_pos(0, 23)),
                  mask.animate.stretch_to_fit_height(out_h - row_h)
                              .align_to(out, DOWN),
                  run_time=2.2)
        # Remaining rows in one accelerating sweep.
        self.play(win.animate.move_to(win_pos(23, 23)),
                  mask.animate.stretch_to_fit_height(1e-3).align_to(out, DOWN),
                  run_time=2.6)

        share = Text("The same 25 numbers, asked at every position",
                     font_size=26, color=ACCENT).to_edge(DOWN)
        self.play(FadeOut(cap), FadeIn(share))
        self.wait(1.4)
        self.play(FadeOut(img), FadeOut(win), FadeOut(in_lbl), FadeOut(out_lbl),
                  FadeOut(out), FadeOut(mask), FadeOut(share))

    # ── Beat 3: pooling keeps the strongest response ────────────────────────
    def _beat_pool(self, digit):
        fmap = naive_conv2d_forward(digit[None, None], VEDGE[None, None],
                                    np.zeros(1))[0, 0]
        relu = np.maximum(fmap, 0.0)
        pooled = relu.reshape(12, 2, 12, 2).max(axis=(1, 3))

        big = _signed_img(relu, 3.4).shift(LEFT * 3.0)
        small = _signed_img(pooled, 1.7).shift(RIGHT * 3.0)
        arrow = Arrow(big.get_right(), small.get_left(), color=GREY, buff=0.3)
        lbl_a = Text("24x24", font_size=22, color=GREY).next_to(big, DOWN, buff=0.2)
        lbl_b = Text("12x12", font_size=22, color=GREY).next_to(small, DOWN, buff=0.2)
        cap = Text("Max pooling: keep each 2x2 window's strongest response",
                   font_size=26, color=GREY).to_edge(DOWN)
        self.play(FadeIn(big), FadeIn(lbl_a), FadeIn(cap))
        self.play(GrowFromEdge(arrow, LEFT), FadeIn(small), FadeIn(lbl_b))
        self.wait(1.4)
        self.play(*[FadeOut(m) for m in (big, small, arrow, lbl_a, lbl_b, cap)])

    # ── Beat 4: the whole stack, shape by shape ─────────────────────────────
    def _beat_stack(self):
        cap = Text("Stack it twice, then decide", font_size=26, color=GREY).to_edge(DOWN)
        stages = [
            ("input", "1x28x28", 2.6),
            ("conv 6@5x5\n+ ReLU", "6x24x24", 2.2),
            ("pool 2x2", "6x12x12", 1.6),
            ("conv 16@5x5\n+ ReLU", "16x8x8", 1.2),
            ("pool 2x2", "16x4x4", 0.9),
            ("flatten\n256", "120 -> 84", 0.7),
            ("10 logits", "softmax", 0.5),
        ]
        boxes = VGroup()
        for name, shape, h in stages:
            box = Rectangle(width=1.45, height=h, color=ACCENT, stroke_width=3)
            top = Text(name, font_size=17, line_spacing=0.9).next_to(box, UP, buff=0.12)
            bot = Text(shape, font_size=16, color=GREY).next_to(box, DOWN, buff=0.12)
            boxes.add(VGroup(box, top, bot))
        boxes.arrange(RIGHT, buff=0.42).move_to(ORIGIN).shift(DOWN * 0.2)

        self.play(FadeIn(cap))
        for g in boxes:
            self.play(FadeIn(g), run_time=0.45)
        self.wait(1.6)
        self.play(FadeOut(boxes), FadeOut(cap))

    # ── Beat 5: what the filters became ─────────────────────────────────────
    def _beat_learned(self, filters, digit):
        cap = Text("Nobody designed these: conv1's six learned filters",
                   font_size=26, color=GREY).to_edge(DOWN)
        kimgs = Group(*[_signed_img(k, 1.15, scale=24) for k in filters])
        for i, im in enumerate(kimgs):
            im.move_to(np.array([-5.0 + i * 2.0, 1.6, 0.0]))

        # Each filter's response map on the same digit.
        rimgs = Group()
        for i, k in enumerate(filters):
            r = naive_conv2d_forward(digit[None, None], k[None, None],
                                     np.zeros(1))[0, 0]
            im = _signed_img(r, 1.5)
            im.move_to(np.array([-5.0 + i * 2.0, -0.6, 0.0]))
            rimgs.add(im)

        self.play(FadeIn(cap), *[FadeIn(im) for im in kimgs])
        self.wait(0.6)
        self.play(*[FadeIn(im) for im in rimgs])
        self.wait(2.0)

        punch = Text("Edge detectors, learned from gradients alone  ->  Module 04: AlexNet",
                     font_size=25, color=WHITE).to_edge(DOWN)
        self.play(FadeOut(cap), FadeIn(punch))
        self.wait(1.8)
