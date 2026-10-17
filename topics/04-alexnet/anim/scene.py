"""
Manim scene for Module 04 — AlexNet.

One story in six beats:
  1. The scoreboard: ImageNet top-5 error, 2010 -> 2012, and the gap
     SuperVision opened on September 30, 2012.
  2. The machine: AlexNet's eight layers, split across two GPUs, shape by
     shape — and where its 61 million parameters (dense layers) and its
     arithmetic (convolutions) actually live.
  3. ReLU: tanh's gradient dies at the tails, ReLU's doesn't — and our own
     CIFAR-10 training curves, ReLU vs tanh (the paper's Figure 1, redone).
  4. Dropout: every training step silences a random half of the hidden
     units; a different thinned network learns each time.
  5. Augmentation: one photo, many honest training examples — random crops
     and mirrors.
  6. What it learned: conv1 of our CIFAR AlexNet as RGB filters -- still
     mostly its random start after 15 epochs -- then what training changed
     (final minus initial weights), where the structure is.

Data (filters, sample photos, curves) comes from alexnet_assets.json, written
by python/alexnet.py's export_assets() from the notebook's training runs — the
same numbers the post and notebook report.

Render (Windows, from repo root, with the conda ffmpeg on PATH):
    python -m manim -qm --media_dir topics/04-alexnet/anim/media \
      topics/04-alexnet/anim/scene.py AlexNetStory

No LaTeX required — all labels use Text (Pango), not MathTex.
"""

import json
import os

import numpy as np
from manim import (
    Scene, Rectangle, Circle, Line, Text, VGroup, Group, ImageMobject, Axes,
    FadeIn, FadeOut, Create, GrowFromEdge,
    GREY, WHITE, UP, DOWN, LEFT, RIGHT, ORIGIN,
)

ACCENT = "#6d4aff"   # the model / the winner
ACTION = "#d81b60"   # change: dropped units, the gap
DATA_A = "#0072b2"
DATA_B = "#e69f00"
_HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(_HERE, "alexnet_assets.json")


def _load_assets():
    with open(ASSETS, encoding="utf-8") as fh:
        return json.load(fh)


def _rgb_img(arr, height, scale=8):
    """Blocky RGB ImageMobject from an (H, W, 3) array in [0, 1]."""
    big = np.kron(np.asarray(arr, dtype=np.float64), np.ones((scale, scale, 1)))
    img = ImageMobject((np.clip(big, 0, 1) * 255).astype(np.uint8))
    img.height = height
    return img


class AlexNetStory(Scene):
    def construct(self):
        assets = _load_assets()
        self._beat_scoreboard()
        self._beat_machine()
        self._beat_relu(assets)
        self._beat_dropout()
        self._beat_augment(assets)
        self._beat_filters(assets)

    # ── Beat 1: the scoreboard ──────────────────────────────────────────────
    def _beat_scoreboard(self):
        title = Text("ImageNet, top-5 error (lower is better)", font_size=34).to_edge(UP)
        rows = [("2010 winner", 28.2, False), ("2011 winner", 25.8, False),
                ("2012, next best team", 26.2, False),
                ("2012, SuperVision (AlexNet)", 15.3, True)]
        bars = VGroup()
        scale = 0.25                         # scene units per percentage point
        for k, (name, err, winner) in enumerate(rows):
            y = 1.3 - k * 1.0
            lbl = Text(name, font_size=22, color=WHITE if winner else GREY)
            lbl.move_to(np.array([-3.6, y, 0.0])).align_to(np.array([-6.2, 0.0, 0.0]), LEFT)
            bar = Rectangle(width=err * scale, height=0.5,
                            fill_color=ACCENT if winner else GREY,
                            fill_opacity=0.85, stroke_width=0)
            bar.move_to(np.array([-1.5 + err * scale / 2, y, 0.0]))
            val = Text(f"{err:.1f}%", font_size=22).next_to(bar, RIGHT, buff=0.15)
            bars.add(VGroup(lbl, bar, val))
        self.play(FadeIn(title))
        for g in bars[:3]:
            self.play(FadeIn(g[0]), GrowFromEdge(g[1], LEFT), FadeIn(g[2]), run_time=0.6)
        self.wait(0.4)
        g = bars[3]
        self.play(FadeIn(g[0]), GrowFromEdge(g[1], LEFT), FadeIn(g[2]), run_time=1.0)
        cap = Text("Submitted September 30, 2012: ten points ahead of the field",
                   font_size=26, color=ACCENT).to_edge(DOWN)
        self.play(FadeIn(cap))
        self.wait(1.6)
        self.play(FadeOut(title), FadeOut(bars), FadeOut(cap))

    # ── Beat 2: the machine, and where its numbers live ──────────────────────
    def _beat_machine(self):
        title = Text("AlexNet: 5 convolutions, 3 dense layers, 2 GPUs", font_size=32).to_edge(UP)
        stages = [("input", "224x224x3", 2.4), ("conv1 11x11/4", "55x55x96", 2.0),
                  ("conv2 5x5", "27x27x256", 1.6), ("conv3 3x3", "13x13x384", 1.2),
                  ("conv4 3x3", "13x13x384", 1.2), ("conv5 3x3", "13x13x256", 1.2),
                  ("fc6", "4096", 0.9), ("fc7", "4096", 0.9), ("fc8", "1000", 0.6)]
        cols = VGroup()
        for name, shape, h in stages:
            # Two boxes per stage: the network was split down the middle, half
            # its kernels on each GPU (the input and the classifier excepted).
            if name in ("input", "fc8"):
                box = Rectangle(width=1.05, height=h * 1.25, color=ACCENT, stroke_width=3)
            else:
                top = Rectangle(width=1.05, height=h * 0.6, color=ACCENT, stroke_width=3)
                bot = Rectangle(width=1.05, height=h * 0.6, color=DATA_B, stroke_width=3)
                bot.next_to(top, DOWN, buff=0.06)
                box = VGroup(top, bot)
            lab = Text(name, font_size=14).next_to(box, UP, buff=0.1)
            shp = Text(shape, font_size=13, color=GREY).next_to(box, DOWN, buff=0.1)
            cols.add(VGroup(box, lab, shp))
        cols.arrange(RIGHT, buff=0.32).move_to(ORIGIN).shift(DOWN * 0.1)
        gpu = Text("purple: GPU 1   amber: GPU 2", font_size=18, color=GREY).to_edge(DOWN)
        self.play(FadeIn(title))
        for c in cols:
            self.play(FadeIn(c), run_time=0.3)
        self.play(FadeIn(gpu))
        self.wait(1.0)

        convs, dense = cols[1:6], cols[6:9]

        def shade(group, level):
            """Dim (level < 1) or restore columns: outlines and labels fade,
            boxes stay unfilled."""
            anims = []
            for box, lab, shp in group:
                rects = box if isinstance(box, VGroup) else [box]
                anims += [r.animate.set_stroke(opacity=level) for r in rects]
                anims += [lab.animate.set_opacity(level), shp.animate.set_opacity(level)]
            return anims

        p1 = Text("96% of the 61 M parameters live in the 3 dense layers",
                  font_size=24, color=ACTION).to_edge(DOWN)
        p2 = Text("92% of the arithmetic lives in the 5 convolutions",
                  font_size=24, color=ACCENT).to_edge(DOWN)
        note = Text("Dropout goes where the parameters are. GPUs go where the arithmetic is.",
                    font_size=22).to_edge(DOWN)
        self.play(FadeOut(gpu), FadeIn(p1), *shade(convs, 0.2))
        self.wait(1.6)
        self.play(FadeOut(p1), FadeIn(p2), *shade(convs, 1.0), *shade(dense, 0.2))
        self.wait(1.6)
        self.play(FadeOut(p2), FadeIn(note), *shade(dense, 1.0))
        self.wait(1.8)
        self.play(FadeOut(title), FadeOut(cols), FadeOut(note))

    # ── Beat 3: ReLU keeps the gradient alive ───────────────────────────────
    def _beat_relu(self, assets):
        title = Text("Why ReLU: the slope that doesn't die", font_size=32).to_edge(UP)
        ax = Axes(x_range=[-4, 4, 1], y_range=[-0.2, 1.2, 0.5], x_length=5.0,
                  y_length=2.8, tips=False, axis_config={"color": GREY})
        ax.shift(LEFT * 3.2 + DOWN * 0.3)
        d_tanh = ax.plot(lambda z: 1 - np.tanh(z) ** 2, color=DATA_A)
        d_relu = ax.plot(lambda z: 1.0 if z > 0 else 0.0, color=DATA_B,
                         discontinuities=[0], use_smoothing=False)
        l1 = Text("tanh'(z): fades to 0 at both ends", font_size=18, color=DATA_A)
        l2 = Text("ReLU'(z): exactly 1 for every z > 0", font_size=18, color=DATA_B)
        legend = VGroup(l1, l2).arrange(DOWN, aligned_edge=LEFT).next_to(ax, DOWN, buff=0.3)
        self.play(FadeIn(title), Create(ax))
        self.play(Create(d_tanh), FadeIn(l1))
        self.play(Create(d_relu), FadeIn(l2))
        self.wait(0.8)

        # Right: our CIFAR-10 training error per epoch, ReLU vs tanh.
        curves = assets["relu_tanh"]
        relu, tanh = curves["relu"], curves["tanh"]
        n = max(len(relu), len(tanh))
        ax2 = Axes(x_range=[0, n, max(1, n // 5)], y_range=[0, 0.9, 0.25], x_length=4.6,
                   y_length=2.8, tips=False, axis_config={"color": GREY})
        ax2.shift(RIGHT * 3.3 + DOWN * 0.3)
        g_t = ax2.plot_line_graph(list(range(1, len(tanh) + 1)), tanh,
                                  line_color=DATA_A, add_vertex_dots=False)
        g_r = ax2.plot_line_graph(list(range(1, len(relu) + 1)), relu,
                                  line_color=DATA_B, add_vertex_dots=False)
        level = curves.get("level", 0.25)
        target = ax2.plot(lambda _: level, color=GREY, stroke_width=1.5)
        target_lbl = Text(f"{level:.0%}", font_size=14, color=GREY).next_to(
            ax2.c2p(0, level), LEFT, buff=0.1)
        x_lbl = Text("epoch", font_size=14, color=GREY).next_to(ax2.x_axis, DOWN, buff=0.1)
        y_lbl = Text("training error", font_size=14, color=GREY).next_to(
            ax2.y_axis, UP, buff=0.1)
        cap2 = Text("redoing the paper's Figure 1 on our CIFAR-10 net", font_size=18,
                    color=GREY).next_to(x_lbl, DOWN, buff=0.25)
        self.play(Create(ax2), FadeIn(cap2), FadeIn(x_lbl), FadeIn(y_lbl),
                  Create(target), FadeIn(target_lbl))
        self.play(Create(g_t), Create(g_r), run_time=2.0)
        msg = Text(curves["caption"], font_size=20, color=ACCENT).to_edge(DOWN)
        self.play(FadeIn(msg))
        self.wait(1.8)
        self.play(*[FadeOut(m) for m in (title, ax, d_tanh, d_relu, legend, ax2, g_t, g_r,
                                          target, target_lbl, x_lbl, y_lbl, cap2, msg)])

    # ── Beat 4: dropout, one thinned network per step ───────────────────────
    def _beat_dropout(self):
        title = Text("Dropout: a different half of the network every step",
                     font_size=30).to_edge(UP)
        sizes = [4, 7, 7, 2]
        xs = [-4.5, -1.5, 1.5, 4.5]
        layers = []
        for x, k in zip(xs, sizes):
            ys = np.linspace(1.9, -2.1, k) if k > 1 else [0.0]
            layers.append([Circle(radius=0.22, color=WHITE, fill_opacity=0.9,
                                  fill_color=ACCENT).move_to(np.array([x, y, 0.0])) for y in ys])
        edges = {}
        for l in range(3):
            for i, a in enumerate(layers[l]):
                for j, b in enumerate(layers[l + 1]):
                    edges[(l, i, j)] = Line(a.get_center(), b.get_center(),
                                            stroke_width=1.2, color=GREY,
                                            stroke_opacity=0.5)
        nodes = VGroup(*[c for layer in layers for c in layer])
        wires = VGroup(*edges.values())
        self.play(FadeIn(title), Create(wires), FadeIn(nodes), run_time=1.2)

        rng = np.random.default_rng(4)
        cap = Text("training step 1", font_size=22, color=GREY).to_edge(DOWN)
        self.play(FadeIn(cap))
        for step in range(1, 5):
            # Drop each hidden unit with p = 0.5; its wires go dark with it.
            dropped = {(l, i) for l in (1, 2) for i in range(sizes[l])
                       if rng.random() < 0.5}
            anims = []
            for l in (1, 2):
                for i, c in enumerate(layers[l]):
                    off = (l, i) in dropped
                    anims.append(c.animate.set_fill(GREY if off else ACCENT,
                                                    opacity=0.15 if off else 0.9))
            for (l, i, j), e in edges.items():
                off = (l, i) in dropped or (l + 1, j) in dropped
                anims.append(e.animate.set_stroke(opacity=0.05 if off else 0.5))
            new_cap = Text(f"training step {step}: {len(dropped)} of 14 hidden units dropped",
                           font_size=22, color=GREY).to_edge(DOWN)
            self.play(*anims, FadeOut(cap), FadeIn(new_cap), run_time=0.7)
            cap = new_cap
            self.wait(0.6)

        # Test time: everyone is back (inverted dropout already scaled them).
        anims = [c.animate.set_fill(ACCENT, opacity=0.9) for l in (1, 2) for c in layers[l]]
        anims += [e.animate.set_stroke(opacity=0.5) for e in edges.values()]
        final = Text("At test time: every unit, no randomness. No unit could rely on any other.",
                     font_size=22, color=ACCENT).to_edge(DOWN)
        self.play(*anims, FadeOut(cap), FadeIn(final), run_time=0.8)
        self.wait(1.8)
        self.play(FadeOut(title), FadeOut(nodes), FadeOut(wires), FadeOut(final))

    # ── Beat 5: augmentation ────────────────────────────────────────────────
    def _beat_augment(self, assets):
        title = Text("Augmentation: one photo, many honest examples", font_size=30).to_edge(UP)
        sample = assets["samples"][0]
        img = np.asarray(sample["img"], dtype=np.float64) / 255.0       # (32, 32, 3)
        orig = _rgb_img(img, 2.6).shift(LEFT * 4.4)
        lab = Text(f"a CIFAR-10 '{sample['label']}', 32x32", font_size=18,
                   color=GREY).next_to(orig, DOWN, buff=0.2)
        self.play(FadeIn(title), FadeIn(orig), FadeIn(lab))

        rng = np.random.default_rng(1)
        crops = Group()
        for _ in range(8):
            dy, dx = rng.integers(0, 9, 2)
            c = img[dy:dy + 24, dx:dx + 24]
            if rng.random() < 0.5:
                c = c[:, ::-1]
            crops.add(_rgb_img(c, 1.5, scale=10))
        crops.arrange_in_grid(rows=2, cols=4, buff=0.25).shift(RIGHT * 1.6)
        self.play(FadeIn(crops, lag_ratio=0.15), run_time=1.6)
        cap = Text("Random 24x24 crops, mirrored half the time: still the same class",
                   font_size=22).to_edge(DOWN)
        self.play(FadeIn(cap))
        self.wait(1.2)
        cap2 = Text("AlexNet: 32 x 32 offsets x 2 mirrors = 2048 views of every image",
                    font_size=22, color=ACCENT).to_edge(DOWN)
        self.play(FadeOut(cap), FadeIn(cap2))
        self.wait(1.6)
        self.play(FadeOut(title), FadeOut(orig), FadeOut(lab), FadeOut(crops), FadeOut(cap2))

    # ── Beat 6: what conv1 learned ──────────────────────────────────────────
    def _beat_filters(self, assets):
        title = Text("Our CIFAR-10 AlexNet's first layer", font_size=30).to_edge(UP)

        def grid(key):
            tiles = Group(*[_rgb_img(f, 0.75, scale=24)
                            for f in np.asarray(assets[key], dtype=np.float64)])
            return tiles.arrange_in_grid(rows=4, buff=0.12).move_to(ORIGIN).shift(DOWN * 0.2)

        raw, learned = grid("filters"), grid("filters_learned")       # (F, 5, 5, 3) tiles
        moved = assets.get("conv1_moved", 0.0)
        cap = Text(f"After 15 epochs: still mostly the random start (weights moved {moved:.0%})",
                   font_size=22, color=GREY).to_edge(DOWN)
        self.play(FadeIn(title), FadeIn(raw, lag_ratio=0.03), run_time=1.8)
        self.play(FadeIn(cap))
        self.wait(1.6)
        cap2 = Text("What training changed: color blobs and light-dark gradients",
                    font_size=22, color=ACCENT).to_edge(DOWN)
        self.play(FadeOut(raw), FadeIn(learned), FadeOut(cap), FadeIn(cap2), run_time=1.2)
        self.wait(2.0)
        cap = cap2
        punch = Text("Deeper should be better.  Module 05: why it wasn't, and the fix (ResNet)",
                     font_size=22, color=WHITE).to_edge(DOWN)
        self.play(FadeOut(cap), FadeIn(punch))
        self.wait(2.0)
