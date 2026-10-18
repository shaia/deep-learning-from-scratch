"""
Manim scene for Module 05 — ResNet.

One story in seven beats:
  1. The depth race: ILSVRC winners 2012 -> 2015, layers against top-5 error —
     8, 8, 22, 152 — deeper kept winning.
  2. Until it didn't: our toy's training accuracy against depth for plain
     networks (fine to 18 layers, falling after), then CIFAR-10's plain-20 vs
     plain-56 training error — deeper is worse ON THE TRAINING DATA.
  3. The fix: the residual block, y = relu(F(x) + x), the identity shortcut
     lit up as the path the gradient can take.
  4. Why it works: the gradient each block receives at initialization, plain
     (exploding toward the input) vs residual (level) — and the input-gradient
     of an 18-layer plain net, already shattered into noise, next to its
     residual twin's, still structured.
  5. Depth works again: the toy's residual twins fit at every depth, and the
     CIFAR ResNet-56 trains as well as ResNet-20 -- depth no longer hurts.
  6. What a ResNet is: three blocks unravel into eight paths; delete a block
     from a trained ResNet-56 and almost nothing happens, from plain-56 and it
     collapses.
  7. Next: depth through time (Module 06).

Data comes from resnet_assets.json, written by the notebook (python/resnet.py
export_assets()) from its own runs — the same numbers the post reports.

Render (Windows, from repo root, with ffmpeg on PATH):
    python -m manim -qm --media_dir topics/05-resnet/anim/media \
      topics/05-resnet/anim/scene.py ResNetStory

No LaTeX required — all labels use Text (Pango), not MathTex.
"""

import json
import os

import numpy as np
from manim import (
    Scene, Rectangle, RoundedRectangle, Circle, Line, Arrow, CurvedArrow, Dot, Text,
    VGroup, Axes, FadeIn, FadeOut, Create, GrowFromEdge, Transform, Indicate,
    GREY, WHITE, UP, DOWN, LEFT, RIGHT, ORIGIN,
)

ACCENT = "#6d4aff"   # the residual network / the winner
ACTION = "#d81b60"   # the plain network's failure, the gradient
DATA_A = "#0072b2"
DATA_B = "#e69f00"
PLAIN = "#e15759"    # plain networks
RES = "#4e79a7"      # residual networks
_HERE = os.path.dirname(os.path.abspath(__file__))
# RESNET_ASSETS overrides the path (for test renders before the notebook has run).
ASSETS = os.environ.get("RESNET_ASSETS", os.path.join(_HERE, "resnet_assets.json"))


def _load_assets():
    with open(ASSETS, encoding="utf-8") as fh:
        return json.load(fh)


def _axes(x_range, y_range, x_len, y_len, pos):
    ax = Axes(x_range=x_range, y_range=y_range, x_length=x_len, y_length=y_len,
              tips=False, axis_config={"color": GREY, "include_ticks": False})
    return ax.move_to(pos)


def _curve(run):
    """(steps, training error) of one Figure-6 run: its 40-step trace if the
    export has one, else one point per epoch."""
    if run.get("trace_step"):
        return list(run["trace_step"]), list(run["trace_err"])
    errs = run["train_err"]
    return list(range(1, len(errs) + 1)), list(errs)


def _label(text, size=16, color=None):
    """A Text label; grey unless a color (hex string or Manim color) is given."""
    return Text(text, font_size=size, color=GREY if color is None else color)


class ResNetStory(Scene):
    def construct(self):
        a = _load_assets()
        self._beat_race()
        self._beat_degradation(a)
        self._beat_block()
        self._beat_gradients(a)
        self._beat_fixed(a)
        self._beat_ensemble(a)
        self._beat_next()

    # ── Beat 1: the depth race ──────────────────────────────────────────────
    def _beat_race(self):
        title = Text("ImageNet winners: deeper kept winning", font_size=34).to_edge(UP)
        rows = [("2012  AlexNet", 8, 15.3), ("2013  Clarifai (ZFNet)", 8, 11.7),
                ("2014  GoogLeNet", 22, 6.7), ("2015  ResNet", 152, 3.57)]
        bars = VGroup()
        scale = 0.032                        # scene units per layer
        for k, (name, depth, err) in enumerate(rows):
            yy = 1.4 - k * 1.0
            lbl = _label(name, 22, WHITE if depth == 152 else GREY)
            lbl.move_to(np.array([-4.2, yy, 0.0])).align_to(np.array([-6.4, 0.0, 0.0]), LEFT)
            bar = Rectangle(width=max(0.08, depth * scale), height=0.5, stroke_width=0,
                            fill_color=ACCENT if depth == 152 else GREY, fill_opacity=0.85)
            bar.move_to(np.array([-2.4 + depth * scale / 2, yy, 0.0]))
            val = _label(f"{depth} layers   {err}% top-5 error", 20, WHITE).next_to(bar, RIGHT, buff=0.15)
            bars.add(VGroup(lbl, bar, val))
        self.play(FadeIn(title))
        for g in bars[:3]:
            self.play(FadeIn(g[0]), GrowFromEdge(g[1], LEFT), FadeIn(g[2]), run_time=0.6)
        self.wait(0.3)
        self.play(FadeIn(bars[3][0]), GrowFromEdge(bars[3][1], LEFT), FadeIn(bars[3][2]), run_time=1.2)
        cap = Text("But in between, stacking more plain layers had stopped working.",
                   font_size=24, color=ACTION).to_edge(DOWN)
        self.play(FadeIn(cap))
        self.wait(1.6)
        self.play(FadeOut(title), FadeOut(bars), FadeOut(cap))

    # ── Beat 2: the degradation problem ─────────────────────────────────────
    def _beat_degradation(self, a):
        title = Text("Deeper plain networks fit their OWN training data worse", font_size=30).to_edge(UP)
        toy = a["toy"]
        layers, acc = toy["layers"], toy["plain_acc"]
        ax = _axes([0, len(layers) - 1, 1], [0.4, 1.05, 0.2], 5.0, 3.0, LEFT * 3.3 + DOWN * 0.2)
        pts = [ax.c2p(i, v) for i, v in enumerate(acc)]
        line = VGroup(*[Line(p, q, color=PLAIN, stroke_width=4) for p, q in zip(pts, pts[1:])])
        dots = VGroup(*[Dot(p, color=PLAIN, radius=0.06) for p in pts])
        xt = VGroup(*[_label(str(L), 14).next_to(ax.c2p(i, 0.4), DOWN, buff=0.1)
                      for i, L in enumerate(layers)])
        xl = _label("layers (toy: 128 spiral points)", 16).next_to(xt, DOWN, buff=0.15)
        yl = _label("training accuracy", 16).next_to(ax, UP, buff=0.1).align_to(ax, LEFT)
        self.play(FadeIn(title), Create(ax), FadeIn(xt), FadeIn(xl), FadeIn(yl))
        self.play(Create(line), FadeIn(dots), run_time=1.6)

        # CIFAR-10: plain-20 vs plain-56 training error per epoch.
        f6 = a["fig6"]
        (s20, e20), (s56, e56) = _curve(f6["plain20"]), _curve(f6["plain56"])
        n = max(s20[-1], s56[-1])
        top = max(max(e20), max(e56)) * 1.05
        ax2 = _axes([0, n, n / 4], [0, top, 0.2], 4.6, 3.0, RIGHT * 3.3 + DOWN * 0.2)
        g20 = ax2.plot_line_graph(s20, e20, line_color=DATA_B, add_vertex_dots=False, stroke_width=2)
        g56 = ax2.plot_line_graph(s56, e56, line_color=PLAIN, add_vertex_dots=False, stroke_width=2)
        l20 = _label("plain-20", 16, DATA_B).next_to(ax2.c2p(s20[-1], e20[-1]), RIGHT, buff=0.1)
        l56 = _label("plain-56", 16, PLAIN).next_to(ax2.c2p(s56[-1], e56[-1]), RIGHT, buff=0.1)
        cap2 = _label("CIFAR-10: training error vs SGD step", 16).next_to(ax2, DOWN, buff=0.35)
        self.play(Create(ax2), FadeIn(cap2))
        self.play(Create(g20), Create(g56), run_time=1.8)
        self.play(FadeIn(l20), FadeIn(l56))
        msg = Text("Not overfitting. Not too little capacity. An optimization failure.",
                   font_size=24, color=ACTION).to_edge(DOWN)
        self.play(FadeIn(msg))
        self.wait(2.0)
        self.play(*[FadeOut(m) for m in (title, ax, line, dots, xt, xl, yl, ax2, g20, g56,
                                          l20, l56, cap2, msg)])

    # ── Beat 3: the residual block ──────────────────────────────────────────
    def _beat_block(self):
        title = Text("The fix: let each block learn a change, F(x), and add x back",
                     font_size=28).to_edge(UP)
        names = ["weight layer", "BN · ReLU", "weight layer", "BN"]
        boxes = VGroup()
        for k, nm in enumerate(names):
            b = RoundedRectangle(width=2.0, height=0.8, corner_radius=0.12, color=WHITE, stroke_width=2)
            b.move_to(np.array([-3.6 + 2.4 * k, 0.0, 0.0]))
            boxes.add(VGroup(b, _label(nm, 18, WHITE).move_to(b)))
        x_in = _label("x", 30, WHITE).move_to(np.array([-6.0, 0.0, 0.0]))
        plus = Circle(radius=0.32, color=ACCENT, stroke_width=3).move_to(np.array([5.7, 0.0, 0.0]))
        plus_t = _label("+", 34, ACCENT).move_to(plus)
        arrows = VGroup(Arrow(x_in.get_right(), boxes[0].get_left(), buff=0.1, color=GREY))
        for p, q in zip(boxes, boxes[1:]):
            arrows.add(Arrow(p.get_right(), q.get_left(), buff=0.08, color=GREY))
        arrows.add(Arrow(boxes[-1].get_right(), plus.get_left(), buff=0.05, color=GREY))
        out = _label("relu → y", 22, WHITE).next_to(plus, DOWN, buff=0.4)
        out_arrow = Arrow(plus.get_bottom(), out.get_top(), buff=0.05, color=GREY)
        fx = _label("F(x): the residual", 18, GREY).next_to(boxes, DOWN, buff=0.35)
        self.play(FadeIn(title), FadeIn(x_in), FadeIn(boxes), Create(arrows), FadeIn(fx))
        self.play(FadeIn(plus), FadeIn(plus_t), Create(out_arrow), FadeIn(out))
        skip = CurvedArrow(x_in.get_top() + UP * 0.1, plus.get_top() + UP * 0.05, angle=-1.1,
                           color=ACCENT, stroke_width=5)
        skip_l = _label("identity shortcut: x", 20, ACCENT).next_to(skip, UP, buff=0.05)
        self.play(Create(skip), FadeIn(skip_l), run_time=1.2)
        cap = Text("To do nothing, drive F to zero. And the gradient gets a path with no layers on it.",
                   font_size=22).to_edge(DOWN)
        self.play(FadeIn(cap))
        # The gradient flowing back along the shortcut.
        pulse = Dot(plus.get_top(), color=ACTION, radius=0.12)
        self.play(FadeIn(pulse))
        self.play(pulse.animate.move_to(x_in.get_top() + UP * 0.1), run_time=1.0)
        self.play(Indicate(x_in, color=ACTION))
        self.wait(1.4)
        self.play(*[FadeOut(m) for m in (title, x_in, boxes, arrows, fx, plus, plus_t, out_arrow, out,
                                          skip, skip_l, cap, pulse)])

    # ── Beat 4: gradients, plain vs residual ────────────────────────────────
    def _beat_gradients(self, a):
        title = Text("The gradient each block receives (34 layers, step 0)", font_size=30).to_edge(UP)
        gp = np.log10(np.asarray(a["toy_grad_norms"]["plain"], dtype=np.float64))
        gr = np.log10(np.asarray(a["toy_grad_norms"]["res"], dtype=np.float64))
        lo, hi = float(min(gp.min(), gr.min())) - 0.3, float(max(gp.max(), gr.max())) + 0.3
        gp, gr = gp - lo, gr - lo            # bars rise from the chart's floor
        D = len(gp)
        ax = _axes([0, D + 1, 1], [0, hi - lo, 1], 6.0, 3.2, LEFT * 2.6 + DOWN * 0.3)
        width = 6.0 / (D + 1) * 0.36

        def bars(vals, color, shift):
            g = VGroup()
            for k, v in enumerate(vals):
                base, topp = ax.c2p(k + 1, 0), ax.c2p(k + 1, v)
                r = Rectangle(width=width, height=max(0.02, topp[1] - base[1]), stroke_width=0,
                              fill_color=color, fill_opacity=0.9)
                r.move_to(np.array([base[0] + shift, (base[1] + topp[1]) / 2, 0.0]))
                g.add(r)
            return g

        bp, br = bars(gp, PLAIN, -width / 1.8), bars(gr, RES, width / 1.8)
        yl = _label("||dW||, log scale", 16).next_to(ax, UP, buff=0.1).align_to(ax, LEFT)
        xl = _label("block 1 (input side)  …  block 16", 16).next_to(ax, DOWN, buff=0.2)
        key = VGroup(_label("plain", 18, PLAIN), _label("residual", 18, RES)).arrange(DOWN, aligned_edge=LEFT)
        key.next_to(ax, RIGHT, buff=0.3).shift(UP * 1.0)
        self.play(FadeIn(title), Create(ax), FadeIn(yl), FadeIn(xl))
        self.play(FadeIn(bp, lag_ratio=0.05), FadeIn(key[0]), run_time=1.4)
        ratio = 10 ** (gp[0] - gp[-1])
        c1 = Text(f"plain: the first block is pushed {ratio:,.0f}x harder than the last",
                  font_size=22, color=PLAIN).to_edge(DOWN)
        self.play(FadeIn(c1))
        self.wait(1.2)
        self.play(FadeIn(br, lag_ratio=0.05), FadeIn(key[1]), run_time=1.4)
        c2 = Text("residual: level — one learning rate suits every block", font_size=22,
                  color=RES).to_edge(DOWN)
        self.play(FadeOut(c1), FadeIn(c2))
        self.wait(1.4)
        self.play(*[FadeOut(m) for m in (ax, bp, br, yl, xl, key, c2)])

        # Shattered gradients: d(logit)/d(input) along a line.
        sh = a["shattered"]
        s = np.asarray(sh["s"]); fp = np.asarray(sh["plain"]); fr = np.asarray(sh["res"])
        cp_, cr_ = (float(np.corrcoef(g[:-1], g[1:])[0, 1]) for g in (fp, fr))
        t2 = Text(f"How the output responds to the input, point by point ({sh.get('layers', 18)} layers)",
                  font_size=26).to_edge(UP)
        axl = _axes([-1, 1, 0.5], [float(fp.min()), float(fp.max()), 1], 5.0, 2.6, LEFT * 3.3 + DOWN * 0.2)
        axr = _axes([-1, 1, 0.5], [float(fr.min()), float(fr.max()), 1], 5.0, 2.6, RIGHT * 3.3 + DOWN * 0.2)
        cp = axl.plot_line_graph(s.tolist(), fp.tolist(), line_color=PLAIN, add_vertex_dots=False,
                                 stroke_width=1.5)
        cr = axr.plot_line_graph(s.tolist(), fr.tolist(), line_color=RES, add_vertex_dots=False,
                                 stroke_width=2.5)
        lp = _label(f"plain: noise (neighbour correlation {cp_:+.2f})", 18, PLAIN).next_to(axl, DOWN, buff=0.3)
        lr = _label(f"residual: still structured ({cr_:+.2f})", 18, RES).next_to(axr, DOWN, buff=0.3)
        self.play(Transform(title, t2), Create(axl), Create(axr))
        self.play(Create(cp), Create(cr), run_time=2.0)
        self.play(FadeIn(lp), FadeIn(lr))
        self.wait(2.0)
        self.play(*[FadeOut(m) for m in (title, axl, axr, cp, cr, lp, lr)])

    # ── Beat 5: depth works again ───────────────────────────────────────────
    def _beat_fixed(self, a):
        title = Text("Same layers, same weights, one + per block", font_size=30).to_edge(UP)
        toy = a["toy"]
        layers = toy["layers"]
        ax = _axes([0, len(layers) - 1, 1], [0.4, 1.05, 0.2], 5.0, 3.0, LEFT * 3.3 + DOWN * 0.2)
        xt = VGroup(*[_label(str(L), 14).next_to(ax.c2p(i, 0.4), DOWN, buff=0.1)
                      for i, L in enumerate(layers)])

        def curve(vals, color):
            pts = [ax.c2p(i, v) for i, v in enumerate(vals)]
            return VGroup(*[Line(p, q, color=color, stroke_width=4) for p, q in zip(pts, pts[1:])],
                          *[Dot(p, color=color, radius=0.06) for p in pts])

        cp, cr = curve(toy["plain_acc"], PLAIN), curve(toy["res_acc"], RES)
        lab = VGroup(_label("plain", 18, PLAIN), _label("residual", 18, RES)).arrange(RIGHT, buff=0.5)
        lab.next_to(ax, UP, buff=0.15)
        self.play(FadeIn(title), Create(ax), FadeIn(xt), FadeIn(cp), FadeIn(lab[0]))
        self.play(Create(cr), FadeIn(lab[1]), run_time=1.4)

        f6 = a["fig6"]
        (s20, r20), (s56, r56) = _curve(f6["res20"]), _curve(f6["res56"])
        n = max(s20[-1], s56[-1])
        top = max(max(_curve(f6["plain56"])[1]), max(r20)) * 1.05
        ax2 = _axes([0, n, n / 4], [0, top, 0.2], 4.6, 3.0, RIGHT * 3.3 + DOWN * 0.2)
        g20 = ax2.plot_line_graph(s20, r20, line_color=DATA_B, add_vertex_dots=False, stroke_width=2)
        g56 = ax2.plot_line_graph(s56, r56, line_color=RES, add_vertex_dots=False, stroke_width=2)
        l20 = _label("ResNet-20", 16, DATA_B).next_to(ax2.c2p(s20[-1], r20[-1]), RIGHT, buff=0.1).shift(UP * 0.15)
        l56 = _label("ResNet-56", 16, RES).next_to(ax2.c2p(s56[-1], r56[-1]), RIGHT, buff=0.1).shift(DOWN * 0.15)
        cap2 = _label("CIFAR-10: training error vs SGD step", 16).next_to(ax2, DOWN, buff=0.35)
        self.play(Create(ax2), FadeIn(cap2))
        self.play(Create(g20), Create(g56), run_time=1.8)
        self.play(FadeIn(l20), FadeIn(l56))
        msg = Text("Depth no longer hurts.", font_size=26, color=ACCENT).to_edge(DOWN)
        self.play(FadeIn(msg))
        self.wait(1.8)
        self.play(*[FadeOut(m) for m in (title, ax, xt, cp, cr, lab, ax2, g20, g56, l20, l56, cap2, msg)])

    # ── Beat 6: an ensemble of paths ────────────────────────────────────────
    def _beat_ensemble(self, a):
        title = Text("Three blocks, eight paths: a ResNet is many networks at once",
                     font_size=28).to_edge(UP)
        # The unraveled view: each block is either taken or skipped.
        xs = [-4.5, -1.5, 1.5, 4.5]
        nodes = VGroup(*[Circle(radius=0.18, color=WHITE, fill_opacity=1, fill_color=GREY)
                         .move_to(np.array([x, 0.6, 0.0])) for x in xs])
        blocks = VGroup(*[_label(f"F{k + 1}", 18, ACCENT).move_to(np.array([(xs[k] + xs[k + 1]) / 2, 1.25, 0.0]))
                          for k in range(3)])
        edges = VGroup()
        for k in range(3):
            p, q = nodes[k].get_center(), nodes[k + 1].get_center()
            edges.add(Line(p, q, color=GREY, stroke_width=3))                       # skip
            edges.add(CurvedArrow(p, q, angle=-0.9, color=ACCENT, stroke_width=3))  # through F
        self.play(FadeIn(title), FadeIn(nodes), Create(edges), FadeIn(blocks))
        paths = _label("2 × 2 × 2 = 8 paths;  ResNet-56's 27 blocks: 2^27 ≈ 134 million", 20, WHITE)
        paths.next_to(nodes, DOWN, buff=0.5)
        self.play(FadeIn(paths))
        self.wait(1.6)
        self.play(FadeOut(nodes), FadeOut(edges), FadeOut(blocks), FadeOut(paths))

        les = a["lesion"]
        res = [v for v in les["res"] if v is not None]
        pl = [v for v in les["plain"] if v is not None]
        t2 = Text("Delete one block from a trained 56-layer network", font_size=28).to_edge(UP)
        ax = _axes([0, max(len(res), len(pl)) + 1, 5], [0, 1, 0.25], 9.0, 3.4, DOWN * 0.3)
        yl = _label("test accuracy", 16).next_to(ax, UP, buff=0.1).align_to(ax, LEFT)
        chance = ax.plot(lambda _: 0.1, color=GREY, stroke_width=1.5)
        ch_l = _label("chance", 14).next_to(ax.c2p(0, 0.1), LEFT, buff=0.1)
        dr = VGroup(*[Dot(ax.c2p(k + 1, v), color=RES, radius=0.06) for k, v in enumerate(res)])
        dp = VGroup(*[Dot(ax.c2p(k + 1, v), color=PLAIN, radius=0.06) for k, v in enumerate(pl)])
        base_r = ax.plot(lambda _: les["res_base"], color=RES, stroke_width=1.5)
        base_p = ax.plot(lambda _: les["plain_base"], color=PLAIN, stroke_width=1.5)
        self.play(Transform(title, t2), Create(ax), FadeIn(yl), Create(chance), FadeIn(ch_l))
        self.play(Create(base_r), Create(base_p))
        self.play(FadeIn(dr, lag_ratio=0.05), run_time=1.2)
        cap = Text(f"ResNet-56: intact {les['res_base']:.0%}, one block gone: about {np.mean(res):.0%}",
                   font_size=22, color=RES).to_edge(DOWN)
        self.play(FadeIn(cap))
        self.wait(1.2)
        self.play(FadeIn(dp, lag_ratio=0.05), run_time=1.2)
        cap2 = Text(f"plain-56: intact {les['plain_base']:.0%}, one block gone: about {np.mean(pl):.0%}",
                    font_size=22, color=PLAIN).to_edge(DOWN)
        self.play(FadeOut(cap), FadeIn(cap2))
        self.wait(2.0)
        self.play(*[FadeOut(m) for m in (title, ax, yl, chance, ch_l, dr, dp, base_r, base_p, cap2)])

    # ── Beat 7: next ────────────────────────────────────────────────────────
    def _beat_next(self):
        a = Text("Depth through layers: solved by a shortcut.", font_size=30)
        b = Text("Module 06: depth through TIME — the recurrent network.", font_size=26, color=ACCENT)
        VGroup(a, b).arrange(DOWN, buff=0.5)
        self.play(FadeIn(a))
        self.play(FadeIn(b))
        self.wait(2.0)
        self.play(FadeOut(a), FadeOut(b))
