"""
Manim scene for Module 06 — RNN.

One story in eight beats:
  1. Sequences: the input arrives one step at a time, of any length; a network
     needs a memory that it carries from step to step.
  2. The recurrent cell, and unrolling it: one cell with a loop becomes a chain
     of copies reading "h e l l" and predicting "e l l o" — every copy with the
     SAME weights (one color).
  3. Backpropagation through time: a gradient travels back along the chain,
     multiplied by the same matrix at every step — it shrinks below spectral
     radius 1 and grows above (our measured profiles at three gains).
  4. The gradient only teaches what it reaches: the echo toy's truncation
     staircase, delay d against gradient reach k.
  5. Counting needs no reach: the Elman-trained mod-3 counter's hidden states,
     in clusters that each belong to one count — a state machine.
  6. Shakespeare: validation loss against the n-gram baselines, then what the
     model writes after 1, 3 and 20 epochs.
  7. Elman's experiment redone: surprisal along a passage, peaking at the first
     letter of each word.
  8. Next: the LSTM (Module 07).

Data comes from rnn_assets.json, written by the notebook (python/rnn.py
export_assets()) from its own runs — the same numbers the post reports.

Render (Windows, from repo root, with ffmpeg on PATH):
    python -m manim -qm --media_dir topics/06-rnn/anim/media \
      topics/06-rnn/anim/scene.py RNNStory

No LaTeX required — all labels use Text (Pango), not MathTex.
"""

import json
import os

import numpy as np
from manim import (
    Scene, Rectangle, RoundedRectangle, Square, Line, Arrow, CurvedArrow, Dot, Text,
    VGroup, Axes, FadeIn, FadeOut, Create, GrowFromEdge, Indicate,
    GREY, WHITE, UP, DOWN, LEFT, RIGHT,
)

ACCENT = "#6d4aff"   # the shared weights / the model
ACTION = "#d81b60"   # the gradient
DATA_A = "#0072b2"
DATA_B = "#e69f00"
DATA_C = "#009e73"
NGRAM = "#9a9a96"
_HERE = os.path.dirname(os.path.abspath(__file__))
# RNN_ASSETS overrides the path (for test renders before the notebook has run).
ASSETS = os.environ.get("RNN_ASSETS", os.path.join(_HERE, "rnn_assets.json"))
MONO = "Courier New"


def _load_assets():
    with open(ASSETS, encoding="utf-8") as fh:
        return json.load(fh)


def _axes(x_range, y_range, x_len, y_len, pos):
    ax = Axes(x_range=x_range, y_range=y_range, x_length=x_len, y_length=y_len,
              tips=False, axis_config={"color": GREY, "include_ticks": False})
    return ax.move_to(pos)


def _label(text, size=16, color=None, font=None):
    """A Text label; grey unless a color (hex string or Manim color) is given."""
    kw = {"font": font} if font else {}
    return Text(text, font_size=size, color=GREY if color is None else color, **kw)


def _cell(pos, label="h", size=0.9):
    box = RoundedRectangle(width=size, height=size, corner_radius=0.12, color=WHITE, stroke_width=2)
    box.move_to(pos)
    return VGroup(box, _label(label, 24, WHITE).move_to(box))


class RNNStory(Scene):
    def construct(self):
        a = _load_assets()
        self._beat_sequences()
        chain = self._beat_unroll()
        self._beat_bptt(a, chain)
        self._beat_reach(a)
        self._beat_counting(a)
        self._beat_shakespeare(a)
        self._beat_surprisal(a)
        self._beat_next()

    # ── Beat 1: sequences ───────────────────────────────────────────────────
    def _beat_sequences(self):
        title = Text("Language arrives one piece at a time", font_size=34).to_edge(UP)
        words = "to be or not to be that is the question".split()
        toks = VGroup(*[_label(w, 26, WHITE) for w in words]).arrange(RIGHT, buff=0.28)
        toks.move_to(DOWN * 0.2)
        if toks.width > 12.5:
            toks.scale_to_fit_width(12.5)
        self.play(FadeIn(title))
        self.play(FadeIn(toks, lag_ratio=0.25), run_time=2.2)
        cap = Text("Any length. What matters now can depend on something long ago.",
                   font_size=24, color=ACCENT).to_edge(DOWN)
        self.play(FadeIn(cap))
        self.play(Indicate(toks[1], color=ACTION), Indicate(toks[5], color=ACTION))
        self.wait(1.2)
        self.play(FadeOut(title), FadeOut(toks), FadeOut(cap))

    # ── Beat 2: the cell, unrolled ──────────────────────────────────────────
    def _beat_unroll(self):
        title = Text("A recurrent network: one cell, a memory, the same weights every step",
                     font_size=28).to_edge(UP)
        cell = _cell(np.array([0.0, 0.0, 0.0]))
        x = _label("x<t>", 22, WHITE).next_to(cell, DOWN, buff=0.7)
        y = _label("prediction", 20, WHITE).next_to(cell, UP, buff=0.7)
        ax_in = Arrow(x.get_top(), cell.get_bottom(), buff=0.08, color=GREY)
        ax_out = Arrow(cell.get_top(), y.get_bottom(), buff=0.08, color=GREY)
        loop = CurvedArrow(cell.get_right() + UP * 0.2, cell.get_right() + DOWN * 0.2, angle=-4.0,
                           color=ACCENT, stroke_width=4)
        loop_l = _label("Whh", 20, ACCENT).next_to(loop, RIGHT, buff=0.15)
        self.play(FadeIn(title), FadeIn(cell), FadeIn(x), FadeIn(y), Create(ax_in), Create(ax_out))
        self.play(Create(loop), FadeIn(loop_l))
        eq = _label("h<t> = tanh( x<t> Wxh + h<t-1> Whh + bh )", 24, WHITE).to_edge(DOWN).shift(UP * 0.4)
        self.play(FadeIn(eq))
        self.wait(1.2)

        # Unroll into four copies reading "h e l l" and predicting "e l l o".
        reads, preds = "hell", "ello"
        cells, ins, outs, links, glyphs = VGroup(), VGroup(), VGroup(), VGroup(), VGroup()
        for k in range(4):
            c = _cell(np.array([-4.5 + 3.0 * k, 0.0, 0.0]), "h")
            cells.add(c)
            xi = _label(reads[k], 30, WHITE, MONO).next_to(c, DOWN, buff=0.8)
            yo = _label(preds[k], 30, DATA_B, MONO).next_to(c, UP, buff=0.8)
            ins.add(VGroup(xi, Arrow(xi.get_top(), c.get_bottom(), buff=0.08, color=GREY)))
            outs.add(VGroup(yo, Arrow(c.get_top(), yo.get_bottom(), buff=0.08, color=GREY)))
        for p, q in zip(cells, cells[1:]):
            arr = Arrow(p.get_right(), q.get_left(), buff=0.08, color=ACCENT, stroke_width=5)
            links.add(arr)
            glyphs.add(_label("Whh", 18, ACCENT).next_to(arr, UP, buff=0.05))
        h0 = _label("h0 = 0", 18).next_to(cells[0], LEFT, buff=0.4)
        self.play(FadeOut(VGroup(cell, x, y, ax_in, ax_out, loop, loop_l)),
                  FadeIn(cells), FadeIn(h0), run_time=1.0)
        for k in range(4):
            anims = [FadeIn(ins[k]), FadeIn(outs[k])]
            if k > 0:
                anims += [Create(links[k - 1]), FadeIn(glyphs[k - 1])]
            self.play(*anims, run_time=0.6)
        cap = Text("Unrolled, it is a deep network whose layers share one set of weights.",
                   font_size=24, color=ACCENT).to_edge(DOWN)
        self.play(FadeOut(eq), FadeIn(cap))
        self.play(Indicate(outs[1][0], color=ACTION), Indicate(outs[2][0], color=ACTION))
        note = _label("same input 'l', different answers: only the memory differs", 18, ACTION)
        note.next_to(cap, UP, buff=0.25)
        self.play(FadeIn(note))
        self.wait(1.4)
        self.play(FadeOut(title), FadeOut(cap), FadeOut(note), FadeOut(h0))
        return VGroup(cells, ins, outs, links, glyphs)

    # ── Beat 3: backpropagation through time ────────────────────────────────
    def _beat_bptt(self, a, chain):
        cells, ins, outs, links, glyphs = chain
        title = Text("Backpropagation through time: back along the chain", font_size=30).to_edge(UP)
        self.play(FadeIn(title))
        # A gradient pulse travelling back, shrinking at every step.
        pulse = Dot(cells[3].get_center(), color=ACTION, radius=0.22)
        self.play(FadeIn(pulse))
        for k in (2, 1, 0):
            self.play(pulse.animate.move_to(cells[k].get_center()).scale(0.6), run_time=0.6)
        cap = Text("Every step back multiplies the gradient by the SAME matrix, Whh (and tanh's slope).",
                   font_size=22).to_edge(DOWN)
        self.play(FadeIn(cap))
        self.wait(1.2)
        self.play(FadeOut(VGroup(cells, ins, outs, links, glyphs, pulse)), FadeOut(cap))

        # The measured profiles at three gains (log scale).
        g = a["grad_init"]
        gains = [("0.5", DATA_A), ("1.0", ACCENT), ("2.0", DATA_B)]
        curves = {k: np.log10(np.maximum(np.asarray(g[k], dtype=float), 1e-16)) for k, _ in gains}
        lo = max(-8.0, min(float(c.min()) for c in curves.values()))
        hi = max(float(c.max()) for c in curves.values())
        n = len(curves["1.0"])
        ax = _axes([0, n - 1, 10], [lo, hi + 0.5, 2], 8.0, 3.6, DOWN * 0.3)
        graphs, keys = VGroup(), VGroup()
        for k, col in gains:
            ys = np.maximum(curves[k], lo)
            graphs.add(ax.plot_line_graph(list(range(n)), ys.tolist(), line_color=col,
                                          add_vertex_dots=False, stroke_width=3))
            keys.add(_label(f"Whh scaled x{k}", 18, col))
        keys.arrange(DOWN, aligned_edge=LEFT).next_to(ax, RIGHT, buff=0.2).shift(UP * 0.8)
        xl = _label("steps back from the last", 16).next_to(ax, DOWN, buff=0.15)
        yl = _label("gradient size (log scale)", 16).next_to(ax, UP, buff=0.1).align_to(ax, LEFT)
        self.play(Create(ax), FadeIn(xl), FadeIn(yl))
        for gr, kk in zip(graphs, keys):
            self.play(Create(gr), FadeIn(kk), run_time=1.0)
        cap2 = Text("Below spectral radius 1 it vanishes; above it, it explodes.", font_size=24,
                    color=ACTION).to_edge(DOWN)
        self.play(FadeIn(cap2))
        self.wait(1.6)
        self.play(*[FadeOut(m) for m in (title, ax, graphs, keys, xl, yl, cap2)])

    # ── Beat 4: the truncation staircase ────────────────────────────────────
    def _beat_reach(self, a):
        title = Text("Echo the symbol from d steps back: it learns only what the gradient reaches",
                     font_size=26).to_edge(UP)
        r = a["reach"]
        acc = np.asarray(r["acc"], dtype=float)
        ks, ds = r["ks"], r["ds"]
        size = 0.75
        grid = VGroup()
        for i in range(len(ks)):
            for j in range(len(ds)):
                v = float(acc[i, j])
                t = (v - 0.25) / 0.75
                sq = Square(side_length=size, stroke_width=1, stroke_color=GREY,
                            fill_color=ACCENT, fill_opacity=0.08 + 0.85 * max(0.0, t))
                sq.move_to(np.array([-1.6 + j * size, 1.2 - i * size, 0.0]))
                txt = _label(f"{v:.2f}", 14, WHITE).move_to(sq)
                grid.add(VGroup(sq, txt))
        rows = VGroup(*[_label(("reach " + k) if k != "1" else "reach 1 (Elman)", 16).move_to(
            np.array([-1.6 - size * 0.6, 1.2 - i * size, 0.0]), aligned_edge=RIGHT)
            for i, k in enumerate(ks)])
        for lbl, i in zip(rows, range(len(ks))):
            lbl.next_to(grid[i * len(ds)], LEFT, buff=0.2)
        cols = VGroup(*[_label(f"d={d}", 16).next_to(grid[j], UP, buff=0.12) for j, d in enumerate(ds)])
        self.play(FadeIn(title), FadeIn(rows), FadeIn(cols))
        for i in range(len(ks)):
            self.play(FadeIn(VGroup(*grid[i * len(ds):(i + 1) * len(ds)])), run_time=0.5)
        cap = Text("Cut the gradient every k steps: every delay shorter than k is learned, none longer.",
                   font_size=22, color=ACCENT).to_edge(DOWN)
        self.play(FadeIn(cap))
        self.play(Indicate(rows[0], color=ACTION))
        self.wait(1.8)
        self.play(*[FadeOut(m) for m in (title, grid, rows, cols, cap)])

    # ── Beat 5: counting ────────────────────────────────────────────────────
    def _beat_counting(self, a):
        title = Text("Counting 1s mod 3: even Elman's one-step training learns it", font_size=28).to_edge(UP)
        c = a["counting"]
        pc = np.asarray(c["pc"], dtype=float)
        cnt = np.asarray(c["count"], dtype=int)
        span = max(float(np.abs(pc).max()), 1e-6)
        ax = _axes([-span, span, span], [-span, span, span], 4.4, 4.4, LEFT * 2.6 + DOWN * 0.3)
        cols = (DATA_A, DATA_B, DATA_C)
        dots = VGroup(*[Dot(ax.c2p(float(p[0]), float(p[1])), radius=0.035, color=cols[k], fill_opacity=0.7)
                        for p, k in zip(pc, cnt)])
        key = VGroup(*[_label(f"count = {k} (mod 3)", 18, cols[k]) for k in range(3)])
        key.arrange(DOWN, aligned_edge=LEFT).next_to(ax, RIGHT, buff=0.5).shift(UP * 0.8)
        acc = c["acc"]
        longest = max(acc, key=lambda s: int(s))
        note = _label(f"trained on 20 bits; {100 * acc[longest]:.1f}% right on {int(longest):,}", 18, WHITE)
        note.next_to(key, DOWN, buff=0.4, aligned_edge=LEFT)
        self.play(FadeIn(title), Create(ax))
        self.play(FadeIn(dots, lag_ratio=0.002), run_time=1.6)
        self.play(FadeIn(key), FadeIn(note))
        cap = Text("Every step's target IS the state: a state machine built from tanh units, one count per cluster.",
                   font_size=22, color=ACCENT).to_edge(DOWN)
        self.play(FadeIn(cap))
        self.wait(1.8)
        self.play(*[FadeOut(m) for m in (title, ax, dots, key, note, cap)])

    # ── Beat 6: Shakespeare ─────────────────────────────────────────────────
    def _beat_shakespeare(self, a):
        ch = a["char"]
        title = Text("Tiny Shakespeare, one character at a time", font_size=32).to_edge(UP)
        val = np.asarray(ch["val_loss"], dtype=float)
        ng = {int(k): float(v) for k, v in ch["ngram"].items()}
        lo, hi = 1.5, 2.6
        ax = _axes([1, len(val), 5], [lo, hi, 0.5], 6.5, 3.4, LEFT * 1.2 + DOWN * 0.3)
        g = ax.plot_line_graph(list(range(1, len(val) + 1)), np.clip(val, lo, hi).tolist(),
                               line_color=ACCENT, add_vertex_dots=True, stroke_width=3,
                               vertex_dot_radius=0.04)
        lines, tags = VGroup(), VGroup()
        for n in (2, 3, 5):
            if n in ng and lo <= ng[n] <= hi:
                y = ax.c2p(1, ng[n])[1]
                ln = Line(np.array([ax.c2p(1, lo)[0], y, 0]), np.array([ax.c2p(len(val), lo)[0], y, 0]),
                          color=NGRAM, stroke_width=2).set_opacity(0.8)
                lines.add(ln)
                tags.add(_label(f"{n}-gram {ng[n]:.2f}", 16, NGRAM).next_to(ln, RIGHT, buff=0.1))
        yl = _label("validation loss (nats per character)", 16).next_to(ax, UP, buff=0.1).align_to(ax, LEFT)
        xl = _label("epoch", 16).next_to(ax, DOWN, buff=0.15)
        self.play(FadeIn(title), Create(ax), FadeIn(yl), FadeIn(xl))
        self.play(Create(lines), FadeIn(tags))
        self.play(Create(g), run_time=2.0)
        fin = _label(f"RNN: {val[-1]:.2f} — below every table of counts", 20, ACCENT)
        fin.next_to(ax, DOWN, buff=0.55)
        self.play(FadeIn(fin))
        self.wait(1.2)
        self.play(*[FadeOut(m) for m in (ax, g, lines, tags, yl, xl, fin)])

        samples = ch["samples"]
        keys = sorted(samples, key=int)
        prev = None
        for k in (keys[0], keys[1], keys[-1]):
            text = samples[k][:170]
            body = _label(text, 20, WHITE, MONO)
            if body.width > 12.5:
                body.scale_to_fit_width(12.5)
            head = _label(f"after epoch {k}", 22, ACCENT)
            block = VGroup(head, body).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(DOWN * 0.3)
            if prev is None:
                self.play(FadeIn(block), run_time=0.8)
            else:
                self.play(FadeOut(prev), FadeIn(block), run_time=0.8)
            prev = block
            self.wait(2.2)
        self.play(FadeOut(prev), FadeOut(title))

    # ── Beat 7: Elman's word boundaries ─────────────────────────────────────
    def _beat_surprisal(self, a):
        ch = a["char"]
        title = Text("Elman's experiment, redone: surprise peaks where words begin", font_size=28).to_edge(UP)
        text = ch["surprisal_text"][:48]
        vals = np.asarray(ch["surprisal"][:48], dtype=float)
        top = float(vals.max()) * 1.05
        width = 12.0 / len(text)
        bars, glyphs = VGroup(), VGroup()
        base_y = -1.6
        for k, (c, v) in enumerate(zip(text, vals)):
            starts = c.isalpha() and (k == 0 or not (text[k - 1].isalpha() or text[k - 1] == "'"))
            h = max(0.02, 3.4 * v / top)
            col = ACTION if starts else (DATA_A if c.isalpha() else GREY)
            r = Rectangle(width=width * 0.8, height=h, stroke_width=0, fill_color=col, fill_opacity=0.9)
            x = -6.0 + (k + 0.5) * width
            r.move_to(np.array([x, base_y + h / 2, 0.0]))
            bars.add(r)
            glyphs.add(_label("¶" if c == "\n" else c, 18, WHITE, MONO).move_to(np.array([x, base_y - 0.3, 0.0])))
        self.play(FadeIn(title), FadeIn(glyphs))
        self.play(GrowFromEdge(bars, DOWN), run_time=1.6)
        bp = ch["by_position"]
        cap = Text(f"first letter of a word: {bp['1st letter']:.2f} nats; third: {bp['3rd']:.2f}. "
                   "Which word comes next is the hard part.", font_size=22).to_edge(DOWN)
        self.play(FadeIn(cap))
        self.wait(2.2)
        self.play(*[FadeOut(m) for m in (title, bars, glyphs, cap)])

    # ── Beat 8: next ────────────────────────────────────────────────────────
    def _beat_next(self):
        t1 = Text("The gradient through time vanishes — by design, not by bad luck.", font_size=28)
        t2 = Text("Next: the LSTM — a memory that is added to, not rewritten,", font_size=28, color=ACCENT)
        t3 = Text("so the gradient can flow back through it multiplied by one.", font_size=28, color=ACCENT)
        grp = VGroup(t1, t2, t3).arrange(DOWN, buff=0.4)
        self.play(FadeIn(t1))
        self.wait(0.8)
        self.play(FadeIn(t2), FadeIn(t3))
        self.wait(2.2)
        self.play(FadeOut(grp))
