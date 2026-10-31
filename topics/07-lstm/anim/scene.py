"""
Manim scene for Module 07 — LSTM.

One story in ten beats:
  1. The wall: Module 06's gradient through time, fading to millionths 79 steps
     back (our measured profile).
  2. The cell: a conveyor belt, c, with a forget gate (x f) and a write (+ i*g)
     on it, and an output gate reading it out as h.
  3. Holding: four values written into one cell stay put for 200 steps; an RNN
     unit with two attractors merges them into two.
  4. The carousel: along the cell the gradient is multiplied by f -- with the
     gates open it reaches the signal; with the textbook forget bias of 1 it
     fades like the RNN's (our measured profiles).
  5. The latch: test accuracy against the lag, RNN against LSTM, three seeds.
  6. Two biases: the forget bias sets how long the cell remembers, 1/(1 - f);
     the lags each bias learns.
  7. Fade, never explode: the full gradient spikes past the clip; the
     carousel's alone stays calm.
  8. Inside the trained latch: the cell writes a whisper, keeps it, and
     amplifies it until the four signals are apart.
  9. Shakespeare: the LSTM below the RNN at the same budget; how far back each
     looks; a cell that counts along every line of verse.
 10. Next: dropout for recurrent nets (Module 08).

Data comes from lstm_assets.json, written by the notebook (python/lstm.py
export_assets()) from its own runs — the same numbers the post reports.

Render (Windows, from repo root, with ffmpeg on PATH):
    python -m manim -qm --media_dir topics/07-lstm/anim/media \
      topics/07-lstm/anim/scene.py LSTMStory

No LaTeX required — all labels use Text (Pango), not MathTex.
"""

import json
import os

import numpy as np
from manim import (
    Scene, RoundedRectangle, Rectangle, Circle, Line, Arrow, Dot, Text, VGroup, Axes,
    DashedLine, FadeIn, FadeOut, Create, GREY, WHITE, UP, DOWN, LEFT, RIGHT,
)

ACCENT = "#6d4aff"   # the LSTM / the cell
ACTION = "#d81b60"   # the gradient
DATA_A = "#0072b2"   # the RNN
DATA_B = "#e69f00"
DATA_C = "#009e73"
DATA_D = "#cc79a7"
SIGNAL_COLORS = (DATA_A, DATA_B, DATA_C, DATA_D)
_HERE = os.path.dirname(os.path.abspath(__file__))
# LSTM_ASSETS overrides the path (for test renders before the notebook has run).
ASSETS = os.environ.get("LSTM_ASSETS", os.path.join(_HERE, "lstm_assets.json"))
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


def _grid_square(size, color, opacity):
    """A filled grid cell (a Rectangle, so a grid's rows can be squashed)."""
    return Rectangle(width=size, height=size * 0.75, stroke_width=1, stroke_color=GREY,
                     fill_color=color, fill_opacity=opacity)


def _log_curve(ax, ys, color, lo, width=3):
    ys = np.log10(np.maximum(np.asarray(ys, dtype=float), 10.0 ** lo))
    return ax.plot_line_graph(list(range(len(ys))), ys.tolist(), line_color=color,
                              add_vertex_dots=False, stroke_width=width)


class LSTMStory(Scene):
    def construct(self):
        a = _load_assets()
        self._beat_wall(a)
        self._beat_cell()
        self._beat_hold(a)
        self._beat_carousel(a)
        self._beat_latch(a)
        self._beat_biases(a)
        self._beat_explode(a)
        self._beat_fan(a)
        self._beat_shakespeare(a)
        self._beat_next()

    # ── Beat 1: the wall ────────────────────────────────────────────────────
    def _beat_wall(self, a):
        title = Text("Module 06's wall: the gradient through time fades", font_size=32).to_edge(UP)
        g = a["grad"]["RNN (Xavier)"]["x"]
        lo = -6.0
        ax = _axes([0, len(g) - 1, 10], [lo, 0.5, 1], 8.5, 3.6, DOWN * 0.3)
        curve = _log_curve(ax, g, DATA_A, lo)
        xl = _label("steps back from the question", 16).next_to(ax, DOWN, buff=0.15)
        yl = _label("gradient reaching the input (log scale)", 16).next_to(ax, UP, buff=0.1).align_to(ax, LEFT)
        self.play(FadeIn(title), Create(ax), FadeIn(xl), FadeIn(yl))
        self.play(Create(curve), run_time=2.0)
        end = Dot(ax.c2p(len(g) - 1, np.log10(max(g[-1], 1e-6))), color=ACTION, radius=0.08)
        note = _label(f"79 steps back: {g[-1]:.0e} of the last step's", 20, ACTION).next_to(end, UP + LEFT, buff=0.2)
        self.play(FadeIn(end), FadeIn(note))
        cap = Text("A lesson that must travel 79 steps back arrives as a few millionths of itself.",
                   font_size=22).to_edge(DOWN)
        self.play(FadeIn(cap))
        self.wait(1.4)
        self.play(*[FadeOut(m) for m in (title, ax, curve, xl, yl, end, note, cap)])

    # ── Beat 2: the cell ────────────────────────────────────────────────────
    def _beat_cell(self):
        title = Text("Long short-term memory: a cell that is added to, not rewritten", font_size=28).to_edge(UP)
        y_belt = 1.2
        belt = Arrow(np.array([-6.0, y_belt, 0]), np.array([6.0, y_belt, 0]), color=ACCENT,
                     stroke_width=8, buff=0, max_tip_length_to_length_ratio=0.03)
        c_in = _label("c<t-1>", 22, ACCENT).next_to(belt.get_start(), DOWN, buff=0.15)
        c_out = _label("c<t>", 22, ACCENT).next_to(belt.get_end(), DOWN, buff=0.15).shift(LEFT * 0.3)

        def node(sym, x, color):
            circ = Circle(radius=0.32, color=color, stroke_width=3, fill_color="#000000", fill_opacity=1)
            circ.move_to(np.array([x, y_belt, 0]))
            return VGroup(circ, _label(sym, 28, color).move_to(circ))

        times = node("×", -2.5, ACTION)
        plus = node("+", 0.5, DATA_C)
        box = RoundedRectangle(width=9.0, height=1.0, corner_radius=0.15, color=GREY, stroke_width=2)
        box.move_to(np.array([0.0, -1.5, 0]))
        inside = _label("z = x<t> Wx + h<t-1> Wh + b      i, f, o = σ(·)      g = tanh(·)", 20, WHITE).move_to(box)
        f_arrow = Arrow(np.array([-2.5, -1.0, 0]), times.get_bottom(), color=ACTION, buff=0.05, stroke_width=3)
        f_lab = _label("f: keep", 20, ACTION).next_to(f_arrow, LEFT, buff=0.1)
        ig_arrow = Arrow(np.array([0.5, -1.0, 0]), plus.get_bottom(), color=DATA_C, buff=0.05, stroke_width=3)
        ig_lab = _label("i·g: write", 20, DATA_C).next_to(ig_arrow, LEFT, buff=0.1)
        tap = Line(np.array([3.3, y_belt, 0]), np.array([3.3, -0.6, 0]), color=ACCENT, stroke_width=3)
        out = _label("h<t> = o · tanh(c<t>): show", 20, DATA_B).next_to(tap.get_end(), RIGHT, buff=0.15)
        self.play(FadeIn(title), Create(belt), FadeIn(c_in), FadeIn(c_out))
        self.play(FadeIn(times), FadeIn(plus))
        self.play(FadeIn(box), FadeIn(inside))
        self.play(Create(f_arrow), FadeIn(f_lab), Create(ig_arrow), FadeIn(ig_lab))
        self.play(Create(tap), FadeIn(out))
        eq = _label("c<t> = f · c<t-1> + i · g", 30, ACCENT).to_edge(DOWN).shift(UP * 0.3)
        self.play(FadeIn(eq))
        cap = Text("With i = 0 and f = 1 the cell is unchanged, whatever arrives.", font_size=22,
                   color=ACCENT).to_edge(DOWN).shift(DOWN * 0.15)
        self.play(eq.animate.shift(UP * 0.4), FadeIn(cap))
        self.wait(1.6)
        self.play(*[FadeOut(m) for m in (title, belt, c_in, c_out, times, plus, box, inside, f_arrow,
                                         f_lab, ig_arrow, ig_lab, tap, out, eq, cap)])

    # ── Beat 3: holding ─────────────────────────────────────────────────────
    def _beat_hold(self, a):
        title = Text("Four values written at step 0, then 200 distractors", font_size=30).to_edge(UP)
        traces = np.asarray(a["hold"]["lstm"], dtype=float)           # (4, 101), every other step
        steps = np.arange(traces.shape[1]) * 2
        vals = traces[:, 0]
        rnn = np.empty((4, 201))                                        # h = tanh(2 h): two attractors
        rnn[:, 0] = vals
        for t in range(1, 201):
            rnn[:, t] = np.tanh(2.0 * rnn[:, t - 1])
        axl = _axes([0, 200, 50], [-1, 1, 0.5], 5.2, 3.2, LEFT * 3.2 + DOWN * 0.4)
        axr = _axes([0, 200, 50], [-1, 1, 0.5], 5.2, 3.2, RIGHT * 3.2 + DOWN * 0.4)
        gl = VGroup(*[axl.plot_line_graph(steps.tolist(), traces[k].tolist(), line_color=SIGNAL_COLORS[k],
                                          add_vertex_dots=False, stroke_width=4) for k in range(4)])
        gr = VGroup(*[axr.plot_line_graph(list(range(201)), rnn[k].tolist(), line_color=SIGNAL_COLORS[k],
                                          add_vertex_dots=False, stroke_width=4) for k in range(4)])
        tl = _label("LSTM cell: input gate shut, forget gate open", 18, ACCENT).next_to(axl, UP, buff=0.15)
        tr = _label("RNN unit, h = tanh(2h + ...)", 18, DATA_A).next_to(axr, UP, buff=0.15)
        self.play(FadeIn(title), Create(axl), Create(axr), FadeIn(tl), FadeIn(tr))
        self.play(Create(gl), Create(gr), run_time=2.0)
        cap = Text("The cell keeps all four. The RNN keeps one bit: an attractor for A and B, one for C and D.",
                   font_size=21).to_edge(DOWN)
        self.play(FadeIn(cap))
        self.wait(1.8)
        self.play(*[FadeOut(m) for m in (title, axl, axr, gl, gr, tl, tr, cap)])

    # ── Beat 4: the carousel ────────────────────────────────────────────────
    def _beat_carousel(self, a):
        title = Text("Along the cell, one step back multiplies the gradient by f — nothing else",
                     font_size=26).to_edge(UP)
        g = a["grad"]
        lo = -6.0
        ax = _axes([0, 79, 10], [lo, 0.5, 1], 8.0, 3.8, LEFT * 1.0 + DOWN * 0.3)
        items = [("RNN (Xavier)", "x", DATA_A, "RNN"),
                 ("LSTM, forget bias 1", "c", DATA_B, "LSTM, forget bias 1 (f ≈ 0.73)"),
                 ("LSTM, gates open", "c", ACCENT, "LSTM, gates open (f ≈ 0.993)")]
        graphs, keys = VGroup(), VGroup()
        for name, key, col, lab in items:
            graphs.add(_log_curve(ax, g[name][key], col, lo))
            keys.add(_label(lab, 17, col))
        keys.arrange(DOWN, aligned_edge=LEFT).next_to(ax, RIGHT, buff=0.25).shift(UP * 0.9)
        xl = _label("steps back from the question", 16).next_to(ax, DOWN, buff=0.15)
        yl = _label("gradient, relative to the last step (log scale)", 16).next_to(ax, UP, buff=0.1).align_to(ax, LEFT)
        self.play(FadeIn(title), Create(ax), FadeIn(xl), FadeIn(yl))
        for gr, kk in zip(graphs, keys):
            self.play(Create(gr), FadeIn(kk), run_time=1.0)
        c79 = g["LSTM, gates open"]["c"][-1]
        cap = Text(f"0.993 to the 79th power is {c79:.2f}: the error rides the carousel all the way back.",
                   font_size=22, color=ACCENT).to_edge(DOWN)
        self.play(FadeIn(cap))
        self.wait(1.8)
        self.play(*[FadeOut(m) for m in (title, ax, graphs, keys, xl, yl, cap)])

    # ── Beat 5: the latch ───────────────────────────────────────────────────
    def _beat_latch(self, a):
        title = Text("The latch: a signal, T − 1 distractors, then the question", font_size=30).to_edge(UP)
        tape = _label("C  w z y x w w z y x z y w ...  →  C ?", 24, WHITE, MONO).next_to(title, DOWN, buff=0.3)
        lag = a["lag"]
        Ts = lag["T"]
        ax = _axes([0, len(Ts) - 1, 1], [0, 1.05, 0.25], 7.5, 3.2, DOWN * 0.8)
        chance = DashedLine(ax.c2p(0, 0.25), ax.c2p(len(Ts) - 1, 0.25), color=GREY, stroke_width=2)
        dots = VGroup()
        for key, col, dx in (("rnn", DATA_A, -0.08), ("lstm", ACCENT, 0.08)):
            for i, accs in enumerate(lag[key]):
                for v in accs:
                    dots.add(Dot(ax.c2p(i + dx, v), radius=0.07, color=col))
        ticks = VGroup(*[_label(f"T={T}", 16).next_to(ax.c2p(i, 0), DOWN, buff=0.15) for i, T in enumerate(Ts)])
        key = VGroup(_label("RNN", 20, DATA_A), _label("LSTM, gates open", 20, ACCENT),
                     _label("chance", 16)).arrange(DOWN, aligned_edge=LEFT).next_to(ax, RIGHT, buff=0.3)
        self.play(FadeIn(title), FadeIn(tape))
        self.play(Create(ax), Create(chance), FadeIn(ticks), FadeIn(key))
        self.play(FadeIn(dots, lag_ratio=0.03), run_time=2.0)
        cap = Text("Three seeds each: the RNN rarely gets past 20 steps; the LSTM holds the signal 320.",
                   font_size=22, color=ACCENT).to_edge(DOWN)
        self.play(FadeIn(cap))
        self.wait(1.8)
        self.play(*[FadeOut(m) for m in (title, tape, ax, chance, dots, ticks, key, cap)])

    # ── Beat 6: two biases ──────────────────────────────────────────────────
    def _beat_biases(self, a):
        title = Text("The forget bias sets how long the cell remembers: about 1 / (1 − σ(b))",
                     font_size=26).to_edge(UP)
        b = a["bias"]
        acc = np.asarray(b["acc"], dtype=float)
        size = 0.8
        grid = VGroup()
        for i, bb in enumerate(b["b"]):
            for j in range(len(b["T"])):
                v = float(acc[i, j])
                t = (v - 0.25) / 0.75
                sq = _grid_square(size, ACCENT, 0.08 + 0.85 * max(0.0, t))
                sq.move_to(np.array([-0.4 + j * size, 1.6 - i * size * 0.75, 0.0]))
                grid.add(VGroup(sq, _label(f"{v:.2f}", 14, WHITE).move_to(sq)))
        rows = VGroup()
        for i, bb in enumerate(b["b"]):
            mem = 1.0 / (1.0 - 1.0 / (1.0 + np.exp(-bb)))
            rows.add(_label(f"b_f = {bb}, b_i = {-bb}: memory {mem:,.0f} steps", 16).next_to(
                grid[i * len(b["T"])], LEFT, buff=0.25))
        cols = VGroup(*[_label(f"T = {T}", 16).next_to(grid[j], UP, buff=0.12) for j, T in enumerate(b["T"])])
        self.play(FadeIn(title), FadeIn(rows), FadeIn(cols))
        for i in range(len(b["b"])):
            self.play(FadeIn(VGroup(*grid[i * len(b["T"]):(i + 1) * len(b["T"])])), run_time=0.4)
        cap = Text("Each row learns lags a few times its memory. The input bias keeps the noise out while it waits.",
                   font_size=21, color=ACCENT).to_edge(DOWN)
        self.play(FadeIn(cap))
        self.wait(1.8)
        self.play(*[FadeOut(m) for m in (title, grid, rows, cols, cap)])

    # ── Beat 7: fade, never explode ─────────────────────────────────────────
    def _beat_explode(self, a):
        title = Text("The full gradient spikes; the carousel alone cannot explode", font_size=30).to_edge(UP)
        n = a["norms"]
        lo, hi = -3.5, 2.0
        ax = _axes([0, len(n["lstm"]) - 1, 25], [lo, hi, 1], 8.5, 3.6, DOWN * 0.3)
        clip = DashedLine(ax.c2p(0, 0), ax.c2p(len(n["lstm"]) - 1, 0), color=GREY, stroke_width=2)
        clip_lab = _label("clip at 1", 16).next_to(clip, RIGHT, buff=0.1)
        g_full = _log_curve(ax, n["lstm"], ACTION, lo, width=2)
        g_cell = _log_curve(ax, n["carousel"], ACCENT, lo, width=2)
        key = VGroup(_label("full BPTT", 18, ACTION), _label("cell only (1997)", 18, ACCENT)).arrange(
            DOWN, aligned_edge=LEFT).next_to(ax, UP, buff=0.1).align_to(ax, RIGHT)
        xl = _label("update", 16).next_to(ax, DOWN, buff=0.15)
        yl = _label("gradient norm (log scale)", 16).next_to(ax, UP, buff=0.1).align_to(ax, LEFT)
        self.play(FadeIn(title), Create(ax), Create(clip), FadeIn(clip_lab), FadeIn(xl), FadeIn(yl), FadeIn(key))
        self.play(Create(g_full), run_time=1.5)
        self.play(Create(g_cell), run_time=1.5)
        cap = Text("Along c the gradient is multiplied by forget gates between 0 and 1: it can fade, never grow.",
                   font_size=21).to_edge(DOWN)
        self.play(FadeIn(cap))
        self.wait(1.8)
        self.play(*[FadeOut(m) for m in (title, ax, clip, clip_lab, g_full, g_cell, key, xl, yl, cap)])

    # ── Beat 8: inside the trained latch ────────────────────────────────────
    def _beat_fan(self, a):
        f = a["fan"]
        title = Text("Inside the trained latch: a whisper, kept, then amplified", font_size=30).to_edge(UP)
        C = np.asarray(f["c"], dtype=float)
        span = float(np.abs(C).max()) * 1.05
        ax = _axes([0, C.shape[1] - 1, 10], [-span, span, 1], 8.0, 4.0, DOWN * 0.3)
        lines = VGroup(*[ax.plot_line_graph(list(range(C.shape[1])), row.tolist(),
                                            line_color=SIGNAL_COLORS[s], add_vertex_dots=False, stroke_width=2)
                         for row, s in zip(C, f["signal"])])
        key = VGroup(*[_label(f"signal {ch}", 18, col) for ch, col in zip("ABCD", SIGNAL_COLORS)]).arrange(
            DOWN, aligned_edge=LEFT).next_to(ax, RIGHT, buff=0.25)
        xl = _label("step", 16).next_to(ax, DOWN, buff=0.15)
        yl = _label(f"cell {f['cell']}", 16).next_to(ax, UP, buff=0.1).align_to(ax, LEFT)
        self.play(FadeIn(title), Create(ax), FadeIn(key), FadeIn(xl), FadeIn(yl))
        self.play(Create(lines), run_time=2.5)
        fmean = float(np.mean(f["f"][1:]))
        cap = Text(f"The input gate barely opens at the signal; the forget gate stays near {fmean:.2f}; "
                   "the mark grows step by step.", font_size=20).to_edge(DOWN)
        self.play(FadeIn(cap))
        self.wait(1.8)
        self.play(*[FadeOut(m) for m in (title, ax, lines, key, xl, yl, cap)])

    # ── Beat 9: Shakespeare ─────────────────────────────────────────────────
    def _beat_shakespeare(self, a):
        ch = a["char"]
        title = Text("Tiny Shakespeare, about 99,000 parameters each", font_size=32).to_edge(UP)
        rv, lv = np.asarray(ch["rnn_val"], dtype=float), np.asarray(ch["lstm_val"], dtype=float)
        lo, hi = 1.55, 2.1
        ax = _axes([1, len(lv), 5], [lo, hi, 0.1], 6.5, 3.4, LEFT * 1.4 + DOWN * 0.3)
        gr = ax.plot_line_graph(list(range(1, len(rv) + 1)), np.clip(rv, lo, hi).tolist(), line_color=DATA_A,
                                add_vertex_dots=True, stroke_width=3, vertex_dot_radius=0.04)
        gl = ax.plot_line_graph(list(range(1, len(lv) + 1)), np.clip(lv, lo, hi).tolist(), line_color=ACCENT,
                                add_vertex_dots=True, stroke_width=3, vertex_dot_radius=0.04)
        key = VGroup(_label(f"RNN, best rate: {rv[-1]:.3f}", 20, DATA_A),
                     _label(f"LSTM, best rate: {lv[-1]:.3f}", 20, ACCENT)).arrange(
            DOWN, aligned_edge=LEFT).next_to(ax, RIGHT, buff=0.3)
        yl = _label("validation loss (nats per character)", 16).next_to(ax, UP, buff=0.1).align_to(ax, LEFT)
        xl = _label("epoch", 16).next_to(ax, DOWN, buff=0.15)
        self.play(FadeIn(title), Create(ax), FadeIn(yl), FadeIn(xl))
        self.play(Create(gr), Create(gl), FadeIn(key), run_time=2.0)
        self.wait(1.4)
        self.play(*[FadeOut(m) for m in (ax, gr, gl, key, yl, xl)])

        # How far back each model looks.
        cx = ch["context"]
        L = cx["L"]
        lo2, hi2 = 1.55, 2.0
        ax2 = _axes([0, len(L) - 1, 1], [lo2, hi2, 0.1], 7.5, 3.2, LEFT * 0.8 + DOWN * 0.4)
        lines = VGroup()
        for k, col in (("RNN", DATA_A), ("LSTM", ACCENT)):
            ys = np.clip(np.asarray(cx[k], dtype=float), lo2, hi2)
            lines.add(ax2.plot_line_graph(list(range(len(L))), ys.tolist(), line_color=col,
                                          add_vertex_dots=True, stroke_width=3, vertex_dot_radius=0.05))
        ticks = VGroup(*[_label(str(x), 14).next_to(ax2.c2p(i, lo2), DOWN, buff=0.12) for i, x in enumerate(L)])
        head = _label("loss on the next character, given L characters of context", 18).next_to(ax2, UP, buff=0.15)
        key2 = VGroup(_label("RNN: no gain after 16", 18, DATA_A), _label("LSTM: gains to 64", 18, ACCENT)).arrange(
            DOWN, aligned_edge=LEFT).next_to(ax2, RIGHT, buff=0.3)
        self.play(Create(ax2), FadeIn(ticks), FadeIn(head))
        self.play(Create(lines), FadeIn(key2), run_time=1.6)
        self.wait(1.6)
        self.play(*[FadeOut(m) for m in (ax2, lines, ticks, head, key2)])

        # A cell that counts along the line: the passage's first four lines of verse.
        full = ch["cell_text"]
        cut = [k for k, cc in enumerate(full) if cc == "\n"][4] + 1       # through the fifth newline
        text = full[:cut]
        c = np.asarray(ch["cell_c"][:cut], dtype=float)
        lo3, hi3 = float(c.min()) - 0.2, float(c.max()) + 0.2
        ax3 = _axes([0, len(text) - 1, 20], [lo3, hi3, 1], 12.0, 2.8, UP * 0.4)
        trace = ax3.plot_line_graph(list(range(len(text))), c.tolist(), line_color=ACCENT,
                                    add_vertex_dots=False, stroke_width=3)
        marks = VGroup(*[DashedLine(ax3.c2p(k, lo3), ax3.c2p(k, hi3), color=GREY, stroke_width=1.5)
                         for k, cc in enumerate(text) if cc == "\n"])
        lines_txt = [ln for ln in text.split("\n") if ln]
        body = VGroup(*[_label(ln, 18, WHITE, MONO) for ln in lines_txt]).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        body.next_to(ax3, DOWN, buff=0.35)
        head3 = _label(f"cell {ch['cell']} while reading (dashed: newlines)", 18, ACCENT).next_to(ax3, UP, buff=0.1)
        self.play(Create(ax3), FadeIn(head3), FadeIn(body))
        self.play(Create(marks), Create(trace), run_time=2.5)
        r2 = ch["r2"]
        cap = Text(f"A counter that every newline resets. Position in the line, read from the cells: R² {r2['lstm']:.2f}; "
                   f"from the RNN: {r2['rnn']:.2f}.", font_size=19, color=ACCENT).to_edge(DOWN)
        self.play(FadeIn(cap))
        self.wait(2.2)
        self.play(*[FadeOut(m) for m in (title, ax3, head3, body, marks, trace, cap)])

    # ── Beat 10: next ───────────────────────────────────────────────────────
    def _beat_next(self):
        t1 = Text("Gates let the gradient travel — if the cell starts out remembering.", font_size=28)
        t2 = Text("Next: both models fit their training text far better than new text.", font_size=28, color=ACCENT)
        t3 = Text("Dropout for recurrent networks — without erasing the memory.", font_size=28, color=ACCENT)
        grp = VGroup(t1, t2, t3).arrange(DOWN, buff=0.4)
        self.play(FadeIn(t1))
        self.wait(0.8)
        self.play(FadeIn(t2), FadeIn(t3))
        self.wait(2.2)
        self.play(FadeOut(grp))

