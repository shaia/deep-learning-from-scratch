# Module 07 — LSTM (gates · the cell · the constant error carousel · gate biases)

Module 06's network rewrites its whole state at every step, so the gradient through time is a power of
one matrix and fades. Hochreiter and Schmidhuber's long short-term memory (1997) keeps a second state,
the cell, that is *added to* under gates instead of rewritten:
$c^{\langle t\rangle} = f \odot c^{\langle t-1\rangle} + i \odot g$, $h^{\langle t\rangle} = o \odot \tanh(c^{\langle t\rangle})$.
Along the cell one step back multiplies the gradient by the forget gate $f$ and by nothing else — the
"constant error carousel". nanograd gains the **LSTM layer** — forward with an initial state and cell,
backpropagation through time along both paths, chunked truncation, and Hochreiter & Schmidhuber's own
**cell-only** gradient — in both languages. A toy that measures nothing but memory, the *latch* (a
signal, $T-1$ distractors, then the question), shows an LSTM holding a symbol across 320 steps where
Module 06's RNN rarely gets past 20 — **if** its cells start out open to the past and shut to new input
(forget bias +5, input bias −5; with the textbook forget bias of 1 it is no better than the RNN), and
shows the cell-only gradient learning as fast as the full one while never exploding. On Tiny
Shakespeare, at the same parameter budget and each at its best learning rate, the LSTM reaches
**1.620 nats/char** against the RNN's 1.644, uses 64 characters of context where the RNN stops at 16,
and grows a cell that counts along every line of verse. C and Python agree to the last printed digit.

> Blog post: [`site/src/content/posts/07-lstm.mdx`](../../site/src/content/posts/07-lstm.mdx)
> Interactive widget: [`site/src/components/viz/LatchWidget.astro`](../../site/src/components/viz/LatchWidget.astro)
> Shared library: [`lib/python/nanograd`](../../lib/python/nanograd) · [`lib/c/nanograd`](../../lib/c/nanograd)
> Dataset: [`data/get_shakespeare.py`](../../data/get_shakespeare.py) (from Module 06)
> Concept & references: [`docs/CURRICULUM.md`](../../docs/CURRICULUM.md), [`docs/references/papers.md`](../../docs/references/papers.md)

## Contents

| Path | What |
|------|------|
| `../../lib/python/nanograd/lstm.py` | **New:** `LSTM` — time-major `(T, m, n)` forward with optional `h0`, `c0`; gate blocks `[i \| f \| g \| o]` (PyTorch's order); `backward(dH, truncate=k, cell_only=False)` is backpropagation through time along both paths (`cell_only=True`: the 1997 gradient, error only along the cell); leaves `dh0`, `dc0`, every step's gate gradient `dZ` and the cell's gradient through time `dC`; `gate_bias(i=, f=, g=, o=)` |
| `../../lib/c/nanograd/lstm.c` | The C mirror — `ng_lstm_forward`, `ng_lstm_backward` (loop orders documented: the Python mirror depends on them) |
| `python/lstm.py` | Canonical: the C-order `ToyLatch` (the four twins, Module 06's C-order RNN imported for the first), the vectorized latch (`train_latch`, a process-pool `sweep`, `grad_through_time`, `gate_trace`), the LSTM fingerprint, `CharLSTM` and `run_char` (checkpoint cache, target ≤ 1.64 nats/char), the learning-rate bracket (`char_sweep`, `LR_GRID`, `RNN_LR`), `context_curve`, `line_position`, the animation export |
| `c/lstm.c` | The C toy: the latch at T = 80 — Module 06's RNN, an LSTM with open gates, the same LSTM with the cell-only gradient, the same weights with the textbook biases — plus the LSTM fingerprint |
| `notebook.ipynb` | Cell-by-cell: the latch and Module 06's wall; a cell that holds four values while an RNN unit merges them; one LSTM step and BPTT through the cell by hand (gradient-checked); the four twins; the lag sweep; the gradient through time; the gate biases (memory length, internal state drift, the bias tables); the carousel alone over eight seeds; inside the trained latch; Tiny Shakespeare (the learning-rate bracket, samples, context curves, a cell that counts); framework mirrors |
| `assignment.ipynb` | Hands-on: you implement **one step of the cell**, the **forward pass over a sequence**, **backpropagation through time along both paths** (with the 1997 cell-only gradient), and the **gate biases**; your pieces must hold the latch's signal through 79 distractors with the gates open, fail with the textbook biases, and succeed with the carousel alone |
| `tests/check_utils.py` | Self-check harness (`rel_error`, `eval_numerical_gradient*`, `compare_to_canonical`) |
| `tests/test_agreement.py` | Compiles + runs the C, runs the Python, asserts agreement (all four twins, the LSTM layer) — **and** gradient-checks the LSTM, checks truncation against chunked BPTT and the cell-only gradient against a frozen-h network, checks small latch and character networks end to end, and checks the two Python paths |
| `anim/scene.py` | Manim scene: Module 06's fading gradient, the cell as a belt with gates, a cell holding four values, the gradient along the carousel, the latch against the lag, the gate biases, full vs cell-only gradient norms, the trained cell's fan-out, Shakespeare (losses, context, the counting cell) |
| `anim/lstm_assets.json` | Profiles, sweeps, traces and Shakespeare measurements exported by the notebook (the scene's data) |

## Run it

```bash
# C  (standalone; Windows: omit -lm, math is in the CRT)
clang -O2 -std=c11 -Wall -Wextra -I lib/c/nanograd \
  topics/07-lstm/c/lstm.c \
  lib/c/nanograd/rng.c lib/c/nanograd/nn.c lib/c/nanograd/init.c \
  lib/c/nanograd/optim.c lib/c/nanograd/rnn.c lib/c/nanograd/lstm.c -o lstm && ./lstm

# C  (via CMake)
cmake -B build && cmake --build build && ./build/topics/07-lstm/c/lstm

# Python mirror: the four twins and the fingerprint, then the Tiny Shakespeare LSTM with the
# <= 1.64 nats/char assert. The character model is ~10 minutes of NumPy the first time (and Module
# 06's RNN at its best rate another ~6); both are cached to data/shakespeare/checkpoints/ (gitignored).
python topics/07-lstm/python/lstm.py

# C <-> Python agreement + finite-difference gradient checks (~2 min)
python topics/07-lstm/tests/test_agreement.py

# Notebook: open interactively, or run headless to confirm it executes clean. With the checkpoint
# cache it runs in about eight minutes (the latch sweeps run in a pool of worker processes and are
# not cached). From a cold cache, add the learning-rate bracket — eleven 20-epoch character models,
# trained in parallel, about half an hour.
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 \
  topics/07-lstm/notebook.ipynb

# Render the animation (needs manim + ffmpeg on PATH; no LaTeX required)
python -m manim -qm --media_dir topics/07-lstm/anim/media \
  topics/07-lstm/anim/scene.py LSTMStory
```

The rendered video is copied to `site/public/media/07-lstm.mp4` and embedded in the post.

## What you should see

- **toy (the latch, T = 80):** C and Python print identical fingerprints for all four twins. Module
  06's RNN ends at chance (25% on fresh 80-step sequences); the LSTM with the textbook biases (forget 1,
  input 0) at 27%; the LSTM with open gates (forget +5, input −5) at **100%** (77% on sequences four
  times longer), clipped on 45 of 300 updates; the same weights trained with the cell-only gradient at
  **100%** at both lengths, clipped twice. Across lags (three seeds): the RNN solves 10 every time, 20
  two times in three, and from 40 on one run in twelve; the open LSTM solves all eighteen runs up to
  T = 320, in 100–360 updates. At initialization the gradient reaching the signal 79 steps back is
  3.4×10⁻⁶ of the last step's for the RNN, 8.6×10⁻⁵ for the textbook LSTM, 0.11 for the open one (0.59
  along its cell, σ(5)⁷⁹). Each forget bias learns lags up to a few times its initial memory 1 + e^b_f;
  at T = 320 the input bias must be −3 or below. Over eight seeds the cell-only gradient solves T = 80
  as fast as the full one (mean update 119 for both), holds 100% at 4T (full: 97%), and is clipped
  0–7 times per run against 11–27, its largest norm 1.5 against 46.
- **tiny shakespeare:** LSTM(65 → 122) + Linear, 99,739 parameters, against Module 06's RNN(65 → 256),
  99,137; both trained at learning rates 2, 4, 8, 16 and 32 ×10⁻³. Each at its best: LSTM **1.620
  nats/char** (2.34 bits; lr 1.6×10⁻²) against the RNN's 1.644 (lr 4×10⁻³; Module 06's 2×10⁻³ gave
  1.654). At the RNN's original rate the LSTM loses (1.724). A 256-cell LSTM reaches 1.582. Given L
  characters of context from a cold start, the RNN stops improving at L = 16 (1.631), the LSTM at L = 64
  (1.603). Cell 11 tracks the position in the line (correlation +0.79; all cells, linear R² 0.81 against
  the RNN's 0.36), a sawtooth reset at every newline. PyTorch's `nn.LSTM` matches the layer, `.detach()`
  matches truncation and the cell-only gradient, and the char model's gradients match, all to ≤ 1e-12;
  TensorFlow's `LSTM` on the latch ends at 25% with Keras's default biases and 100% with the open ones;
  a Keras character LSTM at the same budget reaches 1.732 in four epochs.
- The agreement test prints `OK`: all five FINAL lines identical across C and Python; the LSTM's
  gradients within `1e-6` of finite differences (~1e-9 in practice); truncation equal to chunked BPTT to
  ~4e-16; the cell-only gradient equal to the frozen-h network's to ~1e-8; the two Python paths equal to
  ~1e-12.

## Design notes

**The latch, not the echo.** Module 06's echo needed the state to hold d symbols at once, so its limit
mixed gradient reach with capacity. The latch holds one symbol (two bits) whatever the lag, and its
only error signal is at the last step, so what it measures is how far back the gradient can teach.
Signals and distractors come from separate alphabets, as in the 1997 paper's task 2c, so the network
need not first learn to recognize the start of a sequence.

**Same weights, one difference.** The three LSTM twins draw the same Wx, Wh and Why; `forget1` differs
from `lstm` in one line of bias initialization, `carousel` in one argument of `ng_lstm_backward`. The
RNN twin is Module 06's code, run on the same batches.

**The gate biases are the module's second subject.** The first version of the toy used the textbook
forget bias of 1 and found the LSTM *worse* than the RNN at long lags. Measuring why led back to the
1997 paper: a cell that cannot forget (its f was 1) and input gates biased toward zero (§4.8). The
notebook's bias tables are those two prescriptions, measured.

**A fair learning rate.** The first character LSTM, trained at Module 06's learning rate, lost to the
RNN (1.724 against 1.654). Both architectures are therefore trained at the same five rates and compared
at their best; every run is cached, and `char_sweep` retrains any that are missing, in parallel.

**The C library's transcendentals.** As in Module 06, the C-order mirror evaluates `exp` and `tanh`
through Python's `math` (the C library's own); the sigmoid uses the C's two-branch formula,
1/(1 + e^−z) or e^z/(1 + e^z), written with e^−|z|.

**Scaled down, honestly.** The character models are single-layer, 64-step truncated BPTT through 64
parallel streams, float32, 20 epochs — about ten minutes of NumPy each. The context curves are
measured from a cold start, which handicaps the LSTM at short contexts.
