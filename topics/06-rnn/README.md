# Module 06 — RNN (recurrence · backpropagation through time · truncation · gradient clipping)

Every network so far read a fixed-size input. A recurrent network reads a sequence one step at a
time, carrying a hidden state forward and applying the same weights at every step:
$h^{\langle t\rangle} = \tanh(x^{\langle t\rangle} W_{xh} + h^{\langle t-1\rangle} W_{hh} + b_h)$ (Elman 1990), trained by
backpropagation through time (Werbos 1990). nanograd gains the **RNN layer** — forward over a
whole sequence, BPTT with optional **truncation** — and **gradient-norm clipping**, in both
languages. Two toys show what a recurrent network can and cannot learn: an *echo* it learns only as
far back as its training gradient reaches (the gradient through time is a power of $W_{hh}$), and a
*counter* it learns even with Elman's one-step training, as a state machine that runs for 10,000
steps. On Tiny Shakespeare a 99,137-parameter character model reaches **1.654 nats/char**, below
every n-gram baseline, and rediscovers Elman's word boundaries. C and Python agree to the last
printed digit.

> Blog post: [`site/src/content/posts/06-rnn.mdx`](../../site/src/content/posts/06-rnn.mdx)
> Interactive widget: [`site/src/components/viz/EchoWidget.astro`](../../site/src/components/viz/EchoWidget.astro)
> Shared library: [`lib/python/nanograd`](../../lib/python/nanograd) · [`lib/c/nanograd`](../../lib/c/nanograd)
> Dataset: [`data/get_shakespeare.py`](../../data/get_shakespeare.py) (new; downloads Tiny Shakespeare once)
> Concept & references: [`docs/CURRICULUM.md`](../../docs/CURRICULUM.md), [`docs/references/papers.md`](../../docs/references/papers.md)

## Contents

| Path | What |
|------|------|
| `../../lib/python/nanograd/rnn.py` | **New:** `RNN` — time-major `(T, m, n)` forward with an optional initial state `h0`; `backward(dH, truncate=k)` is backpropagation through time (`k = 1`: Elman's training, `0`: full); leaves `dh0` and every step's `dZ` for inspection |
| `../../lib/python/nanograd/optim.py` | **New:** `clip_grad_norm(grads, max_norm)` — Pascanu et al.'s global-norm clipping, in place; returns the norm before clipping |
| `../../lib/c/nanograd/rnn.c`, `optim.c` | The C mirror — `ng_rnn_forward`, `ng_rnn_backward` (loop orders documented: the Python mirror depends on them), `ng_clip_grad_norm` |
| `../../data/get_shakespeare.py` | **New:** Tiny Shakespeare (1,115,394 characters, 65 symbols), SHA-256 pinned; 90/10 contiguous train/validation split |
| `python/rnn.py` | Canonical: the C-order `ToyRNN` (both twins, with the C library's `tanh`/`exp`/`log`), the vectorized echo and counting toys and their sweeps, the gradient-through-time probe, the fingerprints, `CharRNN` and `run_char` (checkpoint cache, target ≤ 1.70 nats/char), n-gram baselines, sampling, word positions for the surprisal experiment, the animation export |
| `c/rnn.c` | The C toy: the echo network trained with full BPTT and with Elman's truncation from the same weights, plus RNN and clipping fingerprints |
| `notebook.ipynb` | Cell-by-cell: the recurrence on "hello", BPTT by hand (gradient-checked), the echo twins, the truncation staircase and the delay limit, the gradient through time against the spectral radius, Pascanu's one-unit cliff and clipping, the mod-3 counter as a state machine, Tiny Shakespeare (n-grams, the RNN, samples by epoch and temperature, Elman's word boundaries, truncation on text), n-gram table sizes, framework mirrors |
| `assignment.ipynb` | Hands-on: you implement **one step of the recurrence**, the **forward pass over a sequence**, **backpropagation through time** with truncation, and **gradient-norm clipping**; your pieces must make the echo network remember four steps back with full BPTT and fail with Elman's truncation |
| `tests/check_utils.py` | Self-check harness (`rel_error`, `eval_numerical_gradient*`, `compare_to_canonical`) |
| `tests/test_agreement.py` | Compiles + runs the C, runs the Python, asserts agreement (both twins, the RNN layer, clipping) — **and** gradient-checks the RNN layer, checks truncation against chunked BPTT and clipping against its definition, and checks the two Python paths |
| `anim/scene.py` | Manim scene: sequences, the cell unrolled, the gradient shrinking along the chain and its profile at three spectral radii, the truncation staircase, the counter's state machine, Shakespeare's loss and samples, Elman's word boundaries |
| `anim/rnn_assets.json` | Sweeps, gradient profiles, samples and surprisal exported by the notebook (the scene's data) |

## Run it

```bash
# C  (standalone; Windows: omit -lm, math is in the CRT)
clang -O2 -std=c11 -Wall -Wextra -I lib/c/nanograd \
  topics/06-rnn/c/rnn.c \
  lib/c/nanograd/rng.c lib/c/nanograd/nn.c lib/c/nanograd/init.c \
  lib/c/nanograd/optim.c lib/c/nanograd/rnn.c -o rnn && ./rnn

# C  (via CMake)
cmake -B build && cmake --build build && ./build/topics/06-rnn/c/rnn

# Python mirror: both echo twins and the fingerprints, then the Tiny Shakespeare model with
# the <= 1.70 nats/char assert. The character model is ~10 minutes of NumPy the first time; it is
# cached to data/shakespeare/checkpoints/ (gitignored) and shared with the notebook.
python topics/06-rnn/python/rnn.py

# C <-> Python agreement + finite-difference gradient checks (~1 min)
python topics/06-rnn/tests/test_agreement.py

# Notebook: open interactively, or run headless to confirm it executes clean. With the
# checkpoint cache it runs in about three minutes (the toy sweeps and the framework mirrors are
# not cached; without the cache, add the character model's ten minutes and the four 3-epoch
# truncation runs' two).
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 \
  topics/06-rnn/notebook.ipynb

# Render the animation (needs manim + ffmpeg on PATH; no LaTeX required)
python -m manim -qm --media_dir topics/06-rnn/anim/media \
  topics/06-rnn/anim/scene.py RNNStory
```

The rendered video is copied to `site/public/media/06-rnn.mp4` and embedded in the post.

## What you should see

- **toy (echo, delay 4):** C and Python print identical fingerprints for both twins: full BPTT
  reaches **100%** on 96-step test sequences (test loss 0.0066), Elman's one-step truncation **43.0%**
  (chance 25%), from the same weights on the same batches; the clip never fires (`n_clip=0`). With
  the gradient cut every k steps, every delay d < k is learned perfectly and every other stays at
  chance (the staircase); with full BPTT the 16-unit net echoes 100% at d = 8, 99.6% at 12, 84% at
  16, 59% at 20 and nothing from 24 on. At initialization the gradient 10 steps back is 1.2×10⁻⁴ of
  the last step's at gain 0.5, 0.065 at Xavier's scale (spectral radius 1.03), and 39 steps back it
  is 13,700× at gain 3. Pascanu's one-unit cliff: unclipped descent takes one step 27 long and sticks
  at loss 0.125; clipped at 1 it reaches 1.3×10⁻⁹.
- **toy (counting mod 3):** trained on 20-bit streams, 100% on 1,000 and 10,000 bits — with full BPTT
  and with Elman's truncation alike; the hidden states form about a dozen clusters, each tagged with
  one count — the count and the last two bits read explain over 95% of the state's variance.
- **tiny shakespeare:** RNN(65 → 256) + Linear, 99,137 parameters, 20 epochs: validation **1.654
  nats/char** (2.39 bits; target ≤ 1.70) against the best n-gram's 1.750 (5-gram; trigram 2.046,
  bigram 2.482). Samples go from letter soup (epoch 1) to speaker names and verse lines (epoch 3) to
  mostly real words (epoch 20). Surprisal is 2.96 nats on a word's first letter and 1.39–1.59 on the
  rest. Truncating the gradient at 1, 4, 16 or 64 characters moves the 3-epoch loss by only 0.05.
  PyTorch matches the layer, truncation (`.detach()`), clipping and the char model's gradients to
  ≤ 1e-12; TensorFlow's `SimpleRNN` echoes at 100%; a Keras `SimpleRNN` char model reaches 1.830 in
  four epochs.
- The agreement test prints `OK`: all four FINAL lines identical across C and Python; the RNN layer's
  gradients within `1e-6` of finite differences; truncation equal to chunked BPTT and clipping equal
  to its definition to `~1e-15`; the two Python paths equal to `~1e-14`.

## Design notes

**Full BPTT vs truncated, nothing else.** The C toy's twins share weights, batches and forward
arithmetic; one argument of `ng_rnn_backward` differs. Every truncation result in the module is the
same comparison at other lengths — and the test suite checks that `truncate=k` really is chunked
training (the sum of the chunks' full gradients, each from the state the previous chunk left).

**The C library's transcendentals.** Bit-exact agreement needs every sum in the C's order — handled,
as in Module 05, by vectorizing elementwise work and running sums through `np.add.accumulate` — but
a recurrent network also runs `tanh` hundreds of times per sequence. NumPy's own `tanh` differs from
the C library's in the last bit for about 22% of inputs; `ToyRNN` calls `math.tanh` (and `math.exp`,
`math.log`), which is the C library's.

**Fresh data, longer tests.** Every echo update draws a new batch, so nothing can be memorized, and
the test sequences are four times longer than the training ones: only a network that learned the
rule — the same weights doing the same thing at every step — passes.

**Scaled down, honestly.** The character model is a vanilla tanh RNN with 256 hidden units (99,137
parameters), trained for 20 epochs with Adam and 64-step truncated BPTT through 64 parallel streams,
in float32 — about ten minutes of NumPy. The n-gram baselines' smoothing constant is tuned on the
validation text, which flatters them; the RNN beats them anyway.
