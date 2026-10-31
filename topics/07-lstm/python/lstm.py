"""
Module 07 - LSTM: gates that protect the gradient.

The Python mirror of topics/07-lstm/c/lstm.c. Module 06 ended on a wall: the
gradient through a recurrent network is a power of one matrix, so it fades
geometrically as it travels back in time, and a network can only learn to
remember as far back as that gradient reaches. Hochreiter and Schmidhuber's
long short-term memory (1997) adds a second state, the cell, that is not
rewritten at every step but added to, under gates:

    i, f, o = sigma(.), g = tanh(.)   of z<t> = x<t> Wx + h<t-1> Wh + b
    c<t> = f * c<t-1> + i * g          h<t> = o * tanh(c<t>)

Along the cell the gradient is multiplied by f at each step and by nothing
else, and f is a number the network sets for itself. nanograd gains the LSTM
layer (forward, backpropagation through time along both paths, chunked
truncation, and Hochreiter & Schmidhuber's cell-only gradient), in both
languages.

Five kinds of code live here:

  1. run_toy() -- the *agreement* path, on "the latch": a signal symbol (one of
     A B C D) at the first step, then T - 1 distractors (w x y z); at the last
     step the network must say which signal it saw. Holding one symbol needs
     almost no capacity, so the task measures one thing -- how far back the
     gradient can teach. Four networks train on the SAME batches: Module 06's
     RNN; an LSTM whose cells start open to the past and shut to new input
     (forget bias +5, input bias -5); the same LSTM, same weights, trained with
     the 1997 cell-only gradient; and the same weights with the textbook biases
     (forget 1, input 0). Every sum runs in c/lstm.c's loop order with the C
     library's exp and tanh, so C and Python agree to the last digit (1e-9
     gate). The RNN twin is Module 06's own C-order code, imported.

  2. train_latch() and friends -- the same networks on nanograd's vectorized
     layers, for the notebook's sweeps: how long a lag each network learns,
     what the gate biases do to that, the gradient through time, and the gates
     caught in the act.

  3. The fingerprint -- nanograd's LSTM forward/backward (initial state and
     cell, input gradients, truncation, the cell-only gradient) on fixed
     LCG-drawn tensors, so the C library's version is pinned too.

  4. The real thing: a character-level LSTM of Tiny Shakespeare with as many
     parameters as Module 06's RNN (CharLSTM, run_char with a checkpoint cache,
     target val loss <= CHAR_TARGET), and how much each model gains from
     context it has read further back.

  5. export_assets() -- the data the Manim scene draws.

Run directly to see the toy, then the character model:
    python topics/07-lstm/python/lstm.py

Reference: docs/references/papers.md (Module 07) -- Hochreiter & Schmidhuber
(1997), "Long Short-Term Memory"; Gers, Schmidhuber & Cummins (2000),
"Learning to Forget: Continual Prediction with LSTM".
"""

import hashlib
import json
import math
import os
import sys
import time

import numpy as np

# Make the shared nanograd library and Module 06's canonical code importable.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(_REPO, "lib", "python"))
sys.path.insert(0, os.path.join(_REPO, "topics", "06-rnn", "python"))

import rnn as m06  # noqa: E402  (Module 06: its C-order RNN, its char model, n-grams)
from nanograd import (  # noqa: E402
    LSTM,
    RNN,
    Adam,
    Linear,
    SoftmaxCrossEntropy,
    clip_grad_norm,
    xavier_normal,
)
from nanograd.rng import Rng  # noqa: E402


# ============================================================================
# 1. The bit-exact toy: "the latch".
#
# Input: one symbol per step, one-hot over V = 8: at step 0 a SIGNAL, uniform
# over A B C D; at steps 1..T-1 DISTRACTORS, uniform over w x y z. Target, at
# the last step only: which signal came first (4-way). A network that solves it
# has to write the signal into its state at step 0, keep it through T - 1
# steps of noise, and read it out at the end -- Bengio, Simard & Frasconi's
# (1994) "information latching", and in miniature task 2c of Hochreiter &
# Schmidhuber (1997). The only error signal is at the end, so the lesson that
# says "keep that symbol" must travel T - 1 steps back to the moment of
# keeping. Two test sets: fresh sequences of the training length, and
# sequences four times longer -- which only a cell that truly latched passes.
#
# Architecture: recurrent layer (16 units) -> Linear(16 -> 4) on the LAST
# state -> softmax cross-entropy. Xavier weights, Adam, clipping at CLIP.
#
# The four twins:
#   rnn       Module 06's RNN                         (a different weight shape)
#   lstm      LSTM, forget bias +5, input bias -5     full BPTT
#   carousel  the SAME LSTM weights                   cell-only gradient (1997)
#   forget1   the SAME Wx, Wh; forget bias 1, input 0 full BPTT
# ============================================================================

TOY_SEED = 1          # weight draws
DATA_SEED = 2         # the training batches (one fresh batch per update)
TEST_SEED = 3         # the fixed test sequences
SIGNALS, DISTRACTORS = "ABCD", "wxyz"
NS, ND = len(SIGNALS), len(DISTRACTORS)
V = NS + ND           # input alphabet: signals first, then distractors
N_H = 16              # hidden units / cells
T_TOY, M_TOY = 80, 32          # steps per training sequence, sequences per batch
T_TEST, M_TEST = 320, 64       # test: 64 sequences of 80 steps, and 64 four times longer
TOY_STEPS = 300
TOY_LR = 0.01
CLIP = 1.0
ADAM_B1, ADAM_B2, ADAM_EPS = 0.9, 0.999, 1e-8
OPEN_BIAS = {"i": -5.0, "f": 5.0}      # shut to new input, open to the past
TEXTBOOK_BIAS = {"i": 0.0, "f": 1.0}   # the customary forget bias of 1
TWINS = ("rnn", "lstm", "carousel", "forget1")


def draw_latch(rng, m, T):
    """(T, m) int64 symbols in the exact draw order c/lstm.c replays: sequence
    by sequence, each left to right; step 0 is a signal (0..NS-1), every later
    step a distractor (NS..V-1). The target is S[0]."""
    S = np.empty((T, m), dtype=np.int64)
    for i in range(m):
        S[0, i] = int(NS * rng.uniform())
        for t in range(1, T):
            S[t, i] = NS + int(ND * rng.uniform())
    return S


def one_hot(S, vocab=V):
    """(T, m) symbols -> (T, m, vocab) float64 one-hot vectors."""
    return np.eye(vocab)[S]


# -- the C-order building blocks --------------------------------------------
# Module 06's (its loop sums, linear layer, RNN, softmax cross-entropy and
# clipping), plus the LSTM. Elementwise IEEE arithmetic is order-free and
# vectorized; every SUM goes through _loop_sum in the C's loop order; exp and
# tanh are the C library's (Python's math module).

_loop_sum = m06._loop_sum
_ctanh = m06._ctanh
_exp = np.frompyfunc(math.exp, 1, 1)


def _csigmoid(Z):
    """lib/c/nanograd's sigmoid with the C library's exp: 1/(1 + e^-z) for
    z >= 0, e^z/(1 + e^z) below -- both written with e^-|z|."""
    E = _exp(-np.abs(Z)).astype(np.float64)
    return np.where(Z >= 0, 1.0 / (1.0 + E), E / (1.0 + E))


def _lstm_fwd(X, h0, c0, Wx, Wh, b):
    """ng_lstm_forward: for each step, every gate pre-activation is b + input
    terms (k ascending) + recurrent terms (j ascending; skipped at t = 0 when h0
    is None); then i, f, o = sigmoid, g = tanh; c = f*c_prev + i*g (just i*g at
    t = 0 when c0 is None); h = o*tanh(c). Returns (G, C, TC, H)."""
    T, m, _ = X.shape
    n = Wh.shape[0]
    G = np.empty((T, m, 4 * n))
    C, TC, H = np.empty((T, m, n)), np.empty((T, m, n)), np.empty((T, m, n))
    for t in range(T):
        terms = X[t].T[:, :, None] * Wx[:, None, :]                       # (k, i, o)
        hp = H[t - 1] if t > 0 else h0
        if hp is not None:
            terms = np.concatenate([terms, hp.T[:, :, None] * Wh[:, None, :]])  # (j, i, o)
        Z = _loop_sum(terms, np.tile(b, (m, 1)))
        G[t, :, :2 * n] = _csigmoid(Z[:, :2 * n])
        G[t, :, 2 * n:3 * n] = _ctanh(Z[:, 2 * n:3 * n])
        G[t, :, 3 * n:] = _csigmoid(Z[:, 3 * n:])
        i, f, g, o = G[t, :, :n], G[t, :, n:2 * n], G[t, :, 2 * n:3 * n], G[t, :, 3 * n:]
        cp = C[t - 1] if t > 0 else c0
        C[t] = i * g if cp is None else f * cp + i * g
        TC[t] = _ctanh(C[t])
        H[t] = o * TC[t]
    return G, C, TC, H


def _lstm_bwd(X, h0, c0, G, C, TC, H, Wx, Wh, dH, truncate=0, cell_only=False,
              need_dX=False):
    """ng_lstm_backward: walk t from T-1 down to 0. Per element, dh = dH (+ the
    gates of t+1 through Wh, o ascending, unless cut); dc = dh*o*(1 - tanh(c)^2)
    (+ dc<t+1>*f<t+1> along the cell, unless cut); then the four gate
    gradients. The shared weights' gradients run as one sum over (t
    descending, i ascending). Returns (dZ, dC, dX, dWx, dWh, db)."""
    T, m, n_in = X.shape
    n = Wh.shape[0]
    dZ, dC = np.empty_like(G), np.empty_like(C)
    dWx, dWh, db = np.zeros_like(Wx), np.zeros_like(Wh), np.zeros(4 * n)
    dX = np.empty_like(X) if need_dX else None
    for t in range(T - 1, -1, -1):
        flows = t + 1 < T and not (truncate > 0 and (t + 1) % truncate == 0)
        if flows and not cell_only:      # dh = dH<t> + dz<t+1> Wh^T, o ascending
            dh = _loop_sum(np.moveaxis(dZ[t + 1][:, None, :] * Wh[None, :, :], 2, 0), dH[t])
        else:
            dh = dH[t]
        i, f, g, o = G[t, :, :n], G[t, :, n:2 * n], G[t, :, 2 * n:3 * n], G[t, :, 3 * n:]
        dc = dh * o * (1.0 - TC[t] * TC[t])
        if flows:
            dc = dc + dC[t + 1] * G[t + 1, :, n:2 * n]
        dC[t] = dc
        cp = C[t - 1] if t > 0 else (np.zeros((m, n)) if c0 is None else c0)
        dZ[t, :, :n] = dc * g * (i * (1.0 - i))
        dZ[t, :, n:2 * n] = dc * cp * (f * (1.0 - f))
        dZ[t, :, 2 * n:3 * n] = dc * i * (1.0 - g * g)
        dZ[t, :, 3 * n:] = dh * TC[t] * (o * (1.0 - o))
        dWx = _loop_sum(X[t][:, :, None] * dZ[t][:, None, :], dWx)
        hp = H[t - 1] if t > 0 else h0
        if hp is not None:
            dWh = _loop_sum(hp[:, :, None] * dZ[t][:, None, :], dWh)
        db = _loop_sum(dZ[t], db)
        if need_dX:
            dX[t] = _loop_sum(np.moveaxis(dZ[t][:, None, :] * Wx[None, :, :], 2, 0),
                              np.zeros((m, n_in)))
    return dZ, dC, dX, dWx, dWh, db


def _xavier(rng, n_in, n_out):
    """ng_init_linear(NG_INIT_XAVIER)'s draws: row-major, std sqrt(1/n_in)."""
    std = math.sqrt(1.0 / n_in)
    W = np.empty((n_in, n_out))
    for i in range(n_in):
        for o in range(n_out):
            W[i, o] = std * rng.normal()
    return W


class ToyLatch:
    """One latch twin, mirroring c/lstm.c call for call.

    Parameters, in canonical order (the C struct's): for the RNN Wxh (8x16),
    Whh (16x16), bh, Why (16x4), by; for the LSTMs Wx (8x64), Wh (16x64), b
    (64), Why, by. One Xavier stream from TOY_SEED draws the input weights,
    the recurrent weights and Why, in that order -- so the three LSTM twins
    start from the same Wx, Wh and Why, and differ only in b or in the
    backward pass."""

    def __init__(self, rng, twin):
        self.twin = twin
        self.cell_only = twin == "carousel"
        if twin == "rnn":
            self.W_in, self.W_rec = _xavier(rng, V, N_H), _xavier(rng, N_H, N_H)
            self.b = np.zeros(N_H)
        else:
            self.W_in, self.W_rec = _xavier(rng, V, 4 * N_H), _xavier(rng, N_H, 4 * N_H)
            self.b = np.zeros(4 * N_H)
            bias = TEXTBOOK_BIAS if twin == "forget1" else OPEN_BIAS
            self.b[:N_H], self.b[N_H:2 * N_H] = bias["i"], bias["f"]
        self.Why = _xavier(rng, N_H, NS)
        self.by = np.zeros(NS)
        self.m = [np.zeros_like(p) for p in self.params()]      # Adam's moments
        self.v = [np.zeros_like(p) for p in self.params()]
        self.b1_pow = self.b2_pow = 1.0
        self.n_clipped = 0

    def params(self):
        return [self.W_in, self.W_rec, self.b, self.Why, self.by]

    def _forward(self, X):
        """The recurrence; returns (cache, H)."""
        if self.twin == "rnn":
            H = m06._rnn_fwd(X, None, self.W_in, self.W_rec, self.b)
            return H, H
        cache = _lstm_fwd(X, None, None, self.W_in, self.W_rec, self.b)
        return cache, cache[3]

    def evaluate(self, X, S):
        """(loss, accuracy) of the last step's read-out."""
        _, H = self._forward(X)
        loss, P, _ = m06._softmax_ce(m06._lin_fwd(H[-1], self.Why, self.by), S[0])
        return loss, float((P.argmax(1) == S[0]).mean())

    def loss_and_grads(self, X, S):
        """Forward + backpropagation through time: (loss, gradients in
        params() order)."""
        cache, H = self._forward(X)
        loss, _, dlogits = m06._softmax_ce(m06._lin_fwd(H[-1], self.Why, self.by), S[0])
        dh_last, dWhy, dby = m06._lin_bwd(H[-1], self.Why, dlogits)
        dH = np.zeros_like(H)
        dH[-1] = dh_last                                 # the only error signal
        if self.twin == "rnn":
            _, _, _, dWi, dWr, db = m06._rnn_bwd(X, None, H, self.W_in, self.W_rec, dH, 0)
        else:
            G, C, TC, _ = cache
            _, _, _, dWi, dWr, db = _lstm_bwd(X, None, None, G, C, TC, H, self.W_in,
                                              self.W_rec, dH, cell_only=self.cell_only)
        return loss, [dWi, dWr, db, dWhy, dby]

    def step(self, data_rng):
        """Draw a fresh batch, compute the gradients, clip them, take one Adam
        step. Returns (batch loss, gradient norm before clipping)."""
        S = draw_latch(data_rng, M_TOY, T_TOY)
        loss, grads = self.loss_and_grads(one_hot(S), S)
        norm = m06._clip(grads, CLIP)
        if norm > CLIP:
            self.n_clipped += 1
        self.b1_pow *= ADAM_B1                     # ng_adam_step's bias corrections
        self.b2_pow *= ADAM_B2
        b1c, b2c = 1.0 - self.b1_pow, 1.0 - self.b2_pow
        for p, g, m, v in zip(self.params(), grads, self.m, self.v):
            m[...] = ADAM_B1 * m + (1.0 - ADAM_B1) * g
            v[...] = ADAM_B2 * v + (1.0 - ADAM_B2) * (g * g)
            p -= TOY_LR * (m / b1c) / (np.sqrt(v / b2c) + ADAM_EPS)
        return loss, norm

    def wsum(self):
        """Sum of every parameter in canonical order (a plain loop, as the C)."""
        s = 0.0
        for p in self.params():
            for x in p.reshape(-1).tolist():
                s += x
        return s


def test_sets():
    """The fixed test sequences, one stream: 64 of the training length (80
    steps), then 64 four times longer (320). Returns [(X, S), (X4, S4)]."""
    rng = Rng(TEST_SEED)
    out = []
    for T in (T_TOY, T_TEST):
        S = draw_latch(rng, M_TEST, T)
        out.append((one_hot(S), S))
    return out


def fingerprint(net, batch_loss, grad_norm, tests=None):
    """Ordered scalars that pin down a trained toy net (agreement test)."""
    (X, S), (X4, S4) = tests if tests is not None else test_sets()
    loss, acc = net.evaluate(X, S)
    _, acc4 = net.evaluate(X4, S4)
    return (
        ("loss", loss),
        ("acc", acc),
        ("acc4", acc4),
        ("batch_loss", batch_loss),
        ("gnorm", grad_norm),
        ("n_clip", float(net.n_clipped)),
        ("wsum", net.wsum()),
        ("win_00", float(net.W_in[0, 0])),
        ("wrec_00", float(net.W_rec[0, 0])),
        ("why_00", float(net.Why[0, 0])),
    )


def run_toy(twin, verbose=False, steps=TOY_STEPS, tests=None):
    """Train one bit-exact latch twin and return its fingerprint tuple."""
    net = ToyLatch(Rng(TOY_SEED), twin)
    data_rng = Rng(DATA_SEED)
    if verbose:
        how = {"rnn": "Module 06's RNN, full BPTT",
               "lstm": "LSTM, forget bias +5, input bias -5, full BPTT",
               "carousel": "the same LSTM, gradient along the cell only (1997)",
               "forget1": "the same LSTM weights, forget bias 1, input bias 0"}[twin]
        print(f"[toy {twin:8s}] the latch, T={T_TOY}; {how}; Adam lr={TOY_LR}, clip {CLIP}")
    loss = norm = None
    for t in range(steps):
        loss, norm = net.step(data_rng)
        if verbose and (t % (steps // 4) == 0 or t == steps - 1):
            print(f"  step {t:4d}  batch_loss={loss:.6f}  |grad|={norm:.4f}")
    return fingerprint(net, loss, norm, tests)


# -- the vectorized twins -----------------------------------------------------

def latch_batch(gen, T, m):
    """(T, m) latch symbols from a NumPy generator (the fast path's data)."""
    S = gen.integers(NS, V, size=(T, m))
    S[0] = gen.integers(0, NS, size=m)
    return S


def build_latch_net(kind="lstm", rng=None, n_h=N_H, bias=None, n_in=V, n_out=NS):
    """A recurrent layer (kind "rnn" or "lstm") + Linear(n_h -> n_out) on
    nanograd layers, Xavier weights drawn input, recurrent, read-out from
    `rng`; the LSTM's gate biases from `bias` ({"i": .., "f": ..}, default
    OPEN_BIAS). With the defaults, ToyLatch's exact parameters."""
    if rng is None:
        rng = Rng(TOY_SEED)
    if kind == "rnn":
        layer = RNN(n_in, n_h)
        xavier_normal(layer.Wxh, rng)
        xavier_normal(layer.Whh, rng)
    else:
        layer = LSTM(n_in, n_h)
        xavier_normal(layer.Wx, rng)
        xavier_normal(layer.Wh, rng)
        layer.gate_bias(**(OPEN_BIAS if bias is None else bias))
    head = Linear(n_h, n_out)
    xavier_normal(head.W, rng)
    return layer, head


def _backward(layer, dH, cell_only=False):
    if isinstance(layer, LSTM):
        return layer.backward(dH, cell_only=cell_only)
    return layer.backward(dH)


def latch_loss_and_grads(layer, head, X, S, cell_only=False, loss_layer=None):
    """The latch loss on one batch, forward and backward through nanograd's
    layers (the vectorized ToyLatch.loss_and_grads). Returns the loss; the
    gradients are left in layer.grads() + head.grads()."""
    loss_layer = loss_layer or SoftmaxCrossEntropy()
    H = layer.forward(X)
    loss = loss_layer.forward(head.forward(H[-1]), S[0])
    dH = np.zeros_like(H)
    dH[-1] = head.backward(loss_layer.backward())
    _backward(layer, dH, cell_only)
    return loss


def latch_accuracy(layer, head, T, m=256, seed=0):
    """Accuracy on fresh T-step latch sequences."""
    S = latch_batch(np.random.default_rng(10_000 + seed), T, m)
    pred = head.forward(layer.forward(one_hot(S))[-1]).argmax(1)
    return float((pred == S[0]).mean())


def train_latch(T=T_TOY, kind="lstm", bias=None, cell_only=False, steps=1500, seed=0,
                n_h=N_H, m=M_TOY, lr=TOY_LR, clip=CLIP, stop_after=None, record_every=0):
    """The latch on nanograd's fast path (a NumPy generator draws the batches).
    `seed` varies the weights (Rng(TOY_SEED + 100 * seed)) and the data
    together. stop_after=k ends training once the mean batch loss of the last
    k updates is below 0.01 (a solved run need not burn the rest of the
    budget). Returns a dict: loss and norm (per update; norm before clipping),
    n_clip, solved_at (the first update whose trailing 20-update loss is below
    0.05, or None), acc {T: .., 4T: ..} on fresh sequences, curve [(update,
    acc at T)] if record_every, and the trained layer/head."""
    gen = np.random.default_rng(DATA_SEED + 100 * seed)
    layer, head = build_latch_net(kind, Rng(TOY_SEED + 100 * seed), n_h, bias)
    loss_layer = SoftmaxCrossEntropy()
    opt = Adam(lr=lr)
    params = layer.params() + head.params()
    losses, norms, curve, solved = [], [], [], None
    for k in range(steps):
        S = latch_batch(gen, T, m)
        losses.append(latch_loss_and_grads(layer, head, one_hot(S), S, cell_only, loss_layer))
        grads = layer.grads() + head.grads()
        norms.append(clip_grad_norm(grads, clip))
        opt.step(params, grads)
        if solved is None and k >= 19 and np.mean(losses[-20:]) < 0.05:
            solved = k + 1
        if record_every and (k + 1) % record_every == 0:
            curve.append((k + 1, latch_accuracy(layer, head, T, seed=seed)))
        if stop_after and k >= stop_after and np.mean(losses[-stop_after:]) < 0.01:
            break
    return {"loss": losses, "norm": norms, "n_clip": int(sum(n > clip for n in norms)),
            "solved_at": solved, "curve": curve,
            "acc": {T: latch_accuracy(layer, head, T, seed=seed),
                    4 * T: latch_accuracy(layer, head, 4 * T, seed=seed)},
            "layer": layer, "head": head}


def _train_summary(cfg):
    """train_latch(**cfg) without the trained layers (a sweep's worker)."""
    r = train_latch(**cfg)
    del r["layer"], r["head"]
    return r


def pool_map(fn, items, workers=None):
    """[fn(x) for x in items], in parallel -- one process per core, each with
    NumPy single-threaded. fn must be importable (a module-level function), as
    Windows' process spawn requires. The runs here are independent and seeded,
    so the results are what a serial loop would give."""
    from concurrent.futures import ProcessPoolExecutor
    saved = {k: os.environ.get(k) for k in ("MKL_NUM_THREADS", "OMP_NUM_THREADS",
                                            "OPENBLAS_NUM_THREADS")}
    os.environ.update({k: "1" for k in saved})      # inherited by the workers
    try:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(fn, items))
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def sweep(configs, workers=None):
    """train_latch(**cfg) for every cfg in `configs`, in parallel (pool_map);
    the result dicts in order, without the trained layers."""
    return pool_map(_train_summary, configs, workers)


def grad_through_time(layer, T=T_TOY, m=256, seed=0, relative=True):
    """How much of the gradient at the LAST step reaches each earlier step.

    Run a batch of latch sequences forward, send a unit gradient into the final
    hidden state only (a random direction of norm 1 per sequence), backprop-
    agate through time, and return {"x": ..., "c": ...} indexed by k = steps
    back: "x" is ||dL/dx<T-1-k>||, the gradient reaching the INPUT k steps
    back -- comparable across the RNN and the LSTM; for an LSTM, "c" is
    ||dL/dc<T-1-k>||, the gradient along the carousel. Batch means, relative
    to k = 0 (the last step) unless relative=False."""
    S = latch_batch(np.random.default_rng(20_000 + seed), T, m)
    H = layer.forward(one_hot(S))
    dH = np.zeros_like(H)
    g = np.random.default_rng(30_000 + seed).standard_normal(H[-1].shape)
    dH[-1] = g / np.linalg.norm(g, axis=1, keepdims=True)
    dX = layer.backward(dH)
    x = np.linalg.norm(dX, axis=2).mean(axis=1)[::-1]
    out = {"x": x / x[0] if relative else x}
    if isinstance(layer, LSTM):
        c = np.linalg.norm(layer.dC, axis=2).mean(axis=1)[::-1]
        out["c"] = c / c[0] if relative else c
    return out


def gate_trace(layer, S):
    """Run one sequence (S of shape (T,)) through a trained LSTM and return
    (i, f, o, c), each (T, n_h): the gates and the cell at every step."""
    layer.forward(one_hot(S[:, None]))
    n = layer.n_h
    G = layer.G[:, 0]
    return G[:, :n], G[:, n:2 * n], G[:, 3 * n:], layer.C[:, 0]


# ============================================================================
# 3. The fingerprint of the library itself, for the C mirror. c/lstm.c prints
# the same checksums from the same LCG stream; the agreement test compares
# them. (Vectorized here, loops there: the sums run in different orders, so
# they agree to ~1e-15, well inside the 1e-9 gate.)
# ============================================================================

LSTM_SEED = 6
LSTM_SHAPE = (5, 3, 4, 6)          # T, m, n_in, n_h


def lstm_fingerprint():
    """nanograd's LSTM on (T, m, n_in, n_h) = (5, 3, 4, 6) with an initial
    state and cell: the forward pass, a full-BPTT backward (dX, dh0, dc0, the
    three weight gradients, the gate and cell gradients through time), then a
    backward truncated every 2 steps and a cell-only backward. Draw order from
    Rng(LSTM_SEED): Wx, Wh, b, X, h0, c0, dH."""
    T, m, n_in, n_h = LSTM_SHAPE
    rng = Rng(LSTM_SEED)
    layer = LSTM(n_in, n_h)
    layer.Wx[:] = m06._draw(rng, (n_in, 4 * n_h), 0.6)
    layer.Wh[:] = m06._draw(rng, (n_h, 4 * n_h), 0.6)
    layer.b[:] = m06._draw(rng, (4 * n_h,), 0.5, 0.3)
    X = m06._draw(rng, (T, m, n_in), 1.5)
    h0 = m06._draw(rng, (m, n_h), 0.8)
    c0 = m06._draw(rng, (m, n_h), 1.2)
    dH = m06._draw(rng, (T, m, n_h))
    _sums = m06._sums
    H = layer.forward(X, h0, c0)
    dX = layer.backward(dH)
    out = (_sums("h", H) + _sums("c", layer.C) + _sums("dx", dX) + _sums("dh0", layer.dh0)
           + _sums("dc0", layer.dc0) + _sums("dwx", layer.dWx) + _sums("dwh", layer.dWh)
           + _sums("db", layer.db) + _sums("dz", layer.dZ) + _sums("dc", layer.dC))
    layer.backward(dH, truncate=2)
    out += _sums("dwh2", layer.dWh) + _sums("dc2", layer.dC)
    layer.backward(dH, cell_only=True)
    return out + _sums("dwh3", layer.dWh) + _sums("dc3", layer.dC) + _sums("dc03", layer.dc0)


# ============================================================================
# 4. The real thing: a character-level LSTM of Tiny Shakespeare.
#
# Module 06's task and training, unchanged -- read the text one character at a
# time and predict the next, 64 parallel streams, 64-step truncated BPTT with
# the state carried across updates, Adam, clipping at 5, float32, 20 epochs --
# with an LSTM in place of the RNN, sized to the same parameter budget: 122
# cells give 99,739 parameters against the RNN's 99,137. The customary forget
# bias of 1: text gives an error signal at every step, so nothing has to be
# carried blind for hundreds of steps, as in the latch. The learning rate is
# the LSTM's own: both architectures were trained at every rate in LR_GRID,
# and each is compared at its best.
#
#   one-hot(65) -> LSTM(65 -> 122) -> Linear(122 -> 65) -> softmax
# ============================================================================

CHAR_H = 122
CHAR_FORGET_BIAS = 1.0
CHAR_T, CHAR_B = m06.CHAR_T, m06.CHAR_B
CHAR_LR = 1.6e-2                     # the LSTM's best in LR_GRID (Module 06's RNN used 2e-3)
CHAR_CLIP = m06.CHAR_CLIP
CHAR_EPOCHS, CHAR_LR_DROPS = m06.CHAR_EPOCHS, m06.CHAR_LR_DROPS
CHAR_TARGET = 1.64                   # nats per character on the validation text
SAMPLE_PRIME = m06.SAMPLE_PRIME
CHECKPOINTS = m06.CHECKPOINTS
# Part of every checkpoint's key. Bump after changing CharLSTM or the training
# loop so stale checkpoints stop matching.
CHECKPOINT_VERSION = 1

load_text_data = m06.load_text_data


class CharLSTM:
    """The character model: an LSTM read out by a Linear at every step.

    Parameters are nanograd's: lstm.Wx, lstm.Wh, lstm.b, head.W, head.b
    (Xavier weights drawn Wx, Wh, head.W from Rng(seed); forget bias set).
    The recurrent state is the pair (h, c)."""

    def __init__(self, vocab, n_h=CHAR_H, seed=0, dtype=np.float32,
                 forget_bias=CHAR_FORGET_BIAS):
        self.vocab, self.n_h, self.dtype = vocab, n_h, dtype
        rng = Rng(seed)
        self.lstm = LSTM(vocab, n_h)
        xavier_normal(self.lstm.Wx, rng)
        xavier_normal(self.lstm.Wh, rng)
        self.lstm.gate_bias(f=forget_bias)
        self.head = Linear(n_h, vocab)
        xavier_normal(self.head.W, rng)
        self.lstm.astype(dtype)
        self.head.astype(dtype)
        self.eye = np.eye(vocab, dtype=dtype)
        self.loss_layer = SoftmaxCrossEntropy()

    def params(self):
        return self.lstm.params() + self.head.params()

    def grads(self):
        return self.lstm.grads() + self.head.grads()

    def zero_state(self, m):
        z = np.zeros((m, self.n_h), dtype=self.dtype)
        return z, z.copy()

    def logits(self, ids, state):
        """ids (T, m) -> (logits (T*m, vocab), H (T, m, n_h), the final (h, c))."""
        H = self.lstm.forward(self.eye[ids], *state)
        return self.head.forward(H.reshape(-1, self.n_h)), H, (H[-1], self.lstm.C[-1])

    def loss_and_grads(self, xs, ys, state, truncate=0):
        """One truncated-BPTT update's worth of work on xs, ys (T, m): the mean
        loss, with the gradients left in grads(). Returns (loss, final state)."""
        z, H, final = self.logits(xs, state)
        loss = self.loss_layer.forward(z, ys.reshape(-1))
        dH = self.head.backward(self.loss_layer.backward()).reshape(H.shape)
        self.lstm.backward(dH, truncate=truncate)
        return loss, final

    def nll(self, ids, m=32, chunk=256):
        """Per-character negative log-likelihood of ids[1:] from ids[:-1], as m
        parallel streams with the state carried across chunks (CharRNN.nll).
        Returns an (m, L) array; row r is the stream starting at r*L. Also
        leaves every step's cell in self.cells, (m, L, n_h)."""
        L = (ids.size - 1) // m
        xs = ids[:m * L].reshape(m, L).T
        ys = ids[1:m * L + 1].reshape(m, L).T
        out = np.empty((L, m))
        cells = np.empty((L, m, self.n_h), dtype=self.dtype)
        state = self.zero_state(m)
        for s in range(0, L, chunk):
            z, _, state = self.logits(xs[s:s + chunk], state)
            cells[s:s + chunk] = self.lstm.C
            z = z - z.max(axis=1, keepdims=True)
            logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
            y = ys[s:s + chunk].reshape(-1)
            out[s:s + chunk] = -logp[np.arange(y.size), y].reshape(-1, m)
        self.cells = cells.transpose(1, 0, 2)
        return out.T

    def evaluate(self, ids, m=32):
        """Mean loss in nats per character over the whole of `ids`."""
        return float(self.nll(ids, m).mean())

    def sample(self, chars, n=300, prime=SAMPLE_PRIME, temperature=1.0, seed=0):
        """Read `prime`, then generate n characters at the given temperature
        (CharRNN.sample)."""
        index = {c: i for i, c in enumerate(chars)}
        gen = np.random.default_rng(seed)
        state = self.zero_state(1)
        ids = [index[c] for c in prime]
        for i in ids[:-1]:
            _, _, state = self.logits(np.array([[i]]), state)
        out, i = [], ids[-1]
        for _ in range(n):
            z, _, state = self.logits(np.array([[i]]), state)
            z = z[0].astype(np.float64) / temperature
            p = np.exp(z - z.max())
            i = int(gen.choice(self.vocab, p=p / p.sum()))
            out.append(chars[i])
        return prime + "".join(out)


def _run_key(config):
    blob = json.dumps(config, sort_keys=True)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def run_char(epochs=CHAR_EPOCHS, n_h=CHAR_H, T=CHAR_T, B=CHAR_B, lr=CHAR_LR, clip=CHAR_CLIP,
             lr_drops=CHAR_LR_DROPS, forget_bias=CHAR_FORGET_BIAS, truncate=0, train_n=None,
             seed=0, data=None, cache=False, verbose=False):
    """Train the character LSTM; return (model, val_loss, history) -- Module
    06's run_char with the LSTM in place of the RNN. The canonical run (all
    defaults) targets val loss <= CHAR_TARGET. history holds one dict per
    epoch: lr, train_loss, val_loss, clipped (fraction of updates), seconds,
    and a 300-character sample (temperature 0.8, fixed seed).

    cache=True: weights and history are saved under
    data/shakespeare/checkpoints/ (gitignored), named by a hash of every
    setting here plus CHECKPOINT_VERSION; an identical later call loads them.
    """
    config = dict(module="07-lstm", version=CHECKPOINT_VERSION, epochs=int(epochs),
                  n_h=int(n_h), T=int(T), B=int(B), lr=float(lr), clip=float(clip),
                  lr_drops=[int(d) for d in lr_drops], forget_bias=float(forget_bias),
                  truncate=int(truncate), train_n=None if train_n is None else int(train_n),
                  seed=int(seed))
    ckpt = os.path.join(CHECKPOINTS, _run_key(config))
    train_ids, val_ids, chars = data if data is not None else load_text_data()
    model = CharLSTM(len(chars), n_h, seed, forget_bias=forget_bias)
    if cache and os.path.exists(ckpt + ".json"):          # .json is written last
        with np.load(ckpt + ".npz") as z:
            for k, p in enumerate(model.params()):
                p[...] = z[f"p{k}"]
        with open(ckpt + ".json", encoding="utf-8") as fh:
            history = json.load(fh)["history"]
        for rec in history:
            rec["loaded"] = True
        if verbose:
            print(f"[char] loaded checkpoint {os.path.basename(ckpt)} "
                  f"({len(history)} epochs, val_loss={history[-1]['val_loss']:.4f})")
        return model, history[-1]["val_loss"], history

    if train_n is not None:
        train_ids = train_ids[:train_n]
    L = (train_ids.size - 1) // B
    xs_all = train_ids[:B * L].reshape(B, L).T           # (L, B): B contiguous streams
    ys_all = train_ids[1:B * L + 1].reshape(B, L).T
    opt = Adam(lr=lr)
    if verbose:
        print(f"[char] LSTM({len(chars)} -> {n_h}) on {train_ids.size:,} characters, "
              f"{B} streams x {T} steps per update, {L // T} updates per epoch, "
              f"{sum(p.size for p in model.params()):,} params", flush=True)
    history = []
    for e in range(epochs):
        t0 = time.time()
        opt.lr = lr * 0.5 ** sum(e >= d for d in lr_drops)
        state = model.zero_state(B)
        losses, n_clipped = [], 0
        for s in range(0, L - T + 1, T):
            loss, state = model.loss_and_grads(xs_all[s:s + T], ys_all[s:s + T], state, truncate)
            if clip_grad_norm(model.grads(), clip) > clip:
                n_clipped += 1
            opt.step(model.params(), model.grads())
            losses.append(loss)
        rec = {"epoch": e, "lr": opt.lr, "train_loss": float(np.mean(losses)),
               "val_loss": model.evaluate(val_ids),
               "clipped": n_clipped / len(losses),
               "sample": model.sample(chars, temperature=0.8),
               "seconds": time.time() - t0}
        history.append(rec)
        if verbose:
            print(f"  epoch {e:2d}  lr={opt.lr:.5f}  train_loss={rec['train_loss']:.4f}  "
                  f"val_loss={rec['val_loss']:.4f}  clipped={rec['clipped']:.1%}  "
                  f"({rec['seconds']:.0f}s)", flush=True)
    if cache:
        os.makedirs(CHECKPOINTS, exist_ok=True)
        np.savez(ckpt + ".tmp.npz", **{f"p{k}": p for k, p in enumerate(model.params())})
        os.replace(ckpt + ".tmp.npz", ckpt + ".npz")
        with open(ckpt + ".json.tmp", "w", encoding="utf-8") as fh:
            json.dump({"config": config, "history": history}, fh, indent=1)
        os.replace(ckpt + ".json.tmp", ckpt + ".json")
    return model, history[-1]["val_loss"], history


# Module 06 trained its RNN at a learning rate of 2e-3. To compare the two
# architectures fairly, each is trained at every rate in LR_GRID (char_sweep)
# and compared at its best: the RNN's is RNN_LR, the LSTM's CHAR_LR.
LR_GRID = (2e-3, 4e-3, 8e-3, 1.6e-2, 3.2e-2)
RNN_LR = 4e-3


def load_rnn(lr=RNN_LR, data=None, verbose=False):
    """Module 06's character RNN at learning rate `lr` (from the checkpoint
    cache, or trained once if it is empty): (model, val_loss, history)."""
    return m06.run_char(lr=lr, data=data, cache=True, verbose=verbose)


def _char_job(job):
    """One char_sweep job: ("lstm" | "rnn", run_char keyword arguments)."""
    kind, kw = job
    fn = run_char if kind == "lstm" else m06.run_char
    _, val, history = fn(cache=True, **kw)
    return val, history


def char_sweep(jobs, workers=None):
    """Train (or load from the checkpoint cache) every (kind, kwargs) job --
    kind "lstm" for run_char, "rnn" for Module 06's run_char -- in parallel.
    Returns [(val_loss, history)] in order. Cached, instant; uncached, each
    run is a 20-epoch training on one core (about half an hour of wall time
    for the whole learning-rate bracket)."""
    return pool_map(_char_job, list(jobs), workers)


def context_curve(model, ids, lengths=(1, 2, 4, 8, 16, 32, 64, 128, 256), n=2048, seed=0,
                  chunk=512):
    """How much each model gains from context further back.

    For each context length L: pick the same n random positions p in `ids`
    (all >= max(lengths)), start the model from the zero state L characters
    before p, read ids[p-L:p], and score its prediction of ids[p]. Returns the
    mean loss (nats) per L. Works for CharLSTM and Module 06's CharRNN."""
    gen = np.random.default_rng(50_000 + seed)
    pos = gen.integers(max(lengths), ids.size, size=n)
    out = []
    for L in lengths:
        nll = np.empty(n)
        for s in range(0, n, chunk):
            p = pos[s:s + chunk]
            ctx = ids[p[None, :] + np.arange(-L, 0)[:, None]]          # (L, chunk)
            if isinstance(model, CharLSTM):
                z, _, _ = model.logits(ctx, model.zero_state(p.size))
            else:
                z, _ = model.logits(ctx, model.zero_state(p.size))
            z = z.reshape(L, p.size, -1)[-1].astype(np.float64)
            z = z - z.max(axis=1, keepdims=True)
            logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
            nll[s:s + chunk] = -logp[np.arange(p.size), ids[p]]
        out.append(float(nll.mean()))
    return out


def line_position(text):
    """For every character: how many characters since the last newline (0 for
    a line's first character)."""
    pos, k = [], 0
    for c in text:
        pos.append(k)
        k = 0 if c == "\n" else k + 1
    return np.array(pos)


# ============================================================================
# 5. The animation export.
# ============================================================================

def export_assets(payload, out_path=None):
    """Write topics/07-lstm/anim/lstm_assets.json for the Manim scene: the
    notebook assembles `payload` from its own runs, so the video and the
    chapter show the same numbers."""
    if out_path is None:
        out_path = os.path.join(_REPO, "topics", "07-lstm", "anim", "lstm_assets.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    print(f"[export] wrote {out_path}")


if __name__ == "__main__":
    for twin in TWINS:
        fp = run_toy(twin, verbose=True)
        print(f"FINAL {twin} " + " ".join(f"{name}={val!r}" for name, val in fp))
    print("FINAL cell " + " ".join(f"{name}={val!r}" for name, val in lstm_fingerprint()))

    # Tens of minutes of NumPy the first time; afterwards (or after the
    # notebook ran) the cached checkpoints are loaded and re-evaluated.
    data = load_text_data()
    train_ids, val_ids, chars = data
    model, _, _ = run_char(data=data, cache=True, verbose=True)
    rnn_model, _, _ = load_rnn(data=data, verbose=True)
    val, val_rnn = model.evaluate(val_ids), rnn_model.evaluate(val_ids)
    print(f"\n[char] validation loss: LSTM {val:.4f} nats/char ({val / math.log(2):.3f} bits), "
          f"Module 06's RNN {val_rnn:.4f}  (target <= {CHAR_TARGET})")
    print(model.sample(chars, n=400, temperature=0.8))
    assert val <= CHAR_TARGET, f"Tiny Shakespeare target missed: {val:.4f} > {CHAR_TARGET}"
    assert val < val_rnn, "the LSTM should beat Module 06's RNN at the same parameter budget"
