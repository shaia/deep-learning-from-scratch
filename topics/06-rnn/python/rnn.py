"""
Module 06 - RNN: share weights across time.

The Python mirror of topics/06-rnn/c/rnn.c. Every network so far has read a
fixed-size input. Language, speech and music are sequences, of any length,
where what matters at step 10 can depend on step 1. A recurrent network reads
them one step at a time and carries a hidden state forward -- its memory --
applying the SAME weights at every step:

    h<t> = tanh(x<t> Wxh + h<t-1> Whh + bh),     yhat<t> = softmax(h<t> Why + by)

nanograd gains the RNN layer (forward over a whole sequence, and backprop-
agation through time, with optional truncation) and gradient-norm clipping,
in both languages.

Five kinds of code live here:

  1. run_toy() -- the *agreement* path, on "echo": a stream of random symbols
     from a 4-letter alphabet, and at every step the network must output the
     symbol it read DELAY = 4 steps earlier. A 16-unit RNN is trained twice
     from the SAME initial weights on the SAME batches: once with full
     backpropagation through time (Werbos 1990), once with Elman's (1990)
     one-step truncation, in which the gradient stops at the previous state.
     Same network, same arithmetic forward, one cut in the backward pass.
     ToyRNN runs every sum in c/rnn.c's loop order, and its tanh, exp and log
     are the C library's (math.*; NumPy's own tanh differs from it in the last
     bit for about a fifth of inputs), so C and Python agree to the last digit
     (1e-9 gate).

  2. build_echo_net()/train_echo() and the counting toy -- the same networks on
     nanograd's vectorized layers, for the notebook's sweeps: how far back an
     RNN can learn to remember, what truncation does to that, how the gradient
     decays through time, and a mod-K counter whose hidden state becomes a
     finite-state machine that runs far longer than anything it trained on.

  3. The fingerprints -- nanograd's RNN forward/backward (with an initial
     state, input gradients and truncation) and clip_grad_norm, on fixed
     LCG-drawn tensors, so the C library's versions are pinned too.

  4. The real thing: a character-level language model of Tiny Shakespeare
     (CharRNN, run_char with a checkpoint cache, target val loss <= CHAR_TARGET
     nats per character), the n-gram baselines it must beat, sampling, and
     per-character surprisal (Elman's word-boundary experiment, redone).

  5. export_assets() -- the data the Manim scene draws.

Run directly to see the toy, then the character model:
    python topics/06-rnn/python/rnn.py

Reference: docs/references/papers.md (Module 06) -- Elman (1990), "Finding
Structure in Time"; Werbos (1990), "Backpropagation Through Time: What It Does
and How to Do It"; Pascanu, Mikolov & Bengio (2013), "On the Difficulty of
Training Recurrent Neural Networks".
"""

import hashlib
import json
import math
import os
import sys
import time

import numpy as np

# Make the shared nanograd library importable (lib/python on sys.path).
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(_REPO, "lib", "python"))

from nanograd import (  # noqa: E402
    RNN,
    Adam,
    Linear,
    SoftmaxCrossEntropy,
    clip_grad_norm,
    xavier_normal,
)
from nanograd.rng import Rng  # noqa: E402


# ============================================================================
# 1. The bit-exact toy: "echo".
#
# Input: one symbol per step, uniform over an alphabet of V = 4, one-hot.
# Target at step t: the symbol read at step t - DELAY (no target for the first
# DELAY steps). To do it, the network must hold the last DELAY symbols in its
# hidden state -- a shift register built from 16 tanh units -- and pass each
# one along until it is asked for. Every training step draws a fresh batch, so
# memorizing is impossible; the test set is longer sequences than the
# training ones (96 steps against 24), which only a network that learned the
# RULE -- the same weights doing the same thing at every step -- gets right.
#
# Architecture: RNN(4 -> 16) -> Linear(16 -> 4) -> softmax cross-entropy,
# averaged over every (step >= DELAY, sequence) position. Xavier init (tanh),
# Adam, gradient-norm clipping at CLIP.
#
# The two twins differ only in how far the gradient travels back in time:
#   bptt   full backpropagation through time  (truncate = 0)
#   elman  the gradient stops at h<t-1>, which is treated as just another
#          input -- Elman's 1990 training     (truncate = 1)
# ============================================================================

TOY_SEED = 1          # weight draws
DATA_SEED = 2         # the training batches (one fresh batch per step)
TEST_SEED = 3         # the fixed test sequences
V = 4                 # alphabet size
N_H = 16              # hidden units
T_TOY, M_TOY = 24, 32          # steps per training sequence, sequences per batch
T_TEST, M_TEST = 96, 64        # the test set: four times longer than training
DELAY = 4
TOY_STEPS = 300
TOY_LR = 0.01
CLIP = 1.0
ADAM_B1, ADAM_B2, ADAM_EPS = 0.9, 0.999, 1e-8


def draw_symbols(rng, m, T, vocab=V):
    """(T, m) int64 symbols, uniform over `vocab`, in the exact draw order
    c/rnn.c replays: sequence by sequence, each left to right --
    S[t, i] = floor(vocab * u) for i in 0..m-1, then t in 0..T-1."""
    S = np.empty((T, m), dtype=np.int64)
    for i in range(m):
        for t in range(T):
            S[t, i] = int(vocab * rng.uniform())
    return S


def one_hot(S, vocab=V):
    """(T, m) symbols -> (T, m, vocab) float64 one-hot vectors. Multiplying a
    one-hot row by Wxh just picks out row S[t, i] of Wxh."""
    return np.eye(vocab)[S]


# -- the C-order building blocks --------------------------------------------
# Each is one lib/c/nanograd function. Elementwise IEEE arithmetic is order-
# free and vectorized; every SUM goes through _loop_sum, in the C's loop
# order; tanh, exp and log go through Python's math module -- the C library's
# own functions.

_tanh = np.frompyfunc(math.tanh, 1, 1)
_exp = np.frompyfunc(math.exp, 1, 1)
_log = np.frompyfunc(math.log, 1, 1)


def _ctanh(Z):
    return _tanh(Z).astype(np.float64)


def _loop_sum(terms, start):
    """((start + t[0]) + t[1]) + ... over axis 0: the C loop `s = start;
    for (i ...) s += t[i];`, vectorized over every other axis (Module 05)."""
    return np.add.accumulate(np.concatenate([start[None], terms]), axis=0)[-1]


def _lin_fwd(X, W, b):
    """ng_linear_forward: Y[i,o] = b[o] + sum_k X[i,k] W[k,o], k ascending."""
    return _loop_sum(X.T[:, :, None] * W[:, None, :], np.tile(b, (X.shape[0], 1)))


def _lin_bwd(X, W, dY):
    """ng_linear_backward: dW and db summed over i ascending, dX over o
    ascending, each from 0.0. Returns (dX, dW, db)."""
    dW = _loop_sum(X[:, :, None] * dY[:, None, :], np.zeros_like(W))     # (i, k, o)
    db = _loop_sum(dY, np.zeros(W.shape[1]))
    dX = _loop_sum(np.moveaxis(dY[:, None, :] * W[None, :, :], 2, 0),    # (o, i, k)
                   np.zeros_like(X))
    return dX, dW, db


def _rnn_fwd(X, h0, Wxh, Whh, bh):
    """ng_rnn_forward: for each step, z = bh + input terms (k ascending) +
    recurrent terms (j ascending; skipped at t = 0 when h0 is None)."""
    T, m, _ = X.shape
    H = np.empty((T, m, Whh.shape[0]))
    for t in range(T):
        terms = X[t].T[:, :, None] * Wxh[:, None, :]                       # (k, i, o)
        hp = H[t - 1] if t > 0 else h0
        if hp is not None:
            terms = np.concatenate([terms, hp.T[:, :, None] * Whh[:, None, :]])  # (j, i, o)
        H[t] = _ctanh(_loop_sum(terms, np.tile(bh, (m, 1))))
    return H


def _rnn_bwd(X, h0, H, Wxh, Whh, dH, truncate, need_dX=False, need_dh0=False):
    """ng_rnn_backward: walk t from T-1 down to 0; the shared weights'
    gradients run as one sum over (t descending, i ascending). Returns
    (dZ, dX, dh0, dWxh, dWhh, dbh)."""
    T, m, n_in = X.shape
    n_h = Whh.shape[0]
    dZ = np.empty_like(H)
    dWxh, dWhh, dbh = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros(n_h)
    dX = np.empty_like(X) if need_dX else None
    for t in range(T - 1, -1, -1):
        flows = t + 1 < T and not (truncate > 0 and (t + 1) % truncate == 0)
        if flows:      # dh = dH<t> + dz<t+1> Whh^T, o ascending
            s = _loop_sum(np.moveaxis(dZ[t + 1][:, None, :] * Whh[None, :, :], 2, 0), dH[t])
        else:
            s = dH[t]
        dZ[t] = s * (1.0 - H[t] * H[t])
        dWxh = _loop_sum(X[t][:, :, None] * dZ[t][:, None, :], dWxh)
        hp = H[t - 1] if t > 0 else h0
        if hp is not None:
            dWhh = _loop_sum(hp[:, :, None] * dZ[t][:, None, :], dWhh)
        dbh = _loop_sum(dZ[t], dbh)
        if need_dX:
            dX[t] = _loop_sum(np.moveaxis(dZ[t][:, None, :] * Wxh[None, :, :], 2, 0),
                              np.zeros((m, n_in)))
    dh0 = None
    if need_dh0:
        dh0 = np.zeros((m, n_h))
        if not truncate:
            dh0 = _loop_sum(np.moveaxis(dZ[0][:, None, :] * Whh[None, :, :], 2, 0), dh0)
    return dZ, dX, dh0, dWxh, dWhh, dbh


def _softmax_ce(logits, y):
    """ng_softmax_ce_forward/backward with the C library's exp and log, every
    sum in the C's order. Returns (loss, P, dlogits)."""
    m, k = logits.shape
    mx = logits[:, 0].copy()
    for c in range(1, k):
        mx = np.where(logits[:, c] > mx, logits[:, c], mx)
    e = _exp(logits - mx[:, None]).astype(np.float64)
    P = e / _loop_sum(e.T, np.zeros(m))[:, None]
    nll = -_log(P[np.arange(m), y] + 1e-12).astype(np.float64)
    total = float(_loop_sum(nll, np.array(0.0)))
    onehot = np.zeros_like(P)
    onehot[np.arange(m), y] = 1.0
    return total / m, P, (P - onehot) / m


def _clip(grads, max_norm):
    """ng_clip_grad_norm over the gradients laid end to end, in order."""
    flat = np.concatenate([g.reshape(-1) for g in grads])
    norm = math.sqrt(float(_loop_sum(flat * flat, np.array(0.0))))
    if norm > max_norm:
        scale = max_norm / norm
        for g in grads:
            g *= scale
    return norm


class ToyRNN:
    """The echo network, mirroring c/rnn.c call for call.

    Parameters, in canonical order (the C struct's): Wxh (4x16), Whh (16x16),
    bh, Why (16x4), by. Draw order at init (one Xavier stream): Wxh, Whh, Why;
    biases start at 0. `truncate` is the only difference between the twins.
    """

    def __init__(self, rng, truncate):
        self.truncate = truncate
        self.Wxh = self._xavier(rng, V, N_H)
        self.Whh = self._xavier(rng, N_H, N_H)
        self.bh = np.zeros(N_H)
        self.Why = self._xavier(rng, N_H, V)
        self.by = np.zeros(V)
        self.m = [np.zeros_like(p) for p in self.params()]      # Adam's moments
        self.v = [np.zeros_like(p) for p in self.params()]
        self.b1_pow = self.b2_pow = 1.0
        self.n_clipped = 0

    @staticmethod
    def _xavier(rng, n_in, n_out):
        std = math.sqrt(1.0 / n_in)
        W = np.empty((n_in, n_out))
        for i in range(n_in):
            for o in range(n_out):
                W[i, o] = std * rng.normal()
        return W

    def params(self):
        return [self.Wxh, self.Whh, self.bh, self.Why, self.by]

    def _logits(self, X):
        H = _rnn_fwd(X, None, self.Wxh, self.Whh, self.bh)
        T, m, _ = X.shape
        Hd = H[DELAY:].reshape((T - DELAY) * m, N_H)       # the steps with a target
        return H, Hd, _lin_fwd(Hd, self.Why, self.by)

    def evaluate(self, X, S):
        """(loss, accuracy) over every position that has a target."""
        _, _, logits = self._logits(X)
        y = S[:-DELAY].reshape(-1)
        loss, P, _ = _softmax_ce(logits, y)
        return loss, float((P.argmax(1) == y).mean())

    def loss_and_grads(self, X, S):
        """Forward + backpropagation through time: (loss, gradients in
        params() order)."""
        H, Hd, logits = self._logits(X)
        loss, _, dlogits = _softmax_ce(logits, S[:-DELAY].reshape(-1))
        dHd, dWhy, dby = _lin_bwd(Hd, self.Why, dlogits)
        dH = np.zeros_like(H)
        dH[DELAY:] = dHd.reshape(H[DELAY:].shape)          # no gradient before DELAY
        _, _, _, dWxh, dWhh, dbh = _rnn_bwd(X, None, H, self.Wxh, self.Whh, dH,
                                            self.truncate)
        return loss, [dWxh, dWhh, dbh, dWhy, dby]

    def step(self, data_rng):
        """Draw a fresh batch, compute the gradients, clip them, take one Adam
        step. Returns (batch loss, gradient norm before clipping)."""
        S = draw_symbols(data_rng, M_TOY, T_TOY)
        loss, grads = self.loss_and_grads(one_hot(S), S)
        norm = _clip(grads, CLIP)
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


def test_set():
    """The fixed test sequences: (X, S), 96 steps x 64 sequences."""
    S = draw_symbols(Rng(TEST_SEED), M_TEST, T_TEST)
    return one_hot(S), S


def fingerprint(net, batch_loss, grad_norm):
    """Ordered scalars that pin down a trained toy net (agreement test)."""
    loss, acc = net.evaluate(*test_set())
    return (
        ("loss", loss),
        ("acc", acc),
        ("batch_loss", batch_loss),
        ("gnorm", grad_norm),
        ("n_clip", float(net.n_clipped)),
        ("wsum", net.wsum()),
        ("wxh_00", float(net.Wxh[0, 0])),
        ("whh_00", float(net.Whh[0, 0])),
        ("why_00", float(net.Why[0, 0])),
    )


def run_toy(truncate, verbose=False, steps=TOY_STEPS):
    """Train one bit-exact echo twin and return its fingerprint tuple."""
    net = ToyRNN(Rng(TOY_SEED), truncate)
    data_rng = Rng(DATA_SEED)
    name = "elman" if truncate == 1 else "bptt "
    if verbose:
        how = ("gradient stops at the previous state (Elman 1990)" if truncate == 1
               else "full backpropagation through time")
        print(f"[toy {name}] echo the symbol from {DELAY} steps back; RNN({V}->{N_H}), "
              f"{how}; Adam lr={TOY_LR}, clip {CLIP}")
    loss = norm = None
    for t in range(steps):
        loss, norm = net.step(data_rng)
        if verbose and (t % (steps // 4) == 0 or t == steps - 1):
            print(f"  step {t:4d}  batch_loss={loss:.6f}  |grad|={norm:.4f}")
    return fingerprint(net, loss, norm)


# -- the vectorized twin ------------------------------------------------------

def build_echo_net(rng=None, n_in=V, n_h=N_H, n_out=V, gain=1.0):
    """RNN(n_in -> n_h) + Linear(n_h -> n_out) on nanograd layers, Xavier
    weights drawn Wxh, Whh, Why from `rng` -- with the defaults, ToyRNN's
    exact parameters. gain scales Whh after the draw: Xavier's 1/sqrt(n_h)
    puts Whh's eigenvalues in a disc of radius about 1 (the circular law), so
    its spectral radius is about `gain`."""
    if rng is None:
        rng = Rng(TOY_SEED)
    rnn = RNN(n_in, n_h)
    xavier_normal(rnn.Wxh, rng)
    xavier_normal(rnn.Whh, rng)
    rnn.Whh *= gain
    head = Linear(n_h, n_out)
    xavier_normal(head.W, rng)
    return rnn, head


def echo_loss_and_grads(rnn, head, X, S, delay, truncate=0, loss_layer=None):
    """The echo loss on one batch, forward and backward through nanograd's
    layers (the vectorized ToyRNN.loss_and_grads). Returns the loss; the
    gradients are left in rnn.grads() + head.grads()."""
    loss_layer = loss_layer or SoftmaxCrossEntropy()
    T, m, _ = X.shape
    H = rnn.forward(X)
    n_h = H.shape[2]
    loss = loss_layer.forward(head.forward(H[delay:].reshape(-1, n_h)), S[:-delay].reshape(-1))
    dH = np.zeros_like(H)
    dH[delay:] = head.backward(loss_layer.backward()).reshape(T - delay, m, n_h)
    rnn.backward(dH, truncate=truncate)
    return loss


def echo_accuracy(rnn, head, delay, T=T_TEST, m=256, seed=0, vocab=V):
    """Accuracy on fresh test sequences (T steps, every position >= delay)."""
    S = np.random.default_rng(10_000 + seed).integers(0, vocab, size=(T, m))
    H = rnn.forward(one_hot(S, vocab))
    pred = head.forward(H[delay:].reshape(-1, H.shape[2])).argmax(1)
    return float((pred == S[:-delay].reshape(-1)).mean())


def train_echo(delay=DELAY, truncate=0, steps=TOY_STEPS, seed=0, n_h=N_H, T=T_TOY,
               m=M_TOY, lr=TOY_LR, clip=CLIP, gain=1.0, vocab=V, optimizer="adam",
               momentum=0.9, record_every=0):
    """The echo toy on nanograd's fast path (a NumPy generator draws the
    batches). `seed` varies the weights (Rng(TOY_SEED + 100 * seed)) and the
    data together. optimizer is "adam" or "sgd" (with `momentum`); clip=None
    switches clipping off. Returns a dict: loss (batch loss per step), norm
    (gradient norm before clipping, per step), acc (fresh 96-step test
    sequences), and the trained rnn/head. record_every > 0 also records the
    test accuracy every that many steps (curve: [(step, acc), ...])."""
    from nanograd import SGD
    gen = np.random.default_rng(DATA_SEED + 100 * seed)
    rnn, head = build_echo_net(Rng(TOY_SEED + 100 * seed), vocab, n_h, vocab, gain)
    loss_layer = SoftmaxCrossEntropy()
    opt = Adam(lr=lr) if optimizer == "adam" else SGD(lr=lr, momentum=momentum)
    params = rnn.params() + head.params()
    losses, norms, curve = [], [], []
    for k in range(steps):
        S = gen.integers(0, vocab, size=(T, m))
        losses.append(echo_loss_and_grads(rnn, head, one_hot(S, vocab), S, delay, truncate,
                                          loss_layer))
        grads = rnn.grads() + head.grads()
        if clip is None:
            norms.append(float(np.sqrt(sum(float((g * g).sum()) for g in grads))))
        else:
            norms.append(clip_grad_norm(grads, clip))
        opt.step(params, grads)
        if record_every and (k + 1) % record_every == 0:
            curve.append((k + 1, echo_accuracy(rnn, head, delay, seed=seed, vocab=vocab)))
    return {"loss": losses, "norm": norms, "curve": curve,
            "acc": echo_accuracy(rnn, head, delay, seed=seed, vocab=vocab),
            "rnn": rnn, "head": head}


def grad_through_time(rnn, T=40, m=256, seed=0, vocab=V):
    """How much of the gradient at the LAST step reaches each earlier step.

    Run a batch of random sequences forward, send a unit gradient into the
    final hidden state only (dH<T-1> a random direction of norm 1 per
    sequence), backpropagate through time, and return ||dL/dh<t>|| averaged
    over the batch for t = T-1 down to 0 -- index k is k steps back. The
    curve is the product of k Jacobians diag(1 - h^2) Whh^T applied to one
    vector."""
    S = np.random.default_rng(20_000 + seed).integers(0, vocab, size=(T, m))
    H = rnn.forward(one_hot(S, vocab))
    dH = np.zeros_like(H)
    g = np.random.default_rng(30_000 + seed).standard_normal(H[-1].shape)
    dH[-1] = g / np.linalg.norm(g, axis=1, keepdims=True)
    rnn.backward(dH)
    # dL/dh<t> = dL/dz<t+1> Whh^T for t < T-1 (and dH<T-1> itself at the end).
    dh = np.concatenate([rnn.dZ[1:] @ rnn.Whh.T, dH[-1:]])
    return np.linalg.norm(dh, axis=2).mean(axis=1)[::-1]


def spectral_radius(W):
    return float(np.max(np.abs(np.linalg.eigvals(W))))


# -- the counting toy -----------------------------------------------------------
# Input: a stream of bits. Target at every step: how many 1s so far, mod K.
# The answer depends on the ENTIRE history, but only through a K-state
# summary -- a finite-state machine with K states in a cycle. Unlike the echo,
# every step's target IS the state the network must keep, so a gradient that
# reaches one step back is enough to learn it.

COUNT_K = 3
COUNT_T, COUNT_M = 20, 32
COUNT_STEPS = 1500


def counting_batch(gen, m, T, K=COUNT_K, p=0.5):
    """(X, y): X (T, m, 2) one-hot bits (1 with probability p), y (T, m) the
    running count of 1s mod K."""
    bits = (gen.random((T, m)) < p).astype(np.int64)
    return one_hot(bits, 2), np.cumsum(bits, axis=0) % K


def train_counting(K=COUNT_K, steps=COUNT_STEPS, truncate=0, seed=0, n_h=N_H, T=COUNT_T,
                   m=COUNT_M, lr=TOY_LR, clip=CLIP):
    """Train the mod-K counter (Adam, clipped). Returns dict(loss, rnn, head)."""
    gen = np.random.default_rng(DATA_SEED + 100 * seed)
    rnn, head = build_echo_net(Rng(TOY_SEED + 100 * seed), 2, n_h, K)
    loss_layer = SoftmaxCrossEntropy()
    opt = Adam(lr=lr)
    params = rnn.params() + head.params()
    losses = []
    for _ in range(steps):
        X, y = counting_batch(gen, m, T, K)
        H = rnn.forward(X)
        losses.append(loss_layer.forward(head.forward(H.reshape(-1, n_h)), y.reshape(-1)))
        rnn.backward(head.backward(loss_layer.backward()).reshape(H.shape), truncate=truncate)
        grads = rnn.grads() + head.grads()
        clip_grad_norm(grads, clip)
        opt.step(params, grads)
    return {"loss": losses, "rnn": rnn, "head": head}


def counting_accuracy(rnn, head, T, K=COUNT_K, m=200, seed=0):
    """Accuracy over every step of fresh T-step bit streams; also returns the
    hidden states and the true counts (for the state-machine picture)."""
    X, y = counting_batch(np.random.default_rng(40_000 + seed), m, T, K)
    H = rnn.forward(X)
    pred = head.forward(H.reshape(-1, H.shape[2])).argmax(1).reshape(y.shape)
    return float((pred == y).mean()), H, y


# ============================================================================
# 3. Fingerprints of the library itself, for the C mirror. c/rnn.c prints the
# same checksums from the same LCG streams; the agreement test compares them.
# (Vectorized here, loops there: the sums run in different orders, so they
# agree to ~1e-15, well inside the 1e-9 gate.)
# ============================================================================

RNN_SEED = 6
RNN_SHAPE = (5, 3, 4, 6)           # T, m, n_in, n_h
CLIP_SEED = 7
CLIP_SIZES = (12, 30)              # two "parameter" arrays, clipped as one vector


def _draw(rng, shape, scale=1.0, shift=0.0):
    """shape-many scale*signed()+shift draws from `rng`, row-major."""
    return np.array([scale * rng.signed() + shift
                     for _ in range(int(np.prod(shape)))]).reshape(shape)


def _sums(name, A):
    """(name_sum, name_wsum): a plain sum and a position-weighted sum, both
    left-to-right loops (Python's sum() compensates since 3.12, so no)."""
    s = ws = 0.0
    for j, v in enumerate(np.asarray(A, dtype=np.float64).reshape(-1).tolist()):
        s += v
        ws += v * (j + 1)
    return ((f"{name}_sum", s), (f"{name}_wsum", ws))


def rnn_fingerprint():
    """nanograd's RNN on (T, m, n_in, n_h) = (5, 3, 4, 6) with an initial
    state: the forward pass, a full-BPTT backward (dX, dh0 and the three
    weight gradients), then a backward truncated every 2 steps. Draw order
    from Rng(RNN_SEED): Wxh, Whh, bh, X, h0, dH."""
    T, m, n_in, n_h = RNN_SHAPE
    rng = Rng(RNN_SEED)
    rnn = RNN(n_in, n_h)
    rnn.Wxh[:] = _draw(rng, (n_in, n_h), 0.6)
    rnn.Whh[:] = _draw(rng, (n_h, n_h), 0.6)
    rnn.bh[:] = _draw(rng, (n_h,), 0.2)
    X = _draw(rng, (T, m, n_in), 1.5)
    h0 = _draw(rng, (m, n_h), 0.8)
    dH = _draw(rng, (T, m, n_h))
    H = rnn.forward(X, h0)
    dX = rnn.backward(dH)
    out = (_sums("h", H) + _sums("dx", dX) + _sums("dh0", rnn.dh0) + _sums("dwxh", rnn.dWxh)
           + _sums("dwhh", rnn.dWhh) + _sums("dbh", rnn.dbh) + _sums("dz", rnn.dZ))
    rnn.backward(dH, truncate=2)
    return out + _sums("dwhh2", rnn.dWhh) + _sums("dz2", rnn.dZ)


def clip_fingerprint():
    """clip_grad_norm on two arrays drawn 3*signed() from Rng(CLIP_SEED),
    clipped to norm 2 (so they shrink), then again to norm 100 (a no-op)."""
    rng = Rng(CLIP_SEED)
    gs = [_draw(rng, (n,), 3.0) for n in CLIP_SIZES]
    n1 = clip_grad_norm(gs, 2.0)
    n2 = clip_grad_norm(gs, 100.0)
    flat = np.concatenate(gs)
    return (("norm", n1), ("norm_after", n2)) + _sums("g", flat)


# ============================================================================
# 4. The real thing: a character-level language model of Tiny Shakespeare.
#
# The task is Karpathy's char-rnn (2015) in its plainest form: read the text
# one character at a time and, at every step, predict the next one. The loss
# is the mean negative log-probability the model gave the character that
# actually came next -- nats per character; divide by ln 2 for bits.
#
#   one-hot(65) -> RNN(65 -> 256) -> Linear(256 -> 65) -> softmax
#
# The training text is cut into CHAR_B parallel streams, each a contiguous
# slice; every update reads the next CHAR_T characters of each stream and the
# state carries over from one update to the next (truncated BPTT: the memory
# reaches as far back as the stream goes, the gradient only CHAR_T steps).
# Adam, gradient norm clipped at 5 (char-rnn's default), float32, and the
# learning rate halved for the last epochs. Checked by metric -- validation
# loss below the best n-gram baseline -- and cached like Module 05's CIFAR runs.
# ============================================================================

CHAR_H = 256
CHAR_T = 64
CHAR_B = 64
CHAR_LR = 2e-3
CHAR_CLIP = 5.0
CHAR_EPOCHS = 20
CHAR_LR_DROPS = (15, 18)             # halve the learning rate at these epochs
CHAR_TARGET = 1.70                   # nats per character on the validation text
SAMPLE_PRIME = "ROMEO:\n"

CHECKPOINTS = os.path.join(_REPO, "data", "shakespeare", "checkpoints")
# Part of every checkpoint's key. Bump after changing CharRNN or the training
# loop so stale checkpoints stop matching.
CHECKPOINT_VERSION = 1


def load_text_data():
    """(train_ids, val_ids, chars) from data/get_shakespeare.py."""
    sys.path.insert(0, os.path.join(_REPO, "data"))
    from get_shakespeare import load_shakespeare
    return load_shakespeare()


class CharRNN:
    """The character model: an RNN read out by a Linear at every step.

    Built from build_echo_net (Xavier weights, Rng(seed)), so its parameters
    are nanograd's: rnn.Wxh, rnn.Whh, rnn.bh, head.W, head.b."""

    def __init__(self, vocab, n_h=CHAR_H, seed=0, dtype=np.float32):
        self.vocab, self.n_h, self.dtype = vocab, n_h, dtype
        self.rnn, self.head = build_echo_net(Rng(seed), vocab, n_h, vocab)
        self.rnn.astype(dtype)
        self.head.astype(dtype)
        self.eye = np.eye(vocab, dtype=dtype)
        self.loss_layer = SoftmaxCrossEntropy()

    def params(self):
        return self.rnn.params() + self.head.params()

    def grads(self):
        return self.rnn.grads() + self.head.grads()

    def zero_state(self, m):
        return np.zeros((m, self.n_h), dtype=self.dtype)

    def logits(self, ids, h0):
        """ids (T, m) -> (logits (T*m, vocab), H (T, m, n_h))."""
        H = self.rnn.forward(self.eye[ids], h0)
        return self.head.forward(H.reshape(-1, self.n_h)), H

    def loss_and_grads(self, xs, ys, h0, truncate=0):
        """One truncated-BPTT update's worth of work on xs, ys (T, m): the
        mean loss, with the gradients left in grads(). Returns (loss, h<T>),
        the state to carry into the next chunk."""
        z, H = self.logits(xs, h0)
        loss = self.loss_layer.forward(z, ys.reshape(-1))
        dH = self.head.backward(self.loss_layer.backward()).reshape(H.shape)
        self.rnn.backward(dH, truncate=truncate)
        return loss, H[-1]

    def nll(self, ids, m=32, chunk=256):
        """Per-character negative log-likelihood of predicting ids[1:] from
        ids[:-1], read as m parallel streams (each starting from the zero
        state) with the state carried across chunks. Returns an (m, L) array:
        row r is the stream starting at character r*L."""
        L = (ids.size - 1) // m
        xs = ids[:m * L].reshape(m, L).T
        ys = ids[1:m * L + 1].reshape(m, L).T
        out = np.empty((L, m))
        h = self.zero_state(m)
        for s in range(0, L, chunk):
            z, H = self.logits(xs[s:s + chunk], h)
            h = H[-1]
            z = z - z.max(axis=1, keepdims=True)
            logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
            y = ys[s:s + chunk].reshape(-1)
            out[s:s + chunk] = -logp[np.arange(y.size), y].reshape(-1, m)
        return out.T

    def evaluate(self, ids, m=32):
        """Mean loss in nats per character over the whole of `ids`."""
        return float(self.nll(ids, m).mean())

    def sample(self, chars, n=300, prime=SAMPLE_PRIME, temperature=1.0, seed=0):
        """Read `prime`, then generate n characters, each drawn from the
        model's distribution sharpened by 1/temperature (low: safe and
        repetitive; high: adventurous and misspelled)."""
        index = {c: i for i, c in enumerate(chars)}
        gen = np.random.default_rng(seed)
        h = self.zero_state(1)
        ids = [index[c] for c in prime]
        for i in ids[:-1]:
            h = self.rnn.forward(self.eye[[[i]]], h)[-1]
        out, i = [], ids[-1]
        for _ in range(n):
            h = self.rnn.forward(self.eye[[[i]]], h)[-1]
            z = self.head.forward(h)[0].astype(np.float64) / temperature
            p = np.exp(z - z.max())
            i = int(gen.choice(self.vocab, p=p / p.sum()))
            out.append(chars[i])
        return prime + "".join(out)


def _run_key(config):
    blob = json.dumps(config, sort_keys=True)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def run_char(epochs=CHAR_EPOCHS, n_h=CHAR_H, T=CHAR_T, B=CHAR_B, lr=CHAR_LR, clip=CHAR_CLIP,
             lr_drops=CHAR_LR_DROPS, truncate=0, train_n=None, seed=0, data=None,
             cache=False, verbose=False):
    """Train the character model; return (model, val_loss, history).

    The canonical run (all defaults) targets val loss <= CHAR_TARGET. history
    holds one dict per epoch: lr, train_loss (mean over the epoch's updates),
    val_loss (the whole validation text), the fraction of updates whose
    gradient was clipped, seconds, and a 300-character sample (temperature
    0.8, fixed seed) -- the model learning to write, epoch by epoch.
    truncate cuts the gradient inside each T-step chunk as well (0: none);
    train_n trains on only the first train_n characters.

    cache=True: weights and history are saved under
    data/shakespeare/checkpoints/ (gitignored), named by a hash of every
    setting here plus CHECKPOINT_VERSION; an identical later call loads them
    (history entries then carry "loaded": True).
    """
    config = dict(module="06-rnn", version=CHECKPOINT_VERSION, epochs=int(epochs),
                  n_h=int(n_h), T=int(T), B=int(B), lr=float(lr), clip=float(clip),
                  lr_drops=[int(d) for d in lr_drops], truncate=int(truncate),
                  train_n=None if train_n is None else int(train_n), seed=int(seed))
    ckpt = os.path.join(CHECKPOINTS, _run_key(config))
    train_ids, val_ids, chars = data if data is not None else load_text_data()
    model = CharRNN(len(chars), n_h, seed)
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
        print(f"[char] RNN({len(chars)} -> {n_h}) on {train_ids.size:,} characters, "
              f"{B} streams x {T} steps per update, {L // T} updates per epoch, "
              f"{sum(p.size for p in model.params()):,} params", flush=True)
    history = []
    for e in range(epochs):
        t0 = time.time()
        opt.lr = lr * 0.5 ** sum(e >= d for d in lr_drops)
        h = model.zero_state(B)
        losses, n_clipped = [], 0
        for s in range(0, L - T + 1, T):
            loss, h = model.loss_and_grads(xs_all[s:s + T], ys_all[s:s + T], h, truncate)
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


def ngram_losses(train_ids, val_ids, vocab, orders=(1, 2, 3, 4, 5, 6),
                 alphas=(1.0, 0.3, 0.1, 0.03, 0.01)):
    """The baseline an RNN must beat: count-based n-gram models.

    An order-n model predicts each character from the n-1 before it, by
    counting: P(c | context) = (count(context, c) + alpha) /
    (count(context) + alpha * vocab) -- add-alpha smoothing, so a pair never
    seen in training still gets a little probability. Returns
    {n: (val_loss, best_alpha)}, alpha chosen per order to minimize the
    validation loss (generous to the baseline). Contexts are packed into one
    int64 code, base `vocab` (65^6 < 2^63)."""
    def codes(ids, n):
        c = np.zeros(ids.size - n + 1, dtype=np.int64)
        for k in range(n):
            c = c * vocab + ids[k:ids.size - n + 1 + k]
        return c

    out = {}
    for n in orders:
        full_tr, full_va = codes(train_ids, n), codes(val_ids, n)
        ctx_tr, ctx_va = full_tr // vocab, full_va // vocab
        u_full, c_full = np.unique(full_tr, return_counts=True)
        u_ctx, c_ctx = np.unique(ctx_tr, return_counts=True)

        def lookup(u, c, q):
            k = np.clip(np.searchsorted(u, q), 0, u.size - 1)
            return np.where(u[k] == q, c[k], 0)

        n_full, n_ctx = lookup(u_full, c_full, full_va), lookup(u_ctx, c_ctx, ctx_va)
        best = min((float(-np.log((n_full + a) / (n_ctx + a * vocab)).mean()), a)
                   for a in alphas)
        out[n] = best
    return out


def word_position(text):
    """For every character: its position inside the word it belongs to (0 for
    a word's first letter, 1 for the second, ...), or -1 for characters that
    are not letters (spaces, punctuation, newlines). An apostrophe inside a
    word ("know't") counts as part of it."""
    pos, k = [], -1
    for i, c in enumerate(text):
        inner = c == "'" and 0 < i < len(text) - 1 and text[i - 1].isalpha() and text[i + 1].isalpha()
        if c.isalpha() or inner:
            k += 1
            pos.append(k)
        else:
            k = -1
            pos.append(-1)
    return np.array(pos)


# ============================================================================
# 5. The animation export.
# ============================================================================

def export_assets(payload, out_path=None):
    """Write topics/06-rnn/anim/rnn_assets.json for the Manim scene: the
    notebook assembles `payload` (delay sweeps, gradient profiles, samples)
    from its own runs, so the video and the chapter show the same numbers."""
    if out_path is None:
        out_path = os.path.join(_REPO, "topics", "06-rnn", "anim", "rnn_assets.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    print(f"[export] wrote {out_path}")


if __name__ == "__main__":
    for truncate in (0, 1):
        fp = run_toy(truncate, verbose=True)
        print(f"FINAL {'elman' if truncate else 'bptt'} "
              + " ".join(f"{name}={val!r}" for name, val in fp))
    print("FINAL rnn " + " ".join(f"{name}={val!r}" for name, val in rnn_fingerprint()))
    print("FINAL clip " + " ".join(f"{name}={val!r}" for name, val in clip_fingerprint()))

    # Tens of minutes of NumPy the first time; afterwards (or after the
    # notebook ran) the cached checkpoint is loaded and re-evaluated.
    data = load_text_data()
    model, _, _ = run_char(data=data, cache=True, verbose=True)
    train_ids, val_ids, chars = data
    val = model.evaluate(val_ids)
    best_ngram = min(loss for loss, _ in ngram_losses(train_ids, val_ids, len(chars)).values())
    print(f"\n[char] validation loss = {val:.4f} nats/char ({val / math.log(2):.3f} bits); "
          f"best n-gram {best_ngram:.4f}  (target <= {CHAR_TARGET})")
    print(model.sample(chars, n=400, temperature=0.8))
    assert val <= CHAR_TARGET, f"Tiny Shakespeare target missed: {val:.4f} > {CHAR_TARGET}"
    assert val < best_ngram, "the RNN should beat every n-gram baseline"
