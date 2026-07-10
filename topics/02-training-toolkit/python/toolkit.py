"""
Module 02 - Training toolkit: initialization, optimizers, regularization.

The Python mirror of topics/02-training-toolkit/c/toolkit.c, and the module that
stands up the shared `nanograd` library (lib/python/nanograd). Module 01 proved
a network *can* learn; this module is the box of tricks that makes it learn
*fast and reliably*: better activations (ReLU), fan-in-scaled initialization
(Xavier/He), smarter optimizers (momentum -> RMSProp -> Adam), and L2/early-stop
regularization.

Two kinds of code live here, exactly as in Module 01:

  1. run_toy() -- a *bit-exact* path. A 2->8->2 ReLU network with He init trained
     by Adam on XOR, written with explicit scalar loops in the same operation
     order as c/toolkit.c, drawing weights from the same 64-bit LCG. This is what
     the C<->Python agreement test pins to full double precision. It doubles as
     the smallest possible demo of He init + Adam working together.

  2. The ablations -- vectorized nanograd runs on two-moons and MNIST that answer
     "does this trick actually help?" by swapping one knob at a time. These are
     checked by a metric (final accuracy / loss), not bit-for-bit.

Run directly to see the toy solved, then the optimizer/init/activation ablations:
    python topics/02-training-toolkit/python/toolkit.py

Reference: docs/references/papers.md (Module 02).
"""

import math
import os
import sys

import numpy as np

# Make the shared nanograd library importable (lib/python on sys.path).
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(_REPO, "lib", "python"))

from nanograd import Adam, RMSProp, SGD, SoftmaxCrossEntropy, mlp  # noqa: E402
from nanograd.rng import Rng  # noqa: E402


# ============================================================================
# 1. The bit-exact toy: 2->8->2 ReLU, He init, Adam, on XOR.
#
# Everything below runs in explicit scalar loops in a fixed order so c/toolkit.c
# can reproduce every number. Weights are drawn from the shared LCG; the math is
# plain doubles. This is the C<->Python agreement gate.
# ============================================================================

TOY_SEED = 1
N_IN, N_HID, N_OUT = 2, 8, 2
TOY_STEPS = 2000
ADAM_LR, ADAM_B1, ADAM_B2, ADAM_EPS = 0.05, 0.9, 0.999, 1e-8

# XOR as a 2-class problem (labels are class indices for softmax cross-entropy).
TOY_X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
TOY_Y = [0, 1, 1, 0]


def _he_fill(rows, cols, rng, std):
    """Draw a rows x cols matrix (row-major) of N(0, std^2) -- matches init.py."""
    return [[std * rng.normal() for _ in range(cols)] for _ in range(rows)]


class ToyNet:
    """A 2->8->2 ReLU MLP with He init and Adam, all in explicit doubles.

    Mirrors c/toolkit.c line for line so the two agree bit-for-bit. Parameters,
    gradients, and Adam's moment buffers are plain Python lists of floats.
    """

    def __init__(self, rng):
        # He init: std = sqrt(2/fan_in). W1 fan_in=2 -> 1.0; W2 fan_in=8 -> 0.5.
        self.W1 = _he_fill(N_IN, N_HID, rng, math.sqrt(2.0 / N_IN))
        self.W2 = _he_fill(N_HID, N_OUT, rng, math.sqrt(2.0 / N_HID))
        self.b1 = [0.0] * N_HID
        self.b2 = [0.0] * N_OUT
        # Adam state: first/second moments for every parameter, plus running
        # powers of the betas for bias-correction (tracked by multiplication so
        # C and Python never call pow() and can't disagree on its last bit).
        self.mW1 = _zeros2(N_IN, N_HID); self.vW1 = _zeros2(N_IN, N_HID)
        self.mW2 = _zeros2(N_HID, N_OUT); self.vW2 = _zeros2(N_HID, N_OUT)
        self.mb1 = [0.0] * N_HID; self.vb1 = [0.0] * N_HID
        self.mb2 = [0.0] * N_OUT; self.vb2 = [0.0] * N_OUT
        self.b1_pow = 1.0
        self.b2_pow = 1.0

    def forward(self, x):
        """Return (z1, a1, probs) for one input x -- z1 kept for ReLU's mask."""
        z1 = [0.0] * N_HID
        a1 = [0.0] * N_HID
        for o in range(N_HID):
            s = self.b1[o]
            for k in range(N_IN):
                s += x[k] * self.W1[k][o]          # z1_o = b1_o + sum_k x_k W1_ko
            z1[o] = s
            a1[o] = s if s > 0.0 else 0.0          # ReLU
        logits = [0.0] * N_OUT
        for c in range(N_OUT):
            s = self.b2[c]
            for o in range(N_HID):
                s += a1[o] * self.W2[o][c]         # logit_c = b2_c + sum_o a1_o W2_oc
            logits[c] = s
        return z1, a1, _softmax(logits)

    def loss_and_acc(self):
        total, correct = 0.0, 0
        for i in range(len(TOY_Y)):
            _, _, p = self.forward(TOY_X[i])
            total += -math.log(p[TOY_Y[i]] + 1e-12)
            if _argmax(p) == TOY_Y[i]:
                correct += 1
        return total / len(TOY_Y), correct / len(TOY_Y)

    def step(self):
        """One full-batch Adam step. Gradients summed over the 4 XOR points,
        with the 1/m folded into dlogits, then Adam updates every parameter."""
        n = len(TOY_Y)
        gW1 = _zeros2(N_IN, N_HID); gb1 = [0.0] * N_HID
        gW2 = _zeros2(N_HID, N_OUT); gb2 = [0.0] * N_OUT

        for i in range(n):
            x = TOY_X[i]
            z1, a1, p = self.forward(x)
            dlogits = [p[c] - (1.0 if c == TOY_Y[i] else 0.0) for c in range(N_OUT)]
            for c in range(N_OUT):
                dlogits[c] /= n                    # mean loss -> divide once here
            # Output layer.
            for o in range(N_HID):
                for c in range(N_OUT):
                    gW2[o][c] += a1[o] * dlogits[c]
            for c in range(N_OUT):
                gb2[c] += dlogits[c]
            # Hidden layer: push back through W2, then through ReLU's mask.
            for o in range(N_HID):
                da1 = 0.0
                for c in range(N_OUT):
                    da1 += dlogits[c] * self.W2[o][c]
                dz1 = da1 if z1[o] > 0.0 else 0.0  # ReLU derivative
                for k in range(N_IN):
                    gW1[k][o] += x[k] * dz1
                gb1[o] += dz1

        # Advance the bias-correction powers once per step (t := t+1).
        self.b1_pow *= ADAM_B1
        self.b2_pow *= ADAM_B2
        b1c = 1.0 - self.b1_pow
        b2c = 1.0 - self.b2_pow

        _adam2(self.W1, gW1, self.mW1, self.vW1, N_IN, N_HID, b1c, b2c)
        _adam2(self.W2, gW2, self.mW2, self.vW2, N_HID, N_OUT, b1c, b2c)
        _adam1(self.b1, gb1, self.mb1, self.vb1, N_HID, b1c, b2c)
        _adam1(self.b2, gb2, self.mb2, self.vb2, N_OUT, b1c, b2c)

    def wsum(self):
        s = 0.0
        for k in range(N_IN):
            for o in range(N_HID):
                s += self.W1[k][o]
        for o in range(N_HID):
            for c in range(N_OUT):
                s += self.W2[o][c]
        for o in range(N_HID):
            s += self.b1[o]
        for c in range(N_OUT):
            s += self.b2[c]
        return s


def _zeros2(r, c):
    return [[0.0] * c for _ in range(r)]


def _softmax(logits):
    m = logits[0]
    for v in logits[1:]:
        if v > m:
            m = v
    e = [math.exp(v - m) for v in logits]
    s = 0.0
    for v in e:
        s += v
    return [v / s for v in e]


def _argmax(v):
    best, bi = v[0], 0
    for i in range(1, len(v)):
        if v[i] > best:
            best, bi = v[i], i
    return bi


def _adam_update(p, g, m, v, b1c, b2c):
    """Scalar Adam update; returns the new parameter value."""
    m = ADAM_B1 * m + (1.0 - ADAM_B1) * g
    v = ADAM_B2 * v + (1.0 - ADAM_B2) * (g * g)
    m_hat = m / b1c
    v_hat = v / b2c
    p = p - ADAM_LR * m_hat / (math.sqrt(v_hat) + ADAM_EPS)
    return p, m, v


def _adam1(p, g, m, v, n, b1c, b2c):
    for i in range(n):
        p[i], m[i], v[i] = _adam_update(p[i], g[i], m[i], v[i], b1c, b2c)


def _adam2(p, g, m, v, r, c, b1c, b2c):
    for i in range(r):
        for j in range(c):
            p[i][j], m[i][j], v[i][j] = _adam_update(p[i][j], g[i][j],
                                                     m[i][j], v[i][j], b1c, b2c)


def fingerprint(net):
    """Ordered scalars that pin down the trained toy net (for the agreement test)."""
    loss, acc = net.loss_and_acc()
    return (
        ("loss", loss),
        ("acc", acc),
        ("wsum", net.wsum()),
        ("w1_00", net.W1[0][0]),
        ("w2_00", net.W2[0][0]),
        ("b2_0", net.b2[0]),
    )


def run_toy(verbose=False):
    """Train the bit-exact toy and return its fingerprint tuple."""
    rng = Rng(TOY_SEED)
    net = ToyNet(rng)
    if verbose:
        print("[toy] 2->%d->2 ReLU, He init, Adam on XOR" % N_HID)
    for t in range(TOY_STEPS):
        net.step()
        if verbose and (t % (TOY_STEPS // 10) == 0 or t == TOY_STEPS - 1):
            loss, acc = net.loss_and_acc()
            print("  step %5d  loss=%.6f  acc=%.3f" % (t, loss, acc))
    return fingerprint(net)


# ============================================================================
# 2. Ablations: the toolkit in action on real-ish data, via vectorized nanograd.
#
# Same network, one knob changed at a time. Checked by accuracy/loss, not by
# bit-exactness (nanograd uses NumPy matmuls whose summation order differs from
# the scalar C path). This is where "does the trick help?" gets answered.
# ============================================================================

def make_two_moons(n=400, noise=0.2, seed=0):
    """Two interleaving half-circles (same generator as Module 01)."""
    rng = np.random.default_rng(seed)
    n_out = n // 2
    n_in = n - n_out
    t_out = np.linspace(0.0, np.pi, n_out)
    t_in = np.linspace(0.0, np.pi, n_in)
    outer = np.stack([np.cos(t_out), np.sin(t_out)], axis=1)
    inner = np.stack([1.0 - np.cos(t_in), 0.5 - np.sin(t_in)], axis=1)
    X = np.vstack([outer, inner]) + rng.normal(0.0, noise, (n, 2))
    y = np.array([0] * n_out + [1] * n_in, dtype=np.int64)
    return X.astype(np.float64), y


def _make_optimizer(name, lr, weight_decay=0.0):
    if name == "sgd":
        return SGD(lr=lr, weight_decay=weight_decay)
    if name == "momentum":
        return SGD(lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == "rmsprop":
        return RMSProp(lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return Adam(lr=lr, weight_decay=weight_decay)
    raise ValueError(name)


def train(net, opt, Xtr, ytr, epochs, batch_size=32, seed=0,
          Xval=None, yval=None, record=False):
    """Mini-batch training loop over a nanograd net. Returns a history list of
    (epoch, train_loss, val_acc) when `record`, else the final accuracy on the
    validation set (or on the training set when no validation set is given)."""
    loss_layer = SoftmaxCrossEntropy()
    rng = np.random.default_rng(seed)
    n = Xtr.shape[0]
    history = []
    for e in range(epochs):
        idx = rng.permutation(n)
        for s in range(0, n, batch_size):
            b = idx[s:s + batch_size]
            loss_layer.forward(net.forward(Xtr[b]), ytr[b])
            net.backward(loss_layer.backward())
            opt.step(net.params(), net.grads())
        if record:
            train_loss = loss_layer.forward(net.forward(Xtr), ytr)
            val_acc = _accuracy(net, Xval, yval) if Xval is not None else float("nan")
            history.append((e, train_loss, val_acc))
    if record:
        return history
    if Xval is not None:
        return _accuracy(net, Xval, yval)
    return _accuracy(net, Xtr, ytr)


def _accuracy(net, X, y):
    return float((net.forward(X).argmax(axis=1) == y).mean())


def _split(X, y, seed=0):
    n = X.shape[0]
    perm = np.random.default_rng(seed).permutation(n)
    cut = (3 * n) // 4
    return X[perm[:cut]], y[perm[:cut]], X[perm[cut:]], y[perm[cut:]]


# Reasonable per-optimizer learning rates (each near its own sweet spot, so the
# comparison is fair rather than rigged by one global rate).
OPT_LR = {"sgd": 0.5, "momentum": 0.2, "rmsprop": 0.01, "adam": 0.01}


def run_optimizer_ablation(epochs=60, seed=0, hidden=16):
    """Train the same ReLU/He net on two-moons with each optimizer; return
    {name: {'acc': final_val_acc, 'curve': [(epoch, loss, acc), ...]}}."""
    X, y = make_two_moons(n=400, noise=0.2, seed=seed)
    Xtr, ytr, Xte, yte = _split(X, y, seed=seed)
    out = {}
    for name in ("sgd", "momentum", "rmsprop", "adam"):
        net = mlp([2, hidden, 2], activation="relu", init="he", seed=seed)
        opt = _make_optimizer(name, OPT_LR[name])
        hist = train(net, opt, Xtr, ytr, epochs, batch_size=32, seed=seed,
                     Xval=Xte, yval=yte, record=True)
        out[name] = {"acc": hist[-1][2], "curve": hist}
    return out


def run_init_ablation(epochs=40, seed=0, hidden=64, depth=4):
    """Deeper tanh net where init matters: compare small-uniform vs Xavier."""
    X, y = make_two_moons(n=600, noise=0.2, seed=seed)
    Xtr, ytr, Xte, yte = _split(X, y, seed=seed)
    sizes = [2] + [hidden] * depth + [2]
    out = {}
    for init in ("small", "xavier"):
        net = mlp(sizes, activation="tanh", init=init, seed=seed)
        opt = Adam(lr=0.005)
        acc = train(net, opt, Xtr, ytr, epochs, batch_size=32, seed=seed,
                    Xval=Xte, yval=yte)
        out[init] = acc
    return out


def _load_mnist():
    sys.path.insert(0, os.path.join(_REPO, "data"))
    from get_mnist import load_mnist
    return load_mnist()


def run_mnist(optimizer="adam", epochs=8, hidden=128, seed=0, weight_decay=0.0,
              lr=None, verbose=False):
    """Train a 784->hidden->10 ReLU/He net on MNIST; return test accuracy."""
    Xtr, ytr, Xte, yte = _load_mnist()
    net = mlp([784, hidden, 10], activation="relu", init="he", seed=seed)
    opt = _make_optimizer(optimizer, OPT_LR[optimizer] if lr is None else lr,
                          weight_decay=weight_decay)
    if verbose:
        print(f"[mnist] 784->{hidden}->10 ReLU/He, {optimizer}, {epochs} epochs")
    train(net, opt, Xtr, ytr, epochs, batch_size=64, seed=seed,
          Xval=Xte, yval=yte)
    return _accuracy(net, Xte, yte)


if __name__ == "__main__":
    fp = run_toy(verbose=True)
    print("FINAL toy " + " ".join(f"{name}={val!r}" for name, val in fp))

    print("\n[optimizer ablation] two-moons, same ReLU/He net, 60 epochs:")
    for name, r in run_optimizer_ablation().items():
        print(f"  {name:9s} final val_acc = {r['acc']:.4f}")

    print("\n[init ablation] deep tanh net, small-uniform vs Xavier:")
    for init, acc in run_init_ablation().items():
        print(f"  {init:8s} val_acc = {acc:.4f}")

    acc = run_mnist(optimizer="adam", epochs=8, verbose=True)
    print(f"\n[mnist] Adam test accuracy = {acc:.4f}  (target >= 0.95)")
