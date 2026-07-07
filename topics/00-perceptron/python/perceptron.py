"""
Module 00 - The Perceptron (1943 McCulloch-Pitts, 1957 Rosenblatt)

The Python mirror of topics/00-perceptron/c/perceptron.c. It follows the C code
line-for-line on purpose: the same deterministic RNG, the same data, the same
update order. That is what lets the C<->Python agreement test hold to full
double precision (see topics/00-perceptron/tests/test_agreement.py).

Intuition: a perceptron forms z = w . x + b and fires (predicts 1) when z >= 0.
Learning is Rosenblatt's rule -- whenever it is wrong, nudge w toward the truth:

    w <- w + lr * (y - yhat) * x
    b <- b + lr * (y - yhat)

Linearly-separable classes: it converges. XOR: it is stuck forever (the wall
that motivates Module 01's hidden layer).

Run directly to see both experiments:
    python topics/00-perceptron/python/perceptron.py

Reference: docs/references/papers.md (Module 00).
"""

import numpy as np

# ----------------------------------------------------------------------------
# Deterministic RNG: a 64-bit LCG identical to the C version. We emulate 64-bit
# unsigned overflow with an explicit mask so the bit stream matches exactly.
# ----------------------------------------------------------------------------
_MASK64 = (1 << 64) - 1


class Rng:
    def __init__(self, seed: int):
        self.state = seed & _MASK64

    def next_u64(self) -> int:
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) & _MASK64
        return self.state

    def uniform(self) -> float:  # [0, 1)
        return (self.next_u64() >> 11) * (1.0 / 9007199254740992.0)

    def signed(self) -> float:  # [-1, 1)
        return 2.0 * self.uniform() - 1.0


# ----------------------------------------------------------------------------
# Dataset: two 2D blobs, arithmetic only (no transcendentals) so it matches C
# bit-for-bit. Class 0 at (-2,-2), class 1 at (+2,+2); ideal line is x0+x1=0.
# ----------------------------------------------------------------------------
N_PER_CLASS = 50
N_SAMPLES = 2 * N_PER_CLASS
SPREAD = 1.5


def make_blobs(rng: Rng):
    centers = ((-2.0, -2.0), (2.0, 2.0))
    X = np.zeros((N_SAMPLES, 2), dtype=np.float64)
    y = np.zeros(N_SAMPLES, dtype=np.int64)
    i = 0
    for c in (0, 1):
        for _ in range(N_PER_CLASS):
            X[i, 0] = centers[c][0] + SPREAD * rng.signed()
            X[i, 1] = centers[c][1] + SPREAD * rng.signed()
            y[i] = c
            i += 1
    return X, y


# ----------------------------------------------------------------------------
# The perceptron: two weights and a bias, held in a tiny mutable record.
# ----------------------------------------------------------------------------
class Perceptron:
    def __init__(self):
        self.w0 = 0.0
        self.w1 = 0.0
        self.b = 0.0

    def preactivation(self, x0, x1):  # z = w0*x0 + w1*x1 + b
        return self.w0 * x0 + self.w1 * x1 + self.b

    def predict(self, x0, x1):  # yhat = step(z)
        return 1 if self.preactivation(x0, x1) >= 0.0 else 0


def accuracy(p: Perceptron, X, y, n) -> float:
    correct = 0
    for i in range(n):
        if p.predict(float(X[i, 0]), float(X[i, 1])) == int(y[i]):
            correct += 1
    return correct / n


def train_epoch(p: Perceptron, lr, X, y, n):
    """One pass of Rosenblatt's rule over all samples, fixed order (mirrors C)."""
    for i in range(n):
        x0 = float(X[i, 0])
        x1 = float(X[i, 1])
        yhat = p.predict(x0, x1)
        err = float(int(y[i]) - yhat)  # -1, 0, or +1
        p.w0 += lr * err * x0
        p.w1 += lr * err * x1
        p.b += lr * err


# ----------------------------------------------------------------------------
# Experiment 1: blobs. Returns (w0, w1, b, acc) so the test can compare to C.
# ----------------------------------------------------------------------------
def run_blobs(verbose: bool = False):
    rng = Rng(42)
    X, y = make_blobs(rng)
    p = Perceptron()
    lr, epochs = 0.1, 20
    if verbose:
        print("[blobs] training a perceptron on two linearly-separable classes")
    for e in range(epochs):
        train_epoch(p, lr, X, y, N_SAMPLES)
        if verbose and (e % 5 == 0 or e == epochs - 1):
            print(f"  epoch {e:2d}  acc={accuracy(p, X, y, N_SAMPLES):.3f}  "
                  f"w=({p.w0:.3f}, {p.w1:.3f})  b={p.b:.3f}")
    return p.w0, p.w1, p.b, accuracy(p, X, y, N_SAMPLES)


# ----------------------------------------------------------------------------
# Experiment 2: the XOR wall. Not linearly separable -> accuracy gets stuck.
# ----------------------------------------------------------------------------
def run_xor(verbose: bool = False):
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    y = np.array([0, 1, 1, 0], dtype=np.int64)
    p = Perceptron()
    lr, epochs = 0.1, 100
    if verbose:
        print("\n[xor] training the same perceptron on XOR (not separable)")
    for _ in range(epochs):
        for i in range(4):
            x0 = float(X[i, 0])
            x1 = float(X[i, 1])
            yhat = p.predict(x0, x1)
            err = float(int(y[i]) - yhat)
            p.w0 += lr * err * x0
            p.w1 += lr * err * x1
            p.b += lr * err
    acc = accuracy(p, X, y, 4)
    if verbose:
        print(f"  after {epochs} epochs acc={acc:.3f}  <- stuck below 1.0: the XOR wall")
    return p.w0, p.w1, p.b, acc


if __name__ == "__main__":
    w0, w1, b, acc = run_blobs(verbose=True)
    print(f"FINAL blobs w0={w0!r} w1={w1!r} b={b!r} acc={acc!r}")
    w0, w1, b, acc = run_xor(verbose=True)
    print(f"FINAL xor w0={w0!r} w1={w1!r} b={b!r} acc={acc!r}")
