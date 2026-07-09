"""
Module 01 - MLP + Backpropagation (Rumelhart, Hinton & Williams 1986)

The Python mirror of topics/01-mlp-backprop/c/mlp.c. Where Module 00's perceptron
hit the XOR wall -- no single line can split it -- this module stacks a *hidden
layer* in between and learns it with the chain rule. That is backpropagation:
run the input forward, measure the error, then push the blame backward through
every weight and nudge each one against its own gradient.

    forward:   z1 = W1 x + b1;  a1 = sigmoid(z1)
               z2 = W2 a1 + b2; a2 = sigmoid(z2) = yhat
    loss:      L  = -[ y log a2 + (1-y) log(1-a2) ]        (binary cross-entropy)
    backward:  dz2 = a2 - y                                (BCE through sigmoid)
               dW2 = dz2 a1^T ;  db2 = dz2
               dz1 = (W2^T dz2) * a1 (1-a1)                (sigmoid' = a(1-a))
               dW1 = dz1 x^T   ;  db1 = dz1
    update:    theta <- theta - lr * dL/dtheta            (gradient descent)

The XOR path below uses explicit Python loops with float() scalars so it runs
the *same operation order* as the C, which is what lets the C<->Python agreement
test hold to full double precision. The two-moons and MNIST paths (further down)
are vectorized NumPy -- the same math, batched for speed -- and are checked by an
accuracy metric rather than bit-for-bit.

Run directly to see XOR solved, then the two-moons and MNIST accuracies:
    python topics/01-mlp-backprop/python/mlp.py

Reference: docs/references/papers.md (Module 01).
"""

import math

import numpy as np

# ----------------------------------------------------------------------------
# Deterministic RNG: the same 64-bit LCG as Module 00. Weight initialization
# draws from it in a fixed order so the C mirror starts from identical weights.
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
# Sigmoid activation. Scalar version (math.exp) for the bit-exact XOR path;
# a numerically-stable vectorized version lives in the MNIST section below.
# ----------------------------------------------------------------------------
def sigmoid_scalar(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


# ============================================================================
# XOR: the bit-exact path (explicit loops, mirrors c/mlp.c line for line)
# ============================================================================

# Hyperparameters. Chosen so the fixed-seed run reliably drives XOR to 100%.
XOR_SEED = 1
N_HIDDEN = 4
INIT_SCALE = 1.0
XOR_LR = 0.5
XOR_EPOCHS = 20000


def make_xor():
    """The four XOR points and labels (arithmetic only -> bit-exact across C/Py)."""
    X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    y = [0.0, 1.0, 1.0, 0.0]
    return X, y


class MLP:
    """A 2 -> H -> 1 sigmoid network, written out with explicit loops.

    Parameters live in NumPy arrays (so the gradient check can iterate over them),
    but every forward/backward computation reads them as Python floats and runs in
    a fixed left-to-right order -- the exact order c/mlp.c uses.
    """

    def __init__(self, n_in: int, n_hidden: int, rng: Rng, scale: float = INIT_SCALE):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.W1 = np.zeros((n_hidden, n_in))  # hidden weights, row j = unit j
        self.b1 = np.zeros(n_hidden)          # hidden biases
        self.W2 = np.zeros(n_hidden)          # output weights (single output)
        self.b2 = np.zeros(1)                 # output bias
        # Draw in a fixed order: W1 row-major, then b1, then W2, then b2.
        for j in range(n_hidden):
            for k in range(n_in):
                self.W1[j, k] = scale * rng.signed()
        for j in range(n_hidden):
            self.b1[j] = scale * rng.signed()
        for j in range(n_hidden):
            self.W2[j] = scale * rng.signed()
        self.b2[0] = scale * rng.signed()

    # -- forward pass: input -> hidden (sigmoid) -> output (sigmoid) -----------
    def forward(self, x0: float, x1: float):
        a1 = [0.0] * self.n_hidden
        for j in range(self.n_hidden):
            z1 = float(self.b1[j]) + float(self.W1[j, 0]) * x0 + float(self.W1[j, 1]) * x1
            a1[j] = sigmoid_scalar(z1)              # a1_j = sigmoid(z1_j)
        z2 = float(self.b2[0])
        for j in range(self.n_hidden):
            z2 += float(self.W2[j]) * a1[j]         # z2 = W2 . a1 + b2
        a2 = sigmoid_scalar(z2)                     # yhat = sigmoid(z2)
        return a1, a2

    def predict(self, x0: float, x1: float) -> int:
        _, a2 = self.forward(x0, x1)
        return 1 if a2 >= 0.5 else 0

    # -- loss: mean binary cross-entropy over the dataset ---------------------
    def loss(self, X, y) -> float:
        n = len(y)
        total = 0.0
        for i in range(n):
            _, a2 = self.forward(float(X[i][0]), float(X[i][1]))
            yi = float(y[i])
            total += -(yi * math.log(a2) + (1.0 - yi) * math.log(1.0 - a2))
        return total / n

    # -- backward: the chain rule, accumulated over all samples ---------------
    def backward(self, X, y):
        n = len(y)
        dW1 = np.zeros_like(self.W1)
        db1 = np.zeros_like(self.b1)
        dW2 = np.zeros_like(self.W2)
        db2 = np.zeros_like(self.b2)
        for i in range(n):
            x0, x1, yi = float(X[i][0]), float(X[i][1]), float(y[i])
            a1, a2 = self.forward(x0, x1)
            # Output layer: for BCE through a sigmoid, dL/dz2 collapses to (a2 - y).
            dz2 = a2 - yi
            for j in range(self.n_hidden):
                dW2[j] += dz2 * a1[j]               # dL/dW2_j = dz2 * a1_j
            db2[0] += dz2                           # dL/db2   = dz2
            # Hidden layer: push dz2 back through W2, then through sigmoid'.
            for j in range(self.n_hidden):
                da1 = dz2 * float(self.W2[j])       # dL/da1_j = W2_j * dz2
                dz1 = da1 * a1[j] * (1.0 - a1[j])   # dL/dz1_j = da1_j * sigmoid'
                dW1[j, 0] += dz1 * x0               # dL/dW1_j0 = dz1_j * x0
                dW1[j, 1] += dz1 * x1               # dL/dW1_j1 = dz1_j * x1
                db1[j] += dz1                       # dL/db1_j  = dz1_j
        # Mean over the batch (matches the mean in loss()).
        dW1 /= n
        db1 /= n
        dW2 /= n
        db2 /= n
        return dW1, db1, dW2, db2

    # -- one full-batch gradient-descent step ---------------------------------
    def sgd_step(self, X, y, lr: float):
        dW1, db1, dW2, db2 = self.backward(X, y)
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2


def accuracy(mlp: MLP, X, y) -> float:
    n = len(y)
    correct = 0
    for i in range(n):
        if mlp.predict(float(X[i][0]), float(X[i][1])) == int(y[i]):
            correct += 1
    return correct / n


def fingerprint(mlp: MLP, X, y):
    """A small, ordered set of scalars that pins down the trained network.

    Returned as (name, value) pairs so the C<->Python agreement test can compare
    them one by one. `wsum` is the sum of every parameter -- a cheap checksum that
    catches any divergence anywhere in the two implementations.
    """
    wsum = float(mlp.W1.sum() + mlp.b1.sum() + mlp.W2.sum() + mlp.b2.sum())
    return (
        ("loss", mlp.loss(X, y)),
        ("acc", accuracy(mlp, X, y)),
        ("wsum", wsum),
        ("w1_00", float(mlp.W1[0, 0])),
        ("w2_0", float(mlp.W2[0])),
        ("b2", float(mlp.b2[0])),
    )


def run_xor(verbose: bool = False):
    """Train the MLP on XOR and return the fingerprint tuple for the agreement test."""
    rng = Rng(XOR_SEED)
    X, y = make_xor()
    mlp = MLP(2, N_HIDDEN, rng, scale=INIT_SCALE)
    if verbose:
        print("[xor] training a 2->%d->1 sigmoid MLP with backprop" % N_HIDDEN)
    for e in range(XOR_EPOCHS):
        mlp.sgd_step(X, y, XOR_LR)
        if verbose and (e % (XOR_EPOCHS // 10) == 0 or e == XOR_EPOCHS - 1):
            print("  epoch %6d  loss=%.6f  acc=%.3f" % (e, mlp.loss(X, y), accuracy(mlp, X, y)))
    return fingerprint(mlp, X, y)


# ============================================================================
# Two-moons and MNIST: the vectorized path (identical math, batched in NumPy)
#
# The per-sample derivation above is the whole story. To scale to hundreds or
# tens of thousands of examples we simply stack the samples into rows of a matrix
# and let NumPy run the same forward/backward over the whole batch at once. The
# output layer becomes a softmax over classes; for cross-entropy through softmax
# the output error has the *same* clean form as BCE through sigmoid: dz2 = P - Y.
# ============================================================================


def sigmoid(Z):
    """Vectorized, overflow-safe sigmoid (branch by sign of Z)."""
    out = np.empty_like(Z, dtype=np.float64)
    pos = Z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-Z[pos]))
    ez = np.exp(Z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def softmax(Z):
    """Row-wise softmax, shifted by the row max for numerical stability."""
    Z = Z - Z.max(axis=1, keepdims=True)
    e = np.exp(Z)
    return e / e.sum(axis=1, keepdims=True)


class MLPVec:
    """n_in -> H -> n_out: sigmoid hidden, softmax output, cross-entropy loss.

    The batched twin of `MLP`. Weights use a simple 1/sqrt(fan_in) scaling to keep
    the sigmoids in their responsive range (initialization is studied properly in
    Module 02). Forward stores the activations so backward can reuse them.
    """

    def __init__(self, n_in, n_hidden, n_out, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((n_in, n_hidden)) / math.sqrt(n_in)
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.standard_normal((n_hidden, n_out)) / math.sqrt(n_hidden)
        self.b2 = np.zeros(n_out)

    def forward(self, X):
        self.X = X
        self.A1 = sigmoid(X @ self.W1 + self.b1)      # hidden activations
        self.P = softmax(self.A1 @ self.W2 + self.b2)  # class probabilities
        return self.P

    def loss(self, X, y):
        P = self.forward(X)
        n = X.shape[0]
        return float(-np.log(P[np.arange(n), y] + 1e-12).mean())

    def backward(self, X, y):
        n = X.shape[0]
        P = self.forward(X)
        Y = np.zeros_like(P)
        Y[np.arange(n), y] = 1.0
        dZ2 = (P - Y) / n                    # softmax + CE: same form as BCE
        dW2 = self.A1.T @ dZ2
        db2 = dZ2.sum(axis=0)
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * self.A1 * (1.0 - self.A1)  # through sigmoid'
        dW1 = X.T @ dZ1
        db1 = dZ1.sum(axis=0)
        return dW1, db1, dW2, db2

    def sgd_step(self, X, y, lr):
        dW1, db1, dW2, db2 = self.backward(X, y)
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    def predict(self, X):
        return self.forward(X).argmax(axis=1)


def train_minibatch(model, Xtr, ytr, lr, epochs, batch_size, seed=0,
                    Xval=None, yval=None, verbose=False):
    """Plain mini-batch SGD: shuffle each epoch, step on each batch."""
    rng = np.random.default_rng(seed)
    n = Xtr.shape[0]
    for e in range(epochs):
        idx = rng.permutation(n)
        for s in range(0, n, batch_size):
            b = idx[s:s + batch_size]
            model.sgd_step(Xtr[b], ytr[b], lr)
        if verbose and Xval is not None:
            acc = float((model.predict(Xval) == yval).mean())
            print(f"  epoch {e:3d}  val_acc={acc:.4f}")
    return model


def make_two_moons(n=400, noise=0.2, seed=0):
    """Two interleaving half-circles -- not linearly separable, but smoothly
    separable by a bent boundary an MLP can learn. Uses sin/cos, so it lives on
    the Python/JS side only (not part of the bit-exact C<->Python gate)."""
    rng = np.random.default_rng(seed)
    n_out = n // 2
    n_in = n - n_out
    t_out = np.linspace(0.0, np.pi, n_out)
    t_in = np.linspace(0.0, np.pi, n_in)
    outer = np.stack([np.cos(t_out), np.sin(t_out)], axis=1)
    inner = np.stack([1.0 - np.cos(t_in), 0.5 - np.sin(t_in)], axis=1)
    X = np.vstack([outer, inner]) + rng.normal(0.0, noise, (n, 2))
    y = np.array([0] * n_out + [1] * n_in, dtype=np.int64)
    return X, y


def run_two_moons(verbose=False, n=400, n_hidden=16, lr=0.5, epochs=200, seed=0):
    """Train an MLP to bend a boundary around two moons; return test accuracy."""
    X, y = make_two_moons(n=n, noise=0.2, seed=seed)
    # 75/25 train/test split on a fixed shuffle.
    perm = np.random.default_rng(seed).permutation(n)
    cut = (3 * n) // 4
    tr, te = perm[:cut], perm[cut:]
    model = MLPVec(2, n_hidden, 2, seed=seed)
    train_minibatch(model, X[tr], y[tr], lr, epochs, batch_size=32, seed=seed,
                    Xval=X[te], yval=y[te], verbose=verbose)
    return float((model.predict(X[te]) == y[te]).mean())


def _load_mnist():
    import os
    import sys
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    sys.path.insert(0, os.path.join(repo_root, "data"))
    from get_mnist import load_mnist
    return load_mnist()


def run_mnist(verbose=False, n_hidden=128, lr=0.5, epochs=15, batch_size=64, seed=0):
    """Train the vectorized MLP on MNIST; return test accuracy (target >= 0.95)."""
    Xtr, ytr, Xte, yte = _load_mnist()
    model = MLPVec(784, n_hidden, 10, seed=seed)
    if verbose:
        print(f"[mnist] training a 784->{n_hidden}->10 sigmoid MLP "
              f"({epochs} epochs, lr={lr}, batch={batch_size})")
    train_minibatch(model, Xtr, ytr, lr, epochs, batch_size, seed=seed,
                    Xval=Xte, yval=yte, verbose=verbose)
    return float((model.predict(Xte) == yte).mean())


if __name__ == "__main__":
    fp = run_xor(verbose=True)
    print("FINAL xor " + " ".join(f"{name}={val!r}" for name, val in fp))

    acc = run_two_moons(verbose=False)
    print(f"\n[two-moons] test accuracy = {acc:.4f}")

    acc = run_mnist(verbose=True)
    print(f"\n[mnist] test accuracy = {acc:.4f}  (target >= 0.95)")
