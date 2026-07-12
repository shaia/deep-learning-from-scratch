"""
Module 03 - Convolutions / LeNet-5: weight sharing + locality for images.

The Python mirror of topics/03-lenet/c/lenet.c, and the module that teaches
nanograd to see: Conv2D, MaxPool2D/AvgPool2D and Flatten join the library
(lib/python/nanograd/conv.py, lib/c/nanograd/conv.c). Modules 01-02 treated an
image as 784 loose numbers; this module keeps the geometry, scans a small
learned kernel across it, and gets a better MNIST score with fewer than half
the parameters.

Three kinds of code live here:

  1. run_toy() -- the *agreement* path. A tiny CNN (conv 4@3x3 -> ReLU ->
     maxpool 2x2 -> linear 36->2) trained by Adam on the "bars" problem:
     8x8 images containing one bright vertical or horizontal bar. Written with
     explicit scalar loops in the same operation order as c/lenet.c, drawing
     weights from the same 64-bit LCG, so the C<->Python agreement test holds
     to a tight relative tolerance (1e-9 gate; ~1e-15 observed).

  2. naive_conv2d_forward/backward -- the definition of the operation as
     straight quadruple loops over NumPy arrays. The vectorized im2col layer
     in nanograd must match this to 1e-12 (tests/test_agreement.py gate 2),
     and the notebook teaches from it.

  3. build_lenet()/run_mnist() -- the vectorized real thing: a modernized
     LeNet-5 (ReLU, He, max-pool, Adam) on full MNIST, checked by metric
     (target: test accuracy >= 0.98 after 3 epochs).

Run directly to see the toy solved, then MNIST:
    python topics/03-lenet/python/lenet.py

Reference: docs/references/papers.md (Module 03) -- LeCun, Bottou, Bengio &
Haffner (1998), "Gradient-Based Learning Applied to Document Recognition".
"""

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
    Adam,
    AvgPool2D,
    Conv2D,
    Flatten,
    Linear,
    MaxPool2D,
    ReLU,
    Sequential,
    SoftmaxCrossEntropy,
    Tanh,
    he_normal,
    he_normal_conv2d,
    xavier_normal,
    xavier_normal_conv2d,
)
from nanograd.rng import Rng  # noqa: E402


# ============================================================================
# 1. The bit-exact toy: "bars".
#
# 16 one-channel 8x8 images. Even-indexed images contain a bright VERTICAL bar
# (class 0), odd-indexed a HORIZONTAL bar (class 1), over faint uniform noise.
# The smallest problem where locality pays: a single oriented 3x3 edge kernel,
# slid everywhere, solves it -- and after training you can *look at* the
# kernels and see that they became exactly that.
#
# Everything runs in explicit scalar loops on flat row-major arrays in a fixed
# order so c/lenet.c can reproduce every number. The canonical loop-nest
# contract (shared with the C file and lib/*/nanograd comments):
#   conv forward: for i, f, u, v { acc = b[f]; for c, p, q: acc += K*X }
#   conv backward: same i,f,u,v outer nest; db/dK/dX accumulate in c,p,q.
# ============================================================================

TOY_SEED = 1          # weight draws
DATA_SEED = 2         # data draws (independent stream)
TOY_N = 16            # images
IMG = 8               # image height = width
TOY_F, TOY_KS = 4, 3  # conv: 4 filters, 3x3
CONV_OUT = IMG - TOY_KS + 1        # 6x6 feature maps (valid, stride 1)
POOL_OUT = CONV_OUT // 2           # 3x3 after 2x2 max-pool
N_FLAT = TOY_F * POOL_OUT * POOL_OUT   # 36 inputs to the linear head
N_OUT = 2
TOY_STEPS = 300
ADAM_LR, ADAM_B1, ADAM_B2, ADAM_EPS = 0.01, 0.9, 0.999, 1e-8


def make_bars():
    """The toy dataset, in the exact draw order c/lenet.c replays.

    Per image i: one uniform for the bar position, then 64 uniforms
    (row-major) for the noise floor, then the bar is overwritten at full
    brightness. Class is i % 2 (no draw spent on it): even = vertical bar
    at column `pos`, odd = horizontal bar at row `pos`.
    Returns X:(16,1,8,8) float64, y:(16,) int64.
    """
    rng = Rng(DATA_SEED)
    X = np.empty((TOY_N, 1, IMG, IMG), dtype=np.float64)
    y = np.empty(TOY_N, dtype=np.int64)
    for i in range(TOY_N):
        y[i] = i % 2
        pos = int(rng.uniform() * IMG)
        for r in range(IMG):
            for c in range(IMG):
                X[i, 0, r, c] = 0.25 * rng.uniform()
        if y[i] == 0:
            for r in range(IMG):
                X[i, 0, r, pos] = 1.0     # vertical bar
        else:
            for c in range(IMG):
                X[i, 0, pos, c] = 1.0     # horizontal bar
    return X, y


class ToyCNN:
    """conv(1->4, 3x3) -> ReLU -> maxpool 2x2 -> linear(36->2), in flat
    doubles with explicit loops, mirroring c/lenet.c line for line.

    Parameters live in flat row-major lists exactly like the C buffers:
      K1[((f*1 + c)*3 + p)*3 + q], b1[f], W2[j*2 + cls], b2[cls].
    Draw order at init: K1 (He, flat), then W2 (He, row-major) -- one RNG.
    """

    def __init__(self, rng, X, y):
        # Flat images: x[i][ (r*IMG + c) ] with C=1 (NCHW flattened).
        self.X = [[float(v) for v in X[i, 0].reshape(-1)] for i in range(TOY_N)]
        self.y = [int(v) for v in y]
        # He init. Conv fan_in = C*KH*KW = 9; linear fan_in = 36.
        k_std = math.sqrt(2.0 / (1 * TOY_KS * TOY_KS))
        w_std = math.sqrt(2.0 / N_FLAT)
        self.K1 = [k_std * rng.normal() for _ in range(TOY_F * TOY_KS * TOY_KS)]
        self.b1 = [0.0] * TOY_F
        self.W2 = [w_std * rng.normal() for _ in range(N_FLAT * N_OUT)]
        self.b2 = [0.0] * N_OUT
        # Adam state (moments + running beta powers; powers advance by
        # multiplication, never pow(), so C and Python cannot disagree).
        self.mK1 = [0.0] * len(self.K1); self.vK1 = [0.0] * len(self.K1)
        self.mb1 = [0.0] * TOY_F;        self.vb1 = [0.0] * TOY_F
        self.mW2 = [0.0] * len(self.W2); self.vW2 = [0.0] * len(self.W2)
        self.mb2 = [0.0] * N_OUT;        self.vb2 = [0.0] * N_OUT
        self.b1_pow = 1.0
        self.b2_pow = 1.0

    # -- forward pieces, per sample (flat buffers, canonical loop order) ----

    def forward(self, x):
        """Return (z, pooled, argmax, probs) for one flat 8x8 image.

        z: conv pre-activations, flat (4,6,6); pooled: flat (4,3,3) after
        ReLU + 2x2 max-pool; argmax: winning flat z-index per pooled cell
        (backward routes gradients through it); probs: softmax over 2 logits.
        """
        # Convolution, cross-correlation form: no kernel flip.
        z = [0.0] * (TOY_F * CONV_OUT * CONV_OUT)
        for f in range(TOY_F):
            for u in range(CONV_OUT):
                for v in range(CONV_OUT):
                    acc = self.b1[f]
                    for p in range(TOY_KS):           # C=1: channel loop drops
                        for q in range(TOY_KS):
                            acc += (self.K1[(f * TOY_KS + p) * TOY_KS + q]
                                    * x[(u + p) * IMG + (v + q)])
                    z[(f * CONV_OUT + u) * CONV_OUT + v] = acc
        # ReLU then 2x2 max-pool, keeping the winner's index for backward.
        # Strict > keeps the FIRST max in p,q scan order -- same tie-break as
        # NumPy's argmax and the C mirror.
        pooled = [0.0] * N_FLAT
        argmax = [0] * N_FLAT
        for f in range(TOY_F):
            for u2 in range(POOL_OUT):
                for v2 in range(POOL_OUT):
                    best, bi = -1.0, -1
                    for p in range(2):
                        for q in range(2):
                            zi = (f * CONV_OUT + (2 * u2 + p)) * CONV_OUT + (2 * v2 + q)
                            a = z[zi] if z[zi] > 0.0 else 0.0   # ReLU
                            if bi < 0 or a > best:
                                best, bi = a, zi
                    j = (f * POOL_OUT + u2) * POOL_OUT + v2
                    pooled[j] = best
                    argmax[j] = bi
        # Flatten is a no-op on the flat buffer; straight to the linear head.
        logits = [0.0] * N_OUT
        for cls in range(N_OUT):
            s = self.b2[cls]
            for j in range(N_FLAT):
                s += pooled[j] * self.W2[j * N_OUT + cls]
            logits[cls] = s
        return z, pooled, argmax, _softmax(logits)

    def loss_and_acc(self):
        total, correct = 0.0, 0
        for i in range(TOY_N):
            _, _, _, p = self.forward(self.X[i])
            total += -math.log(p[self.y[i]] + 1e-12)
            if _argmax(p) == self.y[i]:
                correct += 1
        return total / TOY_N, correct / TOY_N

    def step(self):
        """One full-batch Adam step; gradients summed over the 16 images with
        the 1/m folded into dlogits, exactly as in Modules 01-02."""
        gK1 = [0.0] * len(self.K1); gb1 = [0.0] * TOY_F
        gW2 = [0.0] * len(self.W2); gb2 = [0.0] * N_OUT

        for i in range(TOY_N):
            x = self.X[i]
            z, pooled, argmax, p = self.forward(x)
            dlogits = [(p[c] - (1.0 if c == self.y[i] else 0.0)) / TOY_N
                       for c in range(N_OUT)]
            # Linear head.
            dpool = [0.0] * N_FLAT
            for j in range(N_FLAT):
                for cls in range(N_OUT):
                    gW2[j * N_OUT + cls] += pooled[j] * dlogits[cls]
                    dpool[j] += dlogits[cls] * self.W2[j * N_OUT + cls]
            for cls in range(N_OUT):
                gb2[cls] += dlogits[cls]
            # Max-pool backward: the winner takes the whole gradient...
            dz = [0.0] * len(z)
            for j in range(N_FLAT):
                # ...but only if it survived ReLU (z > 0); else the gradient
                # dies here, same as ReLU's mask.
                if z[argmax[j]] > 0.0:
                    dz[argmax[j]] += dpool[j]
            # Conv backward: same f,u,v nest as forward; dK is a correlation
            # of the input with the upstream gradient. (dX is not needed --
            # this conv touches the data directly.)
            for f in range(TOY_F):
                for u in range(CONV_OUT):
                    for v in range(CONV_OUT):
                        g = dz[(f * CONV_OUT + u) * CONV_OUT + v]
                        if g == 0.0:
                            continue
                        gb1[f] += g
                        for p_ in range(TOY_KS):
                            for q in range(TOY_KS):
                                gK1[(f * TOY_KS + p_) * TOY_KS + q] += (
                                    g * x[(u + p_) * IMG + (v + q)])

        # Advance bias-correction powers once per step (t := t+1).
        self.b1_pow *= ADAM_B1
        self.b2_pow *= ADAM_B2
        b1c = 1.0 - self.b1_pow
        b2c = 1.0 - self.b2_pow
        _adam(self.K1, gK1, self.mK1, self.vK1, b1c, b2c)
        _adam(self.b1, gb1, self.mb1, self.vb1, b1c, b2c)
        _adam(self.W2, gW2, self.mW2, self.vW2, b1c, b2c)
        _adam(self.b2, gb2, self.mb2, self.vb2, b1c, b2c)

    def wsum(self):
        """Sum of all parameters in canonical order: K1, b1, W2, b2."""
        s = 0.0
        for v in self.K1:
            s += v
        for v in self.b1:
            s += v
        for v in self.W2:
            s += v
        for v in self.b2:
            s += v
        return s


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


def _adam(p, g, m, v, b1c, b2c):
    """In-place Adam over parallel flat lists (identical to c/lenet.c)."""
    for i in range(len(p)):
        m[i] = ADAM_B1 * m[i] + (1.0 - ADAM_B1) * g[i]
        v[i] = ADAM_B2 * v[i] + (1.0 - ADAM_B2) * (g[i] * g[i])
        m_hat = m[i] / b1c
        v_hat = v[i] / b2c
        p[i] -= ADAM_LR * m_hat / (math.sqrt(v_hat) + ADAM_EPS)


def fingerprint(net):
    """Ordered scalars that pin down the trained toy net (agreement test)."""
    loss, acc = net.loss_and_acc()
    return (
        ("loss", loss),
        ("acc", acc),
        ("wsum", net.wsum()),
        ("k1_0", net.K1[0]),
        ("w2_00", net.W2[0]),
        ("b2_0", net.b2[0]),
    )


def build_toy_cnn(rng=None):
    """The vectorized twin of ToyCNN: same layers, same He draws in the same
    order (K1 flat, then W2 row-major) from the same seed, but running on
    nanograd's im2col path. tests/test_agreement.py gate 3 pins its initial
    loss to the scalar mirror's at 1e-9."""
    if rng is None:
        rng = Rng(TOY_SEED)
    conv = Conv2D(1, TOY_F, TOY_KS, TOY_KS)
    head = Linear(N_FLAT, N_OUT)
    he_normal_conv2d(conv.K, rng)
    he_normal(head.W, rng)
    return Sequential([conv, ReLU(), MaxPool2D(2), Flatten(), head])


def run_toy(verbose=False):
    """Train the bit-exact toy CNN and return its fingerprint tuple."""
    X, y = make_bars()
    net = ToyCNN(Rng(TOY_SEED), X, y)
    if verbose:
        print("[toy] conv(1->%d, %dx%d) -> ReLU -> maxpool -> linear(%d->2), "
              "Adam on bars" % (TOY_F, TOY_KS, TOY_KS, N_FLAT))
    for t in range(TOY_STEPS):
        net.step()
        if verbose and (t % (TOY_STEPS // 10) == 0 or t == TOY_STEPS - 1):
            loss, acc = net.loss_and_acc()
            print("  step %4d  loss=%.6f  acc=%.3f" % (t, loss, acc))
    return fingerprint(net)


# ============================================================================
# 2. The naive reference: convolution as its own definition.
#
# Straight loops over NumPy arrays -- slow, obvious, and the ground truth the
# vectorized im2col layer is tested against (must agree to 1e-12). This is
# also the version the notebook and the assignment teach from.
# ============================================================================

def naive_conv2d_forward(X, K, b):
    """Y[i,f,u,v] = b[f] + sum_{c,p,q} K[f,c,p,q] * X[i,c,u+p,v+q].

    Valid cross-correlation, stride 1. X:(n,C,H,W) K:(F,C,KH,KW) b:(F,)
    -> Y:(n,F,H-KH+1,W-KW+1).
    """
    n, C, H, W = X.shape
    F, _, KH, KW = K.shape
    OH, OW = H - KH + 1, W - KW + 1
    Y = np.empty((n, F, OH, OW), dtype=X.dtype)
    for i in range(n):
        for f in range(F):
            for u in range(OH):
                for v in range(OW):
                    acc = b[f]
                    for c in range(C):
                        for p in range(KH):
                            for q in range(KW):
                                acc += K[f, c, p, q] * X[i, c, u + p, v + q]
                    Y[i, f, u, v] = acc
    return Y


def naive_conv2d_backward(X, K, dY):
    """The three conv gradients, straight from the chain rule (summed over
    the batch). Returns (dX, dK, db). Same loop nest as forward."""
    n, C, H, W = X.shape
    F, _, KH, KW = K.shape
    OH, OW = H - KH + 1, W - KW + 1
    dX = np.zeros_like(X)
    dK = np.zeros_like(K)
    db = np.zeros(F, dtype=X.dtype)
    for i in range(n):
        for f in range(F):
            for u in range(OH):
                for v in range(OW):
                    g = dY[i, f, u, v]
                    db[f] += g
                    for c in range(C):
                        for p in range(KH):
                            for q in range(KW):
                                # Each Y cell touched K[f,c,p,q] once (times
                                # X) and X[i,c,u+p,v+q] once (times K); the
                                # chain rule just runs those products back.
                                dK[f, c, p, q] += g * X[i, c, u + p, v + q]
                                dX[i, c, u + p, v + q] += g * K[f, c, p, q]
    return dX, dK, db


# ============================================================================
# 3. The real thing: a modernized LeNet-5 on MNIST, vectorized.
#
#   conv 6@5x5 -> ReLU -> maxpool 2x2 -> conv 16@5x5 -> ReLU -> maxpool 2x2
#   -> flatten(256) -> 120 -> ReLU -> 84 -> ReLU -> 10 logits
#
# The 1998 shape (6/16 feature maps, 120/84 head) with the Module-02 toolkit
# swapped in for the 1998 machinery: ReLU for scaled tanh, max-pool for
# trainable average pooling, He init, Adam, softmax cross-entropy for RBF
# units. 44,426 parameters -- fewer than half the Module-01 MLP's 101,770.
# Checked by metric, not bit-exactness. Pass activation="tanh", pool="avg",
# init="xavier" to get the notebook's historical ablation.
# ============================================================================

def build_lenet(rng=None, seed=0, activation="relu", pool="max", init="he"):
    """Assemble the LeNet Sequential; weights drawn in a fixed layer order."""
    if rng is None:
        rng = Rng(seed)
    act = {"relu": ReLU, "tanh": Tanh}[activation]
    pool_cls = {"max": MaxPool2D, "avg": AvgPool2D}[pool]
    conv_init = {"he": he_normal_conv2d, "xavier": xavier_normal_conv2d}[init]
    lin_init = {"he": he_normal, "xavier": xavier_normal}[init]

    conv1, conv2 = Conv2D(1, 6, 5, 5), Conv2D(6, 16, 5, 5)
    fc1, fc2, fc3 = Linear(256, 120), Linear(120, 84), Linear(84, 10)
    for layer in (conv1, conv2):
        conv_init(layer.K, rng)
    for layer in (fc1, fc2, fc3):
        lin_init(layer.W, rng)
    return Sequential([
        conv1, act(), pool_cls(2),
        conv2, act(), pool_cls(2),
        Flatten(),
        fc1, act(),
        fc2, act(),
        fc3,                       # bare logits for SoftmaxCrossEntropy
    ])


def n_params(net):
    return sum(p.size for p in net.params())


def _load_mnist():
    sys.path.insert(0, os.path.join(_REPO, "data"))
    from get_mnist import load_mnist
    return load_mnist()


def load_mnist_images():
    """MNIST with the geometry restored: (n,784) rows -> (n,1,28,28) images."""
    Xtr, ytr, Xte, yte = _load_mnist()
    return (Xtr.reshape(-1, 1, 28, 28), ytr,
            Xte.reshape(-1, 1, 28, 28), yte)


def predict(net, X, batch=256):
    """Class predictions in batches (a full-test-set im2col would eat RAM)."""
    out = np.empty(X.shape[0], dtype=np.int64)
    for s in range(0, X.shape[0], batch):
        out[s:s + batch] = net.forward(X[s:s + batch]).argmax(axis=1)
    return out


def accuracy(net, X, y, batch=256):
    return float((predict(net, X, batch) == y).mean())


def run_mnist(epochs=3, batch=128, lr=1e-3, train_n=None, seed=0,
              activation="relu", pool="max", init="he", verbose=False):
    """Train the LeNet on MNIST; return (net, test_acc, history).

    history is a list of (epoch, mean_train_loss, test_acc). The canonical
    config (full 60k, 3 epochs, batch 128, Adam 1e-3) targets >= 0.98 test
    accuracy in a few minutes of NumPy.
    """
    Xtr, ytr, Xte, yte = load_mnist_images()
    if train_n is not None:
        Xtr, ytr = Xtr[:train_n], ytr[:train_n]
    net = build_lenet(seed=seed, activation=activation, pool=pool, init=init)
    opt = Adam(lr=lr)
    loss_layer = SoftmaxCrossEntropy()
    shuffle = np.random.default_rng(seed)
    if verbose:
        print(f"[mnist] LeNet ({activation}/{pool}/{init}), {n_params(net)} "
              f"params, {Xtr.shape[0]} train images, {epochs} epochs")
    history = []
    for e in range(epochs):
        t0 = time.time()
        idx = shuffle.permutation(Xtr.shape[0])
        total, batches = 0.0, 0
        for s in range(0, Xtr.shape[0], batch):
            b = idx[s:s + batch]
            total += loss_layer.forward(net.forward(Xtr[b]), ytr[b])
            batches += 1
            net.backward(loss_layer.backward())
            opt.step(net.params(), net.grads())
        test_acc = accuracy(net, Xte, yte)
        history.append((e, total / batches, test_acc))
        if verbose:
            print(f"  epoch {e}  train_loss={total / batches:.4f}  "
                  f"test_acc={test_acc:.4f}  ({time.time() - t0:.0f}s)")
    return net, history[-1][2], history


# ============================================================================
# 4. Asset export: the trained first-layer filters + a few digits, as JSON,
# for the site widget (ConvWidget) and the Manim scene -- one source of truth.
# ============================================================================

def export_filters(out_path=None, net=None):
    """Write conv1's six trained 5x5 kernels and three sample test digits to
    site/src/components/viz/lenetFilters.json. Trains the canonical net if
    one isn't passed in. Run manually once per retrain:
        python topics/03-lenet/python/lenet.py --export-filters
    """
    if out_path is None:
        out_path = os.path.join(_REPO, "site", "src", "components", "viz",
                                "lenetFilters.json")
    if net is None:
        net, acc, _ = run_mnist(verbose=True)
        print(f"[export] trained to test_acc={acc:.4f}")
    _, _, Xte, yte = load_mnist_images()
    # Three visually distinct digits: the first 7, 4 and 0 in the test set.
    digits = []
    for target in (7, 4, 0):
        i = int(np.argmax(yte == target))
        digits.append([[int(round(v * 255)) for v in row] for row in Xte[i, 0]])
    conv1 = net.layers[0]
    filters = [[[round(float(v), 4) for v in row] for row in k[0]]
               for k in conv1.K]                      # (6,5,5); C=1 dropped
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"filters": filters, "digits": digits}, fh)
    print(f"[export] wrote {out_path}")


if __name__ == "__main__":
    if "--export-filters" in sys.argv:
        export_filters()
        sys.exit(0)

    fp = run_toy(verbose=True)
    print("FINAL toy " + " ".join(f"{name}={val!r}" for name, val in fp))

    net, acc, _ = run_mnist(verbose=True)
    print(f"\n[mnist] LeNet test accuracy = {acc:.4f}  (target >= 0.98)")
    assert acc >= 0.98, f"MNIST target missed: {acc:.4f} < 0.98"
