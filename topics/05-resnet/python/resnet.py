"""
Module 05 - ResNet: let the gradient skip, and train a network of any depth.

The Python mirror of topics/05-resnet/c/resnet.c. AlexNet (Module 04) had
eight layers; VGG went to nineteen and got better -- and then deeper plain
networks got WORSE, on their own training data. This module reproduces that
"degradation problem", then fixes it the way He, Zhang, Ren & Sun did in 2015:
every block adds its input back to its output, y = relu(F(x) + x). nanograd
gains BatchNorm, residual blocks, the option-(A) zero-padding shortcut, global
average pooling and bias-free layers, in both languages.

Five kinds of code live here:

  1. run_toy() -- the *agreement* path, on "deep spirals": 128 points on two
     interleaved spirals, classified by a 34-layer MLP -- a stem, 16 blocks
     of [Linear -> BatchNorm -> ReLU -> Linear -> BatchNorm], a head --
     trained twice from the SAME initial weights: once as a plain network,
     once with every block's input added back (the residual twin). Same
     parameters, same arithmetic, one `+` apart. ToyResNet runs every sum in
     c/resnet.c's loop order (the elementwise work is vectorized: IEEE +, -,
     *, / and sqrt round each element the same whatever order the elements
     are visited in, so only the sums need the C's order), so C and Python
     agree to a tight relative tolerance (1e-9 gate).

  2. build_toy_net()/train_toy_vectorized() -- the same networks on
     nanograd's vectorized layers, for the notebook's depth sweeps, gradient
     measurements and seeds.

  3. naive_* and the fingerprints -- BatchNorm by its definition (and its
     backward by the full Jacobian), plus bn_fingerprint()/block_fingerprint():
     nanograd's BatchNorm (train + eval) and a downsampling conv residual
     block with global average pooling, run on fixed LCG-drawn tensors so
     the C mirror's versions are pinned too.

  4. build_resnet()/run_cifar() -- the real thing: the paper's CIFAR-10
     networks (6n+2 layers: a conv, three stages of n basic blocks at widths
     16/32/64, global average pooling, one Linear), residual or plain,
     vectorized, float32, SGD + momentum, checked by metric and cached.

  5. lesioned()/export_assets() -- deleting blocks from trained networks
     (Veit et al. 2016), and the data the Manim scene draws.

Run directly to see the toy, then CIFAR-10:
    python topics/05-resnet/python/resnet.py

Reference: docs/references/papers.md (Module 05) -- He, Zhang, Ren & Sun
(2015), "Deep Residual Learning for Image Recognition"; Ioffe & Szegedy
(2015), "Batch Normalization".
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
    SGD,
    BatchNorm,
    Conv2D,
    GlobalAvgPool2D,
    Linear,
    ReLU,
    Residual,
    Sequential,
    SoftmaxCrossEntropy,
    ZeroPadShortcut,
    he_normal,
    he_normal_conv2d,
    random_crop_flip,
)
from nanograd.rng import Rng  # noqa: E402


# ============================================================================
# 1. The bit-exact toy: "deep spirals".
#
# Two spirals, one per class, wound 1.5 times around each other: a problem a
# small network solves easily -- which is the point. Up to about 18 layers a
# plain network fits all 128 points; past that, the SAME network, made
# deeper, fits them worse. Nothing about the data changed and a deeper network
# can represent everything a shallower one can (let the extra blocks compute
# the identity), so a deeper net that does worse on its own training set has
# an optimization problem, not a capacity or an overfitting one. That is the
# degradation problem of He et al. (2015), in 128 points.
#
# Architecture (WIDTH = 16 units per layer, DEPTH = 16 blocks, 34 layers):
#   stem   Linear(2 -> 16, bias) -> ReLU
#   block  Linear(16 -> 16) -> BN -> ReLU -> Linear(16 -> 16) -> BN
#          -> (+ the block's input, if residual) -> ReLU
#   head   Linear(16 -> 2, bias) -> softmax cross-entropy
# Block Linears have no bias: BatchNorm's beta is the shift. Full-batch SGD
# with momentum, so BatchNorm's batch statistics are the whole data set's.
# ============================================================================

TOY_SEED = 1          # weight draws
DATA_SEED = 2         # the spirals
N_TOY = 128
TURNS = 1.5
WIDTH = 16
DEPTH = 16                         # blocks -> 2*DEPTH + 2 = 34 layers
TOY_STEPS = 400
TOY_LR, TOY_MOM = 0.05, 0.9
BN_EPS, BN_MOM = 1e-5, 0.1


def make_spirals(n=N_TOY, seed=DATA_SEED, turns=TURNS):
    """Two interleaved spirals, in the exact draw order c/resnet.c replays.

    Point i has class i % 2 and one uniform draw t: radius 0.15 + 0.85 t,
    angle 2*pi*turns*t + pi*class -- class 1 is class 0 rotated half a turn.
    Returns X:(n, 2) float64, y:(n,) int64.
    """
    rng = Rng(seed)
    X = np.empty((n, 2), dtype=np.float64)
    y = np.empty(n, dtype=np.int64)
    for i in range(n):
        y[i] = i % 2
        t = rng.uniform()
        r = 0.15 + 0.85 * t
        a = 2.0 * math.pi * turns * t + math.pi * int(y[i])
        X[i, 0] = r * math.cos(a)
        X[i, 1] = r * math.sin(a)
    return X, y


# -- the C-order building blocks --------------------------------------------
# Each is one lib/c/nanograd function. Elementwise work is vectorized freely;
# every SUM goes through _loop_sum, which adds in exactly the order the C's
# for-loop does: np.add.accumulate is defined as the running sum
# r[0] = t[0], r[i] = r[i-1] + t[i] -- the loop, done by NumPy.

def _loop_sum(terms, start):
    """((start + t[0]) + t[1]) + ... over axis 0: the C loop `s = start;
    for (i ...) s += t[i];`, vectorized over every other axis."""
    return np.add.accumulate(np.concatenate([start[None], terms]), axis=0)[-1]


def _lin_fwd(X, W, b=None):
    """ng_linear_forward: Y[i,o] = b[o] + sum_k X[i,k] W[k,o], k ascending
    (from 0.0 when the layer has no bias)."""
    m, n_out = X.shape[0], W.shape[1]
    start = np.zeros((m, n_out)) if b is None else np.tile(b, (m, 1))
    return _loop_sum(X.T[:, :, None] * W[:, None, :], start)       # terms (k, i, o)


def _lin_bwd(X, W, dY, need_dX=True, bias=False):
    """ng_linear_backward: dW[k,o] and db[o] summed over i ascending, dX[i,k]
    over o ascending, each from 0.0."""
    dW = _loop_sum(X[:, :, None] * dY[:, None, :], np.zeros_like(W))  # (i, k, o)
    db = _loop_sum(dY, np.zeros(W.shape[1])) if bias else None
    dX = None
    if need_dX:
        terms = np.moveaxis(dY[:, None, :] * W[None, :, :], 2, 0)      # (o, i, k)
        dX = _loop_sum(terms, np.zeros_like(X))
    return dX, dW, db


def _bn_fwd(Z, gamma, beta, run_mean, run_var, training):
    """ng_batchnorm_forward with HW = 1. Updates the running stats in place
    when training. Returns (Y, XHAT, invstd)."""
    m, C = Z.shape
    if training:
        mean = _loop_sum(Z, np.zeros(C)) / m
        d = Z - mean
        var = _loop_sum(d * d, np.zeros(C)) / m
        run_mean[:] = (1.0 - BN_MOM) * run_mean + BN_MOM * mean
        run_var[:] = (1.0 - BN_MOM) * run_var + BN_MOM * (var * m / (m - 1))
    else:
        mean, var = run_mean.copy(), run_var.copy()
    invstd = 1.0 / np.sqrt(var + BN_EPS)
    xhat = (Z - mean) * invstd
    return gamma * xhat + beta, xhat, invstd


def _bn_bwd(xhat, gamma, invstd, dY):
    """ng_batchnorm_backward (training mode): returns (dX, dgamma, dbeta)."""
    m, C = dY.shape
    sb = _loop_sum(dY, np.zeros(C))
    sg = _loop_sum(dY * xhat, np.zeros(C))
    kc = gamma * invstd / m
    return kc * (m * dY - sb - xhat * sg), sg, sb


def _relu(Z):
    return np.where(Z > 0.0, Z, 0.0)


def _softmax_ce(logits, y):
    """ng_softmax_ce_forward/backward on (m, 2) logits, with the C library's
    exp and log (math.*), summed in i order. Returns (loss, P, dlogits)."""
    m, k = logits.shape
    P = np.empty_like(logits)
    total = 0.0
    for i in range(m):
        row = logits[i].tolist()
        mx = row[0]
        for c in range(1, k):
            if row[c] > mx:
                mx = row[c]
        e = [math.exp(v - mx) for v in row]
        s = 0.0
        for v in e:
            s += v
        for c in range(k):
            P[i, c] = e[c] / s
        total += -math.log(P[i, int(y[i])] + 1e-12)
    onehot = np.zeros_like(P)
    onehot[np.arange(m), y] = 1.0
    return total / m, P, (P - onehot) / m


class ToyResNet:
    """The toy network, plain or residual, mirroring c/resnet.c call for call.

    Parameters, in canonical order (the C struct's, and nanograd's
    params() order for build_toy_net):
      W0 (2x16), b0 | per block: W1, g1, be1, W2, g2, be2 | Wh (16x2), bh.
    Draw order at init (one He stream): W0, then W1 and W2 of each block in
    turn, then Wh. Biases and betas start at 0, gammas at 1. `residual`
    switches the `+ x` on -- nothing else differs between the twins.
    """

    def __init__(self, rng, X, y, residual, depth=DEPTH, width=WIDTH):
        self.X, self.y = X, y
        self.residual, self.depth = residual, depth
        self.W0 = self._he(rng, 2, width)
        self.b0 = np.zeros(width)
        self.blocks = []
        for _ in range(depth):
            W1 = self._he(rng, width, width)
            W2 = self._he(rng, width, width)
            self.blocks.append({
                "W1": W1, "g1": np.ones(width), "be1": np.zeros(width),
                "W2": W2, "g2": np.ones(width), "be2": np.zeros(width),
                "rm1": np.zeros(width), "rv1": np.ones(width),
                "rm2": np.zeros(width), "rv2": np.ones(width)})
        self.Wh = self._he(rng, width, 2)
        self.bh = np.zeros(2)
        self.vel = [np.zeros_like(p) for p in self.params()]

    @staticmethod
    def _he(rng, n_in, n_out):
        std = math.sqrt(2.0 / n_in)
        W = np.empty((n_in, n_out))
        for i in range(n_in):
            for o in range(n_out):
                W[i, o] = std * rng.normal()
        return W

    def params(self):
        ps = [self.W0, self.b0]
        for b in self.blocks:
            ps += [b["W1"], b["g1"], b["be1"], b["W2"], b["g2"], b["be2"]]
        return ps + [self.Wh, self.bh]

    def forward(self, X, training):
        """Logits, plus the caches backward needs."""
        H0 = _lin_fwd(X, self.W0, self.b0)
        A = _relu(H0)
        caches = []
        for b in self.blocks:
            Z1 = _lin_fwd(A, b["W1"])
            U1, XH1, IS1 = _bn_fwd(Z1, b["g1"], b["be1"], b["rm1"], b["rv1"], training)
            A1 = _relu(U1)
            Z2 = _lin_fwd(A1, b["W2"])
            U2, XH2, IS2 = _bn_fwd(Z2, b["g2"], b["be2"], b["rm2"], b["rv2"], training)
            S = U2 + A if self.residual else U2      # the one difference
            caches.append((A, U1, XH1, IS1, A1, XH2, IS2, S))
            A = _relu(S)
        return _lin_fwd(A, self.Wh, self.bh), (H0, A, caches)

    def evaluate(self):
        """(loss, accuracy) in evaluation mode: BatchNorm's running stats."""
        logits, _ = self.forward(self.X, training=False)
        loss, P, _ = _softmax_ce(logits, self.y)
        return loss, float((P.argmax(1) == self.y).mean())

    def step(self):
        """One full-batch SGD + momentum step; returns the batch loss."""
        loss, grads = self.loss_and_grads()
        for p, g, v in zip(self.params(), grads, self.vel):
            v[...] = TOY_MOM * v + g           # ng_sgd_step, weight decay 0
            p -= TOY_LR * v
        return loss

    def loss_and_grads(self):
        """Training-mode forward + the reversed walk: (batch loss, gradients
        in params() order)."""
        logits, (H0, A, caches) = self.forward(self.X, training=True)
        loss, _, dlogits = _softmax_ce(logits, self.y)
        dA, gWh, gbh = _lin_bwd(A, self.Wh, dlogits, bias=True)
        grads = []
        for b, (Ain, U1, XH1, IS1, A1, XH2, IS2, S) in zip(reversed(self.blocks),
                                                           reversed(caches)):
            dS = np.where(S > 0.0, dA, 0.0)
            dZ2, gg2, gbe2 = _bn_bwd(XH2, b["g2"], IS2, dS)
            dA1, gW2, _ = _lin_bwd(A1, b["W2"], dZ2)
            dU1 = np.where(U1 > 0.0, dA1, 0.0)
            dZ1, gg1, gbe1 = _bn_bwd(XH1, b["g1"], IS1, dU1)
            dAin, gW1, _ = _lin_bwd(Ain, b["W1"], dZ1)
            # The fork: the block's input fed the body AND the shortcut, so
            # its gradient is the sum of both routes back.
            dA = dAin + dS if self.residual else dAin
            grads = [gW1, gg1, gbe1, gW2, gg2, gbe2] + grads
        dH0 = np.where(H0 > 0.0, dA, 0.0)
        _, gW0, gb0 = _lin_bwd(self.X, self.W0, dH0, need_dX=False, bias=True)
        return loss, [gW0, gb0] + grads + [gWh, gbh]

    def wsum(self):
        """Sum of every parameter in canonical order (a plain left-to-right
        loop, as the C does)."""
        s = 0.0
        for p in self.params():
            for v in p.reshape(-1).tolist():
                s += v
        return s


def fingerprint(net, batch_loss):
    """Ordered scalars that pin down a trained toy net (agreement test)."""
    loss, acc = net.evaluate()
    last = net.blocks[-1]
    return (
        ("loss", loss),
        ("acc", acc),
        ("batch_loss", batch_loss),
        ("wsum", net.wsum()),
        ("w0_00", float(net.W0[0, 0])),
        ("w1_00", float(net.blocks[0]["W1"][0, 0])),
        ("g2_last", float(last["g2"][0])),
        ("rv2_last", float(last["rv2"][0])),
    )


def run_toy(residual, verbose=False, steps=TOY_STEPS):
    """Train one bit-exact toy twin and return its fingerprint tuple."""
    X, y = make_spirals()
    net = ToyResNet(Rng(TOY_SEED), X, y, residual)
    name = "res  " if residual else "plain"
    if verbose:
        print(f"[toy {name}] {2 * DEPTH + 2} layers ({DEPTH} blocks of width {WIDTH}"
              f"{', each adding its input back' if residual else ''}); "
              f"full-batch SGD lr={TOY_LR} momentum={TOY_MOM}")
    loss = None
    for t in range(steps):
        loss = net.step()
        if verbose and (t % (steps // 4) == 0 or t == steps - 1):
            print(f"  step {t:4d}  batch_loss={loss:.6f}")
    return fingerprint(net, loss)


# -- the vectorized twin ------------------------------------------------------

def toy_block(width, rng, residual):
    """One block of the toy, on nanograd layers: the residual block, or its
    plain twin (the same five layers and a ReLU, no shortcut)."""
    l1 = Linear(width, width, bias=False)
    l2 = Linear(width, width, bias=False)
    he_normal(l1.W, rng)
    he_normal(l2.W, rng)
    body = [l1, BatchNorm(width, BN_EPS, BN_MOM), ReLU(),
            l2, BatchNorm(width, BN_EPS, BN_MOM)]
    return Residual(body) if residual else Sequential(body + [ReLU()])


def build_toy_net(depth=DEPTH, residual=True, rng=None, width=WIDTH, in_dim=2):
    """The vectorized twin of ToyResNet: same layers, same He draws in the
    same order from the same seed, so params() lines up one-for-one."""
    if rng is None:
        rng = Rng(TOY_SEED)
    stem = Linear(in_dim, width)
    he_normal(stem.W, rng)
    blocks = [toy_block(width, rng, residual) for _ in range(depth)]
    head = Linear(width, 2)
    he_normal(head.W, rng)
    return Sequential([stem, ReLU(), *blocks, head])


def block_grad_norms(net):
    """||dW|| of each block's first Linear, input side first -- read after a
    backward pass. How hard each block is being pushed."""
    out = []
    for layer in net.layers:
        if isinstance(layer, (Residual, Sequential)):
            body = layer.body if isinstance(layer, Residual) else layer
            out.append(float(np.linalg.norm(body.layers[0].dW)))
    return out


def train_toy_vectorized(depth=DEPTH, residual=True, seed=0, steps=TOY_STEPS,
                         lr=TOY_LR, n=N_TOY):
    """The toy on nanograd's fast path. `seed` varies the weights and the
    spirals together (seed=0 is run_toy's data and init). Returns a dict:
    loss (batch loss per step), acc (final, evaluation mode) and grad_norms
    (block_grad_norms at the first step)."""
    X, y = make_spirals(n, DATA_SEED + 100 * seed)
    net = build_toy_net(depth, residual, Rng(TOY_SEED + 100 * seed))
    loss_layer = SoftmaxCrossEntropy()
    opt = SGD(lr=lr, momentum=TOY_MOM)
    losses, grad_norms = [], None
    for t in range(steps):
        net.train()
        losses.append(loss_layer.forward(net.forward(X), y))
        net.backward(loss_layer.backward())
        if t == 0:
            grad_norms = block_grad_norms(net)
        opt.step(net.params(), net.grads())
    net.eval()
    acc = float((net.forward(X).argmax(1) == y).mean())
    return {"loss": losses, "acc": acc, "grad_norms": grad_norms, "net": net}


def input_gradients(depth, residual, n_grid=400, seed=0, width=WIDTH):
    """Balduzzi et al. (2017)'s "shattered gradients", at initialization.

    Walk the input along a line, x = (s, 0) for s in [-1, 1], and record
    d(logit 1)/ds at every point. BatchNorm is first calibrated on the grid
    itself (running stats = the grid's statistics), then frozen (eval mode),
    so each point's gradient is its own. Returns (s, grads).
    """
    net = build_toy_net(depth, residual, Rng(TOY_SEED + 100 * seed), width)
    s = np.linspace(-1.0, 1.0, n_grid)
    G = np.stack([s, np.zeros_like(s)], axis=1)
    bns = _find(net, BatchNorm)
    for bn in bns:
        bn.momentum = 1.0                 # one pass: running stats = the grid's
    net.train()
    net.forward(G)
    net.eval()
    out = net.forward(G)
    dY = np.zeros_like(out)
    dY[:, 1] = 1.0
    dX = net.backward(dY)
    return s, dX[:, 0]


def _find(layer, cls):
    """Every layer of type `cls` inside a (nested) network, in order."""
    if isinstance(layer, cls):
        return [layer]
    if isinstance(layer, Sequential):
        return [x for sub in layer.layers for x in _find(sub, cls)]
    if isinstance(layer, Residual):
        return _find(layer.body, cls)
    return []


# ============================================================================
# 2. The naive references.
# ============================================================================

def naive_batchnorm_forward(X, gamma, beta, eps=BN_EPS):
    """Ioffe & Szegedy's Algorithm 1, literally, per channel over every
    other axis. X: (m, C) or (n, C, H, W). Returns Y."""
    Xc = np.moveaxis(X, 1, -1).reshape(-1, X.shape[1])     # (m, C)
    Y = np.empty_like(Xc)
    for c in range(Xc.shape[1]):
        col = Xc[:, c]
        mu = sum(col) / len(col)
        var = sum((v - mu) ** 2 for v in col) / len(col)
        Y[:, c] = gamma[c] * (col - mu) / math.sqrt(var + eps) + beta[c]
    shape = (X.shape[0],) + X.shape[2:] + (X.shape[1],)
    return np.moveaxis(Y.reshape(shape), -1, 1)


def naive_batchnorm_backward(X, gamma, dY, eps=BN_EPS):
    """dL/dX by the full Jacobian: for each channel, build the m x m matrix
    J[i, j] = dy_i/dx_j = gamma * invstd * ([i=j] - 1/m - xhat_i xhat_j / m)
    and multiply. O(m^2) per channel -- the definition, not the fast way."""
    Xc = np.moveaxis(X, 1, -1).reshape(-1, X.shape[1])
    Dc = np.moveaxis(dY, 1, -1).reshape(-1, X.shape[1])
    dXc = np.empty_like(Xc)
    m = Xc.shape[0]
    for c in range(Xc.shape[1]):
        col = Xc[:, c]
        mu = col.mean()
        invstd = 1.0 / math.sqrt(((col - mu) ** 2).mean() + eps)
        xhat = (col - mu) * invstd
        J = gamma[c] * invstd * (np.eye(m) - 1.0 / m - np.outer(xhat, xhat) / m)
        dXc[:, c] = J.T @ Dc[:, c]
    shape = (X.shape[0],) + X.shape[2:] + (X.shape[1],)
    return np.moveaxis(dXc.reshape(shape), -1, 1)


# ============================================================================
# 3. Fingerprints of the conv-side machinery, for the C mirror. c/resnet.c
# prints the same checksums from the same LCG streams; the agreement test
# compares them. (Vectorized here, loops there: the sums run in different
# orders, so they agree to ~1e-15, well inside the 1e-9 gate.)
# ============================================================================

BN_SEED = 6
BN_SHAPE = (2, 3, 4, 4)            # n, C, H, W
BLOCK_SEED = 7
BLOCK_SHAPE = (2, 2, 6, 6)         # the block widens 2 -> 4 channels, stride 2


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


def bn_fingerprint():
    """BatchNorm2d on (2, 3, 4, 4): a training-mode forward + backward (which
    also updates the running stats), then an evaluation-mode forward +
    backward on a second input. Draw order from Rng(BN_SEED): X, gamma, beta,
    dY, X2, dY2."""
    rng = Rng(BN_SEED)
    C = BN_SHAPE[1]
    X = _draw(rng, BN_SHAPE, 3.0, 1.0)
    bn = BatchNorm(C, BN_EPS, BN_MOM)
    bn.gamma[:] = _draw(rng, (C,), 0.5, 1.0)
    bn.beta[:] = _draw(rng, (C,))
    dY = _draw(rng, BN_SHAPE)
    X2 = _draw(rng, BN_SHAPE, 3.0, 1.0)
    dY2 = _draw(rng, BN_SHAPE)
    Y = bn.forward(X)
    dX = bn.backward(dY)
    dg, db = bn.dgamma.copy(), bn.dbeta.copy()
    bn.training = False
    Y2 = bn.forward(X2)
    dX2 = bn.backward(dY2)
    return (_sums("y", Y) + _sums("dx", dX) + _sums("dg", dg) + _sums("db", db)
            + _sums("rm", bn.running_mean) + _sums("rv", bn.running_var)
            + _sums("ye", Y2) + _sums("dxe", dX2))


def build_test_block(rng):
    """A downsampling basic block (2 -> 4 channels, stride 2) with the
    option-(A) shortcut, followed by global average pooling. Draw order: K1,
    K2 (He), then gamma1, beta1, gamma2, beta2 (each 0.5*signed + 1 /
    signed)."""
    c1 = Conv2D(2, 4, 3, 3, stride=2, pad=1, bias=False)
    c2 = Conv2D(4, 4, 3, 3, pad=1, bias=False)
    he_normal_conv2d(c1.K, rng)
    he_normal_conv2d(c2.K, rng)
    bn1, bn2 = BatchNorm(4, BN_EPS, BN_MOM), BatchNorm(4, BN_EPS, BN_MOM)
    for bn in (bn1, bn2):
        bn.gamma[:] = _draw(rng, (4,), 0.5, 1.0)
        bn.beta[:] = _draw(rng, (4,))
    blk = Residual([c1, bn1, ReLU(), c2, bn2], ZeroPadShortcut(2, 4, 2))
    return Sequential([blk, GlobalAvgPool2D()])


def block_fingerprint():
    """Forward + backward through build_test_block on X ~ Rng(BLOCK_SEED)
    (drawn after the block's parameters), with upstream gradient dY ~ the
    same stream: output, input gradient, and every parameter gradient."""
    rng = Rng(BLOCK_SEED)
    net = build_test_block(rng)
    X = _draw(rng, BLOCK_SHAPE, 2.0)
    dY = _draw(rng, (BLOCK_SHAPE[0], 4))
    Y = net.forward(X)
    dX = net.backward(dY)
    blk = net.layers[0]
    c1, bn1, _, c2, bn2 = blk.body.layers
    return (_sums("y", Y) + _sums("dx", dX) + _sums("dk1", c1.dK) + _sums("dk2", c2.dK)
            + _sums("dg1", bn1.dgamma) + _sums("db2", bn2.dbeta))


# ============================================================================
# 4. The real thing: the paper's CIFAR-10 ResNets (Section 4.2).
#
#   32x32x3 -> conv 3x3, 16 -> BN -> ReLU                        32x32
#           -> n basic blocks, 16 channels                        32x32
#           -> n basic blocks, 32 channels (the first at stride 2) 16x16
#           -> n basic blocks, 64 channels (the first at stride 2)  8x8
#           -> global average pool -> Linear(64 -> 10)
#   basic block: conv3x3 -> BN -> ReLU -> conv3x3 -> BN (+ shortcut) -> ReLU
#
# 6n + 2 weight layers: n = 3, 5, 9 give the paper's ResNet-20/32/56. Where a
# block halves the resolution and doubles the width, the shortcut is option
# (A) -- subsample + zero channels -- so a ResNet has EXACTLY the plain
# network's parameters (residual=False drops every shortcut, nothing else).
# The paper's recipe, kept: He init, BN after every conv, no dropout, SGD
# momentum 0.9, lr 0.1 divided by 10 twice, batch 128, 4-pixel pad + random
# 32x32 crop + flip. Scaled to a laptop: tens of epochs, not the paper's 182,
# and weight decay 5e-4 (the paper's 1e-4 is tuned for the long schedule).
# ============================================================================

CIFAR_WIDTHS = (16, 32, 64)
# Threads for im2col/col2im (bit-identical at any count; see nanograd/conv.py).
WORKERS = max(1, min(8, (os.cpu_count() or 1) // 2))


def build_resnet(n=3, residual=True, widths=CIFAR_WIDTHS, seed=0, dtype=np.float32,
                 workers=WORKERS, cache_cols=None):
    """The CIFAR ResNet (or its plain twin) with 6n+2 layers; kernels drawn
    He-normal in layer order, then the classifier, then cast to `dtype`.

    workers / cache_cols are Conv2D's speed and memory knobs (see conv.py);
    neither changes a single bit of the result. cache_cols=None keeps the
    patch matrices for nets up to 20 layers and rebuilds them in backward for
    deeper ones, whose patch matrices would not fit in a laptop's memory."""
    rng = Rng(seed)
    if cache_cols is None:
        cache_cols = 6 * n + 2 <= 20

    def conv(cin, cout, stride=1, first=False):
        c = Conv2D(cin, cout, 3, 3, stride=stride, pad=1, bias=False,
                   input_grad=not first, cache_cols=cache_cols, workers=workers)
        he_normal_conv2d(c.K, rng)
        return c

    w0 = widths[0]
    layers = [conv(3, w0, first=True), BatchNorm(w0), ReLU()]
    cin = w0
    for stage, w in enumerate(widths):
        for k in range(n):
            stride = 2 if (stage > 0 and k == 0) else 1
            body = [conv(cin, w, stride), BatchNorm(w), ReLU(), conv(w, w), BatchNorm(w)]
            if residual:
                shortcut = ZeroPadShortcut(cin, w, stride) if (stride != 1 or cin != w) else None
                layers.append(Residual(body, shortcut))
            else:
                layers.append(Sequential(body + [ReLU()]))
            cin = w
    fc = Linear(widths[-1], 10)
    he_normal(fc.W, rng)
    layers += [GlobalAvgPool2D(), fc]
    return Sequential(layers).astype(dtype)


def n_params(net):
    return sum(p.size for p in net.params())


def n_layers(net):
    """Weight layers (convs + Linears), the depth the paper counts."""
    return len(_find(net, Conv2D)) + len(_find(net, Linear))


def load_cifar():
    """CIFAR-10 as uint8 NCHW (n,3,32,32) + labels, plus the training set's
    per-channel mean/std (in [0,1] units) for standardizing each batch."""
    sys.path.insert(0, os.path.join(_REPO, "data"))
    from get_cifar10 import load_cifar10
    Xtr, ytr, Xte, yte = load_cifar10()
    mean = Xtr.mean(axis=(0, 2, 3)) / 255.0
    var = np.array([((Xtr[:, c].astype(np.float32) / 255.0) ** 2).mean()
                    for c in range(3)]) - mean ** 2
    return Xtr, ytr, Xte, yte, mean, np.sqrt(var)


def standardize(X_uint8, mean, std, dtype=np.float32):
    """uint8 images -> per-channel standardized floats."""
    X = X_uint8.astype(dtype) / 255.0
    return (X - mean.astype(dtype)[None, :, None, None]) / std.astype(dtype)[None, :, None, None]


def pad_crop_flip(X_uint8, rng, pad=4):
    """The paper's CIFAR augmentation: pad 4 black pixels on every side,
    take a random 32x32 crop, mirror half the time (Module 04's
    random_crop_flip, on the padded image)."""
    Xp = np.pad(X_uint8, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    return random_crop_flip(Xp, X_uint8.shape[2], rng)


def downsample(X_uint8, factor=2, chunk=5000):
    """Average-pool uint8 images by `factor` (32x32 -> 16x16 for factor 2),
    rounded back to uint8: the cheap CIFAR the depth experiment trains on."""
    n, C, H, W = X_uint8.shape
    out = np.empty((n, C, H // factor, W // factor), dtype=np.uint8)
    for s in range(0, n, chunk):
        blk = X_uint8[s:s + chunk].astype(np.float32)
        blk = blk.reshape(-1, C, H // factor, factor, W // factor, factor).mean(axis=(3, 5))
        out[s:s + chunk] = np.rint(blk).astype(np.uint8)
    return out


def evaluate(net, X_uint8, y, mean, std, batch=500):
    """Test accuracy on the whole 32x32 image, BatchNorm in evaluation mode."""
    net.eval()
    correct = 0
    for s in range(0, X_uint8.shape[0], batch):
        logits = net.forward(standardize(X_uint8[s:s + batch], mean, std))
        correct += int((logits.argmax(axis=1) == y[s:s + batch]).sum())
    return correct / X_uint8.shape[0]


def cifar_grad_norms(net, X_uint8, y, mean, std):
    """One training-mode forward/backward on a batch: ||dK|| of every conv,
    input side first -- the gradient each layer would get on its first step."""
    loss_layer = SoftmaxCrossEntropy()
    net.train()
    loss_layer.forward(net.forward(standardize(X_uint8, mean, std)), y)
    net.backward(loss_layer.backward())
    norms = [float(np.linalg.norm(c.dK)) for c in _find(net, Conv2D)]
    # The forward pass nudged BatchNorm's running stats; restore none of it --
    # callers measure on a freshly built network they then discard.
    return norms


# The degradation experiment (the paper's Figure 6, redone): plain and
# residual nets of 20 and 56 layers, trained for 3,128 SGD steps on all 50,000
# images. To fit a laptop, at half the paper's width and on 16x16 photos (each
# pixel the mean of a 2x2 block) -- degradation is about depth, not width or
# resolution. All four warm up at lr 0.01 for 400 steps, the paper's own
# recipe for its deepest network. The notebook and anything that pre-trains
# its runs must pass exactly these settings, or the checkpoint names differ.
FIG6 = dict(epochs=8, lr_drops=(5, 7), widths=(8, 16, 32), size=16, warmup=400, eval_n=2000)
FIG6_DEPTHS = (3, 9)                    # n blocks per stage -> 20 and 56 layers

TRACE_EVERY = 40                        # steps per point of history's training-error trace

CHECKPOINTS = os.path.join(_REPO, "data", "cifar10", "checkpoints")

# Part of every checkpoint's key (Module 04 used 1; these live in the same
# folder under different hashes). Bump after changing build_resnet or the
# training loop so stale checkpoints stop matching.
CHECKPOINT_VERSION = 5


def _run_key(config):
    blob = json.dumps(config, sort_keys=True)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def run_cifar(n=3, residual=True, epochs=8, batch=128, lr=0.1, momentum=0.9,
              weight_decay=5e-4, lr_drops=(5, 7), train_n=None, augment=True,
              seed=0, widths=CIFAR_WIDTHS, eval_n=None, size=32, warmup=0, data=None,
              cache=False, verbose=False, workers=WORKERS, cache_cols=None):
    """Train a CIFAR ResNet (residual=False: its plain twin) with 6n+2 layers;
    return (net, test_acc, history).

    The canonical run (all defaults: the paper's ResNet-20, full 50k,
    pad+crop+flip, SGD lr 0.1 divided by 10 at epochs 5 and 7) targets >= 0.80
    test accuracy. history holds one dict per epoch: lr, running train loss
    and error (on the augmented batches, BatchNorm in training mode -- the
    paper's Figure 6 dashed curves), test accuracy, seconds. eval_n scores
    each epoch on only the first eval_n test images (cheaper, for the long
    experiments); the returned accuracy is then measured once more on all
    10,000 at the end (history[-1]["test_acc_full"]). size=16 trains and
    tests on downsample()d photos (the augmentation's padding scales with it:
    size // 8 pixels). warmup runs the first `warmup` steps at lr / 10, as the
    paper did for ResNet-110. workers/cache_cols
    are build_resnet's speed and memory knobs: bit-identical, so they are not
    part of the checkpoint's name.

    cache=True: weights, BatchNorm running statistics and history are saved
    under data/cifar10/checkpoints/ (gitignored), named by a hash of every
    setting here plus CHECKPOINT_VERSION; an identical later call loads them
    (history entries then carry "loaded": True). The returned net is in eval
    mode either way.
    """
    assert epochs >= 1, "train for at least one epoch"
    config = dict(module="05-resnet", version=CHECKPOINT_VERSION, n=int(n),
                  residual=bool(residual), epochs=int(epochs), batch=int(batch),
                  lr=float(lr), momentum=float(momentum),
                  weight_decay=float(weight_decay),
                  lr_drops=[int(d) for d in lr_drops],
                  train_n=None if train_n is None else int(train_n),
                  augment=bool(augment), seed=int(seed),
                  widths=[int(w) for w in widths],
                  eval_n=None if eval_n is None else int(eval_n))
    # Later knobs join the name only when used, so earlier checkpoints keep theirs.
    if size != 32:
        config["size"] = int(size)
    if warmup:
        config["warmup"] = int(warmup)
    ckpt = os.path.join(CHECKPOINTS, _run_key(config))
    net = build_resnet(n=n, residual=residual, widths=widths, seed=seed,
                       workers=workers, cache_cols=cache_cols)
    state = lambda: net.params() + net.buffers()          # noqa: E731
    if cache and os.path.exists(ckpt + ".json"):          # .json is written last
        with np.load(ckpt + ".npz") as z:
            for k, p in enumerate(state()):
                p[...] = z[f"p{k}"]
        with open(ckpt + ".json", encoding="utf-8") as fh:
            history = json.load(fh)["history"]
        for rec in history:
            rec["loaded"] = True
        final = history[-1].get("test_acc_full", history[-1]["test_acc"])
        if verbose:
            print(f"[cifar] loaded checkpoint {os.path.basename(ckpt)} "
                  f"({len(history)} epochs, test_acc={final:.4f})")
        return net.eval(), final, history

    Xtr, ytr, Xte, yte, mean, std = data if data is not None else load_cifar()
    if train_n is not None:
        Xtr, ytr = Xtr[:train_n], ytr[:train_n]
    if size != Xtr.shape[2]:
        f = Xtr.shape[2] // size
        Xtr, Xte = downsample(Xtr, f), downsample(Xte, f)
    opt = SGD(lr=lr, momentum=momentum, weight_decay=weight_decay)
    loss_layer = SoftmaxCrossEntropy()
    gen = np.random.default_rng(seed)
    if verbose:
        kind = "ResNet" if residual else "plain net"
        print(f"[cifar] {kind}-{6 * n + 2} (widths {tuple(widths)}), {n_params(net):,} params, "
              f"{Xtr.shape[0]} train images, {epochs} epochs", flush=True)
    history = []
    step = 0
    for e in range(epochs):
        t0 = time.time()
        epoch_lr = lr * 0.1 ** sum(e >= d for d in lr_drops)
        net.train()
        idx = gen.permutation(Xtr.shape[0])
        total, wrong, seen = 0.0, 0, 0
        trace, c_wrong, c_seen = [], 0, 0      # training error every TRACE_EVERY steps
        for s in range(0, Xtr.shape[0], batch):
            b = idx[s:s + batch]
            opt.lr = epoch_lr * (0.1 if step < warmup else 1.0)
            step += 1
            xb = pad_crop_flip(Xtr[b], gen, pad=size // 8) if augment else Xtr[b]
            logits = net.forward(standardize(xb, mean, std))
            total += loss_layer.forward(logits, ytr[b]) * len(b)
            miss = int((logits.argmax(axis=1) != ytr[b]).sum())
            wrong += miss
            seen += len(b)
            c_wrong, c_seen = c_wrong + miss, c_seen + len(b)
            if step % TRACE_EVERY == 0:
                trace.append([step, c_wrong / c_seen])
                c_wrong, c_seen = 0, 0
            net.backward(loss_layer.backward())
            opt.step(net.params(), net.grads())
        k = Xte.shape[0] if eval_n is None else eval_n
        rec = {"epoch": e, "lr": epoch_lr, "train_loss": total / seen,
               "train_err": wrong / seen, "trace": trace,
               "test_acc": evaluate(net, Xte[:k], yte[:k], mean, std),
               "seconds": time.time() - t0}
        history.append(rec)
        if verbose:
            print(f"  epoch {e:2d}  lr={opt.lr:.4f}  train_loss={rec['train_loss']:.4f}  "
                  f"train_err={rec['train_err']:.4f}  test_acc={rec['test_acc']:.4f}  "
                  f"({rec['seconds']:.0f}s)", flush=True)
    final = history[-1]["test_acc"]
    if eval_n is not None:
        final = history[-1]["test_acc_full"] = evaluate(net, Xte, yte, mean, std)
        if verbose:
            print(f"  final test_acc on all {Xte.shape[0]} = {final:.4f}", flush=True)
    if cache:
        os.makedirs(CHECKPOINTS, exist_ok=True)
        np.savez(ckpt + ".tmp.npz", **{f"p{k}": p for k, p in enumerate(state())})
        os.replace(ckpt + ".tmp.npz", ckpt + ".npz")
        with open(ckpt + ".json.tmp", "w", encoding="utf-8") as fh:
            json.dump({"config": config, "history": history}, fh, indent=1)
        os.replace(ckpt + ".json.tmp", ckpt + ".json")
    return net.eval(), final, history


# ============================================================================
# 5. Lesions (Veit, Wilber & Belongie 2016) and the animation export.
# ============================================================================

class _Bypass:
    """A deleted block: only its shortcut is left (identity, or option A),
    followed by the block's ReLU -- for a plain block, just the identity."""

    def __init__(self, block):
        self.shortcut = block.shortcut if isinstance(block, Residual) else None
        self.relu = isinstance(block, Residual)

    def forward(self, X):
        S = X if self.shortcut is None else self.shortcut.forward(X)
        return np.maximum(S, 0.0) if self.relu else S

    def params(self): return []
    def grads(self): return []


def block_indices(net):
    """Positions in net.layers of the blocks (residual or plain)."""
    return [i for i, layer in enumerate(net.layers) if isinstance(layer, (Residual, Sequential))]


def deletable(net, i):
    """A plain block can only be deleted where its input and output shapes
    agree (no stride, no widening): there is nothing else to put there."""
    layer = net.layers[i]
    if isinstance(layer, Residual):
        return True
    first = layer.layers[0]
    return first.stride == 1 and first.K.shape[0] == first.K.shape[1]


def lesioned(net, i):
    """The same trained network with block net.layers[i] deleted (layers are
    shared, not copied). Evaluate it in eval mode."""
    return Sequential(net.layers[:i] + [_Bypass(net.layers[i])] + net.layers[i + 1:])


def export_assets(payload, out_path=None):
    """Write topics/05-resnet/anim/resnet_assets.json for the Manim scene: the
    notebook assembles `payload` (curves, gradient norms, lesion results)
    from its own runs, so the video and the chapter show the same numbers."""
    if out_path is None:
        out_path = os.path.join(_REPO, "topics", "05-resnet", "anim", "resnet_assets.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    print(f"[export] wrote {out_path}")


if __name__ == "__main__":
    for residual in (False, True):
        fp = run_toy(residual, verbose=True)
        print(f"FINAL {'res' if residual else 'plain'} "
              + " ".join(f"{name}={val!r}" for name, val in fp))
    print("FINAL bn " + " ".join(f"{name}={val!r}" for name, val in bn_fingerprint()))
    print("FINAL block " + " ".join(f"{name}={val!r}" for name, val in block_fingerprint()))

    # Hours of NumPy the first time; afterwards (or after the notebook ran)
    # the cached checkpoint is loaded and re-evaluated from its weights.
    data = load_cifar()
    net, _, _ = run_cifar(data=data, cache=True, verbose=True)
    _, _, Xte, yte, mean, std = data
    acc = evaluate(net, Xte, yte, mean, std)
    print(f"\n[cifar] ResNet-20 test accuracy = {acc:.4f}  (target >= 0.80)")
    assert acc >= 0.80, f"CIFAR-10 target missed: {acc:.4f} < 0.80"
