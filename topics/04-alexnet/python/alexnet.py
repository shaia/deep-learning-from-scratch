"""
Module 04 - AlexNet: depth + ReLU + dropout + augmentation, at scale.

The Python mirror of topics/04-alexnet/c/alexnet.c. Module 03's convolution
learns to read digits; this module is about what it took to make the same
machinery work on *photographs* -- a network big enough to matter, and the
tricks that kept it from simply memorizing its training set. nanograd gains
conv stride/padding, overlapping max-pooling, Dropout (+ train()/eval()),
LocalResponseNorm, and crop/flip augmentation, in both languages.

Four kinds of code live here:

  1. run_toy() -- the *agreement* path, on the "mirror glyphs" toy: 12x12 RGB
     images of a corner shape whose horizontal stroke sits at the bottom
     (class 0) or top (class 1). The 32 training images all face one way;
     the 64 test images are MIRRORED. A tiny AlexNet -- strided+padded conv,
     ReLU, overlapping max-pool, FC + ReLU + dropout, FC -- trained with
     AlexNet's recipe (random crops + flips, SGD with momentum and weight
     decay). Explicit scalar loops in the same operation order as
     c/alexnet.c, from the same LCG streams, so C and Python agree to a
     tight relative tolerance (1e-9 gate).

  2. naive_* -- the operations as their own definitions (strided/padded conv,
     overlapping max-pool, LRN), the ground truth the vectorized nanograd
     layers are tested against at 1e-12.

  3. lrn_fingerprint() -- LocalResponseNorm forward + backward on a fixed
     LCG-drawn tensor, so the C mirror's LRN is checked too (the toy doesn't
     use it; AlexNet's successors didn't either).

  4. build_alexnet()/run_cifar() -- the real thing: a scaled-down AlexNet
     (five conv layers, three dense, ReLU, overlapping pooling, dropout,
     crop+flip augmentation) on CIFAR-10, vectorized, in float32, trained with
     Adam, checked by metric (>= 0.70) and cached to disk (an hour of NumPy).

Run directly to see the toy, then CIFAR-10:
    python topics/04-alexnet/python/alexnet.py

Reference: docs/references/papers.md (Module 04) -- Krizhevsky, Sutskever &
Hinton (2012), "ImageNet Classification with Deep Convolutional Neural
Networks".
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
    Adam,
    Conv2D,
    Dropout,
    Flatten,
    Linear,
    LocalResponseNorm,
    MaxPool2D,
    ReLU,
    Sequential,
    SoftmaxCrossEntropy,
    Tanh,
    center_crop,
    conv_out_size,
    he_normal,
    he_normal_conv2d,
    random_crop_flip,
    ten_crop,
    xavier_normal,
    xavier_normal_conv2d,
)
from nanograd.rng import Rng  # noqa: E402


# ============================================================================
# 1. The bit-exact toy: "mirror glyphs".
#
# Each 12x12 RGB image holds one 5x5 corner glyph -- a vertical stroke joined
# to a horizontal stroke at the BOTTOM (class 0, an "L") or the TOP (class 1,
# a "Γ") -- in a random color, at a random position, over faint noise. The
# label is about up-vs-down only, so a left-right mirror never changes it.
# The training glyphs all have their vertical stroke on the LEFT; every test
# glyph is mirrored (stroke on the RIGHT). A network that memorized its 32
# training images has never seen a test image's shape; a network trained on
# random crops and flips has seen both.
#
# Everything runs in explicit scalar loops on flat row-major arrays in a fixed
# order so c/alexnet.c can reproduce every number. Loop-nest contract (shared
# with lib/c/nanograd/conv.c): conv forward i,f,u,v { acc = b; c,p,q }; conv
# backward the same nest; padding taps skipped; pooling windows scanned p,q.
# ============================================================================

TOY_SEED = 1          # weight draws
DATA_SEED = 2         # training glyphs
TEST_SEED = 3         # mirrored test glyphs
AUG_SEED = 4          # crop + flip draws
DROP_SEED = 5         # dropout masks
N_TRAIN, N_TEST = 32, 64
IMG, GLYPH, CROP = 12, 5, 10       # image, glyph, network input (crop) sizes
N_CH = 3                           # RGB
TOY_F, TOY_KS, TOY_S, TOY_PAD = 6, 5, 2, 2       # conv: 6 filters 5x5, stride 2, pad 2
CONV_OUT = (CROP + 2 * TOY_PAD - TOY_KS) // TOY_S + 1   # 5x5 feature maps
POOL_P, POOL_S = 3, 2                                   # overlapping max-pool
POOL_OUT = (CONV_OUT - POOL_P) // POOL_S + 1            # 2x2 after pooling
N_FLAT = TOY_F * POOL_OUT * POOL_OUT                    # 24 inputs to the head
N_HID = 16
N_OUT = 2
DROP_P = 0.5
TOY_STEPS = 600
SGD_LR, SGD_MOM, SGD_WD = 0.02, 0.9, 5e-4


def glyph_mask(cls, mirrored):
    """The 5x5 corner: vertical stroke in column 0 (column 4 if mirrored),
    horizontal stroke in row 4 (class 0, bottom) or row 0 (class 1, top)."""
    vcol = GLYPH - 1 if mirrored else 0
    hrow = GLYPH - 1 if cls == 0 else 0
    return [[1 if (c == vcol or r == hrow) else 0 for c in range(GLYPH)]
            for r in range(GLYPH)]


def make_glyphs(n, seed, mirrored=False):
    """The toy dataset, in the exact draw order c/alexnet.c replays.

    Per image i: two uniforms for the glyph's top/left corner (in 1..6, so the
    whole glyph survives the evaluation center crop), three for its RGB tint
    (each in [0.5, 1)), then 3*144 uniforms (row-major NCHW) for the noise
    floor; the glyph's pixels are then overwritten with the tint. Class is
    i % 2 (no draw spent on it). Returns X:(n,3,12,12) float64, y:(n,) int64.
    """
    rng = Rng(seed)
    X = np.empty((n, N_CH, IMG, IMG), dtype=np.float64)
    y = np.empty(n, dtype=np.int64)
    for i in range(n):
        y[i] = i % 2
        top = 1 + int(rng.uniform() * (IMG - GLYPH - 1))
        left = 1 + int(rng.uniform() * (IMG - GLYPH - 1))
        tint = [0.5 + 0.5 * rng.uniform() for _ in range(N_CH)]
        for c in range(N_CH):
            for r in range(IMG):
                for col in range(IMG):
                    X[i, c, r, col] = 0.25 * rng.uniform()
        g = glyph_mask(int(y[i]), mirrored)
        for c in range(N_CH):
            for r in range(GLYPH):
                for col in range(GLYPH):
                    if g[r][col]:
                        X[i, c, top + r, left + col] = tint[c]
    return X, y


def _conv_taps():
    """For each output position (u, v) of the toy conv, its valid kernel taps
    as (kernel offset, input offset) pairs, in the C loop nest's c,p,q order
    with padding taps dropped. Precomputing the C nest's bounds checks once
    keeps the scalar mirror's arithmetic identical and ~4x faster."""
    taps = []
    for u in range(CONV_OUT):
        for v in range(CONV_OUT):
            lst = []
            for c in range(N_CH):
                for p in range(TOY_KS):
                    h = u * TOY_S + p - TOY_PAD
                    if h < 0 or h >= CROP:
                        continue
                    for q in range(TOY_KS):
                        w = v * TOY_S + q - TOY_PAD
                        if w < 0 or w >= CROP:
                            continue
                        lst.append(((c * TOY_KS + p) * TOY_KS + q,
                                    (c * CROP + h) * CROP + w))
            taps.append(lst)
    return taps


class ToyAlexNet:
    """crop 10 -> conv(3->6, 5x5, s2, p2) -> ReLU -> maxpool 3/2 ->
    linear(24->16) -> ReLU -> dropout(0.5) -> linear(16->2), in flat doubles
    with explicit loops, mirroring c/alexnet.c line for line.

    Parameters live in flat row-major lists exactly like the C buffers:
      K1[((f*3 + c)*5 + p)*5 + q], b1[f], W2[k*16 + o], b2[o], W3[o*2 + c], b3[c].
    Draw order at init: K1 (He, flat), W2 (He), W3 (He) -- one RNG.
    """

    def __init__(self, rng, X, y):
        self.X = [[float(v) for v in X[i].reshape(-1)] for i in range(X.shape[0])]
        self.y = [int(v) for v in y]
        k_std = math.sqrt(2.0 / (N_CH * TOY_KS * TOY_KS))
        w2_std = math.sqrt(2.0 / N_FLAT)
        w3_std = math.sqrt(2.0 / N_HID)
        self.K1 = [k_std * rng.normal() for _ in range(TOY_F * N_CH * TOY_KS * TOY_KS)]
        self.b1 = [0.0] * TOY_F
        self.W2 = [w2_std * rng.normal() for _ in range(N_FLAT * N_HID)]
        self.b2 = [0.0] * N_HID
        self.W3 = [w3_std * rng.normal() for _ in range(N_HID * N_OUT)]
        self.b3 = [0.0] * N_OUT
        # SGD momentum buffers, one per parameter array (all start at zero).
        self.vel = [[0.0] * len(p) for p in self.params()]
        self.taps = _conv_taps()
        self.aug_rng = Rng(AUG_SEED)
        self.drop_rng = Rng(DROP_SEED)

    def params(self):
        return [self.K1, self.b1, self.W2, self.b2, self.W3, self.b3]

    # -- the two input views ------------------------------------------------

    def augment(self, x):
        """Random 10x10 crop + mirror of one flat 3x12x12 image: three draws
        (row offset, column offset, flip), as ng_random_crop_flip."""
        dy = int(self.aug_rng.uniform() * (IMG - CROP + 1))
        dx = int(self.aug_rng.uniform() * (IMG - CROP + 1))
        mirror = self.aug_rng.uniform() < 0.5
        out = [0.0] * (N_CH * CROP * CROP)
        for c in range(N_CH):
            for h in range(CROP):
                for w in range(CROP):
                    src_w = CROP - 1 - w if mirror else w
                    out[(c * CROP + h) * CROP + w] = x[(c * IMG + (dy + h)) * IMG + (dx + src_w)]
        return out

    @staticmethod
    def center(x):
        """The evaluation view: the middle 10x10 (offset 1, 1)."""
        o = (IMG - CROP) // 2
        out = [0.0] * (N_CH * CROP * CROP)
        for c in range(N_CH):
            for h in range(CROP):
                for w in range(CROP):
                    out[(c * CROP + h) * CROP + w] = x[(c * IMG + (o + h)) * IMG + (o + w)]
        return out

    # -- forward, per sample --------------------------------------------------

    def forward(self, x, train):
        """One flat 3x10x10 crop -> (z, pooled, argmax, h_pre, mask, d, probs).

        z: conv pre-activations (6,5,5); pooled/argmax: ReLU + overlapping
        3x3/2 max-pool and each window's winning z-index; h_pre: fc1
        pre-activations; mask: this sample's dropout mask (None in eval);
        d: the (dropped) hidden activations fed to fc2; probs: softmax.
        """
        # Strided, padded convolution (cross-correlation), via the tap lists.
        z = [0.0] * (TOY_F * CONV_OUT * CONV_OUT)
        n_k = N_CH * TOY_KS * TOY_KS
        for f in range(TOY_F):
            kf = self.K1[f * n_k:(f + 1) * n_k]          # this filter's weights
            for pos, lst in enumerate(self.taps):
                acc = self.b1[f]
                for ki, xi in lst:
                    acc += kf[ki] * x[xi]
                z[f * CONV_OUT * CONV_OUT + pos] = acc
        # ReLU, then OVERLAPPING max-pool: 3x3 windows every 2 cells, so the
        # middle row/column of the 5x5 map belongs to two windows. Strict >
        # keeps the first max in p,q scan order (the C tie-break).
        pooled = [0.0] * N_FLAT
        argmax = [0] * N_FLAT
        for f in range(TOY_F):
            for u2 in range(POOL_OUT):
                for v2 in range(POOL_OUT):
                    best, bi = 0.0, -1
                    for p in range(POOL_P):
                        for q in range(POOL_P):
                            zi = ((f * CONV_OUT + (u2 * POOL_S + p)) * CONV_OUT
                                  + (v2 * POOL_S + q))
                            a = z[zi] if z[zi] > 0.0 else 0.0     # ReLU
                            if bi < 0 or a > best:
                                best, bi = a, zi
                    j = (f * POOL_OUT + u2) * POOL_OUT + v2
                    pooled[j] = best
                    argmax[j] = bi
        # fc1 + ReLU (flatten is a no-op on the flat buffer).
        h_pre = [0.0] * N_HID
        for o in range(N_HID):
            s = self.b2[o]
            for k in range(N_FLAT):
                s += pooled[k] * self.W2[k * N_HID + o]
            h_pre[o] = s
        h = [v if v > 0.0 else 0.0 for v in h_pre]
        # Dropout (inverted): one uniform per hidden unit, in order.
        mask = None
        if train:
            keep = 1.0 - DROP_P
            mask = [(1.0 / keep) if self.drop_rng.uniform() < keep else 0.0
                    for _ in range(N_HID)]
            d = [h[o] * mask[o] for o in range(N_HID)]
        else:
            d = h
        # fc2 -> softmax.
        logits = [0.0] * N_OUT
        for c in range(N_OUT):
            s = self.b3[c]
            for o in range(N_HID):
                s += d[o] * self.W3[o * N_OUT + c]
            logits[c] = s
        return z, pooled, argmax, h_pre, mask, d, _softmax(logits)

    def loss_and_acc(self, X=None, y=None):
        """Mean cross-entropy and accuracy on center crops, dropout off."""
        X = self.X if X is None else X
        y = self.y if y is None else y
        total, correct = 0.0, 0
        for x, label in zip(X, y):
            p = self.forward(self.center(x), train=False)[-1]
            total += -math.log(p[label] + 1e-12)
            if _argmax(p) == label:
                correct += 1
        return total / len(X), correct / len(X)

    def step(self):
        """One full-batch SGD step: augment every image (all crops first, as
        the C does for the batch), then forward/backward per image, summing
        gradients with the 1/m folded into dlogits."""
        m = len(self.X)
        crops = [self.augment(x) for x in self.X]
        g = [[0.0] * len(p) for p in self.params()]
        gK1, gb1, gW2, gb2, gW3, gb3 = g

        for x, label in zip(crops, self.y):
            z, pooled, argmax, h_pre, mask, d, p = self.forward(x, train=True)
            dlogits = [(p[c] - (1.0 if c == label else 0.0)) / m for c in range(N_OUT)]
            # fc2.
            dd = [0.0] * N_HID
            for o in range(N_HID):
                for c in range(N_OUT):
                    gW3[o * N_OUT + c] += d[o] * dlogits[c]
                    dd[o] += dlogits[c] * self.W3[o * N_OUT + c]
            for c in range(N_OUT):
                gb3[c] += dlogits[c]
            # Dropout, then ReLU: a dropped unit, or a dead one, passes nothing.
            assert mask is not None          # training forward always draws one
            dz2 = [0.0] * N_HID
            for o in range(N_HID):
                dh = dd[o] * mask[o]
                dz2[o] = dh if h_pre[o] > 0.0 else 0.0
            # fc1.
            dpool = [0.0] * N_FLAT
            for k in range(N_FLAT):
                for o in range(N_HID):
                    gW2[k * N_HID + o] += pooled[k] * dz2[o]
                    dpool[k] += dz2[o] * self.W2[k * N_HID + o]
            for o in range(N_HID):
                gb2[o] += dz2[o]
            # Max-pool backward: each window's gradient goes to its winner --
            # ADDED, because overlapping windows can share one -- and survives
            # only if the winner was alive through ReLU (z > 0).
            dz = [0.0] * len(z)
            for j in range(N_FLAT):
                if z[argmax[j]] > 0.0:
                    dz[argmax[j]] += dpool[j]
            # Conv backward: same f,u,v nest as forward, over the valid taps.
            for f in range(TOY_F):
                base = f * N_CH * TOY_KS * TOY_KS
                for pos, lst in enumerate(self.taps):
                    gz = dz[f * CONV_OUT * CONV_OUT + pos]
                    if gz == 0.0:
                        continue
                    gb1[f] += gz
                    for ki, xi in lst:
                        gK1[base + ki] += gz * x[xi]

        for p, gp, v in zip(self.params(), g, self.vel):
            _sgd(p, gp, v)

    def wsum(self):
        """Sum of all parameters in canonical order: K1, b1, W2, b2, W3, b3."""
        s = 0.0
        for p in self.params():
            for v in p:
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


def _sgd(p, g, vel):
    """In-place SGD with momentum + L2 weight decay (ng_sgd_step's formula):
    g' = g + wd*theta;  v = mom*v + g';  theta -= lr*v."""
    for i in range(len(p)):
        gi = g[i] + SGD_WD * p[i]
        vel[i] = SGD_MOM * vel[i] + gi
        p[i] -= SGD_LR * vel[i]


def fingerprint(net, Xte_flat, yte):
    """Ordered scalars that pin down the trained toy net (agreement test)."""
    loss, acc = net.loss_and_acc()
    _, test_acc = net.loss_and_acc(Xte_flat, yte)
    return (
        ("loss", loss),
        ("acc", acc),
        ("test_acc", test_acc),
        ("wsum", net.wsum()),
        ("k1_0", net.K1[0]),
        ("w2_00", net.W2[0]),
        ("w3_00", net.W3[0]),
        ("b3_0", net.b3[0]),
    )


def run_toy(verbose=False):
    """Train the bit-exact toy AlexNet and return its fingerprint tuple."""
    Xtr, ytr = make_glyphs(N_TRAIN, DATA_SEED)
    Xte, yte = make_glyphs(N_TEST, TEST_SEED, mirrored=True)
    net = ToyAlexNet(Rng(TOY_SEED), Xtr, ytr)
    Xte_flat = [[float(v) for v in Xte[i].reshape(-1)] for i in range(N_TEST)]
    if verbose:
        print("[toy] crop%d -> conv(3->%d, %dx%d, s%d, p%d) -> ReLU -> maxpool %d/%d "
              "-> fc %d->%d -> ReLU -> dropout %.1f -> fc %d->2; SGD+momentum, "
              "crops+flips, mirror glyphs"
              % (CROP, TOY_F, TOY_KS, TOY_KS, TOY_S, TOY_PAD, POOL_P, POOL_S,
                 N_FLAT, N_HID, DROP_P, N_HID))
    for t in range(TOY_STEPS):
        net.step()
        if verbose and (t % (TOY_STEPS // 6) == 0 or t == TOY_STEPS - 1):
            loss, acc = net.loss_and_acc()
            print("  step %4d  loss=%.6f  train_acc=%.3f" % (t, loss, acc))
    return fingerprint(net, Xte_flat, list(yte))


def build_toy_alexnet(rng=None, drop_rng=None, drop_p=DROP_P):
    """The vectorized twin of ToyAlexNet: same layers, same He draws in the
    same order (K1, W2, W3) from the same seed, on nanograd's im2col path.
    Pass drop_rng=Rng(DROP_SEED) to draw the scalar mirror's exact masks."""
    if rng is None:
        rng = Rng(TOY_SEED)
    conv = Conv2D(N_CH, TOY_F, TOY_KS, TOY_KS, stride=TOY_S, pad=TOY_PAD)
    fc1 = Linear(N_FLAT, N_HID)
    fc2 = Linear(N_HID, N_OUT)
    he_normal_conv2d(conv.K, rng)
    he_normal(fc1.W, rng)
    he_normal(fc2.W, rng)
    return Sequential([conv, ReLU(), MaxPool2D(POOL_P, POOL_S), Flatten(),
                       fc1, ReLU(), Dropout(drop_p, rng=drop_rng), fc2])


def train_toy_vectorized(augment=True, drop_p=DROP_P, seed=0, steps=TOY_STEPS,
                         n_train=N_TRAIN):
    """The toy's ablation harness on the fast path: same data recipe, same
    net, but `seed` varies the weights, glyphs, crops and masks together, so
    the notebook can show the regularizers' effect is not one lucky seed.
    seed=0 with the default config reproduces run_toy's data and init.
    Returns (train_acc, mirrored_test_acc)."""
    Xtr, ytr = make_glyphs(n_train, DATA_SEED + 100 * seed)
    Xte, yte = make_glyphs(N_TEST, TEST_SEED + 100 * seed, mirrored=True)
    gen = np.random.default_rng(seed)
    net = build_toy_alexnet(Rng(TOY_SEED + 100 * seed),
                            drop_rng=np.random.default_rng(seed + 1), drop_p=drop_p)
    loss_layer = SoftmaxCrossEntropy()
    opt = SGD(lr=SGD_LR, momentum=SGD_MOM, weight_decay=SGD_WD)
    for _ in range(steps):
        net.train()
        Xb = random_crop_flip(Xtr, CROP, gen) if augment else center_crop(Xtr, CROP)
        loss_layer.forward(net.forward(Xb), ytr)
        net.backward(loss_layer.backward())
        opt.step(net.params(), net.grads())
    net.eval()
    acc_tr = float((net.forward(center_crop(Xtr, CROP)).argmax(1) == ytr).mean())
    acc_te = float((net.forward(center_crop(Xte, CROP)).argmax(1) == yte).mean())
    return acc_tr, acc_te


# ============================================================================
# 2. The naive references: each new operation as its own definition.
#
# Straight loops over NumPy arrays -- slow, obvious, and the ground truth the
# vectorized nanograd layers are tested against (1e-12). The notebook and the
# assignment teach from these.
# ============================================================================

def naive_conv2d_forward(X, K, b, stride=1, pad=0):
    """Y[i,f,u,v] = b[f] + sum_{c,p,q} K[f,c,p,q] * X[i,c,u*S+p-pad,v*S+q-pad],
    with taps that fall in the zero padding skipped.
    X:(n,C,H,W) K:(F,C,KH,KW) b:(F,) -> Y:(n,F,OH,OW),
    OH = (H + 2*pad - KH)//S + 1."""
    n, C, H, W = X.shape
    F, _, KH, KW = K.shape
    OH, OW = conv_out_size(H, KH, stride, pad), conv_out_size(W, KW, stride, pad)
    Y = np.empty((n, F, OH, OW), dtype=X.dtype)
    for i in range(n):
        for f in range(F):
            for u in range(OH):
                for v in range(OW):
                    acc = b[f]
                    for c in range(C):
                        for p in range(KH):
                            for q in range(KW):
                                h, w = u * stride + p - pad, v * stride + q - pad
                                if 0 <= h < H and 0 <= w < W:
                                    acc += K[f, c, p, q] * X[i, c, h, w]
                    Y[i, f, u, v] = acc
    return Y


def naive_conv2d_backward(X, K, dY, stride=1, pad=0):
    """The three conv gradients from the chain rule (summed over the batch).
    Returns (dX, dK, db). Same loop nest as forward; padding taps fed nothing
    forward, so they receive nothing back."""
    n, C, H, W = X.shape
    F, _, KH, KW = K.shape
    _, _, OH, OW = dY.shape
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
                                h, w = u * stride + p - pad, v * stride + q - pad
                                if 0 <= h < H and 0 <= w < W:
                                    dK[f, c, p, q] += g * X[i, c, h, w]
                                    dX[i, c, h, w] += g * K[f, c, p, q]
    return dX, dK, db


def naive_maxpool2d_forward(X, P, stride):
    """Y[i,c,u,v] = max over the PxP window at (u*S, v*S). Returns (Y, argmax)
    where argmax holds each window's winning (h, w) -- first max in p,q order."""
    n, C, H, W = X.shape
    OH, OW = conv_out_size(H, P, stride), conv_out_size(W, P, stride)
    Y = np.empty((n, C, OH, OW), dtype=X.dtype)
    argmax = np.empty((n, C, OH, OW, 2), dtype=np.int64)
    for i in range(n):
        for c in range(C):
            for u in range(OH):
                for v in range(OW):
                    best, bh, bw = None, -1, -1
                    for p in range(P):
                        for q in range(P):
                            h, w = u * stride + p, v * stride + q
                            if best is None or X[i, c, h, w] > best:
                                best, bh, bw = X[i, c, h, w], h, w
                    Y[i, c, u, v] = best
                    argmax[i, c, u, v] = (bh, bw)
    return Y, argmax


def naive_maxpool2d_backward(dY, argmax, x_shape):
    """Route each window's gradient to its winner, ADDING where overlapping
    windows picked the same cell."""
    dX = np.zeros(x_shape, dtype=dY.dtype)
    n, C, OH, OW = dY.shape
    for i in range(n):
        for c in range(C):
            for u in range(OH):
                for v in range(OW):
                    h, w = argmax[i, c, u, v]
                    dX[i, c, h, w] += dY[i, c, u, v]
    return dX


def naive_lrn_forward(X, size=5, alpha=1e-4, beta=0.75, k=2.0):
    """AlexNet eq. (Sec. 3.3), literally:
    b[c] = a[c] / (k + alpha * sum_{j=max(0,c-n/2)}^{min(C-1,c+n/2)} a[j]^2)^beta."""
    n, C, H, W = X.shape
    half = size // 2
    Y = np.empty_like(X)
    for i in range(n):
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    s = 0.0
                    for j in range(max(0, c - half), min(C - 1, c + half) + 1):
                        s += X[i, j, h, w] ** 2
                    Y[i, c, h, w] = X[i, c, h, w] / (k + alpha * s) ** beta
    return Y


# ============================================================================
# 3. LRN agreement: c/alexnet.c prints LRN forward/backward checksums on a
# tensor drawn from an LCG stream; this computes the same numbers with
# nanograd's vectorized LocalResponseNorm. AlexNet's own constants, but alpha
# raised to 0.1 so the cross-channel coupling term is large enough to test.
# ============================================================================

LRN_SEED = 6
LRN_SHAPE = (2, 7, 3, 3)           # n, C, H, W
LRN_SIZE, LRN_ALPHA, LRN_BETA, LRN_K = 5, 0.1, 0.75, 2.0


def lrn_fingerprint():
    """Draw X then dY (each 2*signed, row-major) from Rng(LRN_SEED), run LRN
    forward and backward, and return order-sensitive checksums."""
    rng = Rng(LRN_SEED)
    size = int(np.prod(LRN_SHAPE))
    X = np.array([2.0 * rng.signed() for _ in range(size)]).reshape(LRN_SHAPE)
    dY = np.array([2.0 * rng.signed() for _ in range(size)]).reshape(LRN_SHAPE)
    lrn = LocalResponseNorm(LRN_SIZE, LRN_ALPHA, LRN_BETA, LRN_K)
    Y = lrn.forward(X).reshape(-1).tolist()
    dX = lrn.backward(dY).reshape(-1).tolist()
    # Plain left-to-right sums, as the C loop does (Python's built-in sum()
    # compensates rounding since 3.12, so it is deliberately not used here).
    # The position-weighted sums make the checksums order-sensitive.
    y_sum = y_wsum = dx_sum = dx_wsum = 0.0
    for j in range(size):
        y_sum += Y[j]
        y_wsum += Y[j] * (j + 1)
        dx_sum += dX[j]
        dx_wsum += dX[j] * (j + 1)
    return (
        ("y_sum", y_sum),
        ("y_wsum", y_wsum),
        ("dx_sum", dx_sum),
        ("dx_wsum", dx_wsum),
        ("y_0", Y[0]),
        ("dx_0", dX[0]),
    )


# ============================================================================
# 4. The real thing: a scaled-down AlexNet on CIFAR-10.
#
#   24x24 random crop (+ mirror) of a 32x32 photo
#   conv1 5x5  -> ReLU -> [LRN] -> maxpool 3/2     24 -> 11
#   conv2 3x3  -> ReLU -> [LRN] -> maxpool 3/2     11 -> 5
#   conv3 3x3  -> ReLU
#   conv4 3x3  -> ReLU
#   conv5 3x3  -> ReLU -> maxpool 3/2               5 -> 2
#   fc6 -> ReLU -> dropout -> fc7 -> ReLU -> dropout -> fc8 (10 logits)
#
# AlexNet's silhouette -- five convolutions (two pooled early, three stacked,
# one more pool), three dense layers, dropout on the two hidden ones -- sized
# for 32x32 photos and a laptop CPU instead of 224x224 and two GTX 580s. The
# 24x24 crop is Krizhevsky's own CIFAR-10 recipe (cuda-convnet). Trained in
# float32 with Adam (Module 02): AlexNet's SGD + momentum + hand-dropped
# learning rate needed ~90 epochs; without batch normalization (Module 05) it
# is too slow for a budget of 15. The toy keeps AlexNet's SGD recipe exactly.
# ============================================================================

CIFAR_CROP = 24
CIFAR_WIDTHS = (32, 64, 96, 96, 64)
CIFAR_HIDDEN = 256


def build_alexnet(seed=0, widths=CIFAR_WIDTHS, hidden=CIFAR_HIDDEN,
                  activation="relu", dropout=0.5, lrn=False, crop=CIFAR_CROP,
                  dtype=np.float32):
    """Assemble the scaled-down AlexNet; weights drawn in a fixed layer order
    (He for ReLU, Xavier for tanh), then cast to `dtype`."""
    rng = Rng(seed)
    act = {"relu": ReLU, "tanh": Tanh}[activation]
    conv_init = {"relu": he_normal_conv2d, "tanh": xavier_normal_conv2d}[activation]
    lin_init = {"relu": he_normal, "tanh": xavier_normal}[activation]
    w1, w2, w3, w4, w5 = widths
    s = conv_out_size(conv_out_size(conv_out_size(crop, 3, 2), 3, 2), 3, 2)

    # conv1 reads the data itself, so it skips dL/dX (input_grad=False).
    convs = [Conv2D(3, w1, 5, 5, pad=2, input_grad=False), Conv2D(w1, w2, 3, 3, pad=1),
             Conv2D(w2, w3, 3, 3, pad=1), Conv2D(w3, w4, 3, 3, pad=1),
             Conv2D(w4, w5, 3, 3, pad=1)]
    fcs = [Linear(w5 * s * s, hidden), Linear(hidden, hidden), Linear(hidden, 10)]
    for layer in convs:
        conv_init(layer.K, rng)
    for layer in fcs:
        lin_init(layer.W, rng)
    drop_gen = np.random.default_rng(seed + 1)

    def norm():
        return [LocalResponseNorm()] if lrn else []

    layers = ([convs[0], act()] + norm() + [MaxPool2D(3, 2),
              convs[1], act()] + norm() + [MaxPool2D(3, 2),
              convs[2], act(),
              convs[3], act(),
              convs[4], act(), MaxPool2D(3, 2),
              Flatten(),
              fcs[0], act(), Dropout(dropout, rng=drop_gen),
              fcs[1], act(), Dropout(dropout, rng=drop_gen),
              fcs[2]])
    return Sequential(layers).astype(dtype)


def n_params(net):
    return sum(p.size for p in net.params())


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
    """uint8 images -> per-channel standardized floats (AlexNet subtracted the
    training set's mean image; per-channel mean/std is the modern habit)."""
    X = X_uint8.astype(dtype) / 255.0
    return (X - mean.astype(dtype)[None, :, None, None]) / std.astype(dtype)[None, :, None, None]


def evaluate(net, X_uint8, y, mean, std, crop=CIFAR_CROP, views="center",
             batch=500):
    """Accuracy with dropout off. views='center' scores the middle crop;
    views='ten' averages the softmax over AlexNet's ten crops (4 corners +
    center, each plain and mirrored)."""
    net.eval()
    correct = 0
    for s in range(0, X_uint8.shape[0], batch):
        xb = X_uint8[s:s + batch]
        crops = [center_crop(xb, crop)] if views == "center" else ten_crop(xb, crop)
        probs = 0.0
        for v in crops:
            logits = net.forward(standardize(np.ascontiguousarray(v), mean, std))
            e = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = probs + e / e.sum(axis=1, keepdims=True)
        correct += int((np.argmax(probs, axis=1) == y[s:s + batch]).sum())
    return correct / X_uint8.shape[0]


CHECKPOINTS = os.path.join(_REPO, "data", "cifar10", "checkpoints")


# Part of every checkpoint's key. Bump it after changing build_alexnet (layers,
# init order) or the training loop, so checkpoints of the old code stop
# matching instead of silently loading into a different network.
CHECKPOINT_VERSION = 1


def _run_key(config):
    """Short stable hash of a run's normalized configuration (the checkpoint
    name). Callers pass plain Python ints/floats/bools/lists -- see run_cifar."""
    blob = json.dumps(config, sort_keys=True)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def run_cifar(epochs=15, batch=128, lr=1e-3, optimizer="adam", momentum=0.9,
              weight_decay=0.0, lr_drops=(10, 13), train_n=None, augment=True,
              dropout=0.5, activation="relu", lrn=False, seed=0,
              widths=CIFAR_WIDTHS, hidden=CIFAR_HIDDEN, track_train=False,
              data=None, cache=False, verbose=False):
    """Train the scaled-down AlexNet on CIFAR-10; return (net, test_acc, history).

    The canonical configuration (all defaults: full 50k, crops + flips,
    dropout 0.5, Adam 1e-3 divided by 10 at epochs 10 and 13) targets >= 0.70
    test accuracy. optimizer="sgd" uses AlexNet's own SGD with momentum and
    weight decay -- the toy's optimizer, but too slow for this epoch budget
    without batch normalization (see the notebook). history is a list of dicts
    per epoch: running train loss/error (on augmented, dropped-out batches),
    test accuracy (center crop), and, if track_train, a clean train accuracy
    (eval mode, center crop, on up to 10k training images) -- the honest
    measure of memorization. `data` lets callers pass load_cifar()'s tuple so
    repeated runs don't reload it.

    cache=True: an hour of NumPy training is worth keeping. The run's weights
    and history are saved under data/cifar10/checkpoints/ (gitignored), named
    by a hash of every setting above plus the crop size and
    CHECKPOINT_VERSION (not `data`: it is assumed to be load_cifar()'s
    standard CIFAR-10). An identical later call loads them instead of
    retraining (history entries then carry "loaded": True). Delete the folder
    to retrain from scratch. Either way the returned net is in eval mode.
    """
    assert epochs >= 1, "train for at least one epoch"
    # Normalized so that equal settings always hash equally (0 vs 0.0, NumPy
    # scalars); momentum only matters to SGD, so Adam runs ignore it.
    config = dict(version=CHECKPOINT_VERSION, crop=CIFAR_CROP,
                  epochs=int(epochs), batch=int(batch), lr=float(lr),
                  optimizer=optimizer,
                  momentum=float(momentum) if optimizer == "sgd" else None,
                  weight_decay=float(weight_decay),
                  lr_drops=[int(d) for d in lr_drops],
                  train_n=None if train_n is None else int(train_n),
                  augment=bool(augment), dropout=float(dropout),
                  activation=activation, lrn=bool(lrn), seed=int(seed),
                  widths=[int(w) for w in widths], hidden=int(hidden),
                  track_train=bool(track_train))
    ckpt = os.path.join(CHECKPOINTS, _run_key(config))
    net = build_alexnet(seed=seed, widths=widths, hidden=hidden,
                        activation=activation, dropout=dropout, lrn=lrn)
    if cache and os.path.exists(ckpt + ".json"):    # .json is written last
        with np.load(ckpt + ".npz") as z:
            for k, p in enumerate(net.params()):
                p[...] = z[f"p{k}"]
        with open(ckpt + ".json", encoding="utf-8") as fh:
            history = json.load(fh)["history"]
        for rec in history:
            rec["loaded"] = True
        if verbose:
            print(f"[cifar] loaded checkpoint {os.path.basename(ckpt)} "
                  f"({len(history)} epochs, test_acc={history[-1]['test_acc']:.4f})")
        return net.eval(), history[-1]["test_acc"], history

    Xtr, ytr, Xte, yte, mean, std = data if data is not None else load_cifar()
    if train_n is not None:
        Xtr, ytr = Xtr[:train_n], ytr[:train_n]
    opt = (SGD(lr=lr, momentum=momentum, weight_decay=weight_decay) if optimizer == "sgd"
           else Adam(lr=lr, weight_decay=weight_decay))
    loss_layer = SoftmaxCrossEntropy()
    gen = np.random.default_rng(seed)
    if verbose:
        print(f"[cifar] AlexNet-mini ({activation}, dropout={dropout}, "
              f"augment={augment}, lrn={lrn}), {n_params(net):,} params, "
              f"{Xtr.shape[0]} train images, {epochs} epochs")
    history = []
    for e in range(epochs):
        t0 = time.time()
        opt.lr = lr * 0.1 ** sum(e >= d for d in lr_drops)
        net.train()
        idx = gen.permutation(Xtr.shape[0])
        total, wrong, seen = 0.0, 0, 0
        for s in range(0, Xtr.shape[0], batch):
            b = idx[s:s + batch]
            xb = random_crop_flip(Xtr[b], CIFAR_CROP, gen) if augment \
                else np.ascontiguousarray(center_crop(Xtr[b], CIFAR_CROP))
            logits = net.forward(standardize(xb, mean, std))
            total += loss_layer.forward(logits, ytr[b]) * len(b)
            wrong += int((logits.argmax(axis=1) != ytr[b]).sum())
            seen += len(b)
            net.backward(loss_layer.backward())
            opt.step(net.params(), net.grads())
        rec = {"epoch": e, "lr": opt.lr, "train_loss": total / seen,
               "train_err": wrong / seen,
               "test_acc": evaluate(net, Xte, yte, mean, std)}
        if track_train:
            k = min(10000, Xtr.shape[0])
            rec["train_acc"] = evaluate(net, Xtr[:k], ytr[:k], mean, std)
        rec["seconds"] = time.time() - t0
        history.append(rec)
        if verbose:
            extra = f"  clean_train_acc={rec['train_acc']:.4f}" if track_train else ""
            print(f"  epoch {e:2d}  lr={opt.lr:.4f}  train_loss={rec['train_loss']:.4f}  "
                  f"train_err={rec['train_err']:.4f}  test_acc={rec['test_acc']:.4f}"
                  f"{extra}  ({rec['seconds']:.0f}s)")
    if cache:
        # Write to temporary names, then rename: an interrupted save leaves no
        # half-written checkpoint behind. The .json goes last -- it is the
        # marker the loader checks for.
        os.makedirs(CHECKPOINTS, exist_ok=True)
        np.savez(ckpt + ".tmp.npz", **{f"p{k}": p for k, p in enumerate(net.params())})
        os.replace(ckpt + ".tmp.npz", ckpt + ".npz")
        with open(ckpt + ".json.tmp", "w", encoding="utf-8") as fh:
            json.dump({"config": config, "history": history}, fh, indent=1)
        os.replace(ckpt + ".json.tmp", ckpt + ".json")
    return net.eval(), history[-1]["test_acc"], history


# ============================================================================
# 5. Asset export: the trained first-layer filters, a few test photos, and the
# experiment curves, as JSON, for the Manim scene -- one source of truth with
# the notebook that produced them.
# ============================================================================

SAMPLE_CLASSES = ("cat", "ship", "horse", "automobile", "bird", "dog")


def epochs_to(errs, level=0.25):
    """First epoch (1-based) at which a training-error curve is <= level, or
    None if it never gets there."""
    for k, e in enumerate(errs):
        if e <= level:
            return k + 1
    return None


def filter_images(K):
    """Conv kernels (F, 3, KH, KW) -> displayable RGB tiles (F, KH, KW, 3) in
    [0, 1]: each filter scaled symmetrically around 0 (mid-gray = 0), so a
    tile's colors show the sign of its weights, not just their range."""
    K = np.asarray(K, dtype=np.float64).transpose(0, 2, 3, 1)
    return 0.5 + K / (2.0 * np.abs(K).max(axis=(1, 2, 3), keepdims=True))


def export_assets(net, relu_hist, tanh_hist, ablation, data, level=0.25,
                  init_net=None, out_path=None):
    """Write topics/04-alexnet/anim/alexnet_assets.json.

    net: the trained CIFAR AlexNet (conv1 is its first layer); init_net: the
    same network at initialization (default: build_alexnet(), the canonical
    seed), so the export can show what training CHANGED in conv1;
    relu_hist / tanh_hist: run_cifar histories of the ReLU-vs-tanh
    experiment, compared at training-error `level`; ablation: {name: history}
    of the regularization experiment; data: load_cifar()'s tuple.
    """
    if out_path is None:
        out_path = os.path.join(_REPO, "topics", "04-alexnet", "anim",
                                "alexnet_assets.json")
    if init_net is None:
        init_net = build_alexnet()
    sys.path.insert(0, os.path.join(_REPO, "data"))
    from get_cifar10 import CLASSES
    _, _, Xte, yte, _, _ = data
    K = net.layers[0].K.astype(np.float64)
    K0 = init_net.layers[0].K.astype(np.float64)
    filters = np.round(filter_images(K), 3).tolist()
    filters_learned = np.round(filter_images(K - K0), 3).tolist()
    samples = []
    for name in SAMPLE_CLASSES:
        i = int(np.argmax(yte == CLASSES.index(name)))
        samples.append({"label": name, "img": Xte[i].transpose(1, 2, 0).tolist()})
    relu = [round(h["train_err"], 4) for h in relu_hist]
    tanh = [round(h["train_err"], 4) for h in tanh_hist]
    er, et = epochs_to(relu, level), epochs_to(tanh, level)
    pct = f"{level:.0%}"
    # Say what the curves show, whichever way they came out.
    if er and et and er <= et:
        caption = f"ReLU reached {pct} training error at epoch {er}; tanh at epoch {et}"
    elif er and et:
        caption = (f"Our redo: tanh reached {pct} at epoch {et}, ReLU at epoch {er}. "
                   f"The paper's 6x gap did not reproduce here.")
    elif er:
        caption = f"ReLU reached {pct} training error at epoch {er}; tanh never did"
    else:
        caption = f"tanh reached {pct} training error at epoch {et}; ReLU never did"
    payload = {
        "filters": filters,
        "filters_learned": filters_learned,
        "conv1_moved": round(float(np.linalg.norm(K - K0) / np.linalg.norm(K0)), 3),
        "samples": samples,
        "relu_tanh": {"relu": relu, "tanh": tanh, "level": level, "caption": caption},
        "ablation": {k: {"train": [round(h.get("train_acc", float("nan")), 4) for h in v],
                         "test": [round(h["test_acc"], 4) for h in v]}
                     for k, v in ablation.items()},
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    print(f"[export] wrote {out_path}")


if __name__ == "__main__":
    fp = run_toy(verbose=True)
    print("FINAL toy " + " ".join(f"{name}={val!r}" for name, val in fp))
    print("FINAL lrn " + " ".join(f"{name}={val!r}" for name, val in lrn_fingerprint()))

    # ~2 hours of NumPy the first time; afterwards (or after the notebook ran)
    # the cached checkpoint is loaded and re-evaluated from its weights.
    data = load_cifar()
    net, _, _ = run_cifar(data=data, cache=True, verbose=True)
    _, _, Xte, yte, mean, std = data
    acc = evaluate(net, Xte, yte, mean, std)
    print(f"\n[cifar] AlexNet-mini test accuracy = {acc:.4f}  (target >= 0.70)")
    assert acc >= 0.70, f"CIFAR-10 target missed: {acc:.4f} < 0.70"
