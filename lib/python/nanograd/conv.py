"""
Convolution and pooling layers -- Module 03's additions to nanograd, with the
stride and padding dials Module 04 (AlexNet) turns.

A convolutional layer is a Linear layer with two constraints bolted on:
*locality* (each output looks at a small patch, not the whole input) and
*weight sharing* (every patch is scored by the same small kernel). Everything
else -- the caching, the hand-written backward, the params/grads contract --
is exactly the machinery of layers.py, reused.

Two implementations of the same arithmetic live in this module's orbit:

  * Here: the *vectorized* forward/backward via im2col -- convolution
    rearranged into one big matrix multiply. This is what trains MNIST in
    minutes and it is (a rounding-error away from) how real frameworks do it.
  * In topics/03-lenet/python/lenet.py and topics/04-alexnet/python/alexnet.py:
    the *naive* loops, which are the definition itself and the thing the C
    mirror implements verbatim. Each module's tests/test_agreement.py asserts
    the two match to 1e-12.

Conventions (shared with lib/c/nanograd/conv.c -- see nanograd.h):
  * Tensors are NCHW: X[i, c, h, w], C-contiguous, so a flat view of X is
    exactly the row-major buffer the C code walks.
  * Kernels are (F, C, KH, KW): F filters, each spanning all C input channels.
    The array is named K (W is taken -- it means image width here).
  * "conv" is cross-correlation (no kernel flip) -- the same choice every
    modern framework makes. The flip is not lost: it reappears on its own in
    the backward pass for dX.
  * Stride S and zero-padding `pad` default to 1 and 0 (Module 03's "valid,
    stride 1" conv). The general output size, for both axes, is
        OH = (H + 2*pad - KH) // S + 1.
  * Gradients are *summed* over the batch; the 1/m lives in the loss layer.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np


def conv_out_size(H, K, stride=1, pad=0):
    """Output length of a K-wide window slid over H (+pad zeros each side)
    in steps of `stride`: floor((H + 2*pad - K) / stride) + 1."""
    if H + 2 * pad < K:
        raise ValueError(f"a {K}-wide window does not fit in {H} (+2*{pad} padding)")
    return (H + 2 * pad - K) // stride + 1


# ---------------------------------------------------------------------------
# im2col / col2im: convolution as a matrix multiply.
#
# Every output position (u, v) reads the patch starting at (u*S, v*S) of the
# (zero-padded) input. im2col copies each such patch out into one ROW of a
# big matrix, so all patches can be scored against all filters in a single
# GEMM: (n*OH*OW, C*KH*KW) @ (C*KH*KW, F). Pixels are duplicated across rows
# (a 5x5 kernel at stride 1 copies each interior pixel 25 times) -- that
# redundancy is the price of turning a sliding window into the one operation
# computers are fastest at. col2im is the exact adjoint: it scatters
# row-gradients back and *sums* where patches overlapped, which is precisely
# dL/dX (and drops whatever landed on the padding, which is not an input).
#
# workers > 1 (Module 05): NumPy runs every copy on a single core, and at
# CIFAR-ResNet sizes these copies -- not the matrix multiplies -- are most of
# a training step. But image i's patch rows depend only on image i, so the
# batch splits into `workers` slices, each built by its own thread (NumPy
# releases the GIL while it copies). Same matrix, element for element, so
# every downstream result is bit-identical to workers=1.
# ---------------------------------------------------------------------------
def _batch_slices(n, workers):
    """Split range(n) into up to `workers` contiguous (start, stop) slices."""
    edges = np.linspace(0, n, min(workers, n) + 1).astype(int)
    return list(zip(edges[:-1], edges[1:]))


def _parallel(fn, slices):
    """Run fn(a, b) for every slice on its own thread (a pool per call: no
    state outlives it)."""
    with ThreadPoolExecutor(len(slices)) as pool:
        for f in [pool.submit(fn, a, b) for a, b in slices]:
            f.result()                                # re-raise any error


def im2col(X, KH, KW, stride=1, pad=0, workers=1):
    """(n, C, H, W) -> (n*OH*OW, C*KH*KW); row (i,u,v) holds patch (c,p,q)."""
    n, C, H, W = X.shape
    OH, OW = conv_out_size(H, KH, stride, pad), conv_out_size(W, KW, stride, pad)
    if workers > 1 and n > 1:
        out = np.empty((n * OH * OW, C * KH * KW), dtype=X.dtype)

        def work(a, b):
            out[a * OH * OW:b * OH * OW] = im2col(X[a:b], KH, KW, stride, pad)

        _parallel(work, _batch_slices(n, workers))
        return out
    if pad:
        X = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)))  # zero rings
    # cols[i, c, p, q, u, v] = X[i, c, u*S + p, v*S + q]: one strided slice
    # per kernel tap -- KH*KW slice-copies instead of n*OH*OW patch-copies.
    cols = np.empty((n, C, KH, KW, OH, OW), dtype=X.dtype)
    for p in range(KH):
        for q in range(KW):
            cols[:, :, p, q, :, :] = X[:, :, p:p + stride * OH:stride,
                                       q:q + stride * OW:stride]
    # Rows ordered by (i, u, v), columns by (c, p, q) -- matching the flat
    # (F, C*KH*KW) kernel matrix the forward pass multiplies against.
    return cols.transpose(0, 4, 5, 1, 2, 3).reshape(n * OH * OW, C * KH * KW)


def col2im(cols, x_shape, KH, KW, stride=1, pad=0, workers=1):
    """Adjoint of im2col: scatter-ADD patch rows back into image positions."""
    n, C, H, W = x_shape
    OH, OW = conv_out_size(H, KH, stride, pad), conv_out_size(W, KW, stride, pad)
    if workers > 1 and n > 1:
        dX = np.empty(x_shape, dtype=cols.dtype)

        def work(a, b):
            dX[a:b] = col2im(cols[a * OH * OW:b * OH * OW], (b - a, C, H, W),
                             KH, KW, stride, pad)

        _parallel(work, _batch_slices(n, workers))
        return dX
    cols6 = cols.reshape(n, OH, OW, C, KH, KW).transpose(0, 3, 4, 5, 1, 2)
    dX = np.zeros((n, C, H + 2 * pad, W + 2 * pad), dtype=cols.dtype)
    for p in range(KH):
        for q in range(KW):
            # += (not =): interior pixels sit in many patches, and the chain
            # rule says their gradients accumulate.
            dX[:, :, p:p + stride * OH:stride,
               q:q + stride * OW:stride] += cols6[:, :, p, q, :, :]
    if pad:
        # The padding ring was constant zeros, not an input: its gradient
        # has nowhere to go and is simply dropped.
        dX = np.ascontiguousarray(dX[:, :, pad:pad + H, pad:pad + W])
    return dX


# ---------------------------------------------------------------------------
# Conv2D: the learnable layer.
#   forward:  Y[i,f,u,v] = b[f] + sum_{c,p,q} K[f,c,p,q] * Xp[i,c,u*S+p,v*S+q]
#             (Xp = X zero-padded by `pad` on every side)
#   backward: db = sum dY over (i,u,v)
#             dK[f,c,p,q]  = sum_{i,u,v} dY[i,f,u,v] * Xp[i,c,u*S+p,v*S+q]
#                            (a correlation of the input with the upstream
#                             gradient -- same op as forward, roles swapped)
#             dX[i,c,h,w]  = sum over every (f,u,v,p,q) with u*S+p = h+pad and
#                            v*S+q = w+pad of dY[i,f,u,v] * K[f,c,p,q]
#                            (at stride 1, no pad: a FULL correlation with the
#                             kernel flipped -- i.e. a true convolution; the
#                             flip we skipped in forward comes back by itself)
# Via im2col all three are matrix products against the cached patch matrix.
# ---------------------------------------------------------------------------
class Conv2D:
    """input_grad=False skips dL/dX -- for a first layer, whose input is the
    data itself and needs no gradient (the C API's dX = NULL). backward then
    returns None, and the expensive col2im scatter never runs.

    bias=False (Module 05) drops b, for a conv that feeds a BatchNorm: the
    normalization would subtract any per-channel constant right back out, and
    BatchNorm's beta is the shift that remains (the C API's b = NULL).

    cache_cols=False (Module 05) keeps only the input between the passes and
    re-runs im2col in backward. The patch matrix is KH*KW times the size of
    the input -- for a 56-layer ResNet, gigabytes held from the forward pass
    to the backward -- so trading one extra copy for that memory is what lets
    the deep CIFAR nets train on a laptop. Same values, same gradients.

    workers (Module 05) builds im2col/col2im on that many threads -- see
    im2col. Also bit-identical; it only changes how fast."""

    def __init__(self, in_ch: int, out_ch: int, kh: int, kw: int,
                 stride: int = 1, pad: int = 0, input_grad: bool = True,
                 bias: bool = True, cache_cols: bool = True, workers: int = 1):
        self.K = np.zeros((out_ch, in_ch, kh, kw))
        self.b = np.zeros(out_ch) if bias else None
        self.dK = np.zeros_like(self.K)
        self.db = np.zeros_like(self.b) if bias else None
        self.stride = stride
        self.pad = pad
        self.input_grad = input_grad
        self.cache_cols = cache_cols
        self.workers = workers
        self.cols = None       # cached im2col(X) -- both grads need it
        self.X = None          # ...or just X, to rebuild it (cache_cols=False)
        self.x_shape = None

    def forward(self, X):
        F, C, KH, KW = self.K.shape
        n, _, H, W = X.shape
        OH = conv_out_size(H, KH, self.stride, self.pad)
        OW = conv_out_size(W, KW, self.stride, self.pad)
        self.x_shape = X.shape
        cols = im2col(X, KH, KW, self.stride, self.pad, self.workers)  # (n*OH*OW, C*KH*KW)
        self.cols, self.X = (cols, None) if self.cache_cols else (None, X)
        Kmat = self.K.reshape(F, C * KH * KW)            # each filter, one row
        if self.b is None:
            out = cols @ Kmat.T
        else:
            out = cols @ Kmat.T + self.b                 # ONE matmul = all of it
        return out.reshape(n, OH, OW, F).transpose(0, 3, 1, 2)

    def backward(self, dY):
        F, C, KH, KW = self.K.shape
        n, _, OH, OW = dY.shape
        # Rows back in (i, u, v) order to line up with the cached patch rows.
        dout = dY.transpose(0, 2, 3, 1).reshape(n * OH * OW, F)
        if self.b is not None:
            self.db = dout.sum(axis=0)
        cols = (self.cols if self.cache_cols
                else im2col(self.X, KH, KW, self.stride, self.pad, self.workers))
        self.dK = (dout.T @ cols).reshape(F, C, KH, KW)
        if not self.input_grad:
            return None
        dcols = dout @ self.K.reshape(F, C * KH * KW)    # grad of each patch
        return col2im(dcols, self.x_shape, KH, KW,       # scatter-add -> dX
                      self.stride, self.pad, self.workers)

    def params(self):
        return [self.K] if self.b is None else [self.K, self.b]

    def grads(self):
        return [self.dK] if self.b is None else [self.dK, self.db]

    def astype(self, dtype):
        """Re-store the parameters at another precision (see Sequential.astype)."""
        self.K = self.K.astype(dtype)
        if self.b is not None:
            self.b = self.b.astype(dtype)
        return self


# ---------------------------------------------------------------------------
# Pooling: parameter-free downsampling over PxP windows.
#
# Max pooling keeps only the strongest response in each window ("was the
# feature here at all?"), buying a little translation tolerance and a cut in
# resolution. Its backward is pure routing: the winner takes the whole
# gradient, everyone else gets zero -- the same mask trick as ReLU.
#
# The window stride defaults to P (non-overlapping tiles, Module 03). AlexNet
# (Module 04) uses OVERLAPPING pooling -- 3x3 windows every 2 pixels -- so a
# cell can sit in several windows and win more than one of them; backward
# then *adds* every gradient it won, exactly as the chain rule says for a
# value used twice. Average pooling (LeNet-5's original choice) spreads the
# gradient evenly instead; both are here so the notebooks can ablate them.
# ---------------------------------------------------------------------------
class MaxPool2D:
    def __init__(self, P: int, stride: int | None = None):
        self.P = P
        self.stride = P if stride is None else stride
        self.argmax = None
        self.x_shape = None

    def forward(self, X):
        _, _, H, W = X.shape
        P, S = self.P, self.stride
        OH, OW = conv_out_size(H, P, S), conv_out_size(W, P, S)
        self.x_shape = X.shape
        # Tap (p, q) of EVERY window at once is one strided slice of X; stack
        # the P*P taps (row-major: index p*P+q, the C mirror's scan order) and
        # take the argmax across them. argmax returns the FIRST maximum, so
        # ties break exactly like the C code's strict > scan.
        taps = np.stack([X[:, :, p:p + S * OH:S, q:q + S * OW:S]
                         for p in range(P) for q in range(P)])  # (P*P, n, C, OH, OW)
        self.argmax = taps.argmax(axis=0)
        return np.take_along_axis(taps, self.argmax[None], axis=0)[0]

    def backward(self, dY):
        P, S = self.P, self.stride
        _, _, OH, OW = dY.shape
        dX = np.zeros(self.x_shape, dtype=dY.dtype)
        # Route each output gradient to the input cell that won its window:
        # tap by tap, the windows whose winner was tap (p, q) send their dY
        # there. += so a cell that won two overlapping windows gets both.
        for p in range(P):
            for q in range(P):
                won = np.where(self.argmax == p * P + q, dY, 0.0)
                dX[:, :, p:p + S * OH:S, q:q + S * OW:S] += won
        return dX

    def params(self): return []
    def grads(self): return []


class AvgPool2D:
    """LeNet-5's original pooling (sans its trainable coefficient): the mean
    of each non-overlapping PxP window. Backward spreads dY/(P*P) uniformly
    over the window."""

    def __init__(self, P: int):
        self.P = P
        self.x_shape = None

    def forward(self, X):
        n, C, H, W = X.shape
        P = self.P
        self.x_shape = X.shape
        return X.reshape(n, C, H // P, P, W // P, P).mean(axis=(3, 5))

    def backward(self, dY):
        n, C, H, W = self.x_shape
        P = self.P
        share = dY / (P * P)
        # Broadcast each output cell's share back over its PxP window.
        return (np.broadcast_to(share[:, :, :, None, :, None],
                                (n, C, H // P, P, W // P, P))
                  .reshape(n, C, H, W).copy())

    def params(self): return []
    def grads(self): return []


# ---------------------------------------------------------------------------
# Flatten: the bridge from feature maps (n, C, H, W) to a Linear's (m, n_in).
# In NumPy this is a reshape; in C it is NOTHING AT ALL -- a C-contiguous
# NCHW buffer already *is* the row-major (m, n_in) matrix ng_linear_forward
# expects. The layer exists so Sequential can run the stack uniformly (and
# so backward can restore the shape for the conv layers upstream).
# ---------------------------------------------------------------------------
class Flatten:
    def __init__(self):
        self.x_shape = None

    def forward(self, X):
        self.x_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, dY):
        return dY.reshape(self.x_shape)

    def params(self): return []
    def grads(self): return []


# ---------------------------------------------------------------------------
# Module 05 (ResNet): the two pieces a residual CNN adds around its convs.
#
# GlobalAvgPool2D replaces the big dense head of AlexNet-era networks: each
# channel's whole feature map is averaged to ONE number, so a (n, C, H, W)
# map becomes the (n, C) input of a single Linear classifier. No parameters,
# no fixed input size -- and none of the dense layers that held 96% of
# AlexNet's weights. Backward spreads each dY evenly over its H*W cells.
#
# ZeroPadShortcut is the paper's option (A) for a residual block that halves
# the resolution and widens the channels: the skip path can't be a plain
# identity any more (shapes differ), so it keeps every `stride`-th pixel and
# pads the new channels with zeros -- still parameter-free, still a straight
# path for the gradient. The input channels land in the middle of the output
# (equal zero planes on each side, the common CIFAR-ResNet layout). Its
# backward is the same selection, run in reverse: the zero planes were not
# inputs, so their gradient is dropped. (Option (B), a 1x1 strided Conv2D,
# needs nothing new.)
# ---------------------------------------------------------------------------
class GlobalAvgPool2D:
    def __init__(self):
        self.x_shape = None

    def forward(self, X):
        self.x_shape = X.shape
        return X.mean(axis=(2, 3))

    def backward(self, dY):
        n, C, H, W = self.x_shape
        return np.broadcast_to((dY / (H * W))[:, :, None, None], (n, C, H, W)).copy()

    def params(self): return []
    def grads(self): return []


class ZeroPadShortcut:
    """(n, in_ch, H, W) -> (n, out_ch, ceil(H/S), ceil(W/S)): every S-th
    pixel, channels zero-padded (out_ch - in_ch)//2 on the low side."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        assert out_ch >= in_ch, "the shortcut only widens"
        self.in_ch, self.out_ch, self.stride = in_ch, out_ch, stride
        self.lo = (out_ch - in_ch) // 2
        self.x_shape = None

    def forward(self, X):
        self.x_shape = X.shape
        S, lo = self.stride, self.lo
        sub = X[:, :, ::S, ::S]
        Y = np.zeros((X.shape[0], self.out_ch) + sub.shape[2:], dtype=X.dtype)
        Y[:, lo:lo + self.in_ch] = sub
        return Y

    def backward(self, dY):
        S, lo = self.stride, self.lo
        dX = np.zeros(self.x_shape, dtype=dY.dtype)
        dX[:, :, ::S, ::S] = dY[:, lo:lo + self.in_ch]
        return dX

    def params(self): return []
    def grads(self): return []
