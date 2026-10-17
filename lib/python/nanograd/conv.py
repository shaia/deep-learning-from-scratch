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
# ---------------------------------------------------------------------------
def im2col(X, KH, KW, stride=1, pad=0):
    """(n, C, H, W) -> (n*OH*OW, C*KH*KW); row (i,u,v) holds patch (c,p,q)."""
    n, C, H, W = X.shape
    OH, OW = conv_out_size(H, KH, stride, pad), conv_out_size(W, KW, stride, pad)
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


def col2im(cols, x_shape, KH, KW, stride=1, pad=0):
    """Adjoint of im2col: scatter-ADD patch rows back into image positions."""
    n, C, H, W = x_shape
    OH, OW = conv_out_size(H, KH, stride, pad), conv_out_size(W, KW, stride, pad)
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
    returns None, and the expensive col2im scatter never runs."""

    def __init__(self, in_ch: int, out_ch: int, kh: int, kw: int,
                 stride: int = 1, pad: int = 0, input_grad: bool = True):
        self.K = np.zeros((out_ch, in_ch, kh, kw))
        self.b = np.zeros(out_ch)
        self.dK = np.zeros_like(self.K)
        self.db = np.zeros_like(self.b)
        self.stride = stride
        self.pad = pad
        self.input_grad = input_grad
        self.cols = None       # cached im2col(X) -- both grads need it
        self.x_shape = None

    def forward(self, X):
        F, C, KH, KW = self.K.shape
        n, _, H, W = X.shape
        OH = conv_out_size(H, KH, self.stride, self.pad)
        OW = conv_out_size(W, KW, self.stride, self.pad)
        self.x_shape = X.shape
        self.cols = im2col(X, KH, KW, self.stride, self.pad)  # (n*OH*OW, C*KH*KW)
        Kmat = self.K.reshape(F, C * KH * KW)            # each filter, one row
        out = self.cols @ Kmat.T + self.b                # ONE matmul = all of it
        return out.reshape(n, OH, OW, F).transpose(0, 3, 1, 2)

    def backward(self, dY):
        F, C, KH, KW = self.K.shape
        n, _, OH, OW = dY.shape
        # Rows back in (i, u, v) order to line up with the cached patch rows.
        dout = dY.transpose(0, 2, 3, 1).reshape(n * OH * OW, F)
        self.db = dout.sum(axis=0)
        self.dK = (dout.T @ self.cols).reshape(F, C, KH, KW)
        if not self.input_grad:
            return None
        dcols = dout @ self.K.reshape(F, C * KH * KW)    # grad of each patch
        return col2im(dcols, self.x_shape, KH, KW,       # scatter-add -> dX
                      self.stride, self.pad)

    def params(self):
        return [self.K, self.b]

    def grads(self):
        return [self.dK, self.db]

    def astype(self, dtype):
        """Re-store the parameters at another precision (see Sequential.astype)."""
        self.K = self.K.astype(dtype)
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
