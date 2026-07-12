"""
Convolution and pooling layers -- Module 03's additions to nanograd.

A convolutional layer is a Linear layer with two constraints bolted on:
*locality* (each output looks at a small patch, not the whole input) and
*weight sharing* (every patch is scored by the same small kernel). Everything
else -- the caching, the hand-written backward, the params/grads contract --
is exactly the machinery of layers.py, reused.

Two implementations of the same arithmetic live in this module's orbit:

  * Here: the *vectorized* forward/backward via im2col -- convolution
    rearranged into one big matrix multiply. This is what trains MNIST in
    minutes and it is (a rounding-error away from) how real frameworks do it.
  * In topics/03-lenet/python/lenet.py: the *naive* quadruple loop, which is
    the definition itself and the thing the C mirror implements verbatim.
    tests/test_agreement.py asserts the two match to 1e-12.

Conventions (shared with lib/c/nanograd/conv.c -- see nanograd.h):
  * Tensors are NCHW: X[i, c, h, w], C-contiguous, so a flat view of X is
    exactly the row-major buffer the C code walks.
  * Kernels are (F, C, KH, KW): F filters, each spanning all C input channels.
    The array is named K (W is taken -- it means image width here).
  * "conv" is cross-correlation (no kernel flip), stride 1, valid (no padding)
    -- all LeNet needs, and the same choice every modern framework makes. The
    flip is not lost: it reappears on its own in the backward pass for dX.
  * Gradients are *summed* over the batch; the 1/m lives in the loss layer.
"""

import numpy as np


# ---------------------------------------------------------------------------
# im2col / col2im: convolution as a matrix multiply.
#
# Every output position (u, v) of a valid convolution reads the patch
# X[:, u:u+KH, v:v+KW]. im2col copies each such patch out into one ROW of a
# big matrix, so all patches can be scored against all filters in a single
# GEMM: (n*OH*OW, C*KH*KW) @ (C*KH*KW, F). Pixels are duplicated across rows
# (a 5x5 kernel copies each interior pixel 25 times) -- that redundancy is
# the price of turning a sliding window into the one operation computers are
# fastest at. col2im is the exact adjoint: it scatters row-gradients back
# and *sums* where patches overlapped, which is precisely dL/dX.
# ---------------------------------------------------------------------------
def im2col(X, KH, KW):
    """(n, C, H, W) -> (n*OH*OW, C*KH*KW); row (i,u,v) holds patch (c,p,q)."""
    n, C, H, W = X.shape
    OH, OW = H - KH + 1, W - KW + 1
    # cols[i, c, p, q, u, v] = X[i, c, u+p, v+q]: one shifted slice per kernel
    # tap -- KH*KW slice-copies instead of n*OH*OW patch-copies.
    cols = np.empty((n, C, KH, KW, OH, OW), dtype=X.dtype)
    for p in range(KH):
        for q in range(KW):
            cols[:, :, p, q, :, :] = X[:, :, p:p + OH, q:q + OW]
    # Rows ordered by (i, u, v), columns by (c, p, q) -- matching the flat
    # (F, C*KH*KW) kernel matrix the forward pass multiplies against.
    return cols.transpose(0, 4, 5, 1, 2, 3).reshape(n * OH * OW, C * KH * KW)


def col2im(cols, x_shape, KH, KW):
    """Adjoint of im2col: scatter-ADD patch rows back into image positions."""
    n, C, H, W = x_shape
    OH, OW = H - KH + 1, W - KW + 1
    cols6 = cols.reshape(n, OH, OW, C, KH, KW).transpose(0, 3, 4, 5, 1, 2)
    dX = np.zeros(x_shape, dtype=cols.dtype)
    for p in range(KH):
        for q in range(KW):
            # += (not =): interior pixels sit in many patches, and the chain
            # rule says their gradients accumulate.
            dX[:, :, p:p + OH, q:q + OW] += cols6[:, :, p, q, :, :]
    return dX


# ---------------------------------------------------------------------------
# Conv2D: the learnable layer.
#   forward:  Y[i,f,u,v] = b[f] + sum_{c,p,q} K[f,c,p,q] * X[i,c,u+p,v+q]
#   backward: db = sum dY over (i,u,v)
#             dK[f,c,p,q]  = sum_{i,u,v} dY[i,f,u,v] * X[i,c,u+p,v+q]
#                            (a correlation of the input with the upstream
#                             gradient -- same op as forward, roles swapped)
#             dX[i,c,h,w]  = sum_{f,p,q} dY[i,f,h-p,w-q] * K[f,c,p,q]
#                            (a FULL correlation with the kernel flipped --
#                             i.e. a true convolution; the flip we skipped in
#                             forward comes back by itself)
# Via im2col all three are matrix products against the cached patch matrix.
# ---------------------------------------------------------------------------
class Conv2D:
    def __init__(self, in_ch: int, out_ch: int, kh: int, kw: int):
        self.K = np.zeros((out_ch, in_ch, kh, kw))
        self.b = np.zeros(out_ch)
        self.dK = np.zeros_like(self.K)
        self.db = np.zeros_like(self.b)
        self.cols = None       # cached im2col(X) -- both grads need it
        self.x_shape = None

    def forward(self, X):
        F, C, KH, KW = self.K.shape
        n, _, H, W = X.shape
        OH, OW = H - KH + 1, W - KW + 1
        self.x_shape = X.shape
        self.cols = im2col(X, KH, KW)                    # (n*OH*OW, C*KH*KW)
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
        dcols = dout @ self.K.reshape(F, C * KH * KW)    # grad of each patch
        return col2im(dcols, self.x_shape, KH, KW)       # scatter-add -> dX

    def params(self):
        return [self.K, self.b]

    def grads(self):
        return [self.dK, self.db]


# ---------------------------------------------------------------------------
# Pooling: parameter-free downsampling over non-overlapping PxP windows.
#
# Max pooling keeps only the strongest response in each window ("was the
# feature here at all?"), buying a little translation tolerance and a 4x
# cut in resolution per 2x2 stage. Its backward is pure routing: the winner
# takes the whole gradient, everyone else gets zero -- the same mask trick
# as ReLU. Average pooling (LeNet-5's original choice) spreads the gradient
# evenly instead; both are here so the notebook can ablate max vs avg.
# ---------------------------------------------------------------------------
class MaxPool2D:
    def __init__(self, P: int):
        self.P = P
        self.argmax = None
        self.x_shape = None

    def forward(self, X):
        n, C, H, W = X.shape
        P = self.P
        OH, OW = H // P, W // P
        self.x_shape = X.shape
        # Expose each PxP window as one axis of length P*P (row-major within
        # the window: index p*P+q -- the same scan order the C mirror uses,
        # so first-occurrence argmax breaks ties identically).
        win = (X.reshape(n, C, OH, P, OW, P)
                .transpose(0, 1, 2, 4, 3, 5)
                .reshape(n, C, OH, OW, P * P))
        self.argmax = win.argmax(axis=-1)
        return np.take_along_axis(win, self.argmax[..., None], axis=-1)[..., 0]

    def backward(self, dY):
        n, C, H, W = self.x_shape
        P = self.P
        OH, OW = H // P, W // P
        dwin = np.zeros((n, C, OH, OW, P * P), dtype=dY.dtype)
        # Route each output gradient to the input cell that won the max.
        np.put_along_axis(dwin, self.argmax[..., None], dY[..., None], axis=-1)
        return (dwin.reshape(n, C, OH, OW, P, P)
                    .transpose(0, 1, 2, 4, 3, 5)     # undo the window packing
                    .reshape(n, C, H, W))

    def params(self): return []
    def grads(self): return []


class AvgPool2D:
    """LeNet-5's original pooling (sans its trainable coefficient): the mean
    of each PxP window. Backward spreads dY/(P*P) uniformly over the window."""

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
