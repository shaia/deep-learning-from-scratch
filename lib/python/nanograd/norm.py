"""
Normalization layers -- Module 04's historical one (and, from Module 05, the
one that replaced it).

LocalResponseNorm is AlexNet's "brightness normalization" (Krizhevsky,
Sutskever & Hinton 2012, Sec. 3.3), inspired by *lateral inhibition* in real
neurons: a unit that fires strongly suppresses its neighbors -- here, the
units at the same pixel in nearby channels. Each activation is divided by a
power of the local energy across channels:

    s[c] = k + alpha * sum_{j = max(0, c-n//2)}^{min(C-1, c+n//2)} a[j]^2
    b[c] = a[c] * s[c]^(-beta)          (k=2, n=5, alpha=1e-4, beta=0.75)

applied independently at every (sample, row, column). The paper credits it
with ~1.2-1.4% of ImageNet top-5/top-1 error; later work (VGG, 2014) found it
bought nothing, and Batch Normalization (Module 05) made it obsolete. It lives
here because the backward pass is a worthwhile chain-rule workout -- a
preview of BatchNorm's -- and because the notebook's historical corner
ablates it.

The backward pass, derived once. b[i] depends on a[m] two ways: directly
(m = i) and through s[i] whenever m is in i's channel window N(i):

    db[i]/da[m] = [m = i] * s[i]^-beta  -  2*alpha*beta * a[i]*a[m] * s[i]^(-beta-1)

The window is symmetric (m in N(i) exactly when i in N(m)), so summing the
upstream gradient over i collapses to

    da[m] = dB[m] * s[m]^-beta  -  2*alpha*beta * a[m] * sum_{i in N(m)} dB[i]*b[i]/s[i]

-- the same windowed channel sum as forward, run on dB*b/s instead of a^2.

NOTE on PyTorch: torch.nn.functional.local_response_norm divides alpha by n
(b = a / (k + alpha/n * sum a^2)^beta). To match the paper with torch, pass
alpha * n. The notebook's framework-mirrors section does exactly that.

BatchNorm (Module 05; Ioffe & Szegedy 2015) is the replacement -- see its
class below for the derivation.
"""

import numpy as np


def _channel_window_sum(A, half):
    """out[:, c] = sum of A[:, j] over j in [c-half, c+half] clipped to [0, C).

    Accumulated offset by offset, d = -half .. +half: for each channel that is
    the same ascending-j order as the C mirror's inner loop (lrn in nn.c), so
    the two sum identically.
    """
    C = A.shape[1]
    out = np.zeros_like(A)
    for d in range(-half, half + 1):
        lo, hi = max(0, -d), min(C, C - d)       # channels c with c+d in [0, C)
        out[:, lo:hi] += A[:, lo + d:hi + d]
    return out


class LocalResponseNorm:
    def __init__(self, size: int = 5, alpha: float = 1e-4, beta: float = 0.75,
                 k: float = 2.0):
        # The window is c-size//2 .. c+size//2, so an even size would quietly
        # be size+1 wide; the paper (and torch) use odd sizes.
        assert size % 2 == 1, "LRN window size must be odd"
        self.size, self.alpha, self.beta, self.k = size, alpha, beta, k
        self.X = self.S = self.Y = None

    def forward(self, X):
        half = self.size // 2
        self.X = X
        self.S = self.k + self.alpha * _channel_window_sum(X * X, half)
        self.Y = X * self.S ** (-self.beta)
        return self.Y

    def backward(self, dY):
        half = self.size // 2
        direct = dY * self.S ** (-self.beta)                   # through a[c] itself
        T = _channel_window_sum(dY * self.Y / self.S, half)    # through the s[i]'s
        return direct - 2.0 * self.alpha * self.beta * self.X * T

    def params(self): return []
    def grads(self): return []


# ---------------------------------------------------------------------------
# Batch normalization (Module 05; Ioffe & Szegedy 2015). LRN tamed activation
# scales with a fixed local rule; BatchNorm measures them and normalizes
# directly. For every channel, over the m values it takes in one batch (for a
# Linear's (m, C) output, m = batch size; for a conv's (n, C, H, W) output,
# m = n*H*W -- every pixel of every image is a sample of that channel):
#
#     mu  = (1/m) sum x_i                 var = (1/m) sum (x_i - mu)^2
#     xhat_i = (x_i - mu) / sqrt(var + eps)      y_i = gamma * xhat_i + beta
#
# so each channel enters the next layer with mean 0 and variance 1, then a
# learned scale and shift (gamma, beta) give back whatever range the network
# actually wants -- including, with gamma = sqrt(var+eps) and beta = mu, the
# identity. beta does a bias's job, which is why the layer before a BatchNorm
# goes without one (Linear/Conv2D bias=False).
#
# The backward pass, derived once. Every x_j moves every xhat_i -- directly,
# and through mu and var, which all m values share -- so with g_i = dL/dy_i,
#
#     dxhat_i/dx_j = invstd * ([i = j] - 1/m - xhat_i * xhat_j / m)
#
# and summing g_i * gamma over i collapses, with dbeta = sum g_i and
# dgamma = sum g_i * xhat_i, to
#
#     dx_j = (gamma * invstd / m) * (m * g_j - dbeta - xhat_j * dgamma).
#
# The two subtracted terms remove the gradient components along the two
# directions the forward pass made irrelevant: shifting every x of the
# channel by a constant (mu absorbs it -- exactly, so sum_j dx_j = 0) and
# scaling them all (var absorbs it -- up to the eps in sqrt(var + eps), so
# sum_j dx_j * xhat_j is tiny rather than zero). The very sums that give
# dgamma and dbeta assemble dx.
#
# Training vs evaluation. A batch's statistics are only an estimate, and at
# test time there may be no batch at all, so training also keeps running
# averages (PyTorch's convention: momentum = weight of the newest batch)
#
#     running_mean <- (1 - momentum) * running_mean + momentum * mu
#     running_var  <- (1 - momentum) * running_var  + momentum * var * m/(m-1)
#
# (the unbiased variance -- the paper's inference estimate), and eval() swaps
# them in for mu and var. Then each channel is a fixed affine map, so the
# backward is just dx = g * gamma * invstd. Like Dropout, the layer has a
# `training` flag that Sequential.train()/.eval() flip.
#
# lib/c/nanograd/norm.c's ng_batchnorm_* implement the same formulas, written
# in the same operation order.
# ---------------------------------------------------------------------------
class BatchNorm:
    def __init__(self, C: int, eps: float = 1e-5, momentum: float = 0.1):
        self.gamma = np.ones(C)
        self.beta = np.zeros(C)
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)
        self.running_mean = np.zeros(C)
        self.running_var = np.ones(C)
        self.eps, self.momentum = eps, momentum
        self.training = True
        self.xhat = self.invstd = None

    @staticmethod
    def _axes(X):
        """The axes a channel's statistics run over: all but axis 1."""
        return (0,) + tuple(range(2, X.ndim))

    @staticmethod
    def _per_channel(v, X):
        """Reshape a (C,) vector to broadcast against X: (1, C) or (1, C, 1, 1)."""
        return v.reshape((1, -1) + (1,) * (X.ndim - 2))

    def forward(self, X):
        axes, pc = self._axes(X), self._per_channel
        if self.training:
            m = X.size // X.shape[1]
            assert m > 1, "BatchNorm needs more than one value per channel to train"
            mu = X.mean(axis=axes)
            var = ((X - pc(mu, X)) ** 2).mean(axis=axes)     # biased: this batch's spread
            mom = self.momentum
            self.running_mean = (1 - mom) * self.running_mean + mom * mu
            self.running_var = (1 - mom) * self.running_var + mom * (var * m / (m - 1))
            self.invstd = 1.0 / np.sqrt(var + self.eps)
            self.xhat = (X - pc(mu, X)) * pc(self.invstd, X)
        else:
            self.invstd = 1.0 / np.sqrt(self.running_var + self.eps)
            self.xhat = (X - pc(self.running_mean, X)) * pc(self.invstd, X)
        return pc(self.gamma, X) * self.xhat + pc(self.beta, X)

    def backward(self, dY):
        axes, pc = self._axes(dY), self._per_channel
        self.dbeta = dY.sum(axis=axes)
        self.dgamma = (dY * self.xhat).sum(axis=axes)
        if not self.training:            # fixed statistics: a per-channel affine map
            return dY * pc(self.gamma, dY) * pc(self.invstd, dY)
        m = dY.size // dY.shape[1]
        k = pc(self.gamma * self.invstd / m, dY)
        return k * (m * dY - pc(self.dbeta, dY) - self.xhat * pc(self.dgamma, dY))

    def params(self):
        return [self.gamma, self.beta]

    def grads(self):
        return [self.dgamma, self.dbeta]

    def buffers(self):
        """State that is not learned by gradient descent but must be saved
        with the weights: the running statistics evaluation uses."""
        return [self.running_mean, self.running_var]

    def astype(self, dtype):
        """Re-store parameters and running statistics at another precision."""
        self.gamma = self.gamma.astype(dtype)
        self.beta = self.beta.astype(dtype)
        self.running_mean = self.running_mean.astype(dtype)
        self.running_var = self.running_var.astype(dtype)
        return self
