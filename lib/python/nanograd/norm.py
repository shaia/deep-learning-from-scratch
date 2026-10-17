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
