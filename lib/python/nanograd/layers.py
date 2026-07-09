"""
Layers for nanograd -- the modular refactor of Module 01's by-hand backprop.

Module 01 wrote one monolithic forward/backward for a fixed 2->h->1 network.
That was the point: see the whole chain rule with no abstraction. Now that we
have seen it, we factor it into *layers* -- each a tiny object that knows how to
push a signal forward and a gradient backward through itself. There is still no
autograd magic: every `backward()` is the same chain rule, written out by hand,
just localized to one operation. Stacking them (see net.Sequential) reconstructs
the full network, and swapping them is how Module 02 runs its ablations.

Contract for every layer:
    forward(X)  -> output; caches whatever backward needs.
    backward(dOut) -> dInput (dL/dInput); parameter layers also set self.dW/self.db.

Gradients are *summed* over the batch here (not averaged); the 1/m is applied
once, at the loss, exactly as in Module 01 -- so the arithmetic matches.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Linear (fully-connected): the only layer with learnable parameters.
#   forward:  Y = X W + b            X:(m, n_in)  W:(n_in, n_out)  b:(n_out,)
#   backward: dX = dY W^T,  dW = X^T dY,  db = sum_rows dY
# ---------------------------------------------------------------------------
class Linear:
    def __init__(self, n_in: int, n_out: int):
        self.W = np.zeros((n_in, n_out))
        self.b = np.zeros(n_out)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.X = None

    def forward(self, X):
        self.X = X                       # cache input for the weight gradient
        return X @ self.W + self.b

    def backward(self, dY):
        self.dW = self.X.T @ dY          # dL/dW = X^T dY
        self.db = dY.sum(axis=0)         # dL/db = column sums of dY
        return dY @ self.W.T             # dL/dX, handed to the previous layer

    def params(self):
        return [self.W, self.b]

    def grads(self):
        return [self.dW, self.db]


# ---------------------------------------------------------------------------
# Activations: no parameters, so backward just multiplies by the local slope.
# This is the file where Module 02's central lesson lives -- the three curves
# and, crucially, their derivatives, which decide whether gradients survive.
# ---------------------------------------------------------------------------
def _sigmoid(Z):
    """Overflow-safe logistic sigmoid (branch on the sign of Z)."""
    out = np.empty_like(Z, dtype=np.float64)
    pos = Z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-Z[pos]))
    ez = np.exp(Z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


class Sigmoid:
    """sigma(z) = 1/(1+e^-z);  sigma'(z) = a(1-a). Saturates for |z| large."""
    def forward(self, Z):
        self.A = _sigmoid(Z)
        return self.A

    def backward(self, dA):
        return dA * self.A * (1.0 - self.A)

    def params(self): return []
    def grads(self): return []


class Tanh:
    """tanh(z);  tanh'(z) = 1 - a^2. Zero-centered -- usually beats sigmoid,
    but still saturates at the tails (the vanishing-gradient failure mode)."""
    def forward(self, Z):
        self.A = np.tanh(Z)
        return self.A

    def backward(self, dA):
        return dA * (1.0 - self.A * self.A)

    def params(self): return []
    def grads(self): return []


class ReLU:
    """max(0, z);  derivative is 1 where z>0 else 0. No saturation on the
    positive side -- the single change that let very deep nets train (Module 04)."""
    def forward(self, Z):
        self.mask = Z > 0.0
        return Z * self.mask

    def backward(self, dA):
        return dA * self.mask         # gradient passes only through active units

    def params(self): return []
    def grads(self): return []


ACTIVATIONS = {"sigmoid": Sigmoid, "tanh": Tanh, "relu": ReLU}


# ---------------------------------------------------------------------------
# Softmax + cross-entropy, fused. As in Module 01, fusing them makes the
# backward collapse to the clean (P - Y)/m -- the reason this pairing is
# universal. This is a *loss* layer: forward takes the labels too.
# ---------------------------------------------------------------------------
class SoftmaxCrossEntropy:
    def forward(self, logits, y):
        """Row-wise softmax of `logits`, then mean cross-entropy against the
        integer class labels `y`. Caches probabilities and labels for backward."""
        Z = logits - logits.max(axis=1, keepdims=True)   # stabilize
        e = np.exp(Z)
        self.P = e / e.sum(axis=1, keepdims=True)
        self.y = y
        self.m = logits.shape[0]
        return float(-np.log(self.P[np.arange(self.m), y] + 1e-12).mean())

    def backward(self):
        """dL/dlogits = (P - Y) / m, with Y the one-hot of the labels."""
        dZ = self.P.copy()
        dZ[np.arange(self.m), self.y] -= 1.0
        return dZ / self.m
