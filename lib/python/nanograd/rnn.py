"""
Recurrent layers -- Module 06's addition to nanograd (Elman 1990; Werbos 1990).

Every layer so far has mapped one fixed-size input to one output. A recurrent
layer reads a SEQUENCE x<1>, x<2>, ..., x<T>, one step at a time, and carries a
hidden state h from each step to the next -- the network's memory of what it
has read so far:

    z<t> = x<t> Wxh + h<t-1> Whh + bh          h<t> = tanh(z<t>)

The same three parameters are used at every step. That is the whole idea, and
it buys two things at once: the layer handles sequences of any length with a
fixed number of weights, and whatever it learns about step 3 it automatically
knows about step 300 (Module 03's weight sharing, across time instead of
across an image).

Unrolled, a T-step sequence is a T-layer network whose layers share weights.
Backpropagation through that unrolled network is "backpropagation through
time" (BPTT): walk the steps in reverse, and at each one the gradient reaching
h<t> comes from two places -- the step's own output, and the future, through
h<t+1>:

    dh<t> = dH<t> + dz<t+1> Whh^T        dz<t> = dh<t> * (1 - h<t>^2)

A shared weight is a value used T times, so -- the chain rule's rule for a
value used more than once, which Module 05's shortcut also obeyed -- its
gradient is the SUM of its gradients at every step:

    dWxh = sum_t x<t>^T dz<t>     dWhh = sum_t h<t-1>^T dz<t>     dbh = sum_t sum_rows dz<t>

The trouble is the middle line of the first block. Going back k steps
multiplies the gradient by k copies of diag(1 - h^2) Whh^T -- the SAME matrix
over and over -- so it shrinks or grows geometrically with k. That is the
vanishing / exploding gradient problem in its purest form (Hochreiter 1991;
Bengio, Simard & Frasconi 1994), and the reason Module 07 exists.
clip_grad_norm (optim.py) is the standard guard against the exploding half.

Shapes are time-major: X is (T, m, n_in) -- T steps of an (m, n_in) batch, the
same row-major "rows = samples" layout as every other layer, stacked in time.
forward(X, h0) returns H, every hidden state, shape (T, m, n_h); a Linear on
H.reshape(T*m, n_h) reads them all out at once. Passing the previous chunk's
last state as h0 is how a long text is trained in short pieces ("truncated"
BPTT): the state flows forward across the cut, the gradient does not.

Contract (as layers.py): forward caches what backward needs; backward(dH)
returns dX, sets dWxh/dWhh/dbh, and leaves dL/dh0 in self.dh0 and every
step's dL/dz in self.dZ (the gradient through time, for inspection).
"""

import numpy as np


class RNN:
    """A vanilla (Elman) tanh recurrent layer: n_in inputs, n_h hidden units.

    Parameters: Wxh (n_in, n_h), Whh (n_h, n_h), bh (n_h,) -- initialized to
    zero here; fill them with an init scheme (xavier_normal suits tanh)."""

    def __init__(self, n_in: int, n_h: int):
        self.Wxh = np.zeros((n_in, n_h))
        self.Whh = np.zeros((n_h, n_h))
        self.bh = np.zeros(n_h)
        self.dWxh = np.zeros_like(self.Wxh)
        self.dWhh = np.zeros_like(self.Whh)
        self.dbh = np.zeros_like(self.bh)

    def forward(self, X, h0=None):
        """X: (T, m, n_in) -> H: (T, m, n_h). h0: (m, n_h) or None (zeros)."""
        T, m, _ = X.shape
        n_h = self.Whh.shape[0]
        self.X = X
        self.h0 = np.zeros((m, n_h), dtype=X.dtype) if h0 is None else h0
        # The input's contribution has no dependence on the past, so all T steps
        # of it are one matrix product; only the recurrence must be a loop.
        XW = X @ self.Wxh + self.bh
        H = np.empty((T, m, n_h), dtype=XW.dtype)
        h = self.h0
        for t in range(T):
            h = np.tanh(XW[t] + h @ self.Whh)        # the recurrence
            H[t] = h
        self.H = H
        return H

    def backward(self, dH, truncate=0):
        """dH: (T, m, n_h), dL/dh<t> from whatever read each step's state (a
        gradient arriving at the final state from beyond the sequence belongs
        in dH[-1]). Walks time in reverse; returns dX (T, m, n_in).

        truncate=k > 0 cuts the gradient's path into the past every k steps:
        nothing flows from step t back to step t-1 (or to h0) when t % k == 0,
        exactly as if the sequence had been trained in chunks of k steps with
        the state carried across. truncate=1 is Elman's (1990) training -- the
        previous state treated as just another input -- and 0 is full BPTT."""
        H, Whh = self.H, self.Whh
        dZ = np.empty_like(H)
        dh_next = np.zeros_like(H[0])                # nothing flows in from after T
        for t in range(H.shape[0] - 1, -1, -1):
            dh = dH[t] + dh_next                     # from the output at t, and from t+1
            dZ[t] = dh * (1.0 - H[t] * H[t])         # back through tanh
            if truncate and t % truncate == 0:
                dh_next = np.zeros_like(dh_next)     # the cut: no further into the past
            else:
                dh_next = dZ[t] @ Whh.T              # ...one step further into the past
        self.dh0 = dh_next
        self.dZ = dZ
        # The shared weights: every step's share, summed (one product over T*m rows).
        n_in, n_h = self.Wxh.shape
        H_prev = np.concatenate([self.h0[None], H[:-1]])
        Z2 = dZ.reshape(-1, n_h)
        self.dWxh = self.X.reshape(-1, n_in).T @ Z2
        self.dWhh = H_prev.reshape(-1, n_h).T @ Z2
        self.dbh = Z2.sum(axis=0)
        return dZ @ self.Wxh.T

    def params(self):
        return [self.Wxh, self.Whh, self.bh]

    def grads(self):
        return [self.dWxh, self.dWhh, self.dbh]

    def astype(self, dtype):
        """Re-store the parameters at another precision (see Sequential.astype)."""
        self.Wxh = self.Wxh.astype(dtype)
        self.Whh = self.Whh.astype(dtype)
        self.bh = self.bh.astype(dtype)
        return self
