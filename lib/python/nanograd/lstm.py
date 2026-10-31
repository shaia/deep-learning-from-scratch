"""
Long short-term memory -- Module 07's addition to nanograd (Hochreiter &
Schmidhuber 1997; the forget gate from Gers, Schmidhuber & Cummins 2000).

Module 06's recurrent layer rewrites its whole state at every step,
h<t> = tanh(x<t> Wxh + h<t-1> Whh + bh), so the gradient going back k steps is
multiplied by k copies of diag(1 - h^2) Whh^T and vanishes or explodes. The
LSTM keeps a second state, the CELL c, that is not rewritten but ADDED TO,
under three gates -- sigmoids between 0 and 1, computed from the input and the
previous hidden state like any layer:

    z<t> = x<t> Wx + h<t-1> Wh + b                 (all four blocks at once)
    i = sigma(z_i)    input gate:  how much of the candidate to write
    f = sigma(z_f)    forget gate: how much of the old cell to keep
    g = tanh(z_g)     candidate:   what to write
    o = sigma(z_o)    output gate: how much of the cell to show

    c<t> = f * c<t-1> + i * g                      h<t> = o * tanh(c<t>)

Along the cell, one step back multiplies the gradient by f -- no weight
matrix, no squashing slope -- and f is a number the network sets for itself,
per unit and per step. Hold f at 1 and the gradient travels any distance
unchanged: Hochreiter and Schmidhuber's "constant error carousel" (their 1997
cell had no forget gate at all, f = 1 always). Close the input gate and
nothing disturbs what is stored. Module 05's shortcut through depth, run
through time.

Backpropagation through time now carries TWO gradients from the future:

    dh<t> = dH<t> + dz<t+1> Wh^T                   (through the gates, as the RNN)
    dc<t> = dc<t+1> * f<t+1> + dh<t> * o * (1 - tanh(c<t>)^2)   (the carousel)

and every gate's pre-activation gradient follows from the product rule:
dz_i = dc g i(1-i), dz_f = dc c<t-1> f(1-f), dz_g = dc i (1-g^2), dz_o =
dh tanh(c) o(1-o). The shared weights sum their shares over every step, as
the RNN's do.

Parameters: Wx (n_in, 4 n_h), Wh (n_h, 4 n_h), b (4 n_h), the four gate blocks
side by side in the order [i | f | g | o] -- PyTorch's order, so its weights
copy straight in (transposed). Shapes are time-major, as RNN: X is (T, m,
n_in), forward(X, h0, c0) returns H (T, m, n_h) and keeps the cells in self.C.

Contract (as rnn.py): forward caches what backward needs; backward(dH)
returns dX, sets dWx/dWh/db, leaves dL/dh0 and dL/dc0 in self.dh0/self.dc0,
every step's gate gradient in self.dZ (T, m, 4 n_h) and the cell's gradient
through time, dL/dc<t>, in self.dC (T, m, n_h).
"""

import numpy as np


def _sigmoid(Z):
    """Overflow-safe logistic sigmoid in Z's own dtype: 1/(1 + e^-z) for z >= 0,
    e^z/(1 + e^z) below -- the formula lib/c/nanograd uses, so the float64
    results agree with C's to the last bit wherever exp does."""
    E = np.exp(-np.abs(Z))
    return np.where(Z >= 0, 1.0 / (1.0 + E), E / (1.0 + E))


class LSTM:
    """An LSTM layer with forget gate: n_in inputs, n_h cells.

    Parameters are zero here; fill Wx and Wh with an init scheme (xavier_normal
    suits the sigmoid/tanh gates) and set the gate biases with gate_bias().
    The forget gate's bias is the one that matters: sigma(b_f) is how much of
    the cell survives each step at the start of training, so the cell's memory
    initially lasts about 1/(1 - sigma(b_f)) steps -- 2 at b_f = 0, 3.7 at the
    customary b_f = 1 (Jozefowicz, Zaremba & Sutskever 2015), 150 at b_f = 5."""

    def __init__(self, n_in: int, n_h: int):
        self.Wx = np.zeros((n_in, 4 * n_h))
        self.Wh = np.zeros((n_h, 4 * n_h))
        self.b = np.zeros(4 * n_h)
        self.dWx = np.zeros_like(self.Wx)
        self.dWh = np.zeros_like(self.Wh)
        self.db = np.zeros_like(self.b)

    @property
    def n_h(self):
        return self.Wh.shape[0]

    def gate_bias(self, i=None, f=None, g=None, o=None):
        """Set whole bias blocks in place -- gate_bias(f=1.0), say, for the
        customary open forget gate; blocks not named are left alone. Returns
        self."""
        n = self.n_h
        for k, value in enumerate((i, f, g, o)):
            if value is not None:
                self.b[k * n:(k + 1) * n] = value
        return self

    def forward(self, X, h0=None, c0=None):
        """X: (T, m, n_in) -> H: (T, m, n_h). h0, c0: (m, n_h) or None (zeros)."""
        T, m, _ = X.shape
        n = self.n_h
        zeros = np.zeros((m, n), dtype=X.dtype)
        self.X = X
        self.h0 = zeros if h0 is None else h0
        self.c0 = zeros if c0 is None else c0
        # As in RNN: the input's share of every step is one matrix product.
        XW = X @ self.Wx + self.b
        G = np.empty((T, m, 4 * n), dtype=XW.dtype)     # the activated gates i, f, g, o
        C = np.empty((T, m, n), dtype=XW.dtype)
        H = np.empty((T, m, n), dtype=XW.dtype)
        h, c = self.h0, self.c0
        for t in range(T):
            z = XW[t] + h @ self.Wh
            gates = G[t]
            gates[:, :2 * n] = _sigmoid(z[:, :2 * n])     # i, f
            gates[:, 2 * n:3 * n] = np.tanh(z[:, 2 * n:3 * n])   # g
            gates[:, 3 * n:] = _sigmoid(z[:, 3 * n:])      # o
            i, f, g, o = gates[:, :n], gates[:, n:2 * n], gates[:, 2 * n:3 * n], gates[:, 3 * n:]
            c = f * c + i * g                              # keep some, write some
            h = o * np.tanh(c)                             # show some
            C[t], H[t] = c, h
        self.G, self.C, self.H = G, C, H
        self.TC = np.tanh(C)
        return H

    def backward(self, dH, truncate=0, cell_only=False):
        """dH: (T, m, n_h), dL/dh<t> from whatever read each step's state.
        Walks time in reverse; returns dX (T, m, n_in).

        truncate=k > 0 cuts BOTH paths into the past every k steps (no gradient
        from step t back to step t-1, or to h0/c0, when t % k == 0): chunked
        training, as RNN.backward. cell_only=True is Hochreiter & Schmidhuber's
        1997 truncated gradient: nothing flows from the gates back into h<t-1>
        at any step, so the error travels back in time only along the cell --
        the constant error carousel, alone."""
        n = self.n_h
        G, C, TC, H = self.G, self.C, self.TC, self.H
        T = H.shape[0]
        dZ = np.empty_like(G)
        dC = np.empty_like(C)
        dh_next = np.zeros_like(H[0])                   # nothing flows in from after T
        dc_next = np.zeros_like(C[0])
        for t in range(T - 1, -1, -1):
            i, f, g, o = G[t, :, :n], G[t, :, n:2 * n], G[t, :, 2 * n:3 * n], G[t, :, 3 * n:]
            c_prev = C[t - 1] if t > 0 else self.c0
            dh = dH[t] + dh_next                        # from the output at t, and from t+1
            dc = dc_next + dh * o * (1.0 - TC[t] * TC[t])   # along the carousel, and via h<t>
            dC[t] = dc
            dz = dZ[t]
            dz[:, :n] = dc * g * (i * (1.0 - i))              # input gate
            dz[:, n:2 * n] = dc * c_prev * (f * (1.0 - f))    # forget gate
            dz[:, 2 * n:3 * n] = dc * i * (1.0 - g * g)       # candidate
            dz[:, 3 * n:] = dh * TC[t] * (o * (1.0 - o))      # output gate
            if truncate and t % truncate == 0:          # the cut: no further into the past
                dh_next = np.zeros_like(dh_next)
                dc_next = np.zeros_like(dc_next)
            else:
                dc_next = dc * f                        # one step back along the cell
                dh_next = np.zeros_like(dh_next) if cell_only else dz @ self.Wh.T
        self.dh0, self.dc0 = dh_next, dc_next
        self.dZ, self.dC = dZ, dC
        # The shared weights: every step's share, summed (one product over T*m rows).
        n_in = self.Wx.shape[0]
        H_prev = np.concatenate([self.h0[None], H[:-1]])
        Z2 = dZ.reshape(-1, 4 * n)
        self.dWx = self.X.reshape(-1, n_in).T @ Z2
        self.dWh = H_prev.reshape(-1, n).T @ Z2
        self.db = Z2.sum(axis=0)
        return dZ @ self.Wx.T

    def params(self):
        return [self.Wx, self.Wh, self.b]

    def grads(self):
        return [self.dWx, self.dWh, self.db]

    def astype(self, dtype):
        """Re-store the parameters at another precision (see Sequential.astype)."""
        self.Wx = self.Wx.astype(dtype)
        self.Wh = self.Wh.astype(dtype)
        self.b = self.b.astype(dtype)
        return self
