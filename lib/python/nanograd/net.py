"""
Sequential container + an MLP builder -- how the layers get stacked.

`Sequential` is the whole "framework" and it is deliberately tiny: forward runs
the layers in order, backward runs them in reverse handing each the gradient
from the next. That reversed walk *is* backpropagation; here it is four lines.
`params()`/`grads()` flatten the learnable tensors (in a stable order) for an
optimizer to consume.

`mlp(...)` wires the Module-01 network -- Linear, activation, Linear, ... -- but
now every knob Module 02 studies (activation function, initialization) is an
argument, which is exactly what makes the ablations one-liners.
"""

from . import init as init_mod
from .layers import ACTIVATIONS, Linear
from .rng import Rng


class Sequential:
    """A straight stack of layers. The last layer's output is the logits; the
    loss layer (SoftmaxCrossEntropy) is applied separately by the trainer."""

    def __init__(self, layers):
        self.layers = layers

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, dOut):
        for layer in reversed(self.layers):     # backprop = the reversed walk
            dOut = layer.backward(dOut)
        return dOut

    def params(self):
        return [p for layer in self.layers for p in layer.params()]

    def grads(self):
        return [g for layer in self.layers for g in layer.grads()]


def mlp(sizes, activation="relu", init="he", rng=None, seed=0):
    """Build an MLP: `sizes` = [n_in, h1, h2, ..., n_out].

    Between every pair of sizes goes a Linear then the chosen activation; the
    final Linear is left bare (its outputs are logits for SoftmaxCrossEntropy).
    Weights are drawn from `rng` using the named init scheme so the C mirror,
    fed the same seed, produces identical parameters.
    """
    if rng is None:
        rng = Rng(seed)
    init_fn = init_mod.INITS[init]
    act_cls = ACTIVATIONS[activation]

    layers = []
    for k in range(len(sizes) - 1):
        lin = Linear(sizes[k], sizes[k + 1])
        init_fn(lin.W, rng)          # fill weights (row-major draw order)
        init_mod.zeros(lin.b)        # biases start at zero
        layers.append(lin)
        if k < len(sizes) - 2:       # no activation after the output layer
            layers.append(act_cls())
    return Sequential(layers)
