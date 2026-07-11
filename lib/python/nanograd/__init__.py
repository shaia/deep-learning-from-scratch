"""
nanograd -- the shared, from-scratch neural-net library for this book.

Introduced in Module 02 (Making Deep Nets Trainable) and grown module by module.
It is *not* an autograd engine: every layer's backward pass is the chain rule
written out by hand (see layers.py). The library just packages the pieces
Module 01 derived once, so later modules can stack and reuse them.

    from nanograd import Rng, mlp, Adam, SoftmaxCrossEntropy

    net  = mlp([2, 16, 2], activation="relu", init="he", seed=0)
    loss = SoftmaxCrossEntropy()
    opt  = Adam(lr=1e-2)

    L = loss.forward(net.forward(X), y)      # forward
    net.backward(loss.backward())            # backward (fills grads)
    opt.step(net.params(), net.grads())      # update
"""

from .init import he_normal, small_uniform, xavier_normal
from .layers import (
    Linear,
    ReLU,
    Sigmoid,
    SoftmaxCrossEntropy,
    Tanh,
)
from .net import Sequential, mlp
from .optim import SGD, Adam, RMSProp
from .rng import Rng

__all__ = [
    "Rng",
    "Linear", "Sigmoid", "Tanh", "ReLU", "SoftmaxCrossEntropy",
    "Sequential", "mlp",
    "SGD", "RMSProp", "Adam",
    "xavier_normal", "he_normal", "small_uniform",
]
