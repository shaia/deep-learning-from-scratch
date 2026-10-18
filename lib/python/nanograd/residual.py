"""
Residual blocks -- Module 05's addition to nanograd (He, Zhang, Ren & Sun 2015).

A plain block computes y = relu(F(x)), where F is a few layers (Linear or
Conv2D, BatchNorm, ReLU, ...). A residual block adds its input back:

    y = relu( F(x) + shortcut(x) )        shortcut = identity, or ZeroPadShortcut

That one addition changes what the layers are asked to learn. To leave x
unchanged, a plain block must *build* the identity out of weights and ReLUs
-- surprisingly hard for a stack of nonlinear layers -- while a residual block
only has to drive F to zero. The layers learn the *correction* to the identity
(the "residual"), and doing nothing is the default.

It changes the backward pass even more. x reaches the output by two routes,
so by the chain rule's rule for a value used twice, its gradient is the SUM of
the gradients along both:

    dL/dx = dL/dz * dF/dx  +  dL/dz          (z = F(x) + x, before the ReLU)
            ^ through the layers   ^ straight down the shortcut

Through n stacked blocks the second term is never multiplied by any layer's
Jacobian: some of the gradient at the output arrives at the input untouched,
however deep the network is. In code the forward is one `+` and the backward
is one `+` -- the fork's gradients meeting again.

Contract (same as layers.py): forward(X) caches what backward needs;
backward(dY) returns dX; params()/grads()/buffers() flatten the body's (the
option-A shortcut has none); train()/eval()/astype() reach every layer inside,
so a Residual nests inside a Sequential like any other layer.
"""

from .layers import ReLU
from .net import Sequential


class Residual:
    """relu(body(x) + shortcut(x)); body is a Sequential (or a list of
    layers); shortcut=None is the identity. relu=False leaves the sum
    un-activated (the "pre-activation" ResNet of He et al. 2016 does that)."""

    def __init__(self, body, shortcut=None, relu: bool = True):
        self.body = body if isinstance(body, Sequential) else Sequential(list(body))
        self.shortcut = shortcut
        self.act = ReLU() if relu else None

    def forward(self, X):
        S = X if self.shortcut is None else self.shortcut.forward(X)
        Z = self.body.forward(X) + S                 # the residual connection
        return Z if self.act is None else self.act.forward(Z)

    def backward(self, dY):
        dZ = dY if self.act is None else self.act.backward(dY)
        d_body = self.body.backward(dZ)              # the long way, through F
        d_skip = dZ if self.shortcut is None else self.shortcut.backward(dZ)
        return d_body + d_skip                       # the two routes meet again

    def params(self):
        return self.body.params() + (self.shortcut.params() if self.shortcut else [])

    def grads(self):
        return self.body.grads() + (self.shortcut.grads() if self.shortcut else [])

    def buffers(self):
        return self.body.buffers()

    def train(self):
        self.body.train()
        return self

    def eval(self):
        self.body.eval()
        return self

    def astype(self, dtype):
        self.body.astype(dtype)
        return self
