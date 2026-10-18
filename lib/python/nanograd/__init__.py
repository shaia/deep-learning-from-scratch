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

What each module added:
    02  Linear, activations, SoftmaxCrossEntropy, Sequential, inits, optimizers
    03  Conv2D, MaxPool2D, AvgPool2D, Flatten, im2col/col2im, conv inits
    04  conv stride/padding, overlapping max-pool, Dropout (+ train()/eval()),
        LocalResponseNorm, crop/flip augmentation, Rng.random
    05  BatchNorm (+ running statistics, buffers()), Residual blocks,
        ZeroPadShortcut, GlobalAvgPool2D, bias-free Linear/Conv2D
"""

from .augment import center_crop, random_crop_flip, ten_crop
from .conv import (
    AvgPool2D,
    Conv2D,
    Flatten,
    GlobalAvgPool2D,
    MaxPool2D,
    ZeroPadShortcut,
    col2im,
    conv_out_size,
    im2col,
)
from .init import (
    he_normal,
    he_normal_conv2d,
    small_uniform,
    xavier_normal,
    xavier_normal_conv2d,
)
from .layers import (
    Dropout,
    Linear,
    ReLU,
    Sigmoid,
    SoftmaxCrossEntropy,
    Tanh,
)
from .net import Sequential, mlp
from .norm import BatchNorm, LocalResponseNorm
from .optim import SGD, Adam, RMSProp
from .residual import Residual
from .rng import Rng

__all__ = [
    "Rng",
    "Linear", "Sigmoid", "Tanh", "ReLU", "SoftmaxCrossEntropy", "Dropout",
    "Conv2D", "MaxPool2D", "AvgPool2D", "Flatten", "im2col", "col2im",
    "conv_out_size",
    "LocalResponseNorm",
    "BatchNorm", "Residual", "ZeroPadShortcut", "GlobalAvgPool2D",
    "random_crop_flip", "center_crop", "ten_crop",
    "Sequential", "mlp",
    "SGD", "RMSProp", "Adam",
    "xavier_normal", "he_normal", "small_uniform",
    "xavier_normal_conv2d", "he_normal_conv2d",
]
