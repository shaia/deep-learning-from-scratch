"""
Weight initialization schemes -- the second pillar of Module 02's toolkit.

Module 01 initialized weights from a plain uniform in [-1, 1). That works for a
2->4->1 toy but is exactly wrong at depth: too large and every sigmoid/tanh
saturates (flat gradient, no learning); too small and the signal shrinks layer
by layer until it vanishes. The fix, discovered the hard way in the 2000s, is to
scale the initial weights by the *fan-in* so the variance of activations (and of
back-propagated gradients) stays roughly constant from layer to layer.

    Xavier/Glorot (2010): for sigmoid/tanh, Var(W) = 1 / fan_in
                          (the symmetric form uses 2 / (fan_in + fan_out)).
    He/Kaiming  (2015): for ReLU, Var(W) = 2 / fan_in  -- the factor of 2
                        compensates for ReLU zeroing out half the inputs.

Each function fills a weight matrix `W` of shape (fan_in, fan_out) in place by
drawing from an `Rng` in row-major order, so the C mirror produces identical
weights. Biases are initialized to zero (the standard, safe default).
"""

import math

from .rng import Rng


def _fill_normal(W, rng: Rng, std: float) -> None:
    """Fill W in row-major order with N(0, std^2) draws (bit-exact with C)."""
    fan_in, fan_out = W.shape
    for i in range(fan_in):
        for o in range(fan_out):
            W[i, o] = std * rng.normal()


def zeros(b) -> None:
    """Zero a bias vector in place (the standard bias initialization)."""
    b[:] = 0.0


def xavier_normal(W, rng: Rng) -> None:
    """Glorot init for sigmoid/tanh: std = sqrt(1 / fan_in)."""
    fan_in = W.shape[0]
    _fill_normal(W, rng, math.sqrt(1.0 / fan_in))


def he_normal(W, rng: Rng) -> None:
    """He/Kaiming init for ReLU: std = sqrt(2 / fan_in)."""
    fan_in = W.shape[0]
    _fill_normal(W, rng, math.sqrt(2.0 / fan_in))


def small_uniform(W, rng: Rng, scale: float = 1.0) -> None:
    """The Module-01 baseline: uniform [-scale, scale). Kept so ablations can
    show *why* the fan-in scalings above are an improvement."""
    fan_in, fan_out = W.shape
    for i in range(fan_in):
        for o in range(fan_out):
            W[i, o] = scale * rng.signed()


# ---------------------------------------------------------------------------
# Conv kernels (Module 03). Same idea, different fan-in: a conv output unit
# reads C*KH*KW inputs (every channel of one kernel-sized patch), so that is
# the variance that must be controlled. K has shape (F, C, KH, KW) and is
# filled in flat row-major order -- the exact draw order of ng_init_conv2d in
# lib/c/nanograd/conv.c -- so the C mirror gets identical kernels.
# ---------------------------------------------------------------------------
def _fill_normal_flat(K, rng: Rng, std: float) -> None:
    """Fill K in flat row-major order with N(0, std^2) draws (bit-exact w/ C)."""
    flat = K.reshape(-1)             # view, so this writes K in place
    for j in range(flat.size):
        flat[j] = std * rng.normal()


def xavier_normal_conv2d(K, rng: Rng) -> None:
    """Glorot init for a conv kernel: std = sqrt(1 / (C*KH*KW))."""
    _, C, KH, KW = K.shape
    _fill_normal_flat(K, rng, math.sqrt(1.0 / (C * KH * KW)))


def he_normal_conv2d(K, rng: Rng) -> None:
    """He/Kaiming init for a conv kernel: std = sqrt(2 / (C*KH*KW))."""
    _, C, KH, KW = K.shape
    _fill_normal_flat(K, rng, math.sqrt(2.0 / (C * KH * KW)))


# Registry so callers (and the notebook's ablation loop) can pick by name.
INITS = {
    "xavier": xavier_normal,
    "he": he_normal,
    "small": small_uniform,
}

CONV_INITS = {
    "xavier": xavier_normal_conv2d,
    "he": he_normal_conv2d,
}
