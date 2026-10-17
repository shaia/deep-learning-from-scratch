"""
Data augmentation -- Module 04's cheapest regularizer, from AlexNet (Sec. 4.1).

A 60-million-parameter network fed 1.2 million images will memorize them.
AlexNet's first answer was not a layer at all but a change to the *data*:
train on a random 224x224 crop of each 256x256 image, mirrored left-right half
the time, re-drawn every epoch. A crop shifted by a few pixels, or a mirrored
dog, is still the same class -- so every one of those variants is a free,
honestly labeled new example. 32*32 offsets x 2 flips: the paper's "factor of
2048" more training data, at the price of a little indexing.

At test time there is nothing to randomize, so AlexNet took ten fixed views --
the four corner crops and the center crop, each plain and mirrored -- and
averaged the network's softmax over them (`ten_crop` here).

All functions take and return NCHW arrays. Random choices come from
`rng.random(shape)`: a numpy Generator for speed, or a nanograd Rng to draw
exactly what the C mirror draws (three uniforms per image, row-major:
row offset, column offset, flip).
"""

import numpy as np


def random_crop_flip(X, crop, rng, flip=True):
    """AlexNet's training-time augmentation: per image, a random crop x crop
    window and (if `flip`) a left-right mirror with probability 1/2.

    X: (n, C, H, W) -> (n, C, crop, crop). The offsets are uniform over all
    H-crop+1 (resp. W-crop+1) valid positions: dy = floor(u0 * (H-crop+1)).
    """
    n, C, H, W = X.shape
    u = rng.random((n, 3))                      # (row offset, col offset, flip)
    dy = (u[:, 0] * (H - crop + 1)).astype(np.int64)
    dx = (u[:, 1] * (W - crop + 1)).astype(np.int64)
    mirror = flip & (u[:, 2] < 0.5)
    out = np.empty((n, C, crop, crop), dtype=X.dtype)
    for i in range(n):
        patch = X[i, :, dy[i]:dy[i] + crop, dx[i]:dx[i] + crop]
        out[i] = patch[:, :, ::-1] if mirror[i] else patch
    return out


def center_crop(X, crop):
    """The single deterministic view used for evaluation: the middle
    crop x crop window (offset (H-crop)//2, (W-crop)//2)."""
    _, _, H, W = X.shape
    oy, ox = (H - crop) // 2, (W - crop) // 2
    return X[:, :, oy:oy + crop, ox:ox + crop]


def ten_crop(X, crop):
    """AlexNet's test-time views: 4 corners + center, each plain and mirrored.
    Returns a list of ten (n, C, crop, crop) arrays; average the network's
    softmax over them to predict."""
    _, _, H, W = X.shape
    offsets = [(0, 0), (0, W - crop), (H - crop, 0), (H - crop, W - crop),
               ((H - crop) // 2, (W - crop) // 2)]
    views = []
    for oy, ox in offsets:
        v = X[:, :, oy:oy + crop, ox:ox + crop]
        views.append(v)
        views.append(v[:, :, :, ::-1])
    return views
