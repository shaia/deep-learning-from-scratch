"""
Optimizers for nanograd -- the third pillar of Module 02.

Module 01 used one update rule: theta <- theta - lr * grad (plain SGD). It works,
but it crawls through ravines, stalls on plateaus, and forces you to hand-tune a
single global learning rate. The 1980s-2010s produced a short, powerful lineage
of fixes, each one a couple of extra lines on top of the last:

    SGD + momentum : accumulate a velocity so consistent directions build speed
                     and oscillations cancel.        (Polyak 1964; Rumelhart 1986)
    RMSProp        : divide each step by a running RMS of that coordinate's
                     gradients -- a per-parameter learning rate.  (Hinton 2012)
    Adam           : momentum *and* RMSProp together, with bias-correction.
                     The default optimizer of modern deep learning. (Kingma 2014)

Every optimizer implements one method:  step(params, grads)  where `params` and
`grads` are parallel lists of NumPy arrays (from net.params()/net.grads()).
Parameters are updated in place; per-parameter state (velocity, moments) is kept
internally, keyed by position. L2 weight decay is available on each as an extra
`weight_decay * theta` term added to the gradient.
"""

import numpy as np


class SGD:
    """theta <- theta - lr * v,  where v = momentum * v + grad.

    momentum=0 recovers Module 01's plain gradient descent exactly."""

    def __init__(self, lr=0.1, momentum=0.0, weight_decay=0.0):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.v = None

    def step(self, params, grads):
        if self.v is None:
            self.v = [np.zeros_like(p) for p in params]
        for i, (p, g) in enumerate(zip(params, grads)):
            if self.weight_decay:
                g = g + self.weight_decay * p        # L2: adds d/dtheta (wd/2 |theta|^2)
            self.v[i] = self.momentum * self.v[i] + g
            p -= self.lr * self.v[i]                  # in-place update


class RMSProp:
    """Per-coordinate step size: divide by the running RMS of the gradient.

        s <- beta s + (1-beta) g^2
        theta <- theta - lr * g / (sqrt(s) + eps)
    """

    def __init__(self, lr=0.01, beta=0.9, eps=1e-8, weight_decay=0.0):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.weight_decay = weight_decay
        self.s = None

    def step(self, params, grads):
        if self.s is None:
            self.s = [np.zeros_like(p) for p in params]
        for i, (p, g) in enumerate(zip(params, grads)):
            if self.weight_decay:
                g = g + self.weight_decay * p
            self.s[i] = self.beta * self.s[i] + (1.0 - self.beta) * (g * g)
            p -= self.lr * g / (np.sqrt(self.s[i]) + self.eps)


class Adam:
    """Momentum + RMSProp + bias-correction (Kingma & Ba 2014).

        m <- b1 m + (1-b1) g            (1st moment: momentum)
        v <- b2 v + (1-b2) g^2          (2nd moment: RMSProp)
        m_hat = m/(1-b1^t),  v_hat = v/(1-b2^t)     (correct the cold start)
        theta <- theta - lr * m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        self.t += 1
        b1c = 1.0 - self.beta1 ** self.t   # bias-correction denominators
        b2c = 1.0 - self.beta2 ** self.t
        for i, (p, g) in enumerate(zip(params, grads)):
            if self.weight_decay:
                g = g + self.weight_decay * p
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (g * g)
            m_hat = self.m[i] / b1c
            v_hat = self.v[i] / b2c
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


OPTIMIZERS = {"sgd": SGD, "rmsprop": RMSProp, "adam": Adam}
