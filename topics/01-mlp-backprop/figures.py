"""Static figures for the Module 01 chapter (site/src/content/posts/01-mlp-backprop.mdx).

Trains the canonical `python/mlp.py` MNIST model (784 -> 128 -> 10, lr 0.5,
batch 64, 15 epochs, seed 0 -- the `run_mnist` defaults) and renders:

  site/public/media/01-mnist-curve.png          train/test accuracy per epoch
  site/public/media/01-mnist-misclassified.png  ten digits the network gets wrong

Deterministic: same seeds as `run_mnist`, so the figures show exactly the run
the chapter quotes. MNIST downloads to data/mnist/ on first use (via
data/get_mnist.py) and is cached after that.

Run from anywhere:  python topics/01-mlp-backprop/figures.py
"""

import os
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "python"))
from mlp import MLPVec, _load_mnist  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "site", "public", "media")

# Okabe-Ito palette (colorblind-safe, per docs/conventions/viz-style.md).
BLUE = "#0072B2"       # test accuracy / class emphasis
ORANGE = "#E69F00"     # train accuracy
VERMILLION = "#D55E00" # action/attention (target line, wrong labels)
GRAY = "#7f7f7f"       # axes / secondary text

N_HIDDEN, LR, EPOCHS, BATCH, SEED = 128, 0.5, 15, 64, 0


def train_with_history():
    """`train_minibatch` from mlp.py, unrolled so we can record accuracy per
    epoch. Same RNG stream (one permutation per epoch from default_rng(SEED)),
    so the final model is bit-identical to run_mnist()'s."""
    Xtr, ytr, Xte, yte = _load_mnist()
    model = MLPVec(784, N_HIDDEN, 10, seed=SEED)
    rng = np.random.default_rng(SEED)
    n = Xtr.shape[0]

    def acc(X, y):
        return float((model.predict(X) == y).mean())

    train_acc, test_acc = [acc(Xtr, ytr)], [acc(Xte, yte)]  # epoch 0 = untrained
    for e in range(EPOCHS):
        idx = rng.permutation(n)
        for s in range(0, n, BATCH):
            b = idx[s:s + BATCH]
            model.sgd_step(Xtr[b], ytr[b], LR)
        train_acc.append(acc(Xtr, ytr))
        test_acc.append(acc(Xte, yte))
        print(f"epoch {e + 1:2d}/{EPOCHS}  train={train_acc[-1]:.4f}  test={test_acc[-1]:.4f}")
    return model, Xte, yte, train_acc, test_acc


def fig_curve(train_acc, test_acc):
    # Plot from epoch 1: the untrained epoch-0 point (~10%, chance) would crush
    # the interesting range; the chapter states the chance baseline in prose.
    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=150)
    epochs = np.arange(1, len(train_acc))
    ax.plot(epochs, train_acc[1:], color=ORANGE, lw=2, marker="o", ms=4, label="train accuracy")
    ax.plot(epochs, test_acc[1:], color=BLUE, lw=2, marker="o", ms=4, label="test accuracy")
    ax.axhline(0.95, color=VERMILLION, lw=1.2, ls="--")
    ax.text(EPOCHS, 0.951, "95% target", color=VERMILLION, fontsize=9,
            ha="right", va="bottom")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_xlim(0.5, EPOCHS + 0.2)
    ax.set_ylim(0.90, 1.0)
    ax.set_xticks([1] + list(range(3, EPOCHS + 1, 3)))
    ax.grid(alpha=0.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRAY)
    ax.tick_params(colors=GRAY)
    ax.xaxis.label.set_color(GRAY)
    ax.yaxis.label.set_color(GRAY)
    ax.legend(frameon=False, loc="lower right")
    out = os.path.join(OUT_DIR, "01-mnist-curve.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


def fig_misclassified(model, Xte, yte, n_show=10):
    pred = model.predict(Xte)
    wrong = np.flatnonzero(pred != yte)[:n_show]
    fig, axes = plt.subplots(2, 5, figsize=(7.0, 3.4), dpi=150)
    for ax, i in zip(axes.ravel(), wrong):
        ax.imshow(Xte[i].reshape(28, 28), cmap="gray_r")
        ax.set_title(f"read {pred[i]}, was {yte[i]}", fontsize=9, color=VERMILLION)
        ax.axis("off")
    fig.suptitle("the network's mistakes", fontsize=10, color=GRAY)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "01-mnist-misclassified.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    model, Xte, yte, train_acc, test_acc = train_with_history()
    print(f"final test accuracy = {test_acc[-1]:.4f}")
    fig_curve(train_acc, test_acc)
    fig_misclassified(model, Xte, yte)
