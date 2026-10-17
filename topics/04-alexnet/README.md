# Module 04 — AlexNet (depth · ReLU · dropout · augmentation, at scale)

LeNet's convolution, three orders of magnitude bigger — and the tricks that kept a network with
far more parameters than training images from simply memorizing them. nanograd gains **stride
and padding**, **overlapping max-pooling**, **dropout** (with `train()`/`eval()` modes),
AlexNet's **local response normalization**, and **crop + flip augmentation**, in both languages.
The toy shows which regularizer does what (memorizing is easy; recognizing *mirrored* glyphs
takes the flips), and a scaled-down AlexNet reaches **72.3%** on CIFAR-10 in NumPy. C and
Python agree to the last printed digit.

> Blog post: [`site/src/content/posts/04-alexnet.mdx`](../../site/src/content/posts/04-alexnet.mdx)
> Interactive widget: [`site/src/components/viz/DropoutWidget.astro`](../../site/src/components/viz/DropoutWidget.astro)
> Shared library: [`lib/python/nanograd`](../../lib/python/nanograd) · [`lib/c/nanograd`](../../lib/c/nanograd)
> Dataset: [`data/get_cifar10.py`](../../data/get_cifar10.py) (downloads ~170 MB once; the Toronto server can be slow)
> Concept & references: [`docs/CURRICULUM.md`](../../docs/CURRICULUM.md), [`docs/references/papers.md`](../../docs/references/papers.md)

## Contents

| Path | What |
|------|------|
| `../../lib/python/nanograd/conv.py` | **Extended:** `Conv2D(stride, pad, input_grad)`, `im2col`/`col2im` with stride + padding, `MaxPool2D(P, stride)` (overlapping), `conv_out_size` |
| `../../lib/python/nanograd/layers.py`, `net.py` | **New:** `Dropout` (inverted); `Sequential.train()/.eval()/.astype()` |
| `../../lib/python/nanograd/norm.py`, `augment.py` | **New:** `LocalResponseNorm`; `random_crop_flip`, `center_crop`, `ten_crop` |
| `../../lib/c/nanograd/conv.c`, `nn.c`, `norm.c`, `augment.c` | The C mirror — `ng_conv2d_*_strided`, `ng_maxpool2d_*_strided`, `ng_dropout_*`, `ng_lrn_*`, `ng_random_crop_flip`, `ng_center_crop` (Module 03's calls are now the stride-1 special case) |
| `python/alexnet.py` | Canonical: scalar `ToyAlexNet` mirrored line-for-line with the C on the **mirror glyphs** toy, the naive definitions (strided conv, overlapping pool, LRN), and the vectorized AlexNet-mini + CIFAR-10 (`run_cifar`, target ≥ 0.70, checkpoint cache) |
| `c/alexnet.c` | The C toy AlexNet (crops+flips → strided conv → ReLU → overlapping pool → fc → dropout → fc, SGD + momentum + weight decay), plus an LRN fingerprint |
| `notebook.ipynb` | Cell-by-cell: mirror glyphs, stride/padding, ReLU at depth, overlapping pooling, dropout, augmentation, the toy's which-regularizer-does-what, CIFAR-10 (Figure 1 redone, the regularizer ablation, the full run, filters, confusion matrix, fc7 neighbors), the historical corner (parameter/compute breakdown, LRN, PCA color), framework mirrors |
| `assignment.ipynb` | Hands-on: you implement the **strided/padded conv**, **overlapping max-pool routing** (and meet the `np.add.at` trap), **inverted dropout**, and **crops + flips**; your pieces must reproduce the toy's result |
| `tests/check_utils.py` | Self-check harness (`rel_error`, `eval_numerical_gradient*`, `compare_to_canonical`) |
| `tests/test_agreement.py` | Compiles + runs the C, runs the Python, asserts agreement (toy + LRN) — **and** gradient-checks every new layer, pins the fast paths to the naive definitions at `1e-12`, and checks the two Python paths through three training steps |
| `anim/scene.py` | Manim scene: the 2012 scoreboard, the two-GPU architecture and where its parameters/arithmetic live, ReLU vs tanh, dropout thinning, augmentation, learned filters |
| `anim/alexnet_assets.json` | Trained conv1 filters, sample photos and experiment curves, exported by the notebook (the scene's data) |

## Run it

```bash
# C  (standalone; Windows: omit -lm, math is in the CRT)
clang -O2 -std=c11 -Wall -Wextra -I lib/c/nanograd \
  topics/04-alexnet/c/alexnet.c \
  lib/c/nanograd/rng.c lib/c/nanograd/nn.c lib/c/nanograd/init.c \
  lib/c/nanograd/optim.c lib/c/nanograd/conv.c lib/c/nanograd/norm.c \
  lib/c/nanograd/augment.c -o alexnet && ./alexnet

# C  (via CMake)
cmake -B build && cmake --build build && ./build/topics/04-alexnet/c/alexnet

# Python mirror: the toy, then the CIFAR-10 AlexNet with the >= 0.70 assert.
# The CIFAR run is ~2 hours of NumPy on a laptop CPU the first time; it is
# cached to data/cifar10/checkpoints/ (gitignored) and shared with the
# notebook, so later runs load and re-evaluate it in about a minute.
python topics/04-alexnet/python/alexnet.py

# C <-> Python agreement + finite-difference gradient checks (~1 min: the scalar toy)
python topics/04-alexnet/tests/test_agreement.py

# Notebook: open interactively, or run headless to confirm it executes clean.
# With an empty checkpoint cache it retrains every CIFAR experiment (hours on a
# laptop CPU); with the cache it re-evaluates the saved networks in minutes.
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 \
  topics/04-alexnet/notebook.ipynb

# Render the animation (needs manim + ffmpeg on PATH; no LaTeX required)
python -m manim -qm --media_dir topics/04-alexnet/anim/media \
  topics/04-alexnet/anim/scene.py AlexNetStory
```

The rendered video is copied to `site/public/media/04-alexnet.mp4` and embedded in the post.

## What you should see

- **toy (mirror glyphs):** the C and Python runs memorize all 32 training glyphs (train accuracy
  **1.000**) and classify **0.922** of the mirrored test glyphs, agreeing on every printed digit.
  Over ten seeds: no regularization 69% on mirrored glyphs, dropout alone 68%, crops + flips 96%,
  both 93% — augmentation adds the missing invariance; dropout can't.
- **cifar-10:** the AlexNet-mini (`conv 32@5×5 → pool 3/2 → conv 64@3×3 → pool 3/2 →
  conv 96/96/64@3×3 → pool 3/2 → fc 256 → fc 256 → 10`, 348,874 parameters, dropout 0.5,
  24×24 crops + flips, Adam, float32) reaches **72.3%** test accuracy (target ≥ 0.70),
  **75.5%** with ten-crop averaging. On a 10k-image subset over 30 epochs: no regularization
  56.6% test (a 43-point memorization gap), dropout only 57.2%, crops + flips **66.3%**, both
  62.8% — augmentation is the lever; at this scale dropout costs more than it buys. Figure 1
  (ReLU 6× faster than tanh) does **not** reproduce here: with modern initialization tanh trains
  faster, even after a learning-rate check.
- The agreement test prints `OK`: toy and LRN fingerprints identical across C and Python; every
  new layer's gradients within `1e-6` of finite differences (most near `1e-10`); fast paths equal
  the naive definitions to `~1e-15`; vectorized and scalar Python paths equal after three
  training steps. The notebook additionally pins our layers to PyTorch at `1e-13` or better and lands
  TF/Keras AlexNet-minis at 70.8% / 68.8%.

## Design notes

**What's modernized, and why.** The toy trains with AlexNet's own recipe — SGD with momentum 0.9
and weight decay 5×10⁻⁴ — bit-exactly in C and Python. The CIFAR network keeps AlexNet's shape
(five convolutions, overlapping pooling, three dense layers with dropout, crops + flips, ReLU) but
trains with **Adam**: AlexNet's SGD with a hand-dropped learning rate ran ~90 epochs, and on a
15-epoch budget a plain 8-layer network without batch normalization learns several times slower
under SGD (the notebook measures 44% vs 28% training error after 10 epochs). He initialization replaces
the paper's fixed N(0, 0.01²); LRN and PCA color augmentation are implemented and examined in
the historical corner rather than used by default (VGG found LRN unhelpful; batch norm replaced
it).

**Scaled down, honestly.** 24×24 crops of 32×32 CIFAR photos (Krizhevsky's own cuda-convnet
recipe) instead of 224×224 ImageNet crops; widths 32–96 instead of 96–384; 256-unit dense layers
instead of 4,096. AlexNet's 60 M parameters sit 96% in its dense layers; ours, with a 2×2 final
map, only 38% — so dropout matters less here than it did there, which the ablation shows.

**float32 at scale.** The teaching code runs in float64 (the agreement gates need it);
`Sequential.astype(np.float32)` re-stores parameters for the CIFAR runs, halving memory traffic.

**Agreement.** As in Module 03, the C↔Python gate is a line-for-line scalar mirror on the toy —
including every random number: data, weights, crops/flips and dropout masks each come from their
own LCG stream, which both `Rng.random()` (Python) and `ng_rng_uniform` (C) draw identically.
CIFAR runs on the vectorized path and is checked by metric.
