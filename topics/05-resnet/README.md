# Module 05 — ResNet (residual connections · batch normalization · the degradation problem)

Past about twenty layers, plain networks fit their own *training* data worse as they get deeper —
not overfitting, not a lack of capacity, but an optimization failure. He, Zhang, Ren & Sun (2015)
fixed it with one addition per block, $y = \mathrm{relu}(\mathcal{F}(x) + x)$. nanograd gains
**batch normalization** (training and evaluation modes, running statistics), **residual blocks**,
the option-(A) **zero-padding shortcut**, **global average pooling** and **bias-free layers**, in
both languages. A 34-layer toy reproduces the degradation problem on 128 spiral points and shows
why (gradients that explode toward the input, chaotic training, shattered input-gradients); its
residual twin — same weights, one `+` per block — fits every point. On CIFAR-10 the paper's
ResNet-20 reaches **84.0%**. C and Python agree to the last printed digit.

> Blog post: [`site/src/content/posts/05-resnet.mdx`](../../site/src/content/posts/05-resnet.mdx)
> Interactive widget: [`site/src/components/viz/ResidualWidget.astro`](../../site/src/components/viz/ResidualWidget.astro)
> Shared library: [`lib/python/nanograd`](../../lib/python/nanograd) · [`lib/c/nanograd`](../../lib/c/nanograd)
> Dataset: [`data/get_cifar10.py`](../../data/get_cifar10.py) (shared with Module 04)
> Concept & references: [`docs/CURRICULUM.md`](../../docs/CURRICULUM.md), [`docs/references/papers.md`](../../docs/references/papers.md)

## Contents

| Path | What |
|------|------|
| `../../lib/python/nanograd/norm.py` | **New:** `BatchNorm` — per-channel over `(m, C)` or `(n, C, H, W)`, `training` flag, running statistics (unbiased variance), `buffers()` |
| `../../lib/python/nanograd/residual.py` | **New:** `Residual(body, shortcut)` — `relu(body(x) + shortcut(x))`, backward sums the two routes |
| `../../lib/python/nanograd/conv.py` | **New:** `ZeroPadShortcut` (option A), `GlobalAvgPool2D`; `Conv2D(bias=False, cache_cols=, workers=)` |
| `../../lib/python/nanograd/layers.py`, `net.py` | `Linear(bias=False)`; `Sequential.buffers()`, `train()`/`eval()` reach into nested blocks |
| `../../lib/c/nanograd/norm.c`, `nn.c`, `conv.c`, `init.c` | The C mirror — `ng_batchnorm_forward/backward`, `ng_add`, `ng_shortcut_pad_*`, `ng_global_avgpool_*`; `b = NULL` for bias-free linear and conv layers |
| `python/resnet.py` | Canonical: the C-order `ToyResNet` (both twins), the vectorized twin and its depth sweeps, gradient and shattered-gradient probes, BatchNorm by its definition, the C fingerprints, the CIFAR ResNets (`build_resnet`, `run_cifar` with a checkpoint cache, target ≥ 0.80), lesions, the animation export |
| `c/resnet.c` | The C toy: a 34-layer net trained plain and residual from the same weights, plus BatchNorm and conv-block fingerprints |
| `notebook.ipynb` | Cell-by-cell: deep spirals, the degradation problem, is it vanishing gradients? (explosion, chaos, shattering), the residual block, batch normalization (derivation, running stats, why ResNets need it), the fix, CIFAR-10 (gradients at init, Figure 6 redone, the paper's ResNet-20, deleting blocks), the ImageNet ResNets (FLOPs recomputed), framework mirrors |
| `assignment.ipynb` | Hands-on: you implement **BatchNorm's forward** (both modes) and **backward**, the **residual block** with its forked backward, and the **zero-padding shortcut**; your pieces must make the 34-layer toy's residual twin fit while its plain twin fails |
| `tests/check_utils.py` | Self-check harness (`rel_error`, `eval_numerical_gradient*`, `compare_to_canonical`) |
| `tests/test_agreement.py` | Compiles + runs the C, runs the Python, asserts agreement (both toy twins, BatchNorm, conv block) — **and** gradient-checks every new layer, pins BatchNorm to its definition, and checks the two Python paths |
| `anim/scene.py` | Manim scene: the race to depth, the degradation problem, the residual block, gradients per block, shattered gradients, depth working again, an ensemble of paths and the lesion study |
| `anim/resnet_assets.json` | Curves, gradient profiles and lesion results exported by the notebook (the scene's data) |

## Run it

```bash
# C  (standalone; Windows: omit -lm, math is in the CRT)
clang -O2 -std=c11 -Wall -Wextra -I lib/c/nanograd \
  topics/05-resnet/c/resnet.c \
  lib/c/nanograd/rng.c lib/c/nanograd/nn.c lib/c/nanograd/init.c \
  lib/c/nanograd/optim.c lib/c/nanograd/conv.c lib/c/nanograd/norm.c -o resnet && ./resnet

# C  (via CMake)
cmake -B build && cmake --build build && ./build/topics/05-resnet/c/resnet

# Python mirror: both toy twins and the fingerprints, then the CIFAR-10 ResNet-20 with the
# >= 0.80 assert. The CIFAR run is hours of NumPy on a laptop CPU the first time; it is cached
# to data/cifar10/checkpoints/ (gitignored) and shared with the notebook.
python topics/05-resnet/python/resnet.py

# C <-> Python agreement + finite-difference gradient checks (~2 min)
python topics/05-resnet/tests/test_agreement.py

# Notebook: open interactively, or run headless to confirm it executes clean. With the
# checkpoint cache it loads the CIFAR networks and runs in about 80 minutes
# (the toy sweeps and the TensorFlow/Keras mirrors are not cached).
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 \
  topics/05-resnet/notebook.ipynb

# Render the animation (needs manim + ffmpeg on PATH; no LaTeX required)
python -m manim -qm --media_dir topics/05-resnet/anim/media \
  topics/05-resnet/anim/scene.py ResNetStory
```

The rendered video is copied to `site/public/media/05-resnet.mp4` and embedded in the post.

## What you should see

- **toy (deep spirals):** C and Python print identical fingerprints for both 34-layer twins: the
  plain net ends at **59.4%** training accuracy, its residual twin at **100%** (loss 3.4×10⁻⁵).
  Over five seeds, plain networks of 4–18 layers fit 100%, 34 layers 64.4%, 66 layers 52.2%;
  every residual twin fits 100%. At step 0 the 34-layer plain net's first block gets 1,239× the
  gradient of its last (66 layers: 1.1 million×); the residual twins' ratios are 14 and 32.
- **cifar-10:** the paper's ResNet-20 exactly (widths 16/32/64, option-(A) shortcuts, BatchNorm,
  269,722 parameters, SGD lr 0.1 + momentum, pad-crop-flip, float32), eight epochs on all 50,000
  images: **84.0%** test accuracy (target ≥ 0.80) — 11.7 points above Module 04's AlexNet-mini,
  with fewer parameters (the paper's 182 epochs reached 91.25%). Figure 6 redone at half width on
  16×16 photos, 3,128 steps each: plain-20 **40.8%** training error / 61.8% test, plain-56
  **67.5%** / 34.5%, ResNet-20 32.5% / 68.4%, ResNet-56 32.3% / 68.6% — deeper plain nets fit
  worse; deeper ResNets don't (on this budget they tie). At initialization plain-56's first block
  gets ~590× its last conv's gradient (ResNet-56: ~5×). Deleting one block from the trained
  plain-56 drops it from 34.5% to 12.0% on average; from ResNet-56, 68.2% → 63.4% (deep blocks
  cost under a point). PyTorch matches BatchNorm (train and eval, running statistics) and the
  residual block to ≤ 1e-13; TensorFlow/Keras ResNet-20s reach 74.3% / 73.0% in three epochs.
- The agreement test prints `OK`: all four FINAL lines identical across C and Python; every new
  layer's gradients within `1e-6` of finite differences; BatchNorm equal to its definition to
  `~1e-15`. The plain twin's two Python paths agree at step 0 but drift apart within three steps
  (the test reports, but does not gate, that): its training is chaotic.

## Design notes

**Plain vs residual, nothing else.** Every comparison here is between twins that share layers,
parameters, initial weights and arithmetic, differing only in the `+` of each block (and, where
shapes change, the parameter-free option-(A) shortcut) — the paper's own design. The C toy
trains both twins from one weight draw.

**The C-order mirror, vectorized.** Bit-exact agreement needs every sum in the C's order, but
elementwise IEEE operations are order-free, so `ToyResNet` vectorizes everything except sums,
which go through `np.add.accumulate` — defined as the running sum a C `for` loop computes. That
keeps the 800 training steps of the agreement test to about a minute.

**Scaled down, honestly.** The canonical CIFAR network is the paper's ResNet-20 exactly (widths
16/32/64, option A, He init, BatchNorm, SGD 0.1 with momentum, pad-crop-flip), trained for 8
epochs instead of 182 (weight decay 5×10⁻⁴ instead of 10⁻⁴). The depth experiment (Figure 6)
uses half the widths and 16×16 photos (each pixel the mean of a 2×2 block) so its four runs —
3,128 steps each, on all 50,000 images — fit a laptop; all four warm up at lr 0.01 for 400 steps,
the paper's recipe for its deepest network.

**Laptop engineering, bit-identical.** `Conv2D(cache_cols=False)` rebuilds im2col patch
matrices in the backward pass instead of holding them (gigabytes for a 56-layer net), and
`Conv2D(workers=k)` builds them on k threads (each image's rows depend only on that image).
Neither changes a bit of any result, so neither is part of a checkpoint's name.
