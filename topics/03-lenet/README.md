# Module 03 — Convolutions / LeNet-5 (weight sharing · locality · feature maps)

The MLP (Modules 01–02) reads an image as 784 loose numbers; this module keeps the geometry.
A learned **kernel** slides across the image (**weight sharing** + **locality**), its answers
form **feature maps**, **pooling** keeps the strongest responses — and a 44,426-parameter
modernized **LeNet-5** beats Module 01's 101,770-parameter MLP on MNIST (**98.46%** vs 97%).
`Conv2D` / `MaxPool2D` / `AvgPool2D` / `Flatten` join the shared `nanograd` library in both
languages, and the C and Python impls agree to ~`1e-15`.

> Blog post: [`site/src/content/posts/03-lenet.mdx`](../../site/src/content/posts/03-lenet.mdx)
> Interactive widget: [`site/src/components/viz/ConvWidget.astro`](../../site/src/components/viz/ConvWidget.astro)
> Shared library: [`lib/python/nanograd`](../../lib/python/nanograd) · [`lib/c/nanograd`](../../lib/c/nanograd)
> Concept & references: [`docs/CURRICULUM.md`](../../docs/CURRICULUM.md), [`docs/references/papers.md`](../../docs/references/papers.md)

## Contents

| Path | What |
|------|------|
| `../../lib/python/nanograd/conv.py` | **New:** `im2col`/`col2im`, `Conv2D`, `MaxPool2D`, `AvgPool2D`, `Flatten` (plus conv He/Xavier in `init.py`) |
| `../../lib/c/nanograd/conv.c` | **New:** the C mirror — `ng_conv2d_*`, `ng_maxpool2d_*`, `ng_avgpool2d_*`, `ng_init_conv2d` |
| `python/lenet.py` | Canonical: scalar `ToyCNN` mirrored line-for-line with the C on the **bars** toy, the naive-loop conv reference, and the vectorized LeNet + MNIST (`run_mnist`, target ≥ 0.98) |
| `c/lenet.c` | The C toy CNN (conv→ReLU→maxpool→linear on bars), linking nanograd — same seeds/op order as the Python |
| `notebook.ipynb` | Cell-by-cell: bars, conv from its definition, backward derivations + gradient checks, pooling, im2col (~270× speedup), MNIST ≥98%, the 1998 historical corner, framework mirrors |
| `assignment.ipynb` | Hands-on: you implement the **conv forward**, **dK/db backward**, **max-pool routing**, and **He kernel init** in `# TODO` blocks; gradient checks grade you |
| `tests/check_utils.py` | Self-check harness (`rel_error`, `eval_numerical_gradient*`, `compare_to_canonical`) |
| `tests/test_agreement.py` | Compiles + runs the C, runs the Python, asserts agreement — **and** gradient-checks Conv2D/pooling + pins im2col to the naive definition at `1e-12` |
| `anim/scene.py` | Manim scene: the dense-layer waste, the kernel sliding, pooling, the full stack, and the six learned filters |
| `../../site/src/components/viz/lenetFilters.json` | Trained conv1 filters + sample digits, exported by `python/lenet.py --export-filters` (shared by widget and animation) |

## Run it

```bash
# C  (standalone; Windows: omit -lm, math is in the CRT)
clang -O2 -std=c11 -Wall -Wextra -I lib/c/nanograd \
  topics/03-lenet/c/lenet.c \
  lib/c/nanograd/rng.c lib/c/nanograd/nn.c lib/c/nanograd/init.c \
  lib/c/nanograd/optim.c lib/c/nanograd/conv.c -o lenet && ./lenet

# C  (via CMake)
cmake -B build && cmake --build build && ./build/topics/03-lenet/c/lenet

# Python mirror (bars toy + full-MNIST LeNet with the >=0.98 assert; ~3 min NumPy)
python topics/03-lenet/python/lenet.py

# C <-> Python agreement + finite-difference gradient checks of the conv machinery
python topics/03-lenet/tests/test_agreement.py

# Notebook: open interactively, or run headless to confirm it executes clean
jupyter nbconvert --to notebook --execute --inplace topics/03-lenet/notebook.ipynb

# Re-export the widget/animation data after a retrain
python topics/03-lenet/python/lenet.py --export-filters

# Render the animation (needs manim + ffmpeg on PATH; no LaTeX required)
python -m manim -qm --media_dir topics/03-lenet/anim/media \
  topics/03-lenet/anim/scene.py LeNetStory
```

The rendered video is copied to `site/public/media/03-lenet.mp4` and embedded in the post.

## What you should see

- **toy (bars):** vertical-vs-horizontal bars solved (accuracy **1.000**, loss ≈ `0.018`) by a
  114-parameter conv net in 300 Adam steps; the learned 3×3 kernels are oriented edge detectors.
- **mnist:** the modernized LeNet (`conv 6@5×5 → pool → conv 16@5×5 → pool → 120 → 84 → 10`,
  ReLU/He/Adam) reaches **≈ 0.985** in 3 epochs on the full 60k (target ≥ 0.98); 154/10,000
  test digits missed.
- The agreement test prints `OK`: C and Python match to `1e-9` (≈`1e-15` in practice), Conv2D /
  MaxPool2D / AvgPool2D analytic gradients match finite differences to `< 1e-6`, the im2col fast
  path equals the naive definition to `1e-12`, and the two Python paths agree. The notebook
  additionally pins our conv/pool to PyTorch's `F.conv2d`/`F.max_pool2d` at ~`1e-15` and lands
  TF/Keras LeNets at ≈ 0.987.

## Design note

We implement a **modernized** LeNet-5 — the 1998 architecture's shape (6/16 feature maps,
120/84 head) with the Module-02 toolkit inside: ReLU for scaled tanh, max pooling for trainable
average subsampling, He init, Adam, softmax cross-entropy for RBF prototypes, and full
connectivity where the paper's C3 table rationed 1998 multiplications. The post's "What LeNet-5
actually was" section walks the original faithfully, and the notebook runs a tanh/avg-pool/Xavier
ablation head-to-head. Convolution is implemented as **cross-correlation** (no kernel flip),
like every modern framework — the flip reappears naturally in the `dX` backward derivation.
As always, the C↔Python gate is a line-for-line mirrored scalar path on the toy (the canonical
`i,f,u,v / c,p,q` loop-nest contract is documented in `conv.c` and mirrored in `ToyCNN`);
MNIST runs on the vectorized im2col path and is checked by metric.
