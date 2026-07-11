# Module 02 — Making Deep Nets Trainable (initialization · optimizers · regularization)

Backprop (Module 01) *learns*; this module makes it learn **fast and reliably** — and stands up
the shared [`nanograd`](../../lib) library that every later module builds on. Four ideas:
**ReLU** (a non-saturating activation), **Xavier/He** fan-in-scaled initialization, the optimizer
lineage **SGD → momentum → RMSProp → Adam**, and **L2 / early stopping**. The from-scratch Adam
is verified to be byte-for-byte PyTorch's, and the C and Python impls agree to ~`1e-14`.

> Blog post: [`site/src/content/posts/02-training-toolkit.mdx`](../../site/src/content/posts/02-training-toolkit.mdx)
> Interactive widget: [`site/src/components/viz/OptimizerWidget.astro`](../../site/src/components/viz/OptimizerWidget.astro)
> Shared library: [`lib/python/nanograd`](../../lib/python/nanograd) · [`lib/c/nanograd`](../../lib/c/nanograd)
> Concept & references: [`docs/CURRICULUM.md`](../../docs/CURRICULUM.md), [`docs/references/papers.md`](../../docs/references/papers.md)

## Contents

| Path | What |
|------|------|
| `../../lib/python/nanograd/` | Shared NumPy library: `layers` (Linear/Sigmoid/Tanh/ReLU/SoftmaxCE), `init` (xavier/he), `optim` (SGD/RMSProp/Adam), `net` (Sequential/mlp), `rng` |
| `../../lib/c/nanograd/` | C mirror: `nn.c`, `init.c`, `optim.c`, `rng.c`, `nanograd.h` (a static lib) |
| `python/toolkit.py` | Canonical: a line-for-line mirrored `2→8→2` ReLU/He/Adam toy on XOR, plus vectorized optimizer/init ablations and MNIST |
| `c/toolkit.c` | The C toy, **linking nanograd** (first module to do so) — same seed/op order as the Python |
| `notebook.ipynb` | Cell-by-cell walkthrough: activations, init, optimizers on a loss surface, ablations, L2/early-stop, MNIST, framework mirrors |
| `assignment.ipynb` | Hands-on: you implement **ReLU**, **He init**, and the **Adam step** in `# TODO` blocks; a gradient check grades you |
| `tests/check_utils.py` | Self-check harness (`rel_error`, `eval_numerical_gradient*`, `compare_to_canonical`) |
| `tests/test_agreement.py` | Compiles + runs the C, runs the Python, asserts agreement — **and** gradient-checks every nanograd layer |
| `anim/scene.py` | Manim scene: optimizer trajectories on the Beale surface, then init's effect on a deep tanh stack |

## Run it

```bash
# C  (standalone; Windows: omit -lm, math is in the CRT)
clang -O2 -std=c11 -Wall -Wextra -I lib/c/nanograd \
  topics/02-training-toolkit/c/toolkit.c \
  lib/c/nanograd/rng.c lib/c/nanograd/nn.c lib/c/nanograd/init.c lib/c/nanograd/optim.c -o toolkit && ./toolkit

# C  (via CMake — the documented "link nanograd" path)
cmake -B build && cmake --build build && ./build/topics/02-training-toolkit/c/toolkit

# Python mirror (toy + optimizer/init ablations + MNIST; downloads MNIST on first run)
python topics/02-training-toolkit/python/toolkit.py

# C <-> Python agreement + finite-difference gradient checks of the nanograd layers
python topics/02-training-toolkit/tests/test_agreement.py

# Notebook: open interactively, or run headless to confirm it executes clean
jupyter nbconvert --to notebook --execute --inplace topics/02-training-toolkit/notebook.ipynb

# Render the animation (needs manim + ffmpeg on PATH; no LaTeX required)
python -m manim -qm --media_dir topics/02-training-toolkit/anim/media \
  topics/02-training-toolkit/anim/scene.py ToolkitStory
```

The rendered video is copied to `site/public/media/02-training-toolkit.mp4` and embedded in the post.

## What you should see

- **toy:** XOR solved (accuracy **1.000**, loss ≈ `3.7e-5`) by a `2→8→2` ReLU/He net under Adam.
- **optimizer ablation:** all four reach ≥ 0.90 on two-moons; the adaptive methods drop the loss sooner.
- **init ablation:** Xavier beats small-uniform on a deep tanh net; the notebook shows *why* (activation std).
- **mnist:** Adam reaches ≈ **0.965** in 8 epochs (target ≥ 0.95).
- The agreement test prints `OK`: C and Python match to `1e-9` (≈`1e-14` in practice), every
  nanograd layer's analytic gradient matches finite differences to `< 1e-6`, and the two Python
  paths agree. The notebook additionally shows our Adam equals `torch.optim.Adam` to `1e-16`.

## Design note

This is the first module to link the shared **nanograd** library rather than being one standalone
file — the refactor promised in `docs/conventions/c-style.md`. The library is deliberately *not* an
autograd engine: every layer's `backward()` is the Module-01 chain rule, written out by hand and
localized to one op. As in Module 01, the C↔Python gate is a small **line-for-line mirrored** path
(the explicit scalar `ToyNet`, matched by `c/toolkit.c` within the `1e-9` gate — ≈`1e-14` observed,
the only drift being last-ULP `exp`/`log`/`sqrt` rounding across the two math libraries); the
ablations and MNIST run on the vectorized NumPy library and are checked by a metric. Weight init in
both languages draws the same stream from a shared Box–Muller normal on the same 64-bit LCG.
