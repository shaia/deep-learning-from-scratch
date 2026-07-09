# Module 01 — MLP + Backpropagation

The escape from Module 00's XOR wall: stack a **hidden layer** and learn it with the chain rule.
A `2 → h → 1` sigmoid network, trained by **backpropagation** written out by hand (Rumelhart,
Hinton & Williams 1986). It solves XOR, bends a boundary around two moons, and — scaled to
`784 → 128 → 10` with a softmax — hits **~97.7%** on MNIST. From this module on, the honesty gate
is a **finite-difference gradient check**.

> Blog post: [`site/src/content/posts/01-mlp-backprop.mdx`](../../site/src/content/posts/01-mlp-backprop.mdx)
> Interactive widget: [`site/src/components/viz/MlpWidget.astro`](../../site/src/components/viz/MlpWidget.astro)
> Concept & references: [`docs/CURRICULUM.md`](../../docs/CURRICULUM.md), [`docs/references/papers.md`](../../docs/references/papers.md)

## Contents

| Path | What |
|------|------|
| `c/mlp.c` | Standalone C11 implementation (libm-only): a 2→4→1 sigmoid MLP solving XOR, read top-to-bottom |
| `python/mlp.py` | NumPy mirror — same RNG/order as the C for XOR, plus vectorized two-moons + MNIST paths |
| `notebook.ipynb` | Cell-by-cell walkthrough: derivation, gradient check, XOR, two moons, MNIST, framework mirrors |
| `assignment.ipynb` | Hands-on: you implement `forward`, the **backward pass**, and the SGD step in `# TODO` blocks; a numeric gradient check grades you |
| `tests/check_utils.py` | Self-check harness (`rel_error`, `eval_numerical_gradient*`, `compare_to_canonical`) |
| `tests/test_agreement.py` | Compiles + runs the C, runs the Python, asserts agreement — **and** finite-diff gradient-checks backprop |
| `anim/scene.py` | Manim scene: error flowing backward, then the two-moons boundary bending over epochs |
| `../../data/get_mnist.py` | MNIST download + loader (shared), used by the Python MNIST path and the notebook |

## Run it

```bash
# C  (Windows: omit -lm; math is in the CRT)
clang -O2 -std=c11 -Wall -Wextra topics/01-mlp-backprop/c/mlp.c -o mlp && ./mlp

# Python mirror (XOR + two-moons + MNIST; downloads MNIST on first run)
python topics/01-mlp-backprop/python/mlp.py

# C <-> Python agreement + finite-difference gradient check
python topics/01-mlp-backprop/tests/test_agreement.py

# Notebook: open interactively, or run headless to confirm it executes clean
jupyter lab topics/01-mlp-backprop/notebook.ipynb
jupyter nbconvert --to notebook --execute --inplace topics/01-mlp-backprop/notebook.ipynb

# Render the animation (needs manim + ffmpeg on PATH; no LaTeX required)
python -m manim -qm --media_dir topics/01-mlp-backprop/anim/media \
  topics/01-mlp-backprop/anim/scene.py MLPStory
```

The rendered video is copied to `site/public/media/01-mlp-backprop.mp4` and embedded in the post.

## What you should see

- **xor:** loss falls to ~0.0009 and accuracy reaches **1.000** — the wall from Module 00 falls.
- **two-moons:** test accuracy ~**0.97** — a curved boundary threads between the moons.
- **mnist:** test accuracy ≥ **0.95** (≈0.977 at 15 epochs) — real digits, learned from scratch.
- The agreement test prints `OK`: C and Python match to `1e-9`, and every analytic gradient
  matches its finite-difference estimate to `< 1e-6`.

## Design note

The C and Python XOR paths share the 64-bit LCG RNG and run the forward/backward/SGD in the **same
operation order**, so they agree bit-for-bit even through `exp`/`log`. Two-moons (needs `sin`/`cos`)
and MNIST live on the Python side only, validated by an accuracy metric rather than bit-exactness —
which keeps the teaching C small enough to read in one sitting. The finite-difference gradient
check is the new load-bearing test: it is how we know the hand-derived backprop is correct.
