# Module 00 — The Perceptron

The simplest thing that can *learn*: a weighted sum plus a threshold, trained by
Rosenblatt's rule. It converges on linearly-separable data and hits a wall on XOR — the
wall that motivates hidden layers in [Module 01](../01-mlp-backprop/).

> Blog post: [`site/src/content/posts/00-perceptron.mdx`](../../site/src/content/posts/00-perceptron.mdx)
> Interactive widget: [`site/src/components/viz/PerceptronWidget.astro`](../../site/src/components/viz/PerceptronWidget.astro)
> Concept & references: [`docs/CURRICULUM.md`](../../docs/CURRICULUM.md), [`docs/references/papers.md`](../../docs/references/papers.md)

## Contents

| Path | What |
|------|------|
| `c/perceptron.c` | Standalone C11 implementation (libm-free), read top-to-bottom |
| `python/perceptron.py` | NumPy mirror — same RNG, data, and update order as the C |
| `notebook.ipynb` | Cell-by-cell walkthrough with inline plots; verifies itself against `python/perceptron.py` |
| `assignment.ipynb` | Hands-on exercise: you implement `predict` + Rosenblatt's rule in `# TODO` blocks; inline checks grade you against the answer key |
| `tests/check_utils.py` | Self-check harness the assignment imports (`rel_error`, numeric gradient checks, `compare_to_canonical`) |
| `tests/test_agreement.py` | Compiles + runs the C, runs the Python, asserts they agree |
| `anim/scene.py` | Manim scene: boundary learning the blobs, then failing on XOR |

## Run it

```bash
# C  (Windows: omit -lm; math is in the CRT)
clang -O2 -std=c11 topics/00-perceptron/c/perceptron.c -o perceptron && ./perceptron

# Python mirror
python topics/00-perceptron/python/perceptron.py

# C <-> Python agreement test
python topics/00-perceptron/tests/test_agreement.py

# Notebook: open interactively, or run headless to confirm it executes clean
jupyter lab topics/00-perceptron/notebook.ipynb
jupyter nbconvert --to notebook --execute --inplace topics/00-perceptron/notebook.ipynb

# Render the animation (needs manim + ffmpeg on PATH; no LaTeX required)
python -m manim -qm topics/00-perceptron/anim/scene.py PerceptronStory
```

The rendered video is copied to `site/public/media/00-perceptron.mp4` and embedded
in the blog post.

## What you should see

- **blobs:** accuracy reaches **1.000** — the perceptron finds a separating line.
- **xor:** accuracy stuck at **0.500** — no single line can split XOR (the wall).
- The agreement test prints `OK` with all 8 tracked values matching to `1e-9`.

## Design note

Both implementations share an identical 64-bit LCG RNG and generate the dataset with
arithmetic only (no `sin`/`log`/`sqrt`), so C and Python produce **bit-for-bit identical**
numbers on the same seed. That is what makes the agreement test meaningful rather than
approximate.
