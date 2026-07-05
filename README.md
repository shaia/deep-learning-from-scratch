# Deep Learning From Scratch

A coded, animated history of deep learning — re-derived from first principles.

We rebuild the field the way it actually grew: from the 1950s perceptron, through the
backprop era, the vision revolution (AlexNet, ResNet), the sequence models (RNN, LSTM,
attention), the Transformer, scaling laws, pipeline parallelism, and variational autoencoders.

Every idea is:

- **implemented twice** — in **plain C** (as simple and dependency-free as possible) and in
  **Python** (a NumPy reference mirror),
- **explained intuition-first** — plain-language history before any math,
- **animated to the max** — interactive web widgets you can play with, plus Manim videos,
- **read as a book** — a linear, blog-style static site.

The curriculum follows **Ilya Sutskever's reading list**, with a 1943–2010 prehistory
prepended. See [`docs/CURRICULUM.md`](docs/CURRICULUM.md) for the full module list and
[`docs/ROADMAP.md`](docs/ROADMAP.md) for the master plan.

## Why both C and Python?

C forces the ideas into the open — no autograd, no hidden broadcasting, just arrays and the
chain rule written out. Python (NumPy) mirrors the same structure so you can read them side by
side, and lets us scale to real datasets comfortably. Neither leans on a deep-learning
framework for the teaching path; PyTorch appears only to *validate* our from-scratch results.

## Repository layout

```
docs/        roadmap, curriculum, conventions, paper references
lib/         shared "nanograd" C + Python libraries (introduced once topics get heavy)
topics/      one folder per idea: c/ python/ anim/ tests/ + README
site/        the Astro static site (MDX posts + interactive widgets)
animations/  shared Manim configuration and render pipeline
data/        dataset download / preparation scripts
tools/       build & render helpers
```

## Status

Bootstrapping. See the per-module status tracker in
[`docs/CURRICULUM.md`](docs/CURRICULUM.md). First milestone: **Module 00 (Perceptron)**
end-to-end through the full toolchain.

## Getting started (once scaffolding lands)

```bash
# C (standalone early modules)
clang -O2 -std=c11 topics/00-perceptron/c/perceptron.c -o perceptron -lm && ./perceptron

# Python mirror
python topics/00-perceptron/python/perceptron.py

# The site
cd site && npm install && npm run dev
```

See [`CLAUDE.md`](CLAUDE.md) for full build/run commands and conventions.
