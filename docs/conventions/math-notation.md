# Math Notation Guide

One consistent notation across posts, C, and Python so the three read side by side. When code
and math use the same names, the reader never has to translate.

## Core symbols

| Symbol | Code name | Meaning |
|--------|-----------|---------|
| $x$ | `x` | input vector / example |
| $X$ | `X` | input matrix (rows = samples, cols = features) |
| $y$ | `y` | target / label |
| $\hat{y}$ | `yhat` | model prediction |
| $w$ | `w` | weight (vector or matrix $W$) |
| $b$ | `b` | bias |
| $z$ | `z` | pre-activation ($z = Wx + b$) |
| $a$ | `a` | activation ($a = \sigma(z)$) |
| $L$ | `loss` | loss (scalar) |
| $\eta$ | `lr` | learning rate |
| $\sigma$ | `sigmoid` | logistic sigmoid |
| $\nabla$, $\partial L/\partial w$ | `grad_w` | gradient of $L$ w.r.t. `w` |
| $m$ | `n_samples` | number of examples |
| $n$ | `n_features` | input dimension |
| $h$ | `n_hidden` | hidden units |
| $\ell$ | `layer` | layer index |
| $t$ | `t` | timestep (sequences) |

## Conventions

- **Vectors** lowercase ($x$, $w$); **matrices** uppercase ($X$, $W$). In code, `W` for a
  weight matrix, `w` for a weight vector.
- **Row-major, rows = samples.** $X \in \mathbb{R}^{m \times n}$: $m$ examples, $n$ features.
- **Gradients** are named `grad_<var>` and always mean $\partial L / \partial \text{var}$.
- **Hats** for predictions ($\hat{y}$), **stars** for optima ($w^\*$) when needed.
- **Subscripts** index elements ($w_j$), **superscripts in parens** index layers ($W^{(\ell)}$),
  **superscript $\langle t \rangle$** indexes time for sequences.

## Standard forms we reuse

- Linear unit: $z = w^\top x + b$
- Sigmoid: $\sigma(z) = \dfrac{1}{1 + e^{-z}}$, with $\sigma'(z) = \sigma(z)(1-\sigma(z))$
- Binary cross-entropy: $L = -[\,y\log\hat{y} + (1-y)\log(1-\hat{y})\,]$
- SGD update: $w \leftarrow w - \eta\, \partial L/\partial w$
- Perceptron rule: $w \leftarrow w + \eta\,(y - \hat{y})\,x$

## In prose (MDX)

Inline math with `$...$`, display math with `$$...$$` (KaTeX via remark-math/rehype-katex).
Introduce every symbol in words the first time it appears in a post.
