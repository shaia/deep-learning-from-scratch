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

### Convolutions (from Module 03)

| Symbol | Code name | Meaning |
|--------|-----------|---------|
| $K$ | `K` | conv kernel bank, shape $(F, C, K_H, K_W)$ — `K` because `W` means image width |
| $F$ | `F` / `out_ch` | number of filters (output channels) |
| $C$ | `C` / `in_ch` | input channels |
| $H, W$ | `H`, `W` | image height, width (NCHW layout: `X[i, c, h, w]`) |
| $K_H, K_W$ | `KH`, `KW` | kernel height, width |
| $u, v$ | `u`, `v` | position in the output feature map |
| $p, q$ | `p`, `q` | position inside the kernel (the taps) |
| $P$ | `P` | pooling window size (stride $P$ unless given: non-overlapping) |
| $S$ | `stride` | conv / pooling stride (Module 03 uses 1; Module 04's overlapping pool: $P=3, S=2$) |
| $P_{\text{pad}}$ | `pad` | zero-padding rings on each side (Module 04) |

Output size of a window of width $K$ slid over $H$ (conv or pool):
$H_{\text{out}} = \lfloor (H + 2P_{\text{pad}} - K)/S \rfloor + 1$ — `conv_out_size(H, K, stride, pad)`.

### Regularization (from Module 04)

| Symbol | Code name | Meaning |
|--------|-----------|---------|
| $p$ | `p` / `drop_p` | dropout probability (a unit is silenced with probability $p$) |
| $m$ | `mask` | dropout mask: $m_j = [u_j < 1-p]/(1-p)$, $u_j \sim U[0,1)$ (inverted dropout) |
| $\lambda$ | `weight_decay` | L2 weight-decay coefficient |
| $\mu$ | `momentum` | SGD momentum |

### Residual networks & batch normalization (from Module 05)

| Symbol | Code name | Meaning |
|--------|-----------|---------|
| $x_\ell$ | — | input to residual block $\ell$ |
| $\mathcal{F}(x)$ | `body` | a block's residual function (its layers); calligraphic, because $F$ already counts filters |
| $y = \mathrm{relu}(\mathcal{F}(x) + x)$ | `Residual` | the basic block; the $+ x$ is the shortcut (identity, or option (A) zero-padding when shapes change) |
| $n$ | `n` | blocks per stage in the CIFAR ResNets: depth $6n + 2$ (ResNet-20, -56 are $n = 3, 9$) |
| $\mu_c$, $\sigma_c^2$ | `mu`, `var` | batch mean and (biased) variance of channel $c$, over its $m$ values in a batch |
| $\hat{x}$ | `xhat` | normalized activation, $(x - \mu_c)/\sqrt{\sigma_c^2 + \epsilon}$ |
| $\gamma_c$, $\beta_c$ | `gamma`, `beta` | BatchNorm's learned scale and shift |
| $\epsilon$ | `eps` | BatchNorm's variance floor ($10^{-5}$) |
| — | `running_mean`, `running_var` | BatchNorm's evaluation-mode statistics (exponential averages, momentum $0.1$) |

In BatchNorm, $m$ is the number of values each channel's statistics run over: the batch
size after a Linear, batch size $\times H \times W$ after a convolution.

### Recurrent networks (from Module 06)

| Symbol | Code name | Meaning |
|--------|-----------|---------|
| $x^{\langle t\rangle}$ | `X[t]` | input at step $t$, shape $(m, n)$; a sequence is time-major, `X` of shape $(T, m, n)$ |
| $h^{\langle t\rangle}$ | `H[t]` / `h` | hidden state after step $t$, shape $(m, n_h)$; $h^{\langle -1\rangle} = h_0$ (`h0`, zeros by default) |
| $z^{\langle t\rangle}$ | `dZ[t]` (its gradient) | pre-activation, $x^{\langle t\rangle} W_{xh} + h^{\langle t-1\rangle} W_{hh} + b_h$ |
| $W_{xh}, W_{hh}, b_h$ | `Wxh`, `Whh`, `bh` | input-to-hidden, hidden-to-hidden (recurrent) weights, bias — shared by every step |
| $W_{hy}, b_y$ | `head.W`, `head.b` | the read-out applied to every step's state |
| $T$ | `T` | sequence length (steps) |
| $n_h$ | `n_h` | hidden units |
| $k$ | `truncate` | truncated BPTT: the gradient's path into the past is cut every $k$ steps ($k = 1$: Elman's training; 0: full BPTT) |
| $d$ | `delay` | the echo toy's delay: the target at step $t$ is the input at $t - d$ |
| $\rho(W_{hh})$ | `spectral_radius` | largest eigenvalue magnitude of $W_{hh}$ |
| $\tau$ | `max_norm` / `clip` | gradient-norm clipping threshold |
| $V$ | `vocab` | alphabet / vocabulary size (one-hot inputs, $V$-way outputs) |

The recurrence: $h^{\langle t\rangle} = \tanh(x^{\langle t\rangle} W_{xh} + h^{\langle t-1\rangle} W_{hh} + b_h)$ — row
vectors times weight matrices, as for `Linear`. Language-model losses are in nats per character
(divide by $\ln 2$ for bits).

### Long short-term memory (from Module 07)

| Symbol | Code name | Meaning |
|--------|-----------|---------|
| $c^{\langle t\rangle}$ | `C[t]` | the cell after step $t$, shape $(m, n_h)$; $c^{\langle -1\rangle} = c_0$ (`c0`, zeros by default) |
| $i, f, o$ | `G[t]` blocks 0, 1, 3 | input, forget and output gates, $\sigma(\cdot)$ of their pre-activations, in $(0, 1)$ |
| $g$ | `G[t]` block 2 | the candidate written into the cell, $\tanh(\cdot)$ (not the clipping section's gradient vector) |
| $z^{\langle t\rangle}$ | `dZ[t]` (its gradient) | all four pre-activations, $x^{\langle t\rangle} W_x + h^{\langle t-1\rangle} W_h + b$, blocks $[i \mid f \mid g \mid o]$ (PyTorch's order) |
| $W_x, W_h, b$ | `Wx`, `Wh`, `b` | shapes $(n, 4n_h)$, $(n_h, 4n_h)$, $(4n_h)$ — the four gates' weights side by side |
| $b_f$, $b_i$ | `gate_bias(f=…, i=…)` | the forget- and input-gate biases; at the start of training the cell's memory lasts about $1/(1 - \sigma(b_f))$ steps |
| $T$ | `T` | in the latch toy, the lag: the signal at step 0, the question at step $T - 1$ |

The cell: $c^{\langle t\rangle} = f \odot c^{\langle t-1\rangle} + i \odot g$, $h^{\langle t\rangle} = o \odot \tanh(c^{\langle t\rangle})$.
`cell_only` is Hochreiter & Schmidhuber's 1997 truncated gradient: the path through $h^{\langle t-1\rangle}$
is cut at every step, and error travels back in time only along $c$.

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
