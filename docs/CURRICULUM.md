# Curriculum & Status Tracker

The full module list with per-artifact status. Update a module's row as artifacts land.
A module is **Done** only when all five checks pass: C runs · Python matches ·
agreement/gradient test · post builds · animation renders.

**Legend:** ☐ not started · ◐ in progress · ☑ done

## Status

| # | Module | C | Py | NB | Asgn | Test | Anim | Widget | Post | Status |
|---|--------|:-:|:--:|:--:|:----:|:----:|:----:|:------:|:----:|:------:|
| 00 | Perceptron | ☑ | ☑ | ☑ | ☑ | ☑ | ☑ | ☑ | ☑ | ☑ |
| 01 | MLP + Backpropagation | ☑ | ☑ | ☑ | ☑ | ☑ | ☑ | ☑ | ☑ | ☑ |
| 02 | Making deep nets trainable (init/opt/reg) | ☑ | ☑ | ☑ | ☑ | ☑ | ☑ | ☑ | ☑ | ☑ |
| 03 | Convolutions / LeNet-5 | ☑ | ☑ | ☑ | ☑ | ☑ | ☑ | ☑ | ☑ | ☑ |
| 04 | AlexNet | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| 05 | ResNet | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| 06 | RNN | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| 07 | LSTM | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| 08 | RNN Regularization | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| 09 | Seq2Seq | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| 10 | Attention | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| 11 | Pointer Networks | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| 12 | Order Matters: Seq2Seq for Sets | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| 13 | Transformer | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| 14 | Scaling Laws | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| 15 | GPipe | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| 16 | VAE → Variational Lossy Autoencoder | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |

## Per-module detail

Each entry: the **hook** (why it mattered), the **mechanism** to implement, the **toy** and
**real** training targets, and the **animation idea**. Source papers: `references/papers.md`.

### Era 0 — Foundations & prehistory (1943–2006)

**00 Perceptron** — *Can a machine learn a decision from examples?*
Mechanism: weighted sum + threshold; perceptron learning rule (`w += lr·(y−ŷ)·x`).
Toy: linearly-separable 2D blobs; then demonstrate the **XOR wall** (why one layer isn't enough).
Anim: decision boundary rotating into place as points are classified; the XOR failure.

**01 MLP + Backpropagation** — *Stack layers and learn them with the chain rule.*
Mechanism: hidden layer(s), nonlinearity, forward pass, backprop by hand, SGD.
Toy: XOR (now solvable), two-moons. Real: MNIST (target ≥95%).
Anim: gradients flowing backward; the two-moons boundary bending over epochs.

**02 Making deep nets trainable** — *The activations, initialization, optimizers, and regularizers that turn "it learns" into "it learns fast."*
Mechanism: sigmoid/tanh/ReLU; init (Xavier/He); momentum → RMSProp → Adam; L2, early-stopping.
Toy: same MLP, ablations. Anim: optimizer trajectories on a loss surface; init effects.
*Introduces `lib/*/nanograd`.*

### Era 1 — The vision revolution (1998–2015)

**03 Convolutions / LeNet-5** — *Weight sharing + locality for images.*
Mechanism: conv, pooling, feature maps. Toy: "bars" (vertical vs horizontal, 8×8).
Real: MNIST (target ≥98%). Anim: kernel sliding; learned filters.
*Adds conv/pool to `lib/*/nanograd`.*

**04 AlexNet** — *Depth + ReLU + dropout + GPUs break ImageNet.*
Mechanism: ReLU at scale, dropout, data augmentation. Real: CIFAR-10 (scaled-down).
Anim: dropout masking; augmentation pipeline; first-layer filters.

**05 ResNet** — *Let gradients skip — train hundreds of layers.*
Mechanism: residual blocks, BatchNorm, degradation problem. Real: CIFAR-10.
Anim: identity shortcut; gradient magnitude with vs without skips.

### Era 2 — Sequences (1986–2015)

**06 RNN** — *Share weights across time.* Mechanism: recurrence, BPTT.
Toy: counting/echo. Real: char-level text. Anim: unrolling through time.

**07 LSTM** — *Gates that protect the gradient.* Mechanism: input/forget/output gates, cell state.
Real: char-RNN text generation. Anim: gate values over a sequence; vanishing-gradient contrast.

**08 RNN Regularization** — *Dropout done right in recurrent nets.*
Mechanism: dropout on non-recurrent connections. Anim: masked vs recurrent connections.

**09 Seq2Seq** — *Map a sequence to a sequence.* Mechanism: encoder–decoder, thought vector.
Toy: reverse/copy; simple translation. Anim: encoder compressing → decoder emitting.

**10 Attention** — *Let the decoder look back.* Mechanism: alignment scores, context vector.
Anim: the attention matrix lighting up as it translates.

**11 Pointer Networks** — *Output positions in the input.* Mechanism: attention as a pointer.
Toy: convex hull / sorting. Anim: pointer selecting input elements in order.

**12 Order Matters: Seq2Seq for Sets** — *Order-invariant encoding.*
Mechanism: read/process/write, permutation invariance. Anim: same set, shuffled → same output.

### Era 3 — Transformers & scale (2017–2020)

**13 Transformer** — *Attention is all you need.*
Mechanism: scaled dot-product self-attention, multi-head, positional encoding, residual+LN.
Real: tiny char-level Transformer. Anim: multi-head attention maps; positional encodings.

**14 Scaling Laws** — *Loss falls as a power law in compute/data/params.*
Mechanism: train a family of tiny models; fit the power law. Anim: loss-vs-compute curve forming.

**15 GPipe** — *Split a model across devices; pipeline the micro-batches.*
Mechanism: pipeline parallelism, micro-batching, bubble. Anim: the pipeline schedule diagram.

### Era 4 — Generative / representation

**16 VAE → Variational Lossy Autoencoder** — *Learn a probabilistic latent code.*
Mechanism: ELBO, reparameterization trick; then VLAE's lossy-code / autoregressive-decoder idea.
Real: MNIST generation. Anim: latent-space interpolation; the reparameterization trick.
