# Reading List & References

Each module is pinned to its source paper(s). The modern canon (Modules 04–16) closely follows
**Ilya Sutskever's well-known ~30-paper reading list**; Modules 00–03 are the classical
prehistory that leads into it. One-line summaries capture *why the paper mattered*.

## Era 0 — Foundations & prehistory

- **00 Perceptron**
  - McCulloch & Pitts (1943), *A Logical Calculus of the Ideas Immanent in Nervous Activity* —
    the first mathematical model of a neuron.
  - Rosenblatt (1958), *The Perceptron: A Probabilistic Model for Information Storage and
    Organization in the Brain* — a machine that learns weights from examples.
  - Minsky & Papert (1969), *Perceptrons* — proved the single-layer limits (the XOR wall).
- **01 MLP + Backpropagation**
  - Rumelhart, Hinton & Williams (1986), *Learning Representations by Back-propagating Errors* —
    training multi-layer nets with the chain rule.
- **02 Training toolkit**
  - Glorot & Bengio (2010), *Understanding the Difficulty of Training Deep Feedforward Networks*
    — Xavier initialization.
  - He et al. (2015), *Delving Deep into Rectifiers* — ReLU + He initialization.
  - Kingma & Ba (2014), *Adam: A Method for Stochastic Optimization*.
  - Srivastava et al. (2014), *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*.

## Era 1 — The vision revolution

- **03 LeNet-5** — LeCun et al. (1998), *Gradient-Based Learning Applied to Document
  Recognition* — convolutions, pooling, weight sharing.
- **04 AlexNet** — Krizhevsky, Sutskever & Hinton (2012), *ImageNet Classification with Deep
  Convolutional Neural Networks* — the result that started the modern era.
- **05 ResNet** — He et al. (2015), *Deep Residual Learning for Image Recognition* — residual
  connections make very deep nets trainable.
  - Ioffe & Szegedy (2015), *Batch Normalization*.

## Era 2 — Sequences

- **06 RNN** — Elman (1990), *Finding Structure in Time*; Werbos (1990), *Backpropagation
  Through Time*.
- **07 LSTM** — Hochreiter & Schmidhuber (1997), *Long Short-Term Memory* — gates that
  preserve gradients.
- **08 RNN Regularization** — Zaremba, Sutskever & Vinyals (2014), *Recurrent Neural Network
  Regularization* — how to apply dropout in RNNs.
- **09 Seq2Seq** — Sutskever, Vinyals & Le (2014), *Sequence to Sequence Learning with Neural
  Networks*.
  - Cho et al. (2014), *Learning Phrase Representations using RNN Encoder–Decoder* (GRU).
- **10 Attention** — Bahdanau, Cho & Bengio (2014), *Neural Machine Translation by Jointly
  Learning to Align and Translate*.
- **11 Pointer Networks** — Vinyals, Fortunato & Jaitly (2015), *Pointer Networks*.
- **12 Order Matters** — Vinyals, Bengio & Kudlur (2015), *Order Matters: Sequence to Sequence
  for Sets*.

## Era 3 — Transformers & scale

- **13 Transformer** — Vaswani et al. (2017), *Attention Is All You Need*.
- **14 Scaling Laws** — Kaplan et al. (2020), *Scaling Laws for Neural Language Models*.
  - Hoffmann et al. (2022), *Training Compute-Optimal Large Language Models* (Chinchilla) — the
    compute-optimal refinement.
- **15 GPipe** — Huang et al. (2018), *GPipe: Efficient Training of Giant Neural Networks using
  Pipeline Parallelism*.

## Era 4 — Generative / representation

- **16 VAE → Variational Lossy Autoencoder**
  - Kingma & Welling (2013), *Auto-Encoding Variational Bayes* — the VAE and reparameterization
    trick.
  - Chen et al. (2016), *Variational Lossy Autoencoder* — controlling what the latent code
    stores vs. what the decoder models.

## Broader context (optional companion reading)

- Sutskever's reading list (informal) — the collection this curriculum tracks.
- Karpathy, *The Unreasonable Effectiveness of Recurrent Neural Networks* (blog) — pairs with
  Modules 06–07.
- Olah, *Understanding LSTM Networks* (blog) — pairs with Module 07.
- Alammar, *The Illustrated Transformer* (blog) — pairs with Module 13.

> Links: add canonical arXiv/DOI URLs next to each entry as modules are built. The
> `arxiv-to-md` skill can pull any of these into local markdown for reference while writing.
