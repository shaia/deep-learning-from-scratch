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
    training multi-layer nets with the chain rule; framed hidden units as learned representations.
  - Linnainmaa (1970), master's thesis, Univ. of Helsinki — reverse-mode automatic
    differentiation, backprop's engine (no neural nets).
  - Werbos (1974), *Beyond Regression* (PhD thesis, Harvard) — backprop proposed for multilayer
    networks a decade early, and ignored.
  - LeCun (1985), *Une procédure d'apprentissage pour réseau à seuil asymétrique* (Cognitiva 85);
    Parker (1985), *Learning-Logic* (MIT TR) — independent rediscoveries.
  - Cybenko (1989), *Approximation by Superpositions of a Sigmoidal Function* — the universal
    approximation theorem: existence, not learnability.
  - LeCun, Bottou, Bengio & Haffner (1998), *Gradient-Based Learning Applied to Document
    Recognition* — source of MNIST, the module's real-data benchmark (main paper for Module 03).
    [doi:10.1109/5.726791](https://doi.org/10.1109/5.726791)
- **02 Making deep nets trainable**
  - Glorot & Bengio (2010), *Understanding the Difficulty of Training Deep Feedforward Networks*
    — Xavier initialization.
  - He et al. (2015), *Delving Deep into Rectifiers* — ReLU + He initialization.
  - Kingma & Ba (2014), *Adam: A Method for Stochastic Optimization*.
  - Srivastava et al. (2014), *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*.

## Era 1 — The vision revolution

- **03 LeNet-5** — LeCun, Bottou, Bengio & Haffner (1998), *Gradient-Based Learning Applied to
  Document Recognition*, Proc. IEEE 86(11) — convolutions, pooling, weight sharing; also the
  source of MNIST and of graph transformer networks (the check-reading system).
  [doi:10.1109/5.726791](https://doi.org/10.1109/5.726791)
  - Hubel & Wiesel (1962), *Receptive Fields, Binocular Interaction and Functional Architecture
    in the Cat's Visual Cortex* — simple/complex cells: the biological blueprint for
    local-detector + pooling hierarchies.
  - Fukushima (1980), *Neocognitron: A Self-organizing Neural Network Model* — the CNN
    architecture (replicated local features, pooling, hierarchy), without end-to-end learning.
  - LeCun et al. (1989), *Backpropagation Applied to Handwritten Zip Code Recognition* —
    backprop meets weight sharing; the first trained-end-to-end, deployed CNN.
- **04 AlexNet** — Krizhevsky, Sutskever & Hinton (2012), *ImageNet Classification with Deep
  Convolutional Neural Networks*, NIPS 25 — the result that started the modern era: ReLU,
  dropout, crop/flip augmentation, overlapping pooling, LRN, two GPUs; 15.3% top-5 on
  ILSVRC-2012 against 26.2% for the next team. Reprinted in *Communications of the ACM* 60(6),
  2017. [doi:10.1145/3065386](https://doi.org/10.1145/3065386)
  - Deng, Dong, Socher, Li, Li & Fei-Fei (2009), *ImageNet: A Large-Scale Hierarchical Image
    Database*, CVPR — the dataset that made scale possible.
  - Russakovsky et al. (2015), *ImageNet Large Scale Visual Recognition Challenge*, IJCV — the
    competition's own history and numbers. [arXiv:1409.0575](https://arxiv.org/abs/1409.0575)
  - Nair & Hinton (2010), *Rectified Linear Units Improve Restricted Boltzmann Machines*, ICML
    — the ReLU AlexNet adopted.
  - Hinton, Srivastava, Krizhevsky, Sutskever & Salakhutdinov (2012), *Improving Neural Networks
    by Preventing Co-adaptation of Feature Detectors* — dropout's first appearance.
    [arXiv:1207.0580](https://arxiv.org/abs/1207.0580)
  - Krizhevsky (2009), *Learning Multiple Layers of Features from Tiny Images*, U. Toronto tech
    report — the source of CIFAR-10, this module's real dataset.
  - Cireșan, Meier, Masci & Schmidhuber (2012), *Multi-Column Deep Neural Network for Traffic
    Sign Classification*, Neural Networks — GPU-trained CNNs beating humans (IJCNN 2011) a
    year before AlexNet.
  - Simonyan & Zisserman (2014), *Very Deep Convolutional Networks for Large-Scale Image
    Recognition* (VGG) — deeper, all-3×3, and the finding that LRN does not help.
    [arXiv:1409.1556](https://arxiv.org/abs/1409.1556)
- **05 ResNet** — He, Zhang, Ren & Sun (2015), *Deep Residual Learning for Image Recognition*,
  CVPR 2016 (best paper) — the degradation problem (deeper plain nets fit their own training
  data worse), and its fix: blocks that learn a residual added to an identity shortcut. 152
  layers, 3.57% top-5 on ILSVRC 2015, first place in all five ILSVRC/COCO 2015 tracks; by
  Nature's 2025 count the most-cited paper of the 21st century.
  [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
  - Ioffe & Szegedy (2015), *Batch Normalization: Accelerating Deep Network Training by
    Reducing Internal Covariate Shift*, ICML — per-channel normalization with a learned scale
    and shift; 14× fewer steps, 4.9% top-5 (ensemble).
    [arXiv:1502.03167](https://arxiv.org/abs/1502.03167)
  - Srivastava, Greff & Schmidhuber (2015), *Highway Networks* (ICML workshop) and *Training
    Very Deep Networks* (NIPS) — gated shortcuts, inspired by the LSTM, training nets of
    hundreds of layers seven months before ResNet.
    [arXiv:1505.00387](https://arxiv.org/abs/1505.00387) ·
    [arXiv:1507.06228](https://arxiv.org/abs/1507.06228)
  - Szegedy et al. (2014), *Going Deeper with Convolutions* (GoogLeNet) — 22 layers, the
    ILSVRC 2014 winner at 6.67%, with auxiliary classifiers to get gradient to its lower
    layers. [arXiv:1409.4842](https://arxiv.org/abs/1409.4842)
  - He, Zhang, Ren & Sun (2016), *Identity Mappings in Deep Residual Networks*, ECCV — why a
    clean identity path matters; pre-activation blocks; a 1001-layer CIFAR network.
    [arXiv:1603.05027](https://arxiv.org/abs/1603.05027)
  - Veit, Wilber & Belongie (2016), *Residual Networks Behave Like Ensembles of Relatively
    Shallow Networks*, NIPS — the unraveled view (2ⁿ paths) and the lesion study: deleting a
    block barely hurts a ResNet; deleting a layer sends VGG to chance.
    [arXiv:1605.06431](https://arxiv.org/abs/1605.06431)
  - Balduzzi et al. (2017), *The Shattered Gradients Problem: If resnets are the answer, then
    what is the question?*, ICML — in deep plain nets the gradient decorrelates into noise;
    skip connections keep it structured. [arXiv:1702.08591](https://arxiv.org/abs/1702.08591)
  - Santurkar, Tsipras, Ilyas & Madry (2018), *How Does Batch Normalization Help
    Optimization?*, NeurIPS — not internal covariate shift: a smoother loss landscape.
    [arXiv:1805.11604](https://arxiv.org/abs/1805.11604)
  - Yang, Pennington, Rao, Sohl-Dickstein & Schoenholz (2019), *A Mean Field Theory of Batch
    Normalization*, ICLR — batch norm itself makes gradients explode with depth in nets
    without skip connections. [arXiv:1902.08129](https://arxiv.org/abs/1902.08129)
  - Hochreiter & Schmidhuber (1997), *Long Short-Term Memory* — the constant error carousel:
    an identity path through time, eighteen years before it ran through depth (Module 07).

## Era 2 — Sequences

- **06 RNN** — Elman (1990), *Finding Structure in Time*, Cognitive Science 14(2) — the simple
  recurrent network: context units hold a copy of the previous hidden state; trained one step
  deep, it found word boundaries and word classes in streams of letters and words.
  [doi:10.1207/s15516709cog1402_1](https://doi.org/10.1207/s15516709cog1402_1) · Werbos (1990),
  *Backpropagation Through Time: What It Does and How to Do It*, Proc. IEEE 78(10) — unroll the
  recurrence into a deep network with shared weights and backpropagate through it.
  [doi:10.1109/5.58337](https://doi.org/10.1109/5.58337)
  - Rumelhart, Hinton & Williams (1986), *Learning Internal Representations by Error
    Propagation*, in *Parallel Distributed Processing*, vol. 1 — recurrent nets unfolded in time
    into layered nets with tied weights.
  - Jordan (1986), *Serial Order: A Parallel Distributed Processing Approach*, ICS Report 8604,
    UC San Diego — the output fed back as state: the network Elman's simplified.
  - Werbos (1988), *Generalization of Backpropagation with Application to a Recurrent Gas Market
    Model*, Neural Networks 1(4) — BPTT applied, two years before the tutorial.
    [doi:10.1016/0893-6080(88)90007-X](https://doi.org/10.1016/0893-6080(88)90007-X)
  - Williams & Zipser (1989), *A Learning Algorithm for Continually Running Fully Recurrent
    Neural Networks*, Neural Computation 1(2) — real-time recurrent learning: gradients carried
    forward instead of backward. [doi:10.1162/neco.1989.1.2.270](https://doi.org/10.1162/neco.1989.1.2.270)
  - Hochreiter (1991), *Untersuchungen zu dynamischen neuronalen Netzen*, diploma thesis, TU
    München — the first analysis of vanishing and exploding gradients through time.
  - Bengio, Simard & Frasconi (1994), *Learning Long-Term Dependencies with Gradient Descent is
    Difficult*, IEEE Trans. Neural Networks 5(2) — storing information robustly forces the
    gradient from the far past to vanish. [doi:10.1109/72.279181](https://doi.org/10.1109/72.279181)
  - Pascanu, Mikolov & Bengio (2013), *On the Difficulty of Training Recurrent Neural Networks*,
    ICML — exploding gradients as cliffs in the loss surface; gradient-norm clipping.
    [arXiv:1211.5063](https://arxiv.org/abs/1211.5063)
  - Bengio, Ducharme, Vincent & Jauvin (2003), *A Neural Probabilistic Language Model*, JMLR 3 —
    a learned network against n-gram counts; Mikolov et al. (2010), *Recurrent Neural Network
    Based Language Model*, Interspeech — the recurrent version.
  - Sutskever, Martens & Hinton (2011), *Generating Text with Recurrent Neural Networks*, ICML —
    character-level text generation with a (multiplicative) RNN, trained by Hessian-free
    optimization.
  - Karpathy (2015), *The Unreasonable Effectiveness of Recurrent Neural Networks* (blog) — char-rnn
    and the Tiny Shakespeare dataset this module trains on.
    [karpathy.github.io](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)
  - Le, Jaitly & Hinton (2015), *A Simple Way to Initialize Recurrent Networks of Rectified Linear
    Units* — the identity-initialized ReLU RNN (an exercise here).
    [arXiv:1504.00941](https://arxiv.org/abs/1504.00941)
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
