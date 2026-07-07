# Roadmap: "Deep Learning From Scratch" — a coded, animated history

> This is the master reference, mirrored from the approved project plan. Module status is
> tracked separately in [`CURRICULUM.md`](CURRICULUM.md).

## Context

The goal is to build **deep understanding** of deep learning by re-deriving its history in
code and prose: starting from the 1950s perceptron, tracing the field's evolution to ~2010,
then walking the modern canon (AlexNet → ResNet → RNN/LSTM → attention → Transformers →
pointer/set networks → scaling → GPipe → variational lossy autoencoders). The topic list
closely mirrors **Ilya Sutskever's well-known reading list**, with a classical 1943–2010
"prehistory" prepended.

Every idea is implemented twice — in **plain C** (as simple and dependency-free as possible)
and in **Python** (NumPy reference mirror) — and explained from intuition first, with
**maximal graphics**: interactive web widgets to play with *and* Manim videos to watch.
The deliverable is a linear, blog-style static site that reads like a book.

## Locked decisions

| Area | Decision |
|------|----------|
| Animation | **Hybrid**: interactive web widgets (vanilla TS + Canvas, Astro islands) + **Manim** videos |
| Site | **Astro** static site (MDX, KaTeX math, code highlighting) → GitHub Pages at `book.<yourdomain>` |
| Distribution | **Slim Ghost Pro companion**: short teasers link to the canonical Astro module + drive the newsletter (see [`conventions/publishing.md`](conventions/publishing.md)) |
| C code | **Hybrid**: standalone single-file `.c` for early topics; shared `nanograd` C lib from CNNs on |
| Training | **Mix**: toy problem first (XOR, spirals, tiny sequences), then a real dataset (MNIST/CIFAR/tiny text) |

### Additional defaults
- **C:** C11, only `libm`. Build via **CMake** (clang toolchain) + one-line compile for standalone topics.
- **Python:** `venv` + **NumPy** for from-scratch code; `matplotlib` for quick plots; `manim` for animation.
  **PyTorch only in optional "validation" cells** to cross-check from-scratch gradients/outputs.
- **Widgets:** vanilla **TypeScript + Canvas/SVG**, no heavy UI framework.
- **Math notation:** KaTeX; one shared notation guide (`conventions/math-notation.md`).

## Pedagogy principles

1. **Intuition before math before code.** History framing → minimal math → code.
2. **From scratch, minimal magic.** Chain rule written out; abstractions only after the manual version.
3. **C and Python agree.** Numerically matching results on the same seed/data (gradient-check + output-match).
4. **Toy → real.** Millisecond toy problem first, then a real dataset.
5. **See it move.** ≥1 interactive widget and ≥1 Manim scene per module.

## Repository structure

```
depth-learning-from-scratch/
├─ CLAUDE.md                     # project conventions
├─ README.md
├─ docs/
│  ├─ ROADMAP.md                 # this file
│  ├─ CURRICULUM.md              # module list + per-module status tracker
│  ├─ conventions/               # c-style, math-notation, viz-style, new-module
│  └─ references/papers.md       # reading list w/ links + one-line summaries
├─ lib/
│  ├─ c/nanograd/                # shared C tensor/autograd/nn lib (grows over time)
│  └─ python/nanograd/           # Python mirror
├─ topics/NN-name/{c,python,anim,tests,README.md}
├─ site/                         # Astro static site (posts + viz components)
├─ animations/                   # shared Manim config + render pipeline
├─ data/                         # dataset download/prep scripts
├─ tools/                        # build/render helpers
└─ CMakeLists.txt
```

## Curriculum (~17 modules, historical order)

Each module ships the full stack: **C impl · Python impl · ≥1 Manim scene · ≥1 web widget ·
MDX blog post · C↔Py agreement test.**

**Era 0 — Foundations & prehistory (1943–2006)**
- **00 Perceptron** (McCulloch–Pitts 1943, Rosenblatt 1957) — linear unit, perceptron learning rule. Toy: linearly-separable blobs; the XOR wall. *(standalone C)*
- **01 MLP + Backpropagation** (Rumelhart 1986) — hidden layers, chain rule by hand, SGD. Toy: XOR, two-moons. Real: MNIST. *(standalone C)*
- **02 Training toolkit** (1990s–2000s) — activations, initialization, momentum → RMSProp → Adam, L2/early-stopping. *(begins shared `nanograd`)*

**Era 1 — The vision revolution (1998–2015)**
- **03 Convolutions / LeNet-5** (LeCun 1998) — conv, pooling, weight sharing. Real: MNIST.
- **04 AlexNet** (2012) — ReLU at scale, dropout, data augmentation. Real: CIFAR-10 (scaled).
- **05 ResNet** (2015) — residual connections, BatchNorm, why depth was hard. Real: CIFAR-10.

**Era 2 — Sequences (1986–2015)**
- **06 RNN** — recurrence, BPTT. Toy: counting/echo. Real: char-level text.
- **07 LSTM** (1997) — gates, vanishing gradients. Real: char-RNN text generation.
- **08 RNN Regularization** (Zaremba 2014) — where/how to apply dropout in RNNs.
- **09 Seq2Seq** (Sutskever 2014) — encoder–decoder. Toy: reverse/copy; simple translation.
- **10 Attention** (Bahdanau 2014) — soft alignment; visualize the attention matrix.
- **11 Pointer Networks** (Vinyals 2015) — attention as a pointer. Toy: convex hull / sort.
- **12 Order Matters: Seq2Seq for Sets** (Vinyals 2015) — read/process/write, set invariance.

**Era 3 — Transformers & scale (2017–2020)**
- **13 Transformer** (Vaswani 2017) — self-attention, multi-head, positional encoding. Real: tiny char Transformer.
- **14 Scaling Laws / "scaling canon"** (Kaplan 2020) — empirical power laws; reproduce a mini loss-vs-compute curve.
- **15 GPipe** (Huang 2018) — pipeline parallelism, micro-batching. Conceptual + pipeline-schedule visualization.

**Era 4 — Generative / representation**
- **16 VAE → Variational Lossy Autoencoder** (Kingma 2013 → Chen 2016) — ELBO, reparameterization, then the VLAE lossy-code idea. Real: MNIST generation.

## Execution roadmap (phased)

- **Phase 0 — Scaffolding.** Repo skeleton, `CLAUDE.md`, `docs/`, Astro skeleton, `CMakeLists.txt`, Manim config, `data/` scripts, Python venv. *(docs core: done; toolchain installs: pending)*
- **Phase 1 — Tracer bullet: Module 00 Perceptron, full stack.** De-risks the entire toolchain before scaling.
- **Phase 2 — Foundations.** Modules 01–02; stand up shared `nanograd`.
- **Phase 3 — Vision era.** Modules 03–05.
- **Phase 4 — Sequence era.** Modules 06–12.
- **Phase 5 — Transformers & scale.** Modules 13–15.
- **Phase 6 — Generative.** Module 16.
- **Ongoing.** Cross-link posts, maintain `CURRICULUM.md`, polish site nav.

## Verification

**Toolchain (end of Phase 1) — acceptance gate:** `cmake --build` produces the perceptron
binary and it trains to converged accuracy; the Python mirror reproduces the same result on
the same seed; the agreement test matches C↔Python within tolerance; `npm run build` renders
the post with working math, code highlighting, and a live widget; the Manim scene renders and
embeds.

**Per module thereafter:** same five checks (C runs · Python matches · agreement/gradient
test passes · post builds · animation renders) before the module is marked done. Real-dataset
modules also assert a target metric (e.g. MLP ≥95% on MNIST).

## Open questions / assumptions
- Astro as the specific SSG (vs mdBook/Quarto) — chosen for interactive islands; swap-able.
- Manim on Windows needs ffmpeg + LaTeX; Phase 0 verifies this before committing per-module
  (fallback: Manim in WSL/Docker, or matplotlib for simple scenes).
- CIFAR-scale training in pure C may be slow; "scaled-down" configs keep runs to minutes.
