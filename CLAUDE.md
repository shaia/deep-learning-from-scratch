# CLAUDE.md — Deep Learning From Scratch

Project conventions and operating manual. Read this first every session.

## Mission

Build **deep understanding** of deep learning by re-deriving its history in code and prose —
from the 1950s perceptron to modern Transformers, scaling, and variational autoencoders.
Every idea is implemented twice (**plain C** + **NumPy Python**), explained intuition-first,
and animated (**interactive web widgets** + **Manim** videos). The output is a linear,
blog-style static site that reads like a book.

The curriculum spine follows **Ilya Sutskever's reading list**, with a 1943–2010 "prehistory"
prepended. Full module list: `docs/CURRICULUM.md`. Master roadmap: `docs/ROADMAP.md`.

## Pedagogy principles (non-negotiable — every module obeys these)

1. **Intuition → math → code.** Open with "why did anyone need this?" history, then minimal
   math, then code. Never lead with equations.
2. **From scratch, minimal magic.** No autograd black boxes early on — write the chain rule
   out by hand. Introduce a library abstraction only *after* the manual version is understood.
3. **C and Python agree.** For every module the C and Python impls must match numerically on
   the same seed/data (agreement test + finite-difference gradient check from Module 01 on).
4. **Toy → real.** Show each mechanism on a millisecond toy problem first, then scale to a
   real dataset (MNIST/CIFAR/tiny-text) so results are verifiable, not hand-waved.
5. **See it move.** Every module ships ≥1 interactive widget and ≥1 Manim scene. It also ships a
   `notebook.ipynb` that rebuilds the idea cell-by-cell with inline plots and verifies itself
   against the canonical `python/NN.py`.

## Repo map

| Path | What lives here |
|------|-----------------|
| `docs/` | Roadmap, curriculum + status tracker, conventions, paper references |
| `docs/conventions/` | `c-style.md`, `math-notation.md`, `viz-style.md`, `new-module.md` |
| `lib/c/nanograd/` | Shared C tensor/autograd/nn lib (introduced at Module 02, grows over time) |
| `lib/python/nanograd/` | Python mirror of the same |
| `topics/NN-name/` | Per-module code: `c/`, `python/`, `anim/` (Manim), `tests/`, `notebook.ipynb`, `README.md` |
| `site/` | Astro static site (→ GitHub Pages at `book.<yourdomain>`) — `src/content/posts/*.mdx`, `src/components/viz/` widgets. A slim Ghost Pro companion teases + links to it; see `docs/conventions/publishing.md` |
| `animations/` | Shared Manim config + render pipeline → site assets |
| `data/` | Dataset download/prep scripts (MNIST, CIFAR, tiny text) |
| `tools/` | Build/render helper scripts |

Early modules (00–01) use **standalone single-file C** for zero indirection. From Module 02
(CNNs onward) topics link against `lib/c/nanograd`.

## Build / run commands

> Windows 11 + PowerShell primary. Bash tool available for POSIX scripts. C toolchain is
> clang (the clangd MCP is configured globally). Adjust paths as the toolchain firms up.

- **Standalone C topic:** `clang -O2 -std=c11 topics/00-perceptron/c/perceptron.c -o perceptron -lm`
  (On **Windows** clang uses the MSVC linker, which has no libm — **omit `-lm`**; math is in the CRT.)
- **C topic vs nanograd:** built via `CMakeLists.txt` — `cmake -B build && cmake --build build`
- **Python mirror:** `python topics/00-perceptron/python/perceptron.py`
- **C↔Python agreement test:** `python topics/00-perceptron/tests/test_agreement.py`
- **Module notebook (interactive):** `jupyter lab topics/00-perceptron/notebook.ipynb`
- **Notebook runs clean (headless check):**
  `jupyter nbconvert --to notebook --execute --inplace topics/00-perceptron/notebook.ipynb`
- **Site dev / build:** `cd site && npm run dev` / `npm run build`
- **Render a Manim scene:** `manim -qm topics/00-perceptron/anim/scene.py SceneName`

Python lives in a `venv` with `numpy`, `matplotlib`, `manim` (see `requirements.txt`).
**PyTorch is only allowed in optional "validation" cells** that cross-check our from-scratch
gradients — never as the teaching implementation.

## Coding conventions

- **C:** C11, only `libm`. Heavy comment density in teaching files. Prefer explicit,
  written-out chain-rule code over cleverness. Names match `docs/conventions/math-notation.md`.
  See `docs/conventions/c-style.md`.
- **Python:** NumPy from-scratch, mirror the C structure so the two read side-by-side.
- **Widgets:** vanilla TypeScript + Canvas/SVG, no heavy UI framework.
- **Math:** KaTeX; symbols consistent with the notation guide across all posts.
- **Thread-safety:** no function-static mutable state (per global rules).

## Adding a module

Follow the checklist in `docs/conventions/new-module.md`. A module is "done" only when all
five verification checks pass (C runs · Python matches · agreement/gradient test passes ·
post builds · animation renders) and its row in `docs/CURRICULUM.md` is updated.

## Navigation (LSP MCP)

Prefer LSP MCP over Grep/Glob for symbols: `mcp__clangd__*` for C, `mcp__pyright__*` for
Python (`find_definition`, `find_references`, `get_hover`, `workspace_symbol_search`, …).
Fall back to Grep/Glob for macros, string literals, comments, file-pattern matching.
Subagents lack MCP tools — do symbol lookups in the main thread, pass results as context.

## Git

New commit per change (no `--amend`); no `Co-Authored-By`; concise present-tense messages
explaining *why*. This repo is not yet a git repo — `git init` before the first commit.
