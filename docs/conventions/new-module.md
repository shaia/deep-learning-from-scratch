# How to Add a Module

Every module follows the same recipe so the book stays consistent and each idea ships complete.
Work in order — the code comes before the prose, the prose before the polish.

## 0. Set up

- Pick the number and slug: `topics/NN-slug/` (e.g. `topics/03-lenet/`).
- Create `c/ python/ anim/ tests/ notebook.ipynb README.md`.
- Read the module's row in [`../CURRICULUM.md`](../CURRICULUM.md): hook, mechanism, toy, real,
  animation idea. Read the source paper(s) in [`../references/papers.md`](../references/papers.md).

## 1. Python reference first

NumPy is the fastest place to get the math right. Implement forward + backward from scratch,
mirror the intended C structure, train on the **toy** problem, confirm it learns. Add an
optional PyTorch validation cell to cross-check gradients if the module is nontrivial.

## 1b. Notebook

`topics/NN-slug/notebook.ipynb` — the same subject broken into cells, exploiting what notebooks
do well: markdown+KaTeX narration interleaved with runnable code and **inline matplotlib plots**.
Follow the reference notebook [`topics/00-perceptron/notebook.ipynb`](../../topics/00-perceptron/notebook.ipynb)
and the module's **intuition → math → code → play** order:

1. **Rebuild the idea inline**, cell by cell, faithful to `python/NN.py` (RNG → data → model →
   train), with a plot for each moving part.
2. **Verify** at the end: import the canonical `python/NN.py` and `assert` the inline results
   match it to a tight tolerance — this is what keeps the notebook from silently drifting from
   the C↔Python source of truth. If they diverge, the cell fails loudly.
3. Keep it dependency-light (numpy + matplotlib); guard any `ipywidgets`/`torch` cell in
   `try/except` so the notebook still runs top-to-bottom without them.

Commit the notebook **executed with outputs** so the plots render on GitHub.

## 1c. Assignment notebook

`topics/NN-slug/assignment.ipynb` — the active-learning counterpart to §1b. Where
`notebook.ipynb` *shows* the idea worked out, this makes the student *implement* the
load-bearing pieces themselves (`# TODO` blocks) and grade their work with inline numeric
checks. Follow the full spec in [`assignment.md`](assignment.md) and the exemplar
[`topics/00-perceptron/assignment.ipynb`](../../topics/00-perceptron/assignment.ipynb):

1. **Cell cadence:** explain (markdown) → `# TODO` (blank the load-bearing lines, mark the
   rest `# GIVEN`) → check cell (`rel_error` / from Module 01 a numeric gradient check, then
   `compare_to_canonical` against `python/NN.py`) → **Inline Question**. Repeat per concept.
2. **Check harness:** add `tests/check_utils.py` (`rel_error`, `eval_numerical_gradient*`,
   `compare_to_canonical`) if the module doesn't have it yet; import it via `sys.path` like
   `tests/test_agreement.py` does. The answer key is `notebook.ipynb` + `python/NN.py`.
3. **Commit with `# TODO` blocks blank** (the one notebook *not* committed executed). Before
   committing, prove it's solvable: fill a throwaway copy from `python/NN.py` and run it
   headless — every check must pass.

## 2. C implementation

Port the Python, following [`c-style.md`](c-style.md). Standalone single file for early
modules; link `lib/c/nanograd` from Module 02 on. Same fixed seed as Python. It must compile
clean under `-Wall -Wextra` and train the toy problem to the same result.

## 3. Agreement test

`tests/test_agreement.py`: run C (or its dumped outputs) and Python on the same seed/data and
assert they match within tolerance. From Module 01 on, include a **finite-difference gradient
check** against the analytic gradients. This is the gate that keeps both impls honest.

## 4. Scale to the real dataset

Wire in the real target (MNIST/CIFAR/tiny-text via `data/` scripts). Assert the module's
metric (e.g. ≥95% MNIST). Keep runtime to minutes with scaled-down configs where needed.

## 5. Animations (≥1 widget + ≥1 Manim scene)

Per [`viz-style.md`](viz-style.md): a widget for the thing to manipulate, a Manim scene for the
narrated mechanism. Render Manim through `animations/` into the site assets.

## 6. Blog post

`site/src/content/posts/NN-slug.mdx`, structured **intuition → history → math → code → play**:

1. **Hook** — the problem this solved, in plain language and historical context.
2. **Intuition** — the idea before any equation.
3. **Math** — minimal, KaTeX, symbols per [`math-notation.md`](math-notation.md).
4. **Code walkthrough** — key excerpts from the C and Python, side by side.
5. **Interactive** — embed the widget; tell the reader what to try.
6. **Watch** — embed the Manim scene.
7. **Exercises / what broke / what's next** — link forward to the next module.

Frontmatter: `title`, `moduleNumber`, `description`, `draft` (flip to `false` to publish),
and an optional `ghostUrl` back-link (see below). The URL slug is **not** in frontmatter —
Astro derives it from the filename, so name the file `NN-slug.mdx` (e.g. `00-perceptron.mdx`).

**Optional — slim Ghost teaser.** To reach the newsletter audience, write a short teaser in
Ghost that links to this (canonical) Astro module, set the Ghost post's canonical URL to the
Astro URL, and record the teaser link in `ghostUrl`. Full flow: [`publishing.md`](publishing.md).

## 7. Wire up & verify

- Link the post into the site index/nav; cross-link neighbors.
- Point `topics/NN-slug/README.md` at the post and list build/run commands.
- Run the **checks**: C runs · Python matches · agreement/gradient test passes ·
  notebook runs top-to-bottom clean
  (`jupyter nbconvert --to notebook --execute --inplace topics/NN-slug/notebook.ipynb`) ·
  assignment is solvable (a filled copy runs clean; see [`assignment.md`](assignment.md)) ·
  `npm run build` succeeds · Manim scene renders.
- Update the module's row in [`../CURRICULUM.md`](../CURRICULUM.md) to ☑ (including the `NB`
  and `Asgn` columns).

## Definition of done

All checks green (including the notebook executing clean end-to-end and the assignment being
solvable), status row updated, post reachable from the site index, and a fresh reader could go
intuition → running code → animation without leaving the page.
