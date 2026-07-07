# Assignment Notebooks

Every module ships **two** notebooks with opposite jobs:

| Notebook | Job | Committed |
|----------|-----|-----------|
| `notebook.ipynb` | **Worked walkthrough** — the idea rebuilt cell-by-cell with outputs and inline plots; verifies itself against `python/NN.py`. Reads like solutions. | Executed, with outputs |
| `assignment.ipynb` | **Active exercise** — the student *implements* the load-bearing pieces inside `# TODO` blocks and runs inline checks that grade themselves. | With `# TODO` blocks **blank** |

This doc specifies `assignment.ipynb`. It follows the Stanford CS231n format
(explain → implement → numerically check → reflect), adapted to this project's
"intuition → math → code → play" order and its C↔Python answer key.

Reference exemplar: [`topics/00-perceptron/assignment.ipynb`](../../topics/00-perceptron/assignment.ipynb).

## Cell cadence

Repeat this four-beat loop for each concept the module teaches:

1. **Markdown — explain.** Intuition first, then the minimal math (KaTeX, symbols per
   [`math-notation.md`](math-notation.md)). End by naming exactly what to implement.
2. **Code — `# TODO`.** A function or method with the load-bearing lines removed, wrapped
   in the delimiter block (below). Everything *not* the point of the exercise — the
   deterministic RNG, dataset construction, plotting, bit-exact scaffolding — is **given**,
   clearly commented `# GIVEN`.
3. **Code — check.** Runs the student's code and grades it: print a `rel_error` (or, from
   Module 01, a numeric gradient-check error) and `assert` it is small, then
   `compare_to_canonical(...)` against `python/NN.py`. A wrong implementation must fail
   **loudly** here, not drift silently.
4. **Markdown — Inline Question.** A short conceptual question with a fill-in blank, to test
   understanding the check can't (geometry, "why", failure modes).

Close every assignment with a **final self-check** cell that imports the canonical
`python/NN.py` and asserts the student's end-to-end results match it to `1e-9` — the same
gate the worked notebook uses.

## The `# TODO` block

Use CS231n's delimiter verbatim so the boundary is unmistakable and diff-friendly. Leave a
value pre-initialized (`z = None`, `pass`) so the cell still *defines* the symbol and the
failure surfaces at the check cell, not as a `SyntaxError`/`NameError` in the definition:

```python
def preactivation(self, x0, x1):
    z = None
    ###########################################################################
    # TODO: Compute the pre-activation z = w0*x0 + w1*x1 + b and store it in   #
    # `z`. One line — the weighted sum plus the bias.                         #
    ###########################################################################

    ###########################################################################
    #                          END OF YOUR CODE                               #
    ###########################################################################
    return z
```

## The check harness — `tests/check_utils.py`

Each module keeps a small importable helper in its `tests/` folder (no shared package yet):

- `rel_error(x, y)` — the "is my answer close enough?" metric. Correct analytic results land
  around `1e-8` or smaller; `> 1e-4` means a bug.
- `eval_numerical_gradient(f, x)` / `eval_numerical_gradient_array(f, x, df)` — centered
  finite-difference gradient checks. **From Module 01 (backprop) on**, this is the centerpiece
  of every assignment: the student writes the analytic backward pass, the check confirms it
  against finite differences. Module 00 doesn't use them (the perceptron rule isn't gradient
  descent) but ships them so the harness is consistent module-to-module.
- `compare_to_canonical(student, canonical, labels)` — asserts the student's result tuple
  matches `python/NN.py` to a tight tolerance, printing a per-value table.

Import it the way [`tests/test_agreement.py`](../../topics/00-perceptron/tests/test_agreement.py)
resolves paths — add `tests/` (and `python/`) to `sys.path`. The exemplar's setup cell does
this robustly whether Jupyter is launched from the repo root or the module folder.

**The answer key** is `notebook.ipynb` + `python/NN.py`. Students are told not to peek; the
check cells *are* the graded feedback.

## Committing

- Commit `assignment.ipynb` with every `# TODO` **blank**. It is intentionally *not* runnable
  end-to-end — check cells downstream of an unfilled TODO are meant to fail until the student
  fills them. (This is the one notebook we do **not** commit executed.)
- Before committing, verify it is *solvable*: copy it beside the module, fill each TODO from
  `python/NN.py`, and run headless —
  `jupyter nbconvert --to notebook --execute --inplace <copy>` — confirming every check passes.
  Keep the filled copy out of the commit.
