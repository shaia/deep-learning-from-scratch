"""
Check utilities for the Module 02 assignment notebook (see assignment.ipynb).

Same harness as Module 01 (the tools are module-agnostic): a relative-error
metric, centered finite-difference gradient checks, and a helper that compares a
student's result to the canonical answer key (python/toolkit.py + nanograd).

In Module 02 the gradient check earns its keep on the *library* layers: the
student writes ReLU's backward and the Adam step, and these numerically confirm
them. A correct analytic gradient matches the finite-difference one to a
`rel_error` around 1e-7 or smaller.

Imported by adding tests/ to sys.path, the same pattern tests/test_agreement.py uses:

    import sys, os
    sys.path.insert(0, os.path.join("topics", "02-training-toolkit", "tests"))
    from check_utils import rel_error, eval_numerical_gradient, compare_to_canonical
"""

import numpy as np


def rel_error(x, y, eps=1e-12):
    """Relative error max |x - y| / max(eps, |x| + |y|).

    The standard "is my answer close enough?" metric. A correct analytic result
    lands around 1e-8 or smaller against its finite-difference or reference
    value; anything above ~1e-4 means a real bug.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return np.max(np.abs(x - y) / np.maximum(eps, np.abs(x) + np.abs(y)))


def eval_numerical_gradient(f, x, verbose=False, h=1e-5):
    """Naive centered finite-difference gradient of a scalar function f at x.

    - f: callable taking a single numpy array and returning a scalar.
    - x: numpy array; the point to evaluate the gradient at (modified in place
      and restored).

    Returns dx with the same shape as x, where dx[i] ~= df/dx[i].
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h
        fxph = f(x)  # f(x + h)
        x[ix] = oldval - h
        fxmh = f(x)  # f(x - h)
        x[ix] = oldval  # restore
        grad[ix] = (fxph - fxmh) / (2 * h)
        if verbose:
            print(ix, grad[ix])
        it.iternext()
    return grad


def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """Centered finite-difference gradient for a function returning an array.

    Given f: x -> y and an upstream gradient df = dL/dy of the same shape as y,
    returns dL/dx by finite differences. The array-valued companion to
    eval_numerical_gradient, for checking a single layer's backward pass.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval
        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad


def compare_to_canonical(student, canonical, labels=None, tol=1e-9):
    """Assert a student's result tuple matches the canonical answer key.

    - student, canonical: equal-length sequences of floats. `canonical` comes
      from the module's python/toolkit.py (the answer key).
    - labels: optional names for each element, used in the printout.
    - tol: relative tolerance.

    Prints a per-element table and raises AssertionError on the first mismatch,
    so a wrong implementation fails loudly instead of silently drifting.
    """
    student = tuple(float(v) for v in student)
    canonical = tuple(float(v) for v in canonical)
    assert len(student) == len(canonical), (
        f"length mismatch: student has {len(student)}, canonical has {len(canonical)}"
    )
    if labels is None:
        labels = [f"[{i}]" for i in range(len(student))]

    failures = []
    for lab, sv, cv in zip(labels, student, canonical):
        ok = abs(sv - cv) <= tol * max(1.0, abs(sv), abs(cv))
        print(f"  {lab:6s}  you={sv:.17g}  answer={cv:.17g}  {'ok' if ok else 'MISMATCH'}")
        if not ok:
            failures.append((lab, sv, cv))

    assert not failures, (
        "your result does not match the canonical answer key:\n"
        + "\n".join(f"  {lab}: you={sv!r} answer={cv!r} (diff={abs(sv - cv):.3e})"
                    for lab, sv, cv in failures)
    )
    print(f"OK: all {len(student)} values match the answer key within {tol:g}.")
