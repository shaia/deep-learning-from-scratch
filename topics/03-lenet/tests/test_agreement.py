"""
Agreement + gradient tests for Module 03 (convolutions / LeNet-5).

Three gates, all required to pass:

1. C <-> Python agreement. Compile the C toy CNN (linking the shared nanograd
   library, now including conv.c), run it, parse its FINAL line, run the Python
   mirror (ToyCNN) on the same seeds, and assert every tracked value agrees.
   Both train the identical conv(1->4,3x3)/ReLU/maxpool/linear net on the bars
   data in the same op order from the same LCG streams, so they match to
   ~1e-15 -- comfortably inside the 1e-9 gate. (Cross-library exp/log/sqrt
   rounding is the only source of the last-ulp drift, exactly as in Module 02.)

2. Finite-difference gradient check of the NEW nanograd layers -- the honesty
   gate for the conv machinery. Conv2D's hand-written dK/db/dX, max-pool's
   gradient routing and avg-pool's spreading are confirmed against numerical
   gradients, the full toy CNN is checked end-to-end through softmax
   cross-entropy, and the vectorized im2col forward/backward is pinned to the
   naive quadruple-loop definition at 1e-12.

   (Max-pool is checked on continuous random inputs: its kinks -- exact ties
   inside a window -- are a measure-zero event a random draw never hits, so
   finite differences are valid there.)

3. nanograd (NumPy) vs the scalar ToyCNN. A cross-check that the vectorized
   library and the explicit scalar mirror compute the same forward loss, so
   the two code paths in python/lenet.py can't silently diverge.

Run:
    python topics/03-lenet/tests/test_agreement.py

Exits 0 and prints "OK" on success; non-zero with a diff on failure. Plain
asserts, no test framework -- runs anywhere Python + clang do.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOPIC = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(TOPIC))
C_DIR = os.path.join(TOPIC, "c")
PY_DIR = os.path.join(TOPIC, "python")
LIB_C = os.path.join(REPO, "lib", "c", "nanograd")
LIB_PY = os.path.join(REPO, "lib", "python")

TOL = 1e-9        # C<->Python: same op order, cross-library math -> ~1e-15 in practice.
GRAD_TOL = 1e-6   # analytic vs numerical gradient: a correct derivation is ~1e-7.
EXACT_TOL = 1e-12  # im2col vs naive: same arithmetic, different order.

# Make the Python mirror, the shared library, and the check utilities importable.
sys.path.insert(0, PY_DIR)
sys.path.insert(0, LIB_PY)
sys.path.insert(0, HERE)
import lenet as py  # noqa: E402
from check_utils import (  # noqa: E402
    eval_numerical_gradient,
    eval_numerical_gradient_array,
    rel_error,
)
from nanograd import (  # noqa: E402
    AvgPool2D,
    Conv2D,
    MaxPool2D,
    SoftmaxCrossEntropy,
)
from nanograd.rng import Rng  # noqa: E402


def compile_c() -> str:
    """Compile the C toy CNN + nanograd sources; return the binary path."""
    cc = shutil.which("clang") or shutil.which("gcc") or shutil.which("cc")
    if cc is None:
        print("SKIP: no C compiler (clang/gcc/cc) found on PATH")
        sys.exit(0)
    out_dir = tempfile.mkdtemp(prefix="lenet_")
    exe = os.path.join(out_dir, "lenet.exe" if os.name == "nt" else "lenet")
    sources = [os.path.join(C_DIR, "lenet.c")] + [
        os.path.join(LIB_C, f)
        for f in ("rng.c", "nn.c", "init.c", "optim.c", "conv.c")
    ]
    cmd = [cc, "-O2", "-std=c11", "-Wall", "-Wextra", "-I", LIB_C, *sources, "-o", exe]
    # On Windows math is in the CRT; only POSIX toolchains need -lm.
    if os.name != "nt":
        cmd.append("-lm")
    subprocess.run(cmd, check=True)
    return exe


def parse_c_output(text: str):
    """Extract {'toy': {name: value, ...}} from the FINAL line (key=value pairs)."""
    results = {}
    for line in text.splitlines():
        m = re.match(r"FINAL (\w+) (.*)", line)
        if not m:
            continue
        pairs = dict(re.findall(r"(\w+)=(\S+)", m.group(2)))
        results[m.group(1)] = {k: float(v) for k, v in pairs.items()}
    return results


def close(a, b, tol=TOL):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def check_agreement():
    """Gate 1: C and Python train the toy CNN to matching numbers."""
    exe = compile_c()
    proc = subprocess.run([exe], capture_output=True, text=True, check=True)
    c = parse_c_output(proc.stdout)
    assert "toy" in c, f"missing FINAL toy line:\n{proc.stdout}"

    py_fp = dict(py.run_toy())  # (name, value) pairs -> dict

    failures = []
    print("== C <-> Python agreement (toy: conv 4@3x3 / ReLU / maxpool / "
          "linear on bars) ==")
    for name, pv in py_fp.items():
        cv = c["toy"][name]
        ok = close(cv, pv)
        print(f"  {name:6s}  C={cv:.17g}  Py={pv:.17g}  {'ok' if ok else 'MISMATCH'}")
        if not ok:
            failures.append((name, cv, pv))
    if failures:
        print("\nFAIL: C and Python disagree:")
        for name, cv, pv in failures:
            print(f"  {name}: C={cv!r} Py={pv!r} (diff={abs(cv - pv):.3e})")
        return False
    print(f"  -> all {len(py_fp)} values agree within tol={TOL:g}")
    return True


def check_gradients():
    """Gate 2: the new conv/pool backwards match finite differences, and the
    vectorized im2col path matches the naive definition."""
    print("\n== finite-difference gradient check (conv machinery) ==")
    ok = True
    rng = np.random.default_rng(0)

    # (a) Conv2D: dK, db, dX on a multi-channel case, X(2,2,6,6) K(3,2,3,3).
    conv = Conv2D(2, 3, 3, 3)
    conv.K[:] = rng.standard_normal(conv.K.shape)
    conv.b[:] = rng.standard_normal(conv.b.shape)
    X = rng.standard_normal((2, 2, 6, 6))
    dY = rng.standard_normal((2, 3, 4, 4))

    conv.forward(X)
    dX = conv.backward(dY)
    for name, analytic, wrt in (("dK", conv.dK, conv.K),
                                ("db", conv.db, conv.b),
                                ("dX", dX, X)):
        numeric = eval_numerical_gradient_array(
            lambda _: conv.forward(X), wrt, dY)
        err = rel_error(analytic, numeric)
        status = "ok" if err < GRAD_TOL else "MISMATCH"
        print(f"  Conv2D     {name:3s}  rel_error={err:.3e}  {status}")
        ok = ok and err < GRAD_TOL

    # (b) Pooling: dX vs finite differences. Continuous random inputs keep
    # max-pool away from its (measure-zero) tie/kink points.
    for Pool in (MaxPool2D, AvgPool2D):
        pool = Pool(2)
        Xp = rng.standard_normal((2, 3, 6, 6))
        dYp = rng.standard_normal((2, 3, 3, 3))
        pool.forward(Xp)
        analytic = pool.backward(dYp)
        numeric = eval_numerical_gradient_array(
            lambda _: pool.forward(Xp), Xp, dYp)
        err = rel_error(analytic, numeric)
        status = "ok" if err < GRAD_TOL else "MISMATCH"
        print(f"  {Pool.__name__:9s}  dX   rel_error={err:.3e}  {status}")
        ok = ok and err < GRAD_TOL

    # (c) The full toy CNN end-to-end through softmax cross-entropy: every
    # parameter's hand-written gradient vs the numerical one.
    net = py.build_toy_cnn()
    loss_layer = SoftmaxCrossEntropy()
    Xb, yb = py.make_bars()

    def loss():
        return loss_layer.forward(net.forward(Xb), yb)

    loss()
    net.backward(loss_layer.backward())
    names = ["K1", "b1", "W2", "b2"]
    for name, p, g in zip(names, net.params(), net.grads()):
        numeric = eval_numerical_gradient(lambda _: loss(), p)
        err = rel_error(g, numeric)
        status = "ok" if err < GRAD_TOL else "MISMATCH"
        print(f"  toy CNN    d{name:3s} rel_error={err:.3e}  {status}")
        ok = ok and err < GRAD_TOL

    # (d) Vectorized (im2col) vs naive definition: forward and all three
    # backward gradients must be the same arithmetic to ~1e-12.
    Y_fast = conv.forward(X)
    Y_naive = py.naive_conv2d_forward(X, conv.K, conv.b)
    dX_fast = conv.backward(dY)
    dX_naive, dK_naive, db_naive = py.naive_conv2d_backward(X, conv.K, dY)
    for name, a, b in (("fwd Y", Y_fast, Y_naive), ("bwd dX", dX_fast, dX_naive),
                       ("bwd dK", conv.dK, dK_naive), ("bwd db", conv.db, db_naive)):
        err = float(np.max(np.abs(a - b)))
        status = "ok" if err < EXACT_TOL else "MISMATCH"
        print(f"  im2col vs naive  {name:7s}  max|diff|={err:.3e}  {status}")
        ok = ok and err < EXACT_TOL

    if ok:
        print(f"  -> all conv gradients match numerically within {GRAD_TOL:g}")
    return ok


def check_paths_agree():
    """Gate 3: vectorized nanograd and the scalar ToyCNN compute the same loss."""
    print("\n== nanograd (NumPy) vs scalar ToyCNN (forward loss) ==")
    Xb, yb = py.make_bars()
    scalar = py.ToyCNN(Rng(py.TOY_SEED), Xb, yb)
    loss_scalar, _ = scalar.loss_and_acc()

    # Rebuild the same initial net in vectorized nanograd from the same seed
    # and feed it the same bars batch; the forward loss must match tightly.
    vnet = py.build_toy_cnn(Rng(py.TOY_SEED))
    loss_layer = SoftmaxCrossEntropy()
    loss_vec = loss_layer.forward(vnet.forward(Xb), yb)

    err = rel_error(loss_scalar, loss_vec)
    status = "ok" if err < 1e-9 else "MISMATCH"
    print(f"  initial loss  scalar={loss_scalar:.12g}  numpy={loss_vec:.12g}  "
          f"rel_error={err:.3e}  {status}")
    return err < 1e-9


def main():
    a = check_agreement()
    g = check_gradients()
    p = check_paths_agree()
    if a and g and p:
        print("\nOK: C<->Python agree, the conv gradients are correct, "
              "and the two Python paths match.")
        sys.exit(0)
    print("\nFAIL: see mismatches above.")
    sys.exit(1)


if __name__ == "__main__":
    main()
