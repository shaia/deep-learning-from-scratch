"""
Agreement + gradient tests for Module 02 (training toolkit / nanograd).

Three gates, all required to pass:

1. C <-> Python agreement. Compile the C toolkit (linking the shared nanograd
   library), run it, parse its FINAL line, run the Python mirror (ToyNet) on the
   same seed, and assert every tracked value agrees. Both train the identical
   2->8->2 ReLU/He/Adam network on XOR in the same op order from the same LCG, so
   they match to ~1e-14 -- comfortably inside the 1e-9 gate. (Cross-library exp/
   log/sqrt rounding is the only source of the last-ulp drift, exactly as noted
   in Module 01, which is why the gate is 1e-9 rather than bit-exact.)

2. Finite-difference gradient check of the nanograd LIBRARY layers -- the honesty
   gate for the refactor. Every layer's hand-written backward is confirmed
   against numerical gradients of the same loss: the full Linear/ReLU MLP through
   softmax cross-entropy, plus each activation (sigmoid/tanh/relu) on its own.

3. nanograd (NumPy) vs the bit-exact ToyNet. A cross-check that the vectorized
   library and the explicit scalar mirror compute the same forward loss, so the
   two code paths in python/toolkit.py can't silently diverge.

Run:
    python topics/02-training-toolkit/tests/test_agreement.py

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

TOL = 1e-9        # C<->Python: same op order, cross-library math -> ~1e-14 in practice.
GRAD_TOL = 1e-6   # analytic vs numerical gradient: a correct derivation is ~1e-7.

# Make the Python mirror, the shared library, and the check utilities importable.
sys.path.insert(0, PY_DIR)
sys.path.insert(0, LIB_PY)
sys.path.insert(0, HERE)
import toolkit as py  # noqa: E402
from check_utils import (  # noqa: E402
    eval_numerical_gradient,
    eval_numerical_gradient_array,
    rel_error,
)
from nanograd import ReLU, Sigmoid, SoftmaxCrossEntropy, Tanh, mlp  # noqa: E402
from nanograd.rng import Rng  # noqa: E402


def compile_c() -> str:
    """Compile the C toolkit + nanograd sources; return the binary path."""
    cc = shutil.which("clang") or shutil.which("gcc") or shutil.which("cc")
    if cc is None:
        print("SKIP: no C compiler (clang/gcc/cc) found on PATH")
        sys.exit(0)
    out_dir = tempfile.mkdtemp(prefix="toolkit_")
    exe = os.path.join(out_dir, "toolkit.exe" if os.name == "nt" else "toolkit")
    sources = [os.path.join(C_DIR, "toolkit.c")] + [
        os.path.join(LIB_C, f) for f in ("rng.c", "nn.c", "init.c", "optim.c")
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
    """Gate 1: C and Python train the toy to matching numbers."""
    exe = compile_c()
    proc = subprocess.run([exe], capture_output=True, text=True, check=True)
    c = parse_c_output(proc.stdout)
    assert "toy" in c, f"missing FINAL toy line:\n{proc.stdout}"

    py_fp = dict(py.run_toy())  # (name, value) pairs -> dict

    failures = []
    print("== C <-> Python agreement (toy: 2->8->2 ReLU / He / Adam on XOR) ==")
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
    """Gate 2: every nanograd layer's hand-written backward matches finite diffs."""
    print("\n== finite-difference gradient check (nanograd layers) ==")
    ok = True

    # (a) The full MLP: Linear -> ReLU -> Linear through softmax cross-entropy.
    rng = Rng(3)
    net = mlp([3, 6, 4], activation="relu", init="he", rng=rng)
    loss_layer = SoftmaxCrossEntropy()
    X = np.random.default_rng(0).standard_normal((8, 3))
    y = np.array([0, 1, 2, 3, 0, 1, 2, 3])

    def loss():
        return loss_layer.forward(net.forward(X), y)

    loss()
    net.backward(loss_layer.backward())
    names = ["W1", "b1", "W2", "b2"]
    for name, p, g in zip(names, net.params(), net.grads()):
        numeric = eval_numerical_gradient(lambda _: loss(), p)
        err = rel_error(g, numeric)
        status = "ok" if err < GRAD_TOL else "MISMATCH"
        print(f"  MLP  d{name:3s}  rel_error={err:.3e}  {status}")
        ok = ok and err < GRAD_TOL

    # (b) Each activation on its own: dInput vs finite differences.
    for Act in (Sigmoid, Tanh, ReLU):
        layer = Act()
        Z = np.random.default_rng(1).standard_normal((5, 5))
        dA = np.random.default_rng(2).standard_normal((5, 5))
        layer.forward(Z)
        analytic = layer.backward(dA)
        numeric = eval_numerical_gradient_array(lambda z: Act().forward(z), Z, dA)
        err = rel_error(analytic, numeric)
        status = "ok" if err < GRAD_TOL else "MISMATCH"
        print(f"  act  {Act.__name__:8s}  rel_error={err:.3e}  {status}")
        ok = ok and err < GRAD_TOL

    if ok:
        print(f"  -> all nanograd gradients match numerically within {GRAD_TOL:g}")
    return ok


def check_paths_agree():
    """Gate 3: vectorized nanograd and the explicit ToyNet compute the same loss."""
    print("\n== nanograd (NumPy) vs bit-exact ToyNet (forward loss) ==")
    rng = Rng(py.TOY_SEED)
    net = py.ToyNet(rng)                       # explicit scalar path
    loss_scalar, _ = net.loss_and_acc()

    # Rebuild the same initial net in vectorized nanograd from the same seed and
    # feed it the same XOR batch; the forward loss must match to float tolerance.
    rng2 = Rng(py.TOY_SEED)
    vnet = mlp([py.N_IN, py.N_HID, py.N_OUT], activation="relu", init="he", rng=rng2)
    loss_layer = SoftmaxCrossEntropy()
    X = np.array(py.TOY_X, dtype=np.float64)
    y = np.array(py.TOY_Y, dtype=np.int64)
    loss_vec = loss_layer.forward(vnet.forward(X), y)

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
        print("\nOK: C<->Python agree, nanograd gradients are correct, "
              "and the two Python paths match.")
        sys.exit(0)
    print("\nFAIL: see mismatches above.")
    sys.exit(1)


if __name__ == "__main__":
    main()
