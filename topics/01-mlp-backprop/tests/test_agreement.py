"""
Agreement + gradient tests for Module 01 (MLP + backpropagation).

Two gates, both required to pass:

1. C <-> Python agreement (project principle #3). Compile and run the C MLP,
   parse its FINAL line, run the Python mirror on the same seed, and assert every
   tracked value agrees to a tight tolerance. Because both train XOR in the same
   op order from the same LCG, they match to near double precision.

2. Finite-difference gradient check (new from Module 01 on -- the honesty gate for
   backprop). The whole module is hand-derived gradients; here we confirm the
   analytic backward pass matches numerical gradients of the same loss. A correct
   derivation lands around 1e-7 or smaller.

Run:
    python topics/01-mlp-backprop/tests/test_agreement.py

Exits 0 and prints "OK" on success; non-zero with a diff on failure. No test
framework dependency -- plain asserts, so it runs anywhere Python + clang do.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TOPIC = os.path.dirname(HERE)
C_SRC = os.path.join(TOPIC, "c", "mlp.c")
PY_DIR = os.path.join(TOPIC, "python")

TOL = 1e-9        # C<->Python: same op order in both langs, expect near-exact.
GRAD_TOL = 1e-6   # analytic vs numerical gradient: a correct derivation is ~1e-7.

# Make the Python mirror and the check utilities importable.
sys.path.insert(0, PY_DIR)
sys.path.insert(0, HERE)
import mlp as py  # noqa: E402
from check_utils import eval_numerical_gradient, rel_error  # noqa: E402


def compile_c() -> str:
    """Compile the standalone C MLP; return the binary path."""
    cc = shutil.which("clang") or shutil.which("gcc") or shutil.which("cc")
    if cc is None:
        print("SKIP: no C compiler (clang/gcc/cc) found on PATH")
        sys.exit(0)
    out_dir = tempfile.mkdtemp(prefix="mlp_")
    exe = os.path.join(out_dir, "mlp.exe" if os.name == "nt" else "mlp")
    cmd = [cc, "-O2", "-std=c11", "-Wall", "-Wextra", C_SRC, "-o", exe]
    # On Windows the math functions live in the CRT and clang uses the MSVC
    # linker, which has no libm; only add -lm on POSIX toolchains.
    if os.name != "nt":
        cmd.append("-lm")
    subprocess.run(cmd, check=True)
    return exe


def parse_c_output(text: str):
    """Extract {'xor': {name: value, ...}} from the FINAL line (key=value pairs)."""
    results = {}
    for line in text.splitlines():
        m = re.match(r"FINAL (\w+) (.*)", line)
        if not m:
            continue
        name = m.group(1)
        pairs = dict(re.findall(r"(\w+)=(\S+)", m.group(2)))
        results[name] = {k: float(v) for k, v in pairs.items()}
    return results


def close(a, b, tol=TOL):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def check_agreement():
    """Gate 1: C and Python train XOR to bit-for-bit matching numbers."""
    exe = compile_c()
    proc = subprocess.run([exe], capture_output=True, text=True, check=True)
    c = parse_c_output(proc.stdout)
    assert "xor" in c, f"missing FINAL xor line:\n{proc.stdout}"

    py_fp = dict(py.run_xor())  # (name, value) pairs -> dict

    failures = []
    print("== C <-> Python agreement (XOR) ==")
    for name, pv in py_fp.items():
        cv = c["xor"][name]
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
    """Gate 2: the hand-derived backprop matches finite-difference gradients."""
    print("\n== finite-difference gradient check (backprop honesty gate) ==")
    rng = py.Rng(py.XOR_SEED)
    X, y = py.make_xor()
    m = py.MLP(2, py.N_HIDDEN, rng, scale=py.INIT_SCALE)

    dW1, db1, dW2, db2 = m.backward(X, y)  # analytic gradients (hand-derived)
    params = [("W1", m.W1, dW1), ("b1", m.b1, db1),
              ("W2", m.W2, dW2), ("b2", m.b2, db2)]

    ok = True
    for name, P, analytic in params:
        # eval_numerical_gradient perturbs P in place; loss() reads P, so the
        # lambda's argument is ignored.
        numeric = eval_numerical_gradient(lambda _: m.loss(X, y), P)
        err = rel_error(analytic, numeric)
        status = "ok" if err < GRAD_TOL else "MISMATCH"
        print(f"  d{name:3s}  rel_error={err:.3e}  {status}")
        if err >= GRAD_TOL:
            ok = False
    if ok:
        print(f"  -> all gradients match numerically within {GRAD_TOL:g}")
    return ok


def main():
    a = check_agreement()
    g = check_gradients()
    if a and g:
        print("\nOK: C<->Python agree and backprop gradients are correct.")
        sys.exit(0)
    print("\nFAIL: see mismatches above.")
    sys.exit(1)


if __name__ == "__main__":
    main()
