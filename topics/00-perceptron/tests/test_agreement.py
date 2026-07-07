"""
C <-> Python agreement test for Module 00 (the perceptron).

Principle #3 of this project: the C and Python implementations must produce
numerically matching results on the same seed and data. This test compiles and
runs the C perceptron, parses its FINAL lines, runs the Python mirror, and
asserts every number agrees to a tight tolerance.

Run:
    python topics/00-perceptron/tests/test_agreement.py

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
C_SRC = os.path.join(TOPIC, "c", "perceptron.c")
PY_DIR = os.path.join(TOPIC, "python")

TOL = 1e-9  # doubles run the same op order in both langs; expect near-exact.

# Make the Python mirror importable.
sys.path.insert(0, PY_DIR)
import perceptron as py  # noqa: E402


def compile_c() -> str:
    """Compile the standalone C perceptron; return the binary path."""
    cc = shutil.which("clang") or shutil.which("gcc") or shutil.which("cc")
    if cc is None:
        print("SKIP: no C compiler (clang/gcc/cc) found on PATH")
        sys.exit(0)
    out_dir = tempfile.mkdtemp(prefix="perceptron_")
    exe = os.path.join(out_dir, "perceptron.exe" if os.name == "nt" else "perceptron")
    cmd = [cc, "-O2", "-std=c11", "-Wall", "-Wextra", C_SRC, "-o", exe]
    # On Windows the math functions live in the CRT and clang uses the MSVC
    # linker, which has no libm; only add -lm on POSIX toolchains.
    if os.name != "nt":
        cmd.append("-lm")
    subprocess.run(cmd, check=True)
    return exe


def parse_c_output(text: str):
    """Extract {'blobs': (w0,w1,b,acc), 'xor': (...)} from the FINAL lines."""
    results = {}
    pat = re.compile(
        r"FINAL (\w+) w0=(\S+) w1=(\S+) b=(\S+) acc=(\S+)"
    )
    for m in pat.finditer(text):
        name = m.group(1)
        results[name] = tuple(float(v) for v in m.groups()[1:])
    return results


def close(a, b, tol=TOL):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def main():
    exe = compile_c()
    proc = subprocess.run([exe], capture_output=True, text=True, check=True)
    c = parse_c_output(proc.stdout)
    assert "blobs" in c and "xor" in c, f"missing FINAL lines:\n{proc.stdout}"

    py_results = {"blobs": py.run_blobs(), "xor": py.run_xor()}

    labels = ("w0", "w1", "b", "acc")
    failures = []
    for exp in ("blobs", "xor"):
        for i, lab in enumerate(labels):
            cv, pv = c[exp][i], py_results[exp][i]
            ok = close(cv, pv)
            status = "ok" if ok else "MISMATCH"
            print(f"[{exp:5s}] {lab:3s}  C={cv:.17g}  Py={pv:.17g}  {status}")
            if not ok:
                failures.append((exp, lab, cv, pv))

    if failures:
        print("\nFAIL: C and Python disagree:")
        for exp, lab, cv, pv in failures:
            print(f"  {exp}.{lab}: C={cv!r} Py={pv!r} (diff={abs(cv - pv):.3e})")
        sys.exit(1)

    print("\nOK: C and Python agree within tolerance "
          f"(tol={TOL:g}) on all {len(labels) * 2} values.")


if __name__ == "__main__":
    main()
