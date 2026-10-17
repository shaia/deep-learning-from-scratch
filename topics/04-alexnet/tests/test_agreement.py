"""
Agreement + gradient tests for Module 04 (AlexNet).

Three gates, all required to pass:

1. C <-> Python agreement. Compile the C toy AlexNet (linking the shared
   nanograd library, now including norm.c and augment.c), run it, parse its
   FINAL lines, run the Python mirror (ToyAlexNet) on the same seeds, and
   assert every tracked value agrees. Both train the identical strided-conv /
   overlapping-pool / dropout network on the mirror-glyphs data, with the
   same crops, flips and dropout masks, in the same op order from the same LCG
   streams -- so they match to the last digit in practice, comfortably inside
   the 1e-9 gate. A second FINAL line pins the C local response
   normalization (forward + backward) to nanograd's.

2. Finite-difference gradient check of the NEW nanograd machinery -- the
   honesty gate. Strided + padded Conv2D (dK/db/dX), overlapping MaxPool2D,
   Dropout (with its mask held fixed), and LocalResponseNorm are confirmed
   against numerical gradients; the whole toy AlexNet is checked end-to-end
   through softmax cross-entropy; and the vectorized layers are pinned to the
   naive definitions in python/alexnet.py at 1e-12.

   (Max-pool is checked on continuous random inputs: its kinks -- exact ties
   inside a window -- are a measure-zero event a random draw never hits.)

3. nanograd (NumPy) vs the scalar ToyAlexNet. The vectorized library and the
   explicit scalar mirror must compute the same initial loss AND the same
   parameters after three full training steps -- crops, flips, dropout masks
   and momentum included -- so the two code paths can't silently diverge.

Run:
    python topics/04-alexnet/tests/test_agreement.py

Exits 0 and prints "OK" on success; non-zero with a diff on failure. Plain
asserts, no test framework -- runs anywhere Python + clang do.
"""

import atexit
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

TOL = 1e-9         # C<->Python: same op order, cross-library math.
GRAD_TOL = 1e-6    # analytic vs numerical gradient: a correct derivation is ~1e-7.
EXACT_TOL = 1e-12  # vectorized vs naive: same arithmetic, different order.

# Make the Python mirror, the shared library, and the check utilities importable.
sys.path.insert(0, PY_DIR)
sys.path.insert(0, LIB_PY)
sys.path.insert(0, HERE)
import alexnet as py  # noqa: E402
from check_utils import (  # noqa: E402
    eval_numerical_gradient,
    eval_numerical_gradient_array,
    rel_error,
)
from nanograd import (  # noqa: E402
    SGD,
    Conv2D,
    Dropout,
    LocalResponseNorm,
    MaxPool2D,
    SoftmaxCrossEntropy,
    center_crop,
    random_crop_flip,
)
from nanograd.rng import Rng  # noqa: E402

C_SOURCES = ("rng.c", "nn.c", "init.c", "optim.c", "conv.c", "norm.c", "augment.c")


def compile_c() -> str:
    """Compile the C toy AlexNet + nanograd sources; return the binary path."""
    cc = shutil.which("clang") or shutil.which("gcc") or shutil.which("cc")
    if cc is None:
        print("SKIP: no C compiler (clang/gcc/cc) found on PATH")
        sys.exit(0)
    out_dir = tempfile.mkdtemp(prefix="alexnet_")
    # The exe is run after this function returns, so clean up at exit.
    atexit.register(shutil.rmtree, out_dir, ignore_errors=True)
    exe = os.path.join(out_dir, "alexnet.exe" if os.name == "nt" else "alexnet")
    sources = [os.path.join(C_DIR, "alexnet.c")] + [
        os.path.join(LIB_C, f) for f in C_SOURCES]
    cmd = [cc, "-O2", "-std=c11", "-Wall", "-Wextra", "-I", LIB_C, *sources, "-o", exe]
    # On Windows math is in the CRT; only POSIX toolchains need -lm.
    if os.name != "nt":
        cmd.append("-lm")
    subprocess.run(cmd, check=True)
    return exe


def parse_c_output(text: str):
    """Extract {'toy': {...}, 'lrn': {...}} from the FINAL lines (key=value pairs)."""
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


def compare(label, c_vals, py_pairs):
    """Print and check one FINAL line against its Python fingerprint."""
    py_fp = dict(py_pairs)
    missing = sorted(set(py_fp) - set(c_vals))
    assert not missing, f"C FINAL {label} line is missing keys {missing}"
    ok = True
    for name, pv in py_fp.items():
        cv = c_vals[name]
        good = close(cv, pv)
        print(f"  {name:8s}  C={cv:.17g}  Py={pv:.17g}  {'ok' if good else 'MISMATCH'}")
        if not good:
            print(f"    -> diff={abs(cv - pv):.3e}")
        ok = ok and good
    return ok


def check_agreement():
    """Gate 1: C and Python train the toy AlexNet (and run LRN) identically."""
    exe = compile_c()
    proc = subprocess.run([exe], capture_output=True, text=True, check=True)
    c = parse_c_output(proc.stdout)
    for key in ("toy", "lrn"):
        assert key in c, f"missing FINAL {key} line:\n{proc.stdout}"

    print("== C <-> Python agreement (toy AlexNet on mirror glyphs: strided conv, "
          "overlapping pool, dropout, crops+flips, SGD+momentum) ==")
    ok_toy = compare("toy", c["toy"], py.run_toy())
    print("== C <-> Python agreement (local response normalization, fwd + bwd) ==")
    ok_lrn = compare("lrn", c["lrn"], py.lrn_fingerprint())
    if ok_toy and ok_lrn:
        print(f"  -> all values agree within tol={TOL:g}")
    return ok_toy and ok_lrn


def _grad_line(name, analytic, numeric, tol=GRAD_TOL):
    err = rel_error(analytic, numeric)
    good = err < tol
    print(f"  {name:28s} rel_error={err:.3e}  {'ok' if good else 'MISMATCH'}")
    return good


def _exact_line(name, a, b):
    err = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
    good = err < EXACT_TOL
    print(f"  {name:28s} max|diff|={err:.3e}  {'ok' if good else 'MISMATCH'}")
    return good


def check_gradients():
    """Gate 2: the new layers' backwards match finite differences, and the
    vectorized layers match their naive definitions."""
    print("\n== finite-difference gradient check (AlexNet machinery) ==")
    ok = True
    rng = np.random.default_rng(0)

    # (a) Strided + padded Conv2D -- AlexNet's stride-4 economy, our stride 2
    # -- plus a stride-1 "same" conv, on a multi-channel input.
    for stride, pad, H in ((2, 2, 7), (1, 1, 5), (3, 1, 8)):
        conv = Conv2D(2, 3, 3, 3, stride=stride, pad=pad)
        conv.K[:] = rng.standard_normal(conv.K.shape)
        conv.b[:] = rng.standard_normal(conv.b.shape)
        X = rng.standard_normal((2, 2, H, H))
        Y = conv.forward(X)
        dY = rng.standard_normal(Y.shape)
        dX = conv.backward(dY)
        for name, analytic, wrt in (("dK", conv.dK, conv.K), ("db", conv.db, conv.b),
                                    ("dX", dX, X)):
            numeric = eval_numerical_gradient_array(lambda _: conv.forward(X), wrt, dY)
            ok &= _grad_line(f"Conv2D S={stride} pad={pad} {name}", analytic, numeric)
        # ...and the fast path is the naive definition, forward and backward.
        dX_n, dK_n, db_n = py.naive_conv2d_backward(X, conv.K, dY, stride, pad)
        ok &= _exact_line(f"  vs naive S={stride} p={pad} fwd",
                          conv.forward(X), py.naive_conv2d_forward(X, conv.K, conv.b, stride, pad))
        conv.backward(dY)
        ok &= _exact_line(f"  vs naive S={stride} p={pad} dX", dX, dX_n)
        ok &= _exact_line(f"  vs naive S={stride} p={pad} dK", conv.dK, dK_n)
        ok &= _exact_line(f"  vs naive S={stride} p={pad} db", conv.db, db_n)

    # (b) Overlapping max-pool (3x3, stride 2): cells in two windows can win
    # both, and their gradients must add.
    pool = MaxPool2D(3, 2)
    Xp = rng.standard_normal((2, 3, 7, 7))
    Yp = pool.forward(Xp)
    dYp = rng.standard_normal(Yp.shape)
    analytic = pool.backward(dYp)
    numeric = eval_numerical_gradient_array(lambda _: pool.forward(Xp), Xp, dYp)
    ok &= _grad_line("MaxPool2D 3/2 dX", analytic, numeric)
    Yn, arg = py.naive_maxpool2d_forward(Xp, 3, 2)
    ok &= _exact_line("  vs naive fwd", Yp, Yn)
    ok &= _exact_line("  vs naive dX", analytic, py.naive_maxpool2d_backward(dYp, arg, Xp.shape))

    # (c) Dropout: with the mask held fixed (re-seeded per call) it is a
    # linear map, so finite differences must reproduce dY * mask exactly.
    drop = Dropout(0.5)
    Xd = rng.standard_normal((4, 9))
    dYd = rng.standard_normal((4, 9))

    def drop_fwd(_):
        drop.rng = Rng(11)
        return drop.forward(Xd)

    drop_fwd(None)
    analytic = drop.backward(dYd)
    numeric = eval_numerical_gradient_array(drop_fwd, Xd, dYd)
    ok &= _grad_line("Dropout p=0.5 dX", analytic, numeric)

    # (d) Local response normalization (alpha raised so the cross-channel
    # coupling term is not lost in the rounding).
    lrn = LocalResponseNorm(5, 0.1, 0.75, 2.0)
    Xl = rng.standard_normal((2, 7, 3, 3)) * 2.0
    Yl = lrn.forward(Xl)
    dYl = rng.standard_normal(Yl.shape)
    analytic = lrn.backward(dYl)
    numeric = eval_numerical_gradient_array(lambda _: lrn.forward(Xl), Xl, dYl)
    ok &= _grad_line("LocalResponseNorm dX", analytic, numeric)
    ok &= _exact_line("  vs naive fwd", Yl, py.naive_lrn_forward(Xl, 5, 0.1, 0.75, 2.0))

    # (e) The full toy AlexNet end-to-end through softmax cross-entropy, on
    # center crops, with one dropout mask held fixed: every parameter's
    # hand-written gradient vs the numerical one.
    Xg, yg = py.make_glyphs(8, py.DATA_SEED)
    Xc = center_crop(Xg, py.CROP)
    net = py.build_toy_alexnet()
    dropout = net.layers[6]
    loss_layer = SoftmaxCrossEntropy()

    def loss():
        dropout.rng = Rng(py.DROP_SEED)
        return loss_layer.forward(net.forward(Xc), yg)

    loss()
    net.backward(loss_layer.backward())
    for name, p, g in zip(["K1", "b1", "W2", "b2", "W3", "b3"], net.params(), net.grads()):
        numeric = eval_numerical_gradient(lambda _: loss(), p)
        ok &= _grad_line(f"toy AlexNet d{name}", g, numeric)

    if ok:
        print(f"  -> all gradients match numerically within {GRAD_TOL:g}; "
              f"fast paths equal the definitions within {EXACT_TOL:g}")
    return ok


def check_paths_agree():
    """Gate 3: vectorized nanograd and the scalar ToyAlexNet agree on the
    initial loss and on every parameter after three full training steps."""
    print("\n== nanograd (NumPy) vs scalar ToyAlexNet ==")
    Xtr, ytr = py.make_glyphs(py.N_TRAIN, py.DATA_SEED)
    scalar = py.ToyAlexNet(Rng(py.TOY_SEED), Xtr, ytr)
    vnet = py.build_toy_alexnet(Rng(py.TOY_SEED), drop_rng=Rng(py.DROP_SEED))
    loss_layer = SoftmaxCrossEntropy()

    loss_scalar, _ = scalar.loss_and_acc()
    vnet.eval()
    loss_vec = loss_layer.forward(vnet.forward(center_crop(Xtr, py.CROP)), ytr)
    err = rel_error(loss_scalar, loss_vec)
    ok = err < 1e-9
    print(f"  initial loss   scalar={loss_scalar:.12g}  numpy={loss_vec:.12g}  "
          f"rel_error={err:.3e}  {'ok' if ok else 'MISMATCH'}")

    # Three steps each, both drawing crops/flips from Rng(AUG_SEED) and masks
    # from Rng(DROP_SEED) -- the vectorized path through Rng.random().
    aug = Rng(py.AUG_SEED)
    opt = SGD(lr=py.SGD_LR, momentum=py.SGD_MOM, weight_decay=py.SGD_WD)
    for _ in range(3):
        scalar.step()
        vnet.train()
        loss_layer.forward(vnet.forward(random_crop_flip(Xtr, py.CROP, aug)), ytr)
        vnet.backward(loss_layer.backward())
        opt.step(vnet.params(), vnet.grads())
    for name, sp, vp in zip(["K1", "b1", "W2", "b2", "W3", "b3"],
                            scalar.params(), vnet.params()):
        err = rel_error(np.array(sp), vp.reshape(-1))
        good = err < 1e-9
        print(f"  after 3 steps  {name:3s} rel_error={err:.3e}  {'ok' if good else 'MISMATCH'}")
        ok = ok and good
    return ok


def main():
    a = check_agreement()
    g = check_gradients()
    p = check_paths_agree()
    if a and g and p:
        print("\nOK: C<->Python agree, the AlexNet gradients are correct, "
              "and the two Python paths match.")
        sys.exit(0)
    print("\nFAIL: see mismatches above.")
    sys.exit(1)


if __name__ == "__main__":
    main()
