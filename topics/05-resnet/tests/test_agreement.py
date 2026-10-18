"""
Agreement + gradient tests for Module 05 (ResNet).

Three gates, all required to pass:

1. C <-> Python agreement. Compile the C toy (linking the shared nanograd
   library, now with BatchNorm, the residual add, the zero-padding shortcut
   and global average pooling), run it, parse its FINAL lines, run the Python
   mirror (ToyResNet) on the same seeds, and assert every tracked value
   agrees. Both train the same 34-layer network on the deep-spirals data
   twice -- plain, then residual, from the same initial weights -- in the
   same op order from the same LCG streams, so they match to the last digit
   in practice, comfortably inside the 1e-9 gate. Two more FINAL lines pin the
   C BatchNorm over NCHW maps (training + evaluation mode) and a downsampling
   conv residual block with global average pooling to nanograd's.

2. Finite-difference gradient check of the NEW nanograd machinery -- the
   honesty gate. BatchNorm on (m, C) and (n, C, H, W) inputs, in training and
   in evaluation mode (dX, dgamma, dbeta); the residual block, with the
   identity shortcut and with the option-(A) zero-padding shortcut;
   ZeroPadShortcut and GlobalAvgPool2D alone; bias-free Linear and Conv2D;
   and small plain and residual toy networks end-to-end through softmax
   cross-entropy. BatchNorm is also pinned to its definition: the forward to
   Ioffe & Szegedy's Algorithm 1, the backward to the full m x m Jacobian
   (1e-12), and the running statistics to their update rule.

3. nanograd (NumPy) vs the C-order ToyResNet. The vectorized library and the
   mirror must compute the same initial loss and the same gradients for
   BOTH twins, and -- for the residual twin -- the same parameters after
   three training steps. (The plain twin is compared at step 0 only: its
   gradients explode toward the input, and the training dynamics amplify
   the rounding differences between two correct implementations -- BLAS
   versus loop order -- by orders of magnitude per step. The test prints how
   far apart they drift; the chapter discusses why.)

Run:
    python topics/05-resnet/tests/test_agreement.py

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
EXACT_TOL = 1e-12  # fast path vs definition: same arithmetic, different order.
# Finite-difference step for the conv residual block. Its input gradient has
# very small entries (BatchNorm's backward projects two directions out of
# every channel), and at the default h = 1e-5 the outputs' float rounding,
# ~1e-16/h, is no longer small next to them; at 1e-4 the truncation error is
# ~1e-8 and every pre-ReLU value in that block sits > 1e-3 from the kink. (The
# toy nets below keep h = 1e-5: with 32 points, a step of 1e-4 can carry a
# pre-activation across ReLU's kink, which no step size forgives.)
NET_H = 1e-4

# Make the Python mirror, the shared library, and the check utilities importable.
sys.path.insert(0, PY_DIR)
sys.path.insert(0, LIB_PY)
sys.path.insert(0, HERE)
import resnet as py  # noqa: E402
from check_utils import (  # noqa: E402
    eval_numerical_gradient,
    eval_numerical_gradient_array,
    rel_error,
)
from nanograd import (  # noqa: E402
    SGD,
    BatchNorm,
    Conv2D,
    GlobalAvgPool2D,
    Linear,
    ReLU,
    Residual,
    Sequential,
    SoftmaxCrossEntropy,
    ZeroPadShortcut,
)
from nanograd.rng import Rng  # noqa: E402

C_SOURCES = ("rng.c", "nn.c", "init.c", "optim.c", "conv.c", "norm.c")


def compile_c() -> str:
    """Compile the C toy ResNet + nanograd sources; return the binary path."""
    cc = shutil.which("clang") or shutil.which("gcc") or shutil.which("cc")
    if cc is None:
        print("SKIP: no C compiler (clang/gcc/cc) found on PATH")
        sys.exit(0)
    out_dir = tempfile.mkdtemp(prefix="resnet_")
    # The exe is run after this function returns, so clean up at exit.
    atexit.register(shutil.rmtree, out_dir, ignore_errors=True)
    exe = os.path.join(out_dir, "resnet.exe" if os.name == "nt" else "resnet")
    sources = [os.path.join(C_DIR, "resnet.c")] + [
        os.path.join(LIB_C, f) for f in C_SOURCES]
    cmd = [cc, "-O2", "-std=c11", "-Wall", "-Wextra", "-I", LIB_C, *sources, "-o", exe]
    # On Windows math is in the CRT; only POSIX toolchains need -lm.
    if os.name != "nt":
        cmd.append("-lm")
    subprocess.run(cmd, check=True)
    return exe


def parse_c_output(text: str):
    """Extract {'plain': {...}, 'res': {...}, ...} from the FINAL lines."""
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
        print(f"  {name:10s}  C={cv:.17g}  Py={pv:.17g}  {'ok' if good else 'MISMATCH'}")
        if not good:
            print(f"    -> diff={abs(cv - pv):.3e}")
        ok = ok and good
    return ok


def check_agreement():
    """Gate 1: C and Python train both toy twins (and run BN + the conv block)
    identically."""
    exe = compile_c()
    proc = subprocess.run([exe], capture_output=True, text=True, check=True)
    c = parse_c_output(proc.stdout)
    for key in ("plain", "res", "bn", "block"):
        assert key in c, f"missing FINAL {key} line:\n{proc.stdout}"

    ok = True
    print("== C <-> Python agreement (34-layer PLAIN net on deep spirals: BN, full-batch "
          "SGD + momentum) ==")
    ok &= compare("plain", c["plain"], py.run_toy(residual=False))
    print("== C <-> Python agreement (its RESIDUAL twin, same initial weights) ==")
    ok &= compare("res", c["res"], py.run_toy(residual=True))
    print("== C <-> Python agreement (BatchNorm over NCHW maps, train + eval) ==")
    ok &= compare("bn", c["bn"], py.bn_fingerprint())
    print("== C <-> Python agreement (downsampling conv residual block + global avg pool) ==")
    ok &= compare("block", c["block"], py.block_fingerprint())
    if ok:
        print(f"  -> all values agree within tol={TOL:g}")
    return ok


def _grad_line(name, analytic, numeric, tol=GRAD_TOL):
    err = rel_error(analytic, numeric)
    good = err < tol
    print(f"  {name:34s} rel_error={err:.3e}  {'ok' if good else 'MISMATCH'}")
    return good


def _exact_line(name, a, b, tol=EXACT_TOL):
    err = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
    good = err < tol
    print(f"  {name:34s} max|diff|={err:.3e}  {'ok' if good else 'MISMATCH'}")
    return good


def _check_layer(name, layer, X, rng, wrt=()):
    """Finite-difference check of one layer's dX (and of the named
    parameters' gradients) against an upstream gradient dY."""
    Y = layer.forward(X)
    dY = rng.standard_normal(Y.shape)
    dX = layer.backward(dY)
    ok = _grad_line(f"{name} dX", dX,
                    eval_numerical_gradient_array(lambda _: layer.forward(X), X, dY))
    for pname, p, g in wrt:
        layer.forward(X)
        layer.backward(dY)
        ok &= _grad_line(f"{name} d{pname}", g(),
                         eval_numerical_gradient_array(lambda _: layer.forward(X), p, dY))
    return ok


def check_gradients():
    """Gate 2: the new layers' backwards match finite differences, and
    BatchNorm matches its definition."""
    print("\n== finite-difference gradient check (ResNet machinery) ==")
    ok = True
    rng = np.random.default_rng(0)

    # (a) BatchNorm on dense (m, C) and conv (n, C, H, W) inputs, both modes.
    for shape in ((6, 4), (3, 4, 5, 5)):
        for training in (True, False):
            bn = BatchNorm(4)
            bn.gamma[:] = rng.standard_normal(4)
            bn.beta[:] = rng.standard_normal(4)
            bn.running_mean[:] = rng.standard_normal(4)
            bn.running_var[:] = rng.random(4) + 0.5
            bn.training = training
            X = rng.standard_normal(shape) * 2.0 + 1.0
            tag = f"BatchNorm {len(shape)}-D {'train' if training else 'eval '}"
            ok &= _check_layer(tag, bn, X, rng,
                               wrt=(("gamma", bn.gamma, lambda bn=bn: bn.dgamma),
                                    ("beta", bn.beta, lambda bn=bn: bn.dbeta)))

    # ...and against its definition: Algorithm 1 forward, the full Jacobian
    # backward, and the running-statistics update (unbiased variance).
    bn = BatchNorm(3, momentum=0.1)
    bn.gamma[:] = rng.standard_normal(3)
    bn.beta[:] = rng.standard_normal(3)
    X = rng.standard_normal((4, 3, 3, 3)) * 1.5 + 0.5
    dY = rng.standard_normal(X.shape)
    Y = bn.forward(X)
    ok &= _exact_line("  BN fwd vs Algorithm 1", Y, py.naive_batchnorm_forward(X, bn.gamma, bn.beta))
    ok &= _exact_line("  BN bwd vs full Jacobian", bn.backward(dY),
                      py.naive_batchnorm_backward(X, bn.gamma, dY))
    m = X.size // 3
    mu, var = X.mean(axis=(0, 2, 3)), X.var(axis=(0, 2, 3))
    ok &= _exact_line("  BN running mean", bn.running_mean, 0.1 * mu)
    ok &= _exact_line("  BN running var (unbiased)", bn.running_var, 0.9 + 0.1 * var * m / (m - 1))

    # (b) Residual block with the identity shortcut (dense, as in the toy).
    body = [Linear(5, 5, bias=False), BatchNorm(5), ReLU(), Linear(5, 5, bias=False), BatchNorm(5)]
    blk = Residual(body)
    for p in blk.params():
        p[...] = rng.standard_normal(p.shape)
    Xr = rng.standard_normal((7, 5))
    ok &= _check_layer("Residual (identity) ", blk, Xr, rng,
                       wrt=(("W1", body[0].W, lambda: body[0].dW),
                            ("gamma2", body[4].gamma, lambda: body[4].dgamma)))

    # (c) A downsampling conv block: option-(A) shortcut + global average pool
    # -- the block_fingerprint network.
    net = py.build_test_block(Rng(py.BLOCK_SEED))
    Xb = rng.standard_normal((3, 2, 6, 6))
    Yb = net.forward(Xb)
    dYb = rng.standard_normal(Yb.shape)
    dXb = net.backward(dYb)
    ok &= _grad_line("Residual conv/2 + pad shortcut dX", dXb,
                     eval_numerical_gradient_array(lambda _: net.forward(Xb), Xb, dYb, h=NET_H))
    for k, (p, g) in enumerate(zip(net.params(), net.grads())):
        net.forward(Xb)
        net.backward(dYb)
        ok &= _grad_line(f"  ...param {k} {tuple(p.shape)}", g,
                         eval_numerical_gradient_array(lambda _: net.forward(Xb), p, dYb, h=NET_H))

    # (d) The new parameter-free ops alone, and bias-free layers.
    ok &= _check_layer("ZeroPadShortcut 3->6 /2", ZeroPadShortcut(3, 6, 2),
                       rng.standard_normal((2, 3, 5, 5)), rng)
    ok &= _check_layer("GlobalAvgPool2D", GlobalAvgPool2D(), rng.standard_normal((2, 3, 4, 5)), rng)
    lin = Linear(4, 3, bias=False)
    lin.W[:] = rng.standard_normal(lin.W.shape)
    ok &= _check_layer("Linear bias=False", lin, rng.standard_normal((5, 4)), rng,
                       wrt=(("W", lin.W, lambda: lin.dW),))
    conv = Conv2D(2, 3, 3, 3, stride=2, pad=1, bias=False)
    conv.K[:] = rng.standard_normal(conv.K.shape)
    ok &= _check_layer("Conv2D bias=False", conv, rng.standard_normal((2, 2, 6, 6)), rng,
                       wrt=(("K", conv.K, lambda: conv.dK),))
    assert len(lin.params()) == 1 and len(conv.params()) == 1

    # (e) Small toy nets end to end through softmax cross-entropy: every
    # parameter's hand-written gradient vs the numerical one, both twins.
    Xs, ys = py.make_spirals(32)
    for residual in (False, True):
        tnet = py.build_toy_net(depth=3, residual=residual, width=6)
        loss_layer = SoftmaxCrossEntropy()

        def loss():
            tnet.train()
            return loss_layer.forward(tnet.forward(Xs), ys)

        loss()
        tnet.backward(loss_layer.backward())
        worst = max(rel_error(g, eval_numerical_gradient(lambda _: loss(), p))
                    for p, g in zip(tnet.params(), tnet.grads()))
        good = worst < GRAD_TOL
        print(f"  {'toy ' + ('residual' if residual else 'plain') + ' net, all ' + str(len(tnet.params())) + ' params':34s} "
              f"worst rel_error={worst:.3e}  {'ok' if good else 'MISMATCH'}")
        ok &= good

    if ok:
        print(f"  -> all gradients match numerically within {GRAD_TOL:g}; "
              f"BatchNorm equals its definition within {EXACT_TOL:g}")
    return ok


def check_paths_agree():
    """Gate 3: vectorized nanograd and the C-order ToyResNet agree on the
    initial loss and gradients (both twins) and, for the residual twin, on
    every parameter after three training steps."""
    print("\n== nanograd (NumPy) vs C-order ToyResNet ==")
    X, y = py.make_spirals()
    ok = True
    for residual in (False, True):
        name = "res  " if residual else "plain"
        mirror = py.ToyResNet(Rng(py.TOY_SEED), X, y, residual)
        vnet = py.build_toy_net(py.DEPTH, residual, Rng(py.TOY_SEED))
        loss_layer = SoftmaxCrossEntropy()
        loss_m, grads_m = mirror.loss_and_grads()
        vnet.train()
        loss_v = loss_layer.forward(vnet.forward(X), y)
        vnet.backward(loss_layer.backward())
        err_l = rel_error(loss_m, loss_v)
        err_g = max(rel_error(a, b) for a, b in zip(grads_m, vnet.grads()))
        good = err_l < 1e-9 and err_g < 1e-9
        print(f"  {name} step 0: loss rel_error={err_l:.3e}  worst grad rel_error={err_g:.3e}  "
              f"{'ok' if good else 'MISMATCH'}")
        ok &= good

        # Three SGD steps on each path, from the same start.
        mirror = py.ToyResNet(Rng(py.TOY_SEED), X, y, residual)
        vnet = py.build_toy_net(py.DEPTH, residual, Rng(py.TOY_SEED))
        opt = SGD(lr=py.TOY_LR, momentum=py.TOY_MOM)
        for _ in range(3):
            mirror.step()
            vnet.train()
            loss_layer.forward(vnet.forward(X), y)
            vnet.backward(loss_layer.backward())
            opt.step(vnet.params(), vnet.grads())
        err = max(rel_error(a, b) for a, b in zip(mirror.params(), vnet.params()))
        if residual:
            good = err < 1e-9
            print(f"  {name} after 3 steps: worst param rel_error={err:.3e}  "
                  f"{'ok' if good else 'MISMATCH'}")
            ok &= good
        else:
            print(f"  {name} after 3 steps: worst param rel_error={err:.3e}  (not gated: the "
                  f"plain net's exploding gradients amplify rounding differences)")
    return ok


def main():
    a = check_agreement()
    g = check_gradients()
    p = check_paths_agree()
    if a and g and p:
        print("\nOK: C<->Python agree, the ResNet gradients are correct, "
              "and the two Python paths match.")
        sys.exit(0)
    print("\nFAIL: see mismatches above.")
    sys.exit(1)


if __name__ == "__main__":
    main()
