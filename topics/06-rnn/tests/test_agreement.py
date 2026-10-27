"""
Agreement + gradient tests for Module 06 (RNN).

Three gates, all required to pass:

1. C <-> Python agreement. Compile the C toy (linking the shared nanograd
   library, now with the recurrent layer and gradient clipping), run it,
   parse its FINAL lines, run the Python mirror (ToyRNN) on the same seeds,
   and assert every tracked value agrees. Both train the same 16-unit RNN on
   the echo task twice -- full backpropagation through time, then Elman's
   one-step truncation, from the same initial weights on the same batches --
   in the same op order from the same LCG streams, with the C library's tanh,
   exp and log on both sides, so they match to the last digit in practice,
   comfortably inside the 1e-9 gate. Two more FINAL lines pin the C RNN
   forward/backward (initial state, input gradients, truncation) and
   gradient clipping to nanograd's.

2. Finite-difference gradient check of the NEW nanograd machinery -- the
   honesty gate. The RNN layer's dX, dh0, dWxh, dWhh and dbh against finite
   differences; its truncated backward against what truncation MEANS (the
   sum of full-BPTT gradients of the chunks, each starting from the state the
   previous chunk left, held fixed); clip_grad_norm against its definition;
   and small echo and counting networks end to end through softmax
   cross-entropy, every parameter.

3. nanograd (NumPy) vs the C-order ToyRNN. The vectorized library and the
   mirror must compute the same initial loss and gradients for both twins,
   and the same parameters after three Adam steps on the same batches.

Run:
    python topics/06-rnn/tests/test_agreement.py

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

TOL = 1e-9         # C<->Python: same op order, same libm.
GRAD_TOL = 1e-6    # analytic vs numerical gradient: a correct derivation is ~1e-8.
EXACT_TOL = 1e-12  # fast path vs definition: same arithmetic, different order.

# Make the Python mirror, the shared library, and the check utilities importable.
sys.path.insert(0, PY_DIR)
sys.path.insert(0, LIB_PY)
sys.path.insert(0, HERE)
import rnn as py  # noqa: E402
from check_utils import (  # noqa: E402
    eval_numerical_gradient,
    eval_numerical_gradient_array,
    rel_error,
)
from nanograd import RNN, Adam, SoftmaxCrossEntropy, clip_grad_norm  # noqa: E402
from nanograd.rng import Rng  # noqa: E402

C_SOURCES = ("rng.c", "nn.c", "init.c", "optim.c", "rnn.c")


def compile_c() -> str:
    """Compile the C toy RNN + nanograd sources; return the binary path."""
    cc = shutil.which("clang") or shutil.which("gcc") or shutil.which("cc")
    if cc is None:
        print("SKIP: no C compiler (clang/gcc/cc) found on PATH")
        sys.exit(0)
    out_dir = tempfile.mkdtemp(prefix="rnn_")
    # The exe is run after this function returns, so clean up at exit.
    atexit.register(shutil.rmtree, out_dir, ignore_errors=True)
    exe = os.path.join(out_dir, "rnn.exe" if os.name == "nt" else "rnn")
    sources = [os.path.join(C_DIR, "rnn.c")] + [
        os.path.join(LIB_C, f) for f in C_SOURCES]
    cmd = [cc, "-O2", "-std=c11", "-Wall", "-Wextra", "-I", LIB_C, *sources, "-o", exe]
    # On Windows math is in the CRT; only POSIX toolchains need -lm.
    if os.name != "nt":
        cmd.append("-lm")
    subprocess.run(cmd, check=True)
    return exe


def parse_c_output(text: str):
    """Extract {'bptt': {...}, 'elman': {...}, ...} from the FINAL lines."""
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
    """Gate 1: C and Python train both echo twins (and run the RNN + clip
    checks) identically."""
    exe = compile_c()
    proc = subprocess.run([exe], capture_output=True, text=True, check=True)
    c = parse_c_output(proc.stdout)
    for key in ("bptt", "elman", "rnn", "clip"):
        assert key in c, f"missing FINAL {key} line:\n{proc.stdout}"

    ok = True
    print("== C <-> Python agreement (echo, delay 4: full BPTT, Adam + clipping) ==")
    ok &= compare("bptt", c["bptt"], py.run_toy(truncate=0))
    print("== C <-> Python agreement (its ELMAN twin: one-step truncation, same weights) ==")
    ok &= compare("elman", c["elman"], py.run_toy(truncate=1))
    print("== C <-> Python agreement (RNN forward/backward: h0, dX, truncation) ==")
    ok &= compare("rnn", c["rnn"], py.rnn_fingerprint())
    print("== C <-> Python agreement (gradient-norm clipping) ==")
    ok &= compare("clip", c["clip"], py.clip_fingerprint())
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


def _random_rnn(rng, n_in, n_h, scale=0.6):
    layer = RNN(n_in, n_h)
    for p in layer.params():
        p[...] = rng.standard_normal(p.shape) * scale
    return layer


def check_gradients():
    """Gate 2: the RNN's backward matches finite differences, truncation
    matches chunked training, clipping matches its definition, and small
    networks are correct end to end."""
    print("\n== finite-difference gradient check (RNN machinery) ==")
    ok = True
    rng = np.random.default_rng(0)

    # (a) The RNN layer with an initial state: every input and parameter.
    T, m, n_in, n_h = 7, 3, 4, 5
    layer = _random_rnn(rng, n_in, n_h)
    X = rng.standard_normal((T, m, n_in))
    h0 = rng.standard_normal((m, n_h)) * 0.5
    dH = rng.standard_normal((T, m, n_h))

    def f(_):
        return layer.forward(X, h0)

    f(None)
    dX = layer.backward(dH)
    ok &= _grad_line("RNN dX", dX, eval_numerical_gradient_array(f, X, dH))
    f(None)
    layer.backward(dH)
    ok &= _grad_line("RNN dh0", layer.dh0, eval_numerical_gradient_array(f, h0, dH))
    for name, p, g in (("Wxh", layer.Wxh, lambda: layer.dWxh),
                       ("Whh", layer.Whh, lambda: layer.dWhh),
                       ("bh", layer.bh, lambda: layer.dbh)):
        f(None)
        layer.backward(dH)
        ok &= _grad_line(f"RNN d{name}", g().copy(), eval_numerical_gradient_array(f, p, dH))

    # (b) Truncation is chunked training: the truncated backward equals the sum
    # of each chunk's full-BPTT gradients, each chunk starting from the state
    # the previous one left (a constant -- no gradient flows into it).
    for k in (1, 2, 3):
        H = layer.forward(X, h0)
        layer.backward(dH, truncate=k)
        got = [g.copy() for g in layer.grads()]
        want = [np.zeros_like(g) for g in got]
        piece = RNN(n_in, n_h)
        piece.Wxh, piece.Whh, piece.bh = layer.Wxh, layer.Whh, layer.bh
        for s in range(0, T, k):
            piece.forward(X[s:s + k], h0 if s == 0 else H[s - 1])
            piece.backward(dH[s:s + k])
            want = [w + g for w, g in zip(want, piece.grads())]
        err = max(float(np.max(np.abs(a - b))) for a, b in zip(got, want))
        good = err < EXACT_TOL and np.all(layer.dh0 == 0.0)
        print(f"  {'truncate=' + str(k) + ' == chunked BPTT':34s} max|diff|={err:.3e}  "
              f"{'ok' if good else 'MISMATCH'}")
        ok &= good

    # (c) clip_grad_norm: the norm, the direction, the clipped length.
    gs = [rng.standard_normal(6), rng.standard_normal((3, 4))]
    flat = np.concatenate([g.reshape(-1) for g in gs])
    norm = clip_grad_norm(gs, 0.5)
    after = np.concatenate([g.reshape(-1) for g in gs])
    ok &= _exact_line("  clip: reported norm", norm, np.linalg.norm(flat))
    ok &= _exact_line("  clip: new norm = max_norm", np.linalg.norm(after), 0.5)
    ok &= _exact_line("  clip: direction kept", after / 0.5, flat / np.linalg.norm(flat))
    before = [g.copy() for g in gs]
    clip_grad_norm(gs, 10.0)
    ok &= _exact_line("  clip: below max_norm is a no-op", np.concatenate(
        [g.reshape(-1) for g in gs]), np.concatenate([g.reshape(-1) for g in before]))

    # (d) Small networks end to end: every parameter's hand-written gradient
    # vs the numerical one, through the read-out and softmax cross-entropy.
    loss_layer = SoftmaxCrossEntropy()
    rnn, head = py.build_echo_net(Rng(5), 4, 5, 4)
    S = np.random.default_rng(1).integers(0, 4, size=(9, 3))
    Xe = py.one_hot(S)

    def echo_loss():
        return py.echo_loss_and_grads(rnn, head, Xe, S, delay=2, loss_layer=loss_layer)

    echo_loss()
    worst = max(rel_error(g, eval_numerical_gradient(lambda _: echo_loss(), p))
                for p, g in zip(rnn.params() + head.params(),
                                [g.copy() for g in rnn.grads() + head.grads()]))
    good = worst < GRAD_TOL
    print(f"  {'echo net (delay 2), all 5 params':34s} worst rel_error={worst:.3e}  "
          f"{'ok' if good else 'MISMATCH'}")
    ok &= good

    rnn, head = py.build_echo_net(Rng(6), 2, 5, 3)
    Xc, yc = py.counting_batch(np.random.default_rng(2), 3, 8)

    def count_loss():
        H = rnn.forward(Xc)
        L = loss_layer.forward(head.forward(H.reshape(-1, 5)), yc.reshape(-1))
        rnn.backward(head.backward(loss_layer.backward()).reshape(H.shape))
        return L

    count_loss()
    worst = max(rel_error(g, eval_numerical_gradient(lambda _: count_loss(), p))
                for p, g in zip(rnn.params() + head.params(),
                                [g.copy() for g in rnn.grads() + head.grads()]))
    good = worst < GRAD_TOL
    print(f"  {'counting net (mod 3), all 5 params':34s} worst rel_error={worst:.3e}  "
          f"{'ok' if good else 'MISMATCH'}")
    ok &= good

    if ok:
        print(f"  -> all gradients match numerically within {GRAD_TOL:g}; truncation and "
              f"clipping equal their definitions within {EXACT_TOL:g}")
    return ok


def check_paths_agree():
    """Gate 3: vectorized nanograd and the C-order ToyRNN agree on the initial
    loss and gradients and on every parameter after three Adam steps, for
    both twins."""
    print("\n== nanograd (NumPy) vs C-order ToyRNN ==")
    ok = True
    for truncate in (0, 1):
        name = "elman" if truncate else "bptt "
        mirror = py.ToyRNN(Rng(py.TOY_SEED), truncate)
        rnn, head = py.build_echo_net()
        S = py.draw_symbols(Rng(py.DATA_SEED), py.M_TOY, py.T_TOY)
        X = py.one_hot(S)
        loss_m, grads_m = mirror.loss_and_grads(X, S)
        loss_v = py.echo_loss_and_grads(rnn, head, X, S, py.DELAY, truncate)
        err_l = rel_error(loss_m, loss_v)
        err_g = max(rel_error(a, b) for a, b in zip(grads_m, rnn.grads() + head.grads()))
        good = err_l < 1e-9 and err_g < 1e-9
        print(f"  {name} step 0: loss rel_error={err_l:.3e}  worst grad rel_error={err_g:.3e}  "
              f"{'ok' if good else 'MISMATCH'}")
        ok &= good

        # Three Adam steps on each path, on the same batches.
        mirror = py.ToyRNN(Rng(py.TOY_SEED), truncate)
        rnn, head = py.build_echo_net()
        opt = Adam(lr=py.TOY_LR, beta1=py.ADAM_B1, beta2=py.ADAM_B2, eps=py.ADAM_EPS)
        r_m, r_v = Rng(py.DATA_SEED), Rng(py.DATA_SEED)
        for _ in range(3):
            mirror.step(r_m)
            S = py.draw_symbols(r_v, py.M_TOY, py.T_TOY)
            py.echo_loss_and_grads(rnn, head, py.one_hot(S), S, py.DELAY, truncate)
            grads = rnn.grads() + head.grads()
            clip_grad_norm(grads, py.CLIP)
            opt.step(rnn.params() + head.params(), grads)
        err = max(rel_error(a, b) for a, b in zip(mirror.params(), rnn.params() + head.params()))
        good = err < 1e-9
        print(f"  {name} after 3 steps: worst param rel_error={err:.3e}  "
              f"{'ok' if good else 'MISMATCH'}")
        ok &= good
    return ok


def main():
    a = check_agreement()
    g = check_gradients()
    p = check_paths_agree()
    if a and g and p:
        print("\nOK: C<->Python agree, the RNN gradients are correct, "
              "and the two Python paths match.")
        sys.exit(0)
    print("\nFAIL: see mismatches above.")
    sys.exit(1)


if __name__ == "__main__":
    main()
